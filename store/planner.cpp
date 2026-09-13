#include "store/planner.h"

#include <algorithm>
#include <format>

#include "core/log.h"
#include "model/layout.h"

namespace deepmoe::store {

std::string PlannerStats::to_string() const {
    return std::format(
        "planner: {} layers, {} requests ({} hits, {} misses, {:.3f}), "
        "{} evictions (+{} refused), prefetch {} issued / {} used / {} wasted, {} stalls",
        layers_planned, requests, hits, misses, hit_rate(),
        evictions, evict_failures, prefetch_issued, prefetch_used, prefetch_wasted, stall_waits);
}

namespace {

// design §9.3 baseline: evict the slot whose last_use_token is smallest.
class LruPolicy final : public EvictionPolicy {
public:
    const char* name() const override { return "lru"; }
    std::vector<uint32_t> choose_victims(const std::vector<ExpertSlot>& c, size_t count) override {
        std::vector<uint32_t> order(c.size());
        for (size_t i = 0; i < c.size(); ++i) order[i] = static_cast<uint32_t>(i);
        const size_t n = std::min(count, c.size());
        std::partial_sort(order.begin(), order.begin() + static_cast<long>(n), order.end(),
                          [&](uint32_t a, uint32_t b) {
                              if (c[a].last_use_token != c[b].last_use_token)
                                  return c[a].last_use_token < c[b].last_use_token;
                              return c[a].slot < c[b].slot;   // deterministic tie-break
                          });
        std::vector<uint32_t> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) out.push_back(c[order[i]].slot);
        return out;
    }
};

// TODO(design §9.3): rank by (heat, last_use) once tools/cache_sim.py shows
// score-aware beats plain LRU on a real route trace (Q2). Falling back to LRU
// keeps the runtime correct and honest about which policy actually ran.
class ScoreAwarePolicy final : public EvictionPolicy {
public:
    const char* name() const override { return "score-aware-lru(fallback:lru)"; }
    std::vector<uint32_t> choose_victims(const std::vector<ExpertSlot>& c, size_t count) override {
        return lru_.choose_victims(c, count);
    }
private:
    LruPolicy lru_;
};

}  // namespace

std::unique_ptr<EvictionPolicy> make_lru_policy() { return std::make_unique<LruPolicy>(); }
std::unique_ptr<EvictionPolicy> make_score_aware_policy() { return std::make_unique<ScoreAwarePolicy>(); }

std::unique_ptr<EvictionPolicy> make_policy(CachePolicy which) {
    switch (which) {
        case CachePolicy::Lru:            return make_lru_policy();
        case CachePolicy::ScoreAwareLru:  return make_score_aware_policy();
        // TODO(design §9.3): LFU-decay / ARC / static-pin variants, gated on Q2.
        case CachePolicy::Lfu:
        case CachePolicy::StaticPinPlusLru:
            log_warn("planner: policy {} is not implemented, using LRU",
                     static_cast<int>(which));
            return make_lru_policy();
    }
    return make_lru_policy();
}

Result<void> Planner::init(ExpertStore& store, storage::IoEngine& io,
                           const storage::File& experts_file,
                           const CacheConfig& cache, const PrefetchConfig& prefetch,
                           Profiler* profiler) {
    if (!experts_file.is_open()) return fail(Err::InvalidArgument, "experts.bin is not open");
    store_    = &store;
    io_       = &io;
    experts_  = &experts_file;
    cache_    = cache;
    prefetch_ = prefetch;
    profiler_ = profiler;
    policy_   = make_policy(cache.policy);
    log_info("planner: policy '{}', cache {} slots, lookahead d={} K={}",
             policy_->name(), store.slot_count(), prefetch.lookahead_depth, prefetch.lookahead_width);
    return {};
}

uint32_t Planner::reclaim(size_t count) {
    if (count == 0) return 0;
    auto candidates = store_->evictable();
    if (candidates.empty()) return 0;
    // Over-ask a little: a victim can still refuse if its guard advanced
    // between the snapshot and the evict call.
    auto victims = policy_->choose_victims(candidates, std::min(candidates.size(), count * 2));
    uint32_t freed = 0;
    for (uint32_t slot : victims) {
        if (freed >= count) break;
        if (auto r = store_->evict(slot); r) {
            ++freed;
        } else {
            std::lock_guard lk(stats_mutex_);
            ++stats_.evict_failures;
        }
    }
    {
        std::lock_guard lk(stats_mutex_);
        stats_.evictions += freed;
    }
    return freed;
}

Result<storage::IoRequestId> Planner::fetch(ExpertKey key, IoPriority priority,
                                            TokenIndex token, uint32_t deadline_layer,
                                            std::function<void(bool)> on_done) {
    if (!store_ || !io_) return fail(Err::FailedPrecondition, "planner is not initialised");

    auto res = store_->begin_fill(key, Tier::Cached);
    if (!res && res.error().code == Err::ResourceExhausted) {
        if (reclaim(1) == 0)
            return fail(Err::ResourceExhausted,
                        std::format("cache is full and nothing is evictable for ({}, {})",
                                    key.layer, key.expert));
        res = store_->begin_fill(key, Tier::Cached);
    }
    if (!res) return std::unexpected(res.error());

    const uint32_t slot = res->slot;
    // design §5.1: experts.bin is addressed arithmetically, layer-major, so a
    // whole layer is 7.22 GB of contiguous blocks for the prefill stream.
    storage::IoRequest req;
    req.key            = key;
    req.priority       = priority;
    req.file           = experts_;
    req.file_off       = layout::expert_file_offset(key.layer, key.expert);
    req.bytes          = layout::kExpertBytes;
    req.dst            = res->addr.host_ptr;
    req.issue_token    = token;
    req.deadline_layer = deadline_layer;

    ExpertStore* store = store_;
    auto id = io_->submit(req, [store, slot, token, cb = std::move(on_done)](const storage::IoResult& r) {
        // Runs on the IoEngine dispatcher thread: publish the slot and hand
        // control straight back (design §9.6).
        (void)store->finish_fill(slot, r.ok(), token);
        if (cb) cb(r.ok());
    });
    if (!id) {
        (void)store_->finish_fill(slot, false, token);
        return std::unexpected(id.error());
    }
    if (priority == IoPriority::Lookahead || priority == IoPriority::Backfill) {
        std::lock_guard lk(stats_mutex_);
        ++stats_.prefetch_issued;
        if (profiler_) profiler_->note_prefetch_issued();
    }
    return *id;
}

Result<LayerPlan> Planner::plan_layer(const RouteDecision& route, TokenIndex token) {
    if (!store_ || !io_) return fail(Err::FailedPrecondition, "planner is not initialised");

    LayerPlan plan;
    plan.layer = route.layer;
    plan.hits.reserve(route.chosen.size());
    plan.misses.reserve(route.chosen.size());

    // design §9.3: the top-16 "near miss" scores keep an expert warm even when
    // it was not selected, so a bursty expert survives one bad round.
    for (size_t i = 0; i < route.near_ids.size() && i < route.near_scores.size(); ++i)
        store_->note_heat(ExpertKey{static_cast<uint16_t>(route.layer), route.near_ids[i]},
                          route.near_scores[i]);

    for (uint16_t id : route.chosen) {
        const ExpertKey key{static_cast<uint16_t>(route.layer), id};
        if (store_->lookup(key, token)) {
            plan.hits.push_back(key);
            if (profiler_) profiler_->note_expert_lookup(true);
        } else {
            plan.misses.push_back(key);
            if (profiler_) profiler_->note_expert_lookup(false);
        }
    }

    // Free the room for every miss up front, so the fills do not serialise
    // behind one eviction each.
    if (!plan.misses.empty()) {
        const uint32_t free_now = store_->free_slots();
        if (free_now < plan.misses.size()) reclaim(plan.misses.size() - free_now);
    }

    for (ExpertKey key : plan.misses) {
        auto id = fetch(key, IoPriority::BlockingMiss, token, route.layer);
        if (!id) {
            log_warn("planner: P0 fetch of ({}, {}) failed: {}", key.layer, key.expert,
                     id.error().str());
            continue;
        }
        plan.issued.push_back(*id);
        plan.miss_bytes += layout::kExpertBytes;
    }

    {
        std::lock_guard lk(stats_mutex_);
        ++stats_.layers_planned;
        stats_.requests += route.chosen.size();
        stats_.hits     += plan.hits.size();
        stats_.misses   += plan.misses.size();
        if (!plan.misses.empty()) ++stats_.stall_waits;
    }
    return plan;
}

Result<void> Planner::pin(ExpertKey key) {
    if (!store_) return fail(Err::FailedPrecondition, "planner is not initialised");
    auto s = store_->slot_for(key);
    if (!s) return fail(Err::NotFound, std::format("expert ({}, {}) is not held", key.layer, key.expert));
    if (s->state != SlotState::Resident)
        return fail(Err::FailedPrecondition, "cannot pin a slot that is not resident");
    // Tier is set at begin_fill; re-filling as Pinned is the clean path, so
    // this is deliberately a stub rather than a mutation behind the store's back.
    // TODO(design §9.3): give ExpertStore a retier(slot, Tier) once the pinned
    // set is loaded from hot.bin/mtp.bin at startup rather than promoted here.
    return unimplemented("Planner::pin (design §9.3 pinned set is loaded at startup)");
}

// TODO(design §9.4): needs store/predictor.h and the Q4/Q5 answers. Until then
// the runtime runs demand-only, which is exactly the h=0.30 row of design §3.1.
Result<uint32_t> Planner::prefetch_lookahead(uint32_t, TokenIndex) {
    return unimplemented("Planner::prefetch_lookahead (design §9.4, gated on Q4/Q5)");
}

// TODO(design §9.6): idle-time P3 backfill by static heat, gated on Q1.
Result<uint32_t> Planner::backfill(TokenIndex) {
    return unimplemented("Planner::backfill (design §9.6 P3, gated on Q1)");
}

PlannerStats Planner::stats() const {
    std::lock_guard lk(stats_mutex_);
    return stats_;
}

void Planner::reset_stats() {
    std::lock_guard lk(stats_mutex_);
    stats_ = PlannerStats{};
}

}  // namespace deepmoe::store
