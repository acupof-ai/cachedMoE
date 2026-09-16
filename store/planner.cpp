#include "store/planner.h"

#include <algorithm>
#include <atomic>
#include <format>
#include <memory>
#include <cstdio>
#include <fstream>
#include <string_view>

#include "core/log.h"
#include "model/layout.h"

namespace deepmoe::store {

std::string PlannerStats::to_string() const {
    return std::format(
        "planner: {} layers, {} requests ({} hits, {} misses, {:.3f}), "
        "{} evictions (+{} refused), prefetch {} issued / {} used / {} wasted, {} stalls, "
        "{} joined; backfill {} issued / {} done / {} failed; prefill handoff {} resident / "
        "{} kept / {} dropped",
        layers_planned, requests, hits, misses, hit_rate(),
        evictions, evict_failures, prefetch_issued, prefetch_used, prefetch_wasted, stall_waits,
        joined_fills, backfill_issued, backfill_done, backfill_failed, streamed_resident,
        streamed_filled, streamed_dropped);
}

Result<StreamAdmit> admit_streamed(ExpertStore& store, ExpertKey key, TokenIndex stamp,
                                   const ExpertEntry* entry, TimelineValue guard) {
    StreamAdmit out;
    // Held already: a resident key only gets newer; one the backfill is still
    // filling is waited for (a few ms) rather than read twice.
    if (auto slot = store.slot_of(key)) {
        if (!store.touch(key, stamp)) {
            if (store.wait_settled(key, std::chrono::seconds(60))) (void)store.touch(key, stamp);
        }
        if (store.resident(key)) {
            if (auto r = store.set_guard(*slot, guard); !r) return std::unexpected(r.error());
            auto info = store.slot_info(*slot);
            out.kind = StreamKind::Resident;
            out.slot = *slot;
            out.addr = SlotAddress{info->host_ptr, info->dev_addr};
            return out;
        }
    }
    if (store.free_slots() == 0) {
        const auto oldest = store.oldest_evictable_stamp();
        if (!oldest || stamp <= *oldest) return out;          // Drop
        if (auto v = store.evict_lru(); !v) return out;
    }
    auto res = entry ? store.begin_fill(key, *entry, Tier::Cached) : store.begin_fill(key, Tier::Cached);
    if (!res) {
        if (res.error().code == Err::ResourceExhausted) return out;
        return std::unexpected(res.error());
    }
    out.kind = StreamKind::Fill;
    out.slot = res->slot;
    out.addr = res->addr;
    return out;
}

Result<void> finish_streamed(ExpertStore& store, const StreamAdmit& a, bool ok, TokenIndex stamp) {
    if (a.kind != StreamKind::Fill) return {};
    return store.finish_fill(a.slot, ok, stamp);
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

namespace {
struct HeatRow { uint16_t layer, expert; uint32_t count; };
constexpr HeatRow kStaticHeat[] = {
#include "store/static_heat.inc"
};
}  // namespace

std::vector<ExpertKey> static_heat_order() {
    std::vector<ExpertKey> out;
    out.reserve(std::size(kStaticHeat));
    for (const HeatRow& r : kStaticHeat) out.push_back(ExpertKey{r.layer, r.expert});
    return out;
}

std::vector<ExpertKey> static_heat_order(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        log_warn("planner: heat file '{}' not readable; built-in order", path);
        return {};
    }
    std::vector<HeatRow> rows;
    std::string line;
    while (std::getline(f, line)) {
        unsigned int layer = 0, expert = 0, count = 0;
        if (std::sscanf(line.c_str(), " { %u , %u , %u }", &layer, &expert, &count) == 3 ||
            std::sscanf(line.c_str(), "%u, %u, %u", &layer, &expert, &count) == 3) {
            if (layer < 4096 && expert < 4096)
                rows.push_back(HeatRow{static_cast<uint16_t>(layer), static_cast<uint16_t>(expert), count});
        }
    }
    std::stable_sort(rows.begin(), rows.end(),
                     [](const HeatRow& a, const HeatRow& b) { return a.count > b.count; });
    std::vector<ExpertKey> out;
    out.reserve(rows.size());
    for (const HeatRow& r : rows) out.push_back(ExpertKey{r.layer, r.expert});
    if (out.empty()) log_warn("planner: heat file '{}' has no rows; built-in order", path);
    return out;
}

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
                           const Manifest& manifest, const ShardSet& shards,
                           const CacheConfig& cache, const PrefetchConfig& prefetch,
                           Profiler* profiler) {
    if (shards.empty()) return fail(Err::InvalidArgument, "no safetensors shards are open");
    if (shards.size() != manifest.files().size())
        return fail(Err::InvalidArgument,
                    std::format("{} shards open but the manifest lists {}",
                                shards.size(), manifest.files().size()));
    store_    = &store;
    io_       = &io;
    manifest_ = &manifest;
    shards_   = &shards;
    cache_    = cache;
    prefetch_ = prefetch;
    profiler_ = profiler;
    policy_   = make_policy(cache.policy);
    log_info("planner: policy '{}', cache {} slots, {} shards, lookahead d={} K={}",
             policy_->name(), store.slot_count(), shards.size(),
             prefetch.lookahead_depth, prefetch.lookahead_width);
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

namespace {
// Shared between the N run callbacks of one expert fetch: the last one to
// report calls the user's callback, and any failure poisons the result.
struct FetchState {
    std::function<void(bool)> cb;
    std::atomic<bool>         ok{true};
};
}  // namespace

Result<Planner::Fetch> Planner::fetch(ExpertKey key, IoPriority priority,
                                      TokenIndex token, uint32_t deadline_layer,
                                      std::function<void(bool)> on_done, TokenIndex stamp_in) {
    if (!store_ || !io_ || !manifest_ || !shards_)
        return fail(Err::FailedPrecondition, "planner is not initialised");

    // design §5.1 (v0.5): the expert's address is looked up, not computed --
    // there is no experts.bin any more. One IoRequest per run; the manifest has
    // already widened each to 4 KiB boundaries, so the I/O layer sees nothing
    // unusual.
    auto entry = manifest_->require_expert(key);
    if (!entry) return std::unexpected(entry.error());

    auto res = store_->begin_fill(key, **entry, Tier::Cached);
    if (!res && res.error().code == Err::ResourceExhausted) {
        if (reclaim(1) == 0)
            return fail(Err::ResourceExhausted,
                        std::format("cache is full and nothing is evictable for ({}, {})",
                                    key.layer, key.expert));
        res = store_->begin_fill(key, **entry, Tier::Cached);
    }
    if (!res) return std::unexpected(res.error());

    const uint32_t slot = res->slot;
    ExpertStore* store  = store_;
    // The slot's LRU age is the moment it was ASKED for, not the moment its
    // last run lands: that is when the simulator's `admit` touches it.
    const TokenIndex stamp = stamp_in ? stamp_in : next_stamp();
    auto state = std::make_shared<FetchState>();
    state->cb = std::move(on_done);

    Fetch out;
    out.slot = slot;
    out.ids.reserve(res->run_count);

    Status first_error{Err::Ok};
    for (uint32_t i = 0; i < res->run_count; ++i) {
        const FillRun& run = res->runs[i];
        auto file = shards_->require(run.file);
        if (!file) { first_error = file.error(); break; }

        storage::IoRequest req;
        req.key            = key;
        req.priority       = priority;
        req.file           = *file;
        req.file_off       = run.file_off;   // 4 KiB aligned by construction
        req.bytes          = run.bytes;      // 4 KiB multiple by construction
        req.dst            = run.dst;        // slot_base + slot_offset, aligned
        req.issue_token    = token;
        req.deadline_layer = deadline_layer;

        auto id = io_->submit(req, [store, slot, stamp, state](const storage::IoResult& r) {
            // Runs on the IoEngine dispatcher thread: count the run in, publish
            // the slot when it was the last one, hand control straight back
            // (design §9.6, docs/architecture.md §2.2).
            if (!r.ok()) {
                state->ok.store(false, std::memory_order_relaxed);
                // Without this a failed read releases the slot silently and the
                // first anyone hears of it is "not resident" at the MoE dispatch.
                log_warn("planner: a run of expert ({}, {}) failed into slot {}: {}",
                         r.key.layer, r.key.expert, slot, r.status.message);
            }
            auto settled = store->finish_run(slot, r.ok(), stamp);
            if (settled && *settled && state->cb)
                state->cb(state->ok.load(std::memory_order_relaxed));
        });
        if (!id) { first_error = id.error(); break; }
        out.ids.push_back(*id);
        out.bytes += run.bytes;
    }

    if (first_error.code != Err::Ok) {
        // Account for the runs that were never handed to the engine, so the
        // slot cannot sit in Filling for ever. The ones already in flight still
        // report on their own.
        state->ok.store(false, std::memory_order_relaxed);
        for (uint32_t i = static_cast<uint32_t>(out.ids.size()); i < res->run_count; ++i)
            (void)store_->finish_run(slot, false, stamp);
        return std::unexpected(first_error);
    }

    if (priority == IoPriority::Lookahead || priority == IoPriority::Backfill) {
        std::lock_guard lk(stats_mutex_);
        ++stats_.prefetch_issued;
        if (profiler_) profiler_->note_prefetch_issued();
    }
    return out;
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

    // One key at a time, in the gate's order, exactly as tools/cache_sim.py's
    // `simulate` walks a trace: a hit is touched; a miss evicts the oldest
    // resident if the cache is full and is admitted, which also touches it.
    // Doing the evictions for all of a layer's misses up front -- what this
    // did before -- decides victims before the layer's later hits have been
    // touched, and a key that is both this layer's hit and the cache's oldest
    // would be thrown out under it. The fills themselves still run
    // concurrently; only the bookkeeping is ordered.
    const bool lru = std::string_view(policy_->name()) == "lru";
    plan.group = std::make_shared<FetchGroup>();
    for (uint16_t id : route.chosen) {
        const ExpertKey key{static_cast<uint16_t>(route.layer), id};
        if (store_->lookup(key, next_stamp())) {
            plan.hits.push_back(key);
            if (profiler_) profiler_->note_expert_lookup(true);
            continue;
        }
        plan.misses.push_back(key);
        if (profiler_) profiler_->note_expert_lookup(false);
        // Already being filled by another class (the P3 backfill): join it.
        if (store_->slot_of(key)) {
            plan.joined.emplace_back(key, next_stamp());
            std::lock_guard lk(stats_mutex_);
            ++stats_.joined_fills;
            continue;
        }
        if (store_->free_slots() == 0) {
            if (lru) {
                if (auto v = store_->evict_lru(); v) {
                    std::lock_guard lk(stats_mutex_);
                    ++stats_.evictions;
                }
            } else {
                reclaim(1);
            }
        }
        {
            std::lock_guard lk(plan.group->m);
            ++plan.group->pending;
        }
        auto f = fetch(key, IoPriority::BlockingMiss, token, route.layer,
                       [g = plan.group](bool ok) {
                           std::lock_guard lk(g->m);
                           --g->pending;
                           g->failed |= !ok;
                           g->cv.notify_all();
                       });
        if (!f) {
            {
                std::lock_guard lk(plan.group->m);
                --plan.group->pending;
            }
            log_warn("planner: P0 fetch of ({}, {}) failed: {}", key.layer, key.expert,
                     f.error().str());
            continue;
        }
        plan.issued.insert(plan.issued.end(), f->ids.begin(), f->ids.end());
        plan.miss_bytes += f->bytes;
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
    // set is loaded at startup from the manifest rather than promoted here.
    return unimplemented("Planner::pin (design §9.3 pinned set is loaded at startup)");
}

// TODO(design §9.4): needs store/predictor.h and the Q4/Q5 answers. Until then
// the runtime runs demand-only, which is exactly the h=0.30 row of design §3.1.
Result<uint32_t> Planner::prefetch_lookahead(uint32_t, TokenIndex) {
    return unimplemented("Planner::prefetch_lookahead (design §9.4, gated on Q4/Q5)");
}

Result<void> Planner::wait_layer(LayerPlan& plan, std::chrono::milliseconds timeout) {
    if (plan.group) {
        std::unique_lock lk(plan.group->m);
        if (!plan.group->cv.wait_for(lk, timeout, [&] { return plan.group->pending == 0; }))
            return fail(Err::Io,
                        std::format("layer {}: {} expert fills still outstanding after {} ms",
                                    plan.layer, plan.group->pending, timeout.count()));
        if (plan.group->failed)
            return fail(Err::Io, std::format("layer {}: an expert read failed", plan.layer));
    }
    for (const auto& [key, stamp] : plan.joined) {
        if (!store_->wait_settled(key, timeout))
            return fail(Err::Io, std::format("layer {}: the fill of expert {} it joined did "
                                             "not land", plan.layer, key.expert));
        (void)store_->touch(key, stamp);
    }
    return {};
}

Result<StreamAdmit> Planner::admit_streamed(ExpertKey key, TokenIndex stamp, TimelineValue guard) {
    if (!store_ || !manifest_) return fail(Err::FailedPrecondition, "planner is not initialised");
    auto entry = manifest_->require_expert(key);
    if (!entry) return std::unexpected(entry.error());
    auto a = store::admit_streamed(*store_, key, stamp, *entry, guard);
    if (a) {
        std::lock_guard lk(stats_mutex_);
        if (a->kind == StreamKind::Resident) ++stats_.streamed_resident;
        else if (a->kind == StreamKind::Fill) ++stats_.streamed_filled;
        else ++stats_.streamed_dropped;
    }
    return a;
}

Result<void> Planner::finish_streamed(const StreamAdmit& a, bool ok, TokenIndex stamp) {
    if (!store_) return fail(Err::FailedPrecondition, "planner is not initialised");
    return store::finish_streamed(*store_, a, ok, stamp);
}

Result<uint32_t> Planner::backfill(TokenIndex) {
    return unimplemented("Planner::backfill(token): use start_backfill(order) (design §9.6 P3)");
}

Result<void> Planner::start_backfill(std::vector<ExpertKey> order, uint32_t inflight) {
    if (!store_ || !io_) return fail(Err::FailedPrecondition, "planner is not initialised");
    stop_backfill();
    if (order.size() >= kDemandStampBase)
        return fail(Err::InvalidArgument, "backfill order is longer than its stamp space");
    auto b = std::make_shared<Backfill>();
    b->order = std::move(order);
    b->max_inflight = std::max<uint32_t>(1, inflight);
    b->active.store(true);
    backfill_ = b;
    backfill_pump(b);
    return {};
}

void Planner::stop_backfill() {
    if (backfill_) backfill_->active.store(false);
}

// Issues backfill fetches until `max_inflight` are out, the order is spent or
// the cache has no free slot. Runs on the caller's thread at start and on the
// IoEngine dispatcher thread from each completion: short, and it never blocks
// (fetch only queues).
void Planner::backfill_pump(const std::shared_ptr<Backfill>& b) {
    for (;;) {
        ExpertKey key{};
        TokenIndex stamp = 0;
        {
            std::lock_guard lk(b->m);
            if (!b->active.load() || b->inflight >= b->max_inflight) return;
            // Never evict for a guess: only free slots.
            if (store_->free_slots() == 0 || b->next >= b->order.size()) {
                if (b->inflight == 0) b->active.store(false);
                return;
            }
            const size_t rank = b->next++;
            key = b->order[rank];
            if (store_->slot_of(key)) continue;          // held already
            stamp = kDemandStampBase - 1 - rank;          // hotter = newer, all below demand
            ++b->inflight;
        }
        auto f = fetch(key, IoPriority::Backfill, 0, 0,
                       [this, b](bool ok) {
                           {
                               std::lock_guard lk(b->m);
                               --b->inflight;
                           }
                           {
                               std::lock_guard lk(stats_mutex_);
                               ++(ok ? stats_.backfill_done : stats_.backfill_failed);
                           }
                           backfill_pump(b);
                       },
                       stamp);
        if (f) {
            std::lock_guard lk(stats_mutex_);
            ++stats_.backfill_issued;
            continue;
        }
        {
            std::lock_guard lb(b->m);
            --b->inflight;
        }
        if (f.error().code == Err::AlreadyExists) continue;   // demand got there first
        {
            std::lock_guard lk(stats_mutex_);
            ++stats_.backfill_failed;
        }
        b->active.store(false);   // the cache filled up under us, or a real error
        return;
    }
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
