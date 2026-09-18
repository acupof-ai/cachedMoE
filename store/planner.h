// ExpertPlanner: residency decisions (design §4.1, §9.3-§9.5).
//
// The Planner is the only module allowed to decide what leaves the cache and
// what is fetched speculatively. Design §16 is explicit that no policy code is
// written before tools/cache_sim.py has answered Q1-Q5, so exactly one policy
// exists here: the LRU baseline of §9.3, implemented, plus a score-aware
// variant whose ordering function is stubbed until the simulator picks it.
//
// The per-layer flow of design §7.8:
//   1. the gate kernel writes ids[6]/weights[6]/top16 into host-coherent memory
//   2. plan_layer() classifies them into hits and misses
//   3. misses become P0 IoRequests; each needs a slot, so victims are evicted
//   4. when the last fill completes the caller host-signals the timeline value
//      the command buffer is waiting on
//
// Ownership/threading: the Planner borrows the ExpertStore and the IoEngine; it
// owns neither. plan_layer() is called from the GPU submit thread; completion
// callbacks land on the IoEngine dispatcher thread and only touch the store and
// the small atomic counters here. A dedicated planner thread (design
// docs/architecture.md) drives background prefetch and backfill between tokens.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/status.h"
#include "core/types.h"
#include "model/manifest.h"
#include "storage/io_engine.h"
#include "store/expert_store.h"
#include "store/shard_set.h"

namespace deepmoe::store {

// What the gate kernel produced for one layer (design §7.8).
struct RouteDecision {
    uint32_t layer = 0;
    std::span<const uint16_t> chosen;      // top-6 expert ids
    std::span<const float>    weights;     // normalised routing weights
    std::span<const uint16_t> near_ids;    // top-16 ids, for the heat EWMA of §9.3
    std::span<const float>    near_scores;
};

// One layer's outstanding P0 fills, waited on as a group (Track R1). The
// engine used to `IoEngine::drain()`, which also waits on the engram's P2 rows
// and on any P3 backfill -- so a background read could hold a layer's gate.
struct FetchGroup {
    std::mutex              m;
    std::condition_variable cv;
    uint32_t                pending = 0;
    bool                    failed  = false;
};

// Result of classifying one layer.
struct LayerPlan {
    uint32_t layer = 0;
    std::vector<ExpertKey> hits;
    std::vector<ExpertKey> misses;
    std::vector<storage::IoRequestId> issued;   // P0 requests now in flight, one per run
    uint64_t miss_bytes = 0;                    // aligned bytes actually read
    std::shared_ptr<FetchGroup> group;          // the fills `issued` belongs to
    // Misses another class was already filling (the P3 backfill): no second
    // read, waited on by Planner::wait_layer and then stamped as touched here.
    std::vector<std::pair<ExpertKey, TokenIndex>> joined;
};

// --- Track R1: the prefill hands its experts to the decode cache -------------
//
// docs/p3_prefill.md §3.4 / design §9.7.3: the set left in the cache after a
// prefill is the set a global LRU run over the prompt's routing table in token
// order would leave -- i.e. the `slot_count` keys with the newest last access,
// counting what was resident before (older than every prompt access). A
// prefill streams layer by layer, not token by token, so the stamps are
// computed instead of ticked: `base + (pos * layers + layer) * topk + slot` of
// the LAST prompt row routed to the expert. Admitting each streamed expert iff
// the cache has a free slot or its stamp is newer than the oldest evictable
// resident (which it then evicts) is an exact streaming top-`slot_count`
// selection, so the order the layers arrive in does not matter.
enum class StreamKind : uint8_t {
    Drop = 0,    // older than everything the cache keeps: compute from the transit
    Fill,        // read into `addr` (the slot is Filling until finish_streamed)
    Resident,    // already cached: no read; guarded until the caller completes `guard`
};
struct StreamAdmit {
    StreamKind  kind = StreamKind::Drop;
    uint32_t    slot = 0;
    SlotAddress addr{};                 // the slot's base; the runs go at run.slot_offset
};
// `entry` null = the synthetic single-run layout (tests). `guard` protects a
// Resident answer from being evicted by a later admission until the store's
// completed timeline reaches it.
Result<StreamAdmit> admit_streamed(ExpertStore& store, ExpertKey key, TokenIndex stamp,
                                   const ExpertEntry* entry, TimelineValue guard);
// A Fill is published (or released when `ok` is false) with its stamp.
Result<void> finish_streamed(ExpertStore& store, const StreamAdmit& a, bool ok, TokenIndex stamp);

struct PlannerStats {
    uint64_t layers_planned = 0;
    uint64_t requests = 0, hits = 0, misses = 0;
    uint64_t evictions = 0, evict_failures = 0;
    uint64_t prefetch_issued = 0, prefetch_used = 0, prefetch_wasted = 0;
    uint64_t stall_waits = 0;
    uint64_t joined_fills = 0;              // misses already being filled by the backfill
    uint64_t backfill_issued = 0, backfill_done = 0, backfill_failed = 0;
    uint64_t streamed_resident = 0, streamed_filled = 0, streamed_dropped = 0;
    double hit_rate() const { return requests ? static_cast<double>(hits) / requests : 0.0; }
    std::string to_string() const;
};

// Ranks resident slots worst-first; the head of the returned order is evicted
// first. Implementations must never return a pinned or guarded slot -- they
// only ever see ExpertStore::evictable().
class EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;
    virtual const char* name() const = 0;
    // Returns up to `count` slot indices to evict, worst first.
    virtual std::vector<uint32_t> choose_victims(const std::vector<ExpertSlot>& candidates,
                                                 size_t count) = 0;
};

// design §9.3 baseline: strict least-recently-used on last_use_token.
std::unique_ptr<EvictionPolicy> make_lru_policy();

// design §9.3: LRU tie-broken by the router-score EWMA, so a "near miss"
// expert survives an eviction round. Whether this beats plain LRU is a
// cache_sim question (Q2); the ordering function is a stub until then.
std::unique_ptr<EvictionPolicy> make_score_aware_policy();

std::unique_ptr<EvictionPolicy> make_policy(CachePolicy which);

// The P3 backfill order: every (layer, expert) of the model's 40 layers by
// static routing frequency, hottest first, from store/static_heat.inc
// (tools/hitrate_sim.py heat over the 27,399-token route trace).
std::vector<ExpertKey> static_heat_order();
// Same order, read from a tools/hitrate_sim.py heat file (`layer, expert, count,`
// per line, hottest first). Empty when the file is missing or has no rows.
std::vector<ExpertKey> static_heat_order(const std::string& path);

class Planner {
public:
    Planner() = default;
    ~Planner() = default;

    Planner(const Planner&) = delete;
    Planner& operator=(const Planner&) = delete;

    // `manifest` is the address book over the original safetensors shards and
    // `shards` the matching open files; the Planner turns one expert into one
    // IoRequest per run (design §5.1 v0.5). Both are borrowed and must outlive
    // the Planner.
    Result<void> init(ExpertStore& store, storage::IoEngine& io,
                      const Manifest& manifest, const ShardSet& shards,
                      const CacheConfig& cache, const PrefetchConfig& prefetch,
                      Profiler* profiler = nullptr);

    // Classifies one layer's routing decision, updates heat, and issues P0
    // reads for the misses. Does not block: the caller waits on the returned
    // request ids (or on the timeline the completion callback signals).
    Result<LayerPlan> plan_layer(const RouteDecision& route, TokenIndex token);

    // Makes room for `count` more fills by evicting the policy's worst slots.
    // Returns how many were actually freed (a guarded slot can refuse).
    uint32_t reclaim(size_t count);

    // Fetches one expert at the given priority. Used by plan_layer for P0, by
    // the predictor for P1 and by the idle backfill for P3. One expert is one
    // IoRequest per manifest run, so the id list has as many entries as the
    // expert has runs (two, in the shipped checkpoint). `on_done` fires exactly
    // once, when the last run has landed and the slot has settled.
    struct Fetch {
        uint32_t                     slot = 0;
        std::vector<storage::IoRequestId> ids;
        uint64_t                     bytes = 0;   // aligned bytes submitted
    };
    Result<Fetch> fetch(ExpertKey key, IoPriority priority,
                        TokenIndex token, uint32_t deadline_layer,
                        std::function<void(bool)> on_done = {},
                        TokenIndex stamp = 0);   // 0 = the next tick of the LRU clock

    // Blocks until every fill `plan` issued (and every fill it joined) has
    // settled. Err::Io if one failed or `timeout` passed.
    Result<void> wait_layer(LayerPlan& plan,
                            std::chrono::milliseconds timeout = std::chrono::seconds(120));

    // Reserves `n` consecutive LRU stamps for a caller that computes its own
    // (the prefill handoff) and returns the first; every later tick is newer.
    TokenIndex reserve_stamps(uint64_t n) {
        return access_clock_.fetch_add(n, std::memory_order_relaxed) + 1;
    }
    Result<StreamAdmit> admit_streamed(ExpertKey key, TokenIndex stamp, TimelineValue guard);
    Result<void> finish_streamed(const StreamAdmit& a, bool ok, TokenIndex stamp);

    // Marks a key as pinned once it is resident; pinned slots never evict
    // (design §9.3: the attention, shared-expert, mtp and embed set lives here).
    Result<void> pin(ExpertKey key);

    // --- lookahead (design §9.4) ------------------------------------------
    // TODO(design §9.4): drive store/predictor.h over layers L+1..L+d and issue
    // the not-yet-resident predictions at P1. Blocked on Q4/Q5 per design §16.
    Result<uint32_t> prefetch_lookahead(uint32_t from_layer, TokenIndex token);

    // design §9.6 P3 (Track R1): fills FREE slots -- never evicts -- with the
    // keys of `order`, hottest first, keeping at most `inflight` experts in
    // flight; each completion issues the next. Returns immediately; the IoEngine
    // runs P3 only behind P0 (and throttles it while decode is issuing P0s).
    //
    // `keep` (R1 round 2, docs/p4_hitrate.md §7) is what the per-turn reheat
    // pass sets, and it changes the stamp the filled slots get. The default is
    // the P3 behaviour: every backfilled slot is stamped below every demand
    // stamp (`kDemandStampBase`), so the LRU throws it back out first and a
    // startup guess never displaces a measured access. A reheat pass is the
    // opposite intent -- the order IS this conversation's measured access,
    // ordered by the heat that the turn boundary just aged -- so its slots are
    // stamped like demand traffic (the current LRU clock, hotter newer) and an
    // expert chosen twice in a row is not re-read on the next turn.
    Result<void> start_backfill(std::vector<ExpertKey> order, uint32_t inflight = 2,
                                bool keep = false);
    // Stops issuing. Fills already in flight still settle.
    void stop_backfill();
    bool backfill_active() const { return backfill_ && backfill_->active.load(); }
    // The old token-indexed entry point: Unimplemented (see start_backfill).
    Result<uint32_t> backfill(TokenIndex token);

    // Demand stamps start here, so [1, kDemandStampBase) is free for backfill.
    static constexpr uint64_t kDemandStampBase = 1ull << 32;

    const EvictionPolicy& policy() const { return *policy_; }
    PlannerStats stats() const;
    void reset_stats();

private:
    ExpertStore*       store_ = nullptr;
    storage::IoEngine* io_    = nullptr;
    const Manifest*    manifest_ = nullptr;
    const ShardSet*    shards_   = nullptr;
    Profiler*          profiler_ = nullptr;
    CacheConfig        cache_{};
    PrefetchConfig     prefetch_{};
    std::unique_ptr<EvictionPolicy> policy_;

    mutable std::mutex stats_mutex_;
    PlannerStats       stats_{};
    // The LRU clock. One tick per expert ACCESS, in the order the gate lists
    // them, not one per token: tools/cache_sim.py's `LRU` is an OrderedDict
    // that moves each key to the end as it is touched, so within a token layer
    // 0's experts are older than layer 39's and within a layer the first-listed
    // is older than the last. Stamping with the token index instead gave all
    // 240 accesses of a token the same age and let the slot index decide which
    // of them went first -- a different policy from the one simulated.
    std::atomic<uint64_t> access_clock_{kDemandStampBase};
    uint64_t next_stamp() { return access_clock_.fetch_add(1, std::memory_order_relaxed) + 1; }

    struct Backfill {
        std::vector<ExpertKey> order;
        std::mutex             m;
        size_t                 next = 0;
        uint32_t               inflight = 0, max_inflight = 2;
        bool                   keep = false;   // stamp like demand (the reheat pass)
        std::atomic<bool>      active{false};
    };
    std::shared_ptr<Backfill> backfill_;
    void backfill_pump(const std::shared_ptr<Backfill>& b);
};

}  // namespace deepmoe::store
