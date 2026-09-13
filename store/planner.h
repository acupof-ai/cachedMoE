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
#include "storage/io_engine.h"
#include "store/expert_store.h"

namespace deepmoe::store {

// What the gate kernel produced for one layer (design §7.8).
struct RouteDecision {
    uint32_t layer = 0;
    std::span<const uint16_t> chosen;      // top-6 expert ids
    std::span<const float>    weights;     // normalised routing weights
    std::span<const uint16_t> near_ids;    // top-16 ids, for the heat EWMA of §9.3
    std::span<const float>    near_scores;
};

// Result of classifying one layer.
struct LayerPlan {
    uint32_t layer = 0;
    std::vector<ExpertKey> hits;
    std::vector<ExpertKey> misses;
    std::vector<storage::IoRequestId> issued;   // P0 requests now in flight
    uint64_t miss_bytes = 0;
};

struct PlannerStats {
    uint64_t layers_planned = 0;
    uint64_t requests = 0, hits = 0, misses = 0;
    uint64_t evictions = 0, evict_failures = 0;
    uint64_t prefetch_issued = 0, prefetch_used = 0, prefetch_wasted = 0;
    uint64_t stall_waits = 0;
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

class Planner {
public:
    Planner() = default;
    ~Planner() = default;

    Planner(const Planner&) = delete;
    Planner& operator=(const Planner&) = delete;

    // `experts_file` is the opened experts.bin; the Planner computes offsets
    // arithmetically from model/layout.h (design §5.1).
    Result<void> init(ExpertStore& store, storage::IoEngine& io,
                      const storage::File& experts_file,
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
    // the predictor for P1 and by the idle backfill for P3.
    Result<storage::IoRequestId> fetch(ExpertKey key, IoPriority priority,
                                       TokenIndex token, uint32_t deadline_layer,
                                       std::function<void(bool)> on_done = {});

    // Marks a key as pinned once it is resident; pinned slots never evict
    // (design §9.3: hot.bin, mtp.bin and embed live here).
    Result<void> pin(ExpertKey key);

    // --- lookahead (design §9.4) ------------------------------------------
    // TODO(design §9.4): drive store/predictor.h over layers L+1..L+d and issue
    // the not-yet-resident predictions at P1. Blocked on Q4/Q5 per design §16.
    Result<uint32_t> prefetch_lookahead(uint32_t from_layer, TokenIndex token);

    // TODO(design §9.6 P3): fill free slots by static heat when the drive is
    // idle. Needs the Q1 frequency distribution first.
    Result<uint32_t> backfill(TokenIndex token);

    const EvictionPolicy& policy() const { return *policy_; }
    PlannerStats stats() const;
    void reset_stats();

private:
    ExpertStore*       store_ = nullptr;
    storage::IoEngine* io_    = nullptr;
    const storage::File* experts_ = nullptr;
    Profiler*          profiler_ = nullptr;
    CacheConfig        cache_{};
    PrefetchConfig     prefetch_{};
    std::unique_ptr<EvictionPolicy> policy_;

    mutable std::mutex stats_mutex_;
    PlannerStats       stats_{};
};

}  // namespace deepmoe::store
