// ExpertStore: the physical location of every weight block and the
// Free -> Filling -> Resident state machine over the slab pool (design §5.3,
// §5.4).
//
// This module owns *where things are*; store/planner.h owns *what should be
// where*. The split follows design §4.1: ExpertStore knows slabs, slots, tiers
// and the (host_ptr, device_addr) pair, and keeps the LRU/heat metadata that
// the Planner's policy reads, but never decides a victim itself.
//
// The GPU-side expert pointer table (design §5.3) lives here too: a flat
// uint64[layers][experts_per_layer] of device addresses, 0 when not resident,
// which the MoE kernel indexes through buffer_device_address instead of a
// descriptor per expert. With a host-memory backing the table carries host
// pointers, so the same code path is exercised by tests and the CPU oracle.
//
// Ownership/threading: ExpertStore owns the SlabPool and all slot metadata.
// Every public method is safe from any thread -- one mutex guards the tables,
// held only for pointer bookkeeping, never across I/O. The typical callers are
// the planner thread (reserve/evict), the IoEngine dispatcher thread
// (finish_fill from a completion callback) and the GPU submit thread (lookup).
#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/config.h"
#include "core/status.h"
#include "core/types.h"
#include "store/slab.h"

namespace deepmoe::store {

// design §5.4
struct ExpertSlot {
    ExpertKey     key{};
    SlotState     state = SlotState::Free;
    Tier          tier  = Tier::Cached;
    uint16_t      slab  = 0;
    uint32_t      slot  = 0;
    void*         host_ptr = nullptr;
    DeviceAddress dev_addr = kNoDeviceAddress;
    TokenIndex    last_use_token = 0;   // LRU key
    float         heat = 0.0f;          // EWMA of router score, incl. near-misses (§9.3)
    TimelineValue guard_timeline = 0;   // must be <= the GPU's completed value before eviction
};

struct ExpertStoreStats {
    uint64_t lookups = 0, hits = 0, misses = 0;
    uint64_t fills_started = 0, fills_ok = 0, fills_failed = 0;
    uint64_t evictions = 0, eviction_blocked_by_guard = 0;
    uint32_t resident = 0, filling = 0, free = 0, pinned = 0;

    double hit_rate() const { return lookups ? static_cast<double>(hits) / lookups : 0.0; }
    std::string to_string() const;
};

class ExpertStore {
public:
    ExpertStore() = default;
    ~ExpertStore() = default;

    ExpertStore(const ExpertStore&) = delete;
    ExpertStore& operator=(const ExpertStore&) = delete;

    // `layers` includes the three DSpark blocks (design: logical layers 40..42),
    // so the pointer table covers everything the MoE kernel can address.
    Result<void> init(std::unique_ptr<SlabBacking> backing,
                      const CacheConfig& cache,
                      uint32_t layers = layout::kTotalLogicalLayers,
                      uint32_t experts_per_layer = layout::kRoutedExperts);

    // --- hot path ---------------------------------------------------------

    // Resident lookup. On a hit the slot's LRU stamp is refreshed to `token`.
    // Counts one lookup in the stats either way.
    std::optional<SlotAddress> lookup(ExpertKey key, TokenIndex token);

    // Same but without touching LRU or the stats; used by the planner when it
    // is only asking "would this miss?".
    bool resident(ExpertKey key) const;

    // --- fill path --------------------------------------------------------

    // Takes a Free slot for `key` and moves it to Filling. The returned slot
    // index and address are where the IoEngine must land the bytes.
    // ResourceExhausted when nothing is free -- the caller (Planner) must evict
    // first. AlreadyExists when `key` is resident or already filling.
    struct Reservation {
        uint32_t    slot = 0;
        SlotAddress addr{};
    };
    Result<Reservation> begin_fill(ExpertKey key, Tier tier = Tier::Cached);

    // Completion. `ok` publishes the slot (Filling -> Resident, pointer table
    // updated); on failure the slot returns to Free and the key stays absent.
    Result<void> finish_fill(uint32_t slot, bool ok, TokenIndex token = 0);

    // --- eviction ---------------------------------------------------------

    // Resident -> Free. FailedPrecondition when the slot is pinned, not
    // resident, or still guarded by an in-flight command buffer (design §5.3:
    // the GPU timeline must have passed guard_timeline).
    Result<void> evict(uint32_t slot);
    Result<void> evict_key(ExpertKey key);

    // The eviction guard: the value the GPU timeline semaphore has reached.
    // Slots whose guard exceeds it cannot be recycled.
    void          set_completed_timeline(TimelineValue v);
    TimelineValue completed_timeline() const;
    // Records that a command buffer signalling `v` will read this slot.
    Result<void> set_guard(uint32_t slot, TimelineValue v);

    // --- metadata for the planner's policy --------------------------------

    // EWMA update of a routed expert's heat. Called for the chosen top-6 and
    // for the high-scoring near misses of design §9.3.
    void note_heat(ExpertKey key, float score, float alpha = 0.125f);

    std::optional<ExpertSlot> slot_info(uint32_t slot) const;
    std::optional<ExpertSlot> slot_for(ExpertKey key) const;
    // Snapshot of every Resident, non-pinned slot; the Planner ranks these.
    std::vector<ExpertSlot> evictable() const;

    // --- geometry / GPU handoff -------------------------------------------

    uint32_t slot_count()     const { return pool_.slot_count(); }
    uint64_t slot_bytes()     const { return pool_.slot_bytes(); }
    uint32_t free_slots()     const;
    uint64_t capacity_bytes() const { return pool_.bytes(); }
    const SlabPool& pool()    const { return pool_; }

    // Flat [layer][expert] address table (design §5.3). Stable for the life of
    // the store; the Vulkan path maps this buffer host-coherently so the GPU
    // sees updates without a descriptor rewrite.
    const uint64_t* pointer_table() const { return table_.data(); }
    size_t          pointer_table_entries() const { return table_.size(); }
    size_t          pointer_table_bytes()   const { return table_.size() * sizeof(uint64_t); }
    Result<uint64_t> table_entry(ExpertKey key) const;

    ExpertStoreStats stats() const;
    void             reset_stats();

private:
    size_t table_index(ExpertKey k) const {
        return static_cast<size_t>(k.layer) * experts_per_layer_ + k.expert;
    }
    bool key_in_range(ExpertKey k) const {
        return k.layer < layers_ && k.expert < experts_per_layer_;
    }
    void publish_locked(uint32_t slot);
    void unpublish_locked(uint32_t slot);

    mutable std::mutex mutex_;
    SlabPool    pool_;
    CacheConfig cache_{};
    uint32_t    layers_ = 0;
    uint32_t    experts_per_layer_ = 0;

    std::vector<ExpertSlot> slots_;
    std::vector<uint32_t>   free_list_;
    std::unordered_map<ExpertKey, uint32_t, ExpertKeyHash> index_;   // key -> slot
    std::vector<uint64_t>   table_;                                  // the GPU pointer table
    TimelineValue           completed_timeline_ = 0;

    mutable ExpertStoreStats stats_{};
};

}  // namespace deepmoe::store
