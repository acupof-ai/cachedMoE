// ExpertStore: the physical location of every weight block and the
// Free -> Filling -> Resident state machine over the slab pool (design §5.3,
// §5.4).
//
// This module owns *where things are*; store/planner.h owns *what should be
// where*. The split follows design §4.1: ExpertStore knows slabs, slots, tiers
// and the (host_ptr, device_addr) pair, and keeps the LRU/heat metadata that
// the Planner's policy reads, but never decides a victim itself.
//
// Since design §5.1 v0.5 there is no repack: the runtime reads the original
// safetensors shards, whose tensor offsets are 8-byte but never 4 KiB aligned.
// The manifest widens every read to sector boundaries and merges byte-adjacent
// tensors into *runs*, so filling one expert is one IoRequest per run (two, in
// the shipped checkpoint: 17,698,816 B of weights and 1,110,016 B of scales).
// A slot is layout::kExpertSlotBytes and holds those runs back to back;
// `Filling` ends when every run has reported in.
//
// The GPU-side expert pointer table (design §5.3) lives here too: a flat
// uint64[layers][experts_per_layer][6] of device addresses, 0 when not
// resident, which the MoE kernel indexes through buffer_device_address instead
// of a descriptor per expert. Six entries per expert, because each of the six
// tensors sits at `slot_base + run.slot_offset + part.skew` and the skews differ
// per expert. 43 x 384 x 6 x 8 B = 793 KB. With a host-memory backing the table
// carries host pointers, so the same code path is exercised by tests and the
// CPU oracle.
//
// Ownership/threading: ExpertStore owns the SlabPool and all slot metadata.
// Every public method is safe from any thread -- one mutex guards the tables,
// held only for pointer bookkeeping, never across I/O. The typical callers are
// the planner thread (reserve/evict), the IoEngine dispatcher thread
// (finish_run from a completion callback) and the GPU submit thread (lookup).
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/config.h"
#include "core/status.h"
#include "core/types.h"
#include "model/manifest.h"
#include "store/slab.h"

namespace deepmoe::store {

// One sector-aligned read that has to land in a slot before it is Resident.
// A copy of the manifest's Run with the destination resolved (design §5.1).
struct FillRun {
    uint32_t      file     = 0;          // index into Manifest::files()
    uint64_t      file_off = 0;          // == Run::aligned_off, 4 KiB aligned
    uint64_t      bytes    = 0;          // == Run::aligned_bytes, 4 KiB multiple
    void*         dst      = nullptr;    // slot_base + slot_offset, 4 KiB aligned
    DeviceAddress dev_dst  = kNoDeviceAddress;
};

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

    // Fill bookkeeping: Filling ends when every run has reported in.
    uint32_t runs_total = 0;
    uint32_t runs_done  = 0;
    bool     run_failed = false;
    // Slot-relative position of each of the six parts, copied from the manifest.
    uint64_t part_offset[kExpertPartCount] = {};
    uint64_t part_bytes [kExpertPartCount] = {};
};

struct ExpertStoreStats {
    uint64_t lookups = 0, hits = 0, misses = 0;
    uint64_t fills_started = 0, fills_ok = 0, fills_failed = 0;
    uint64_t runs_started = 0, runs_done = 0;
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

    // Gives every slab back to the backing and forgets every slot. The store
    // outlives nothing, but the MEMORY it holds is the GPU allocator's, and the
    // allocator has to be torn down after it -- so a caller that owns both
    // needs a way to say "let go now" that is not the destructor's ordering.
    void reset();

    // --- hot path ---------------------------------------------------------

    // Resident lookup. On a hit the slot's LRU stamp is refreshed to `token`.
    // Counts one lookup in the stats either way. The returned address is the
    // slot's base; the six parts are at pointer_table() offsets from it.
    std::optional<SlotAddress> lookup(ExpertKey key, TokenIndex token);

    // Same but without touching LRU or the stats; used by the planner when it
    // is only asking "would this miss?".
    bool resident(ExpertKey key) const;

    // --- fill path --------------------------------------------------------

    // Takes a Free slot for `key` and moves it to Filling. `entry` is the
    // manifest's record for this expert: it decides how many IoRequests the
    // caller must issue (one per run, `file_off = run.aligned_off`, 4 KiB
    // aligned dst) and where the six parts end up inside the slot.
    // ResourceExhausted when nothing is free -- the caller (Planner) must evict
    // first. AlreadyExists when `key` is resident or already filling.
    struct Reservation {
        uint32_t    slot = 0;
        SlotAddress addr{};                        // the slot's base
        uint32_t    run_count = 0;
        FillRun     runs[layout::kMaxExpertRuns]{};
    };
    Result<Reservation> begin_fill(ExpertKey key, const ExpertEntry& entry,
                                   Tier tier = Tier::Cached);
    // Synthetic single-run fill covering the whole slot, with the six parts laid
    // out planar (w1 | w1.scale | w2 | w2.scale | w3 | w3.scale). No manifest is
    // involved, so this is for tests, benchmarks and the CPU oracle -- the
    // runtime always goes through the overload above.
    Result<Reservation> begin_fill(ExpertKey key, Tier tier = Tier::Cached);

    // One run has landed. Returns true when that was the last outstanding run
    // and the slot has settled (Resident on success, Free on failure). Runs may
    // complete out of order and from different threads; the first failure
    // poisons the fill, but the slot is only released once every run reported.
    Result<bool> finish_run(uint32_t slot, bool ok, TokenIndex token = 0);

    // Completes the whole fill at once, whatever is still outstanding. `ok`
    // publishes the slot (Filling -> Resident, pointer table updated); on
    // failure the slot returns to Free and the key stays absent.
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

    // design §9.3's global LRU exactly as tools/cache_sim.py's `LRU` runs it:
    // evicts the resident, unpinned, unguarded slot with the smallest
    // `last_use_token` and returns its index. One linear scan under the lock
    // and no copy -- `evictable()` + a sort is a 290 KB vector a call at 4,500
    // slots, and the Planner evicts once per miss. NotFound when nothing is
    // evictable.
    Result<uint32_t> evict_lru();

    // --- Track R1 (docs/p4_hitrate.md) --------------------------------------

    // Refreshes a RESIDENT key's LRU stamp to `stamp` if that is newer, without
    // counting a lookup. False when the key is not resident. The prefill
    // handoff uses it for an expert the cache already holds.
    bool touch(ExpertKey key, TokenIndex stamp);
    // The stamp `evict_lru` would evict next, or nullopt when nothing is
    // evictable (every resident slot pinned or guarded).
    std::optional<TokenIndex> oldest_evictable_stamp() const;
    // Blocks until `key` is not Filling (resident, or absent after a failed
    // fill) or `timeout` passes; returns whether it is resident. A key another
    // priority class is already filling -- the P3 backfill -- is waited on
    // here instead of being read a second time.
    bool wait_settled(ExpertKey key, std::chrono::milliseconds timeout);
    // The slot index of a held key (any state).
    std::optional<uint32_t> slot_of(ExpertKey key) const;

    // --- geometry / GPU handoff -------------------------------------------

    uint32_t slot_count()     const { return pool_.slot_count(); }
    uint64_t slot_bytes()     const { return pool_.slot_bytes(); }
    uint32_t free_slots()     const;
    uint64_t capacity_bytes() const { return pool_.bytes(); }
    const SlabPool& pool()    const { return pool_; }

    // Flat [layer][expert][part] address table (design §5.3). Stable for the
    // life of the store; the Vulkan path maps this buffer host-coherently so the
    // GPU sees updates without a descriptor rewrite.
    const uint64_t* pointer_table() const { return table_.data(); }
    size_t          pointer_table_entries() const { return table_.size(); }
    size_t          pointer_table_bytes()   const { return table_.size() * sizeof(uint64_t); }
    static constexpr uint32_t pointer_table_stride() { return kExpertPartCount; }
    Result<uint64_t> table_entry(ExpertKey key, ExpertPart part = ExpertPart::W1Weight) const;
    // All six parts of a RESIDENT expert under one lock; FailedPrecondition if
    // it is not resident. The MoE bridge asks for six experts a layer, and
    // six lookups plus thirty-six `table_entry` calls were 42 lock round trips.
    Result<void> table_row(ExpertKey key, uint64_t out[kExpertPartCount]) const;

    ExpertStoreStats stats() const;
    void             reset_stats();

private:
    size_t table_index(ExpertKey k, ExpertPart p = ExpertPart::W1Weight) const {
        return (static_cast<size_t>(k.layer) * experts_per_layer_ + k.expert) * kExpertPartCount
             + static_cast<uint8_t>(p);
    }
    bool key_in_range(ExpertKey k) const {
        return k.layer < layers_ && k.expert < experts_per_layer_;
    }
    Result<Reservation> reserve_locked(ExpertKey key, Tier tier);
    void publish_locked(uint32_t slot);
    void unpublish_locked(uint32_t slot);
    void release_locked(uint32_t slot);
    void settle_locked(uint32_t slot, bool ok, TokenIndex token);

    mutable std::mutex mutex_;
    std::condition_variable settled_;   // notified whenever a fill settles
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
