// The pinned weight set: everything design §2.2 marks "pin" (attention, shared
// experts, router, mHC, norms, embed, head, engram wkv) loaded once into
// GPU-addressable memory and never evicted.
//
// Why this is not the ExpertStore
// ------------------------------
// ExpertStore is a fixed-size slot machine: every routed expert is exactly
// `kExpertSlotBytes` and slots are recycled. The pinned set is 17.7 GB of
// tensors of a hundred different shapes that are never recycled, so it is a
// bump allocator over regions instead. What the two have in common is the
// SlabBacking interface of store/slab.h, so the pinned set gets path A or path
// B (design §3.3) from exactly the same place the expert cache does, and
// store/ still knows nothing about Vulkan.
//
// Region size
// -----------
// `maxMemoryAllocationSize` is 2 GiB on gfx1151 (design §1.1), so the set spans
// several regions. A tensor never straddles one: each is placed at a 4 KiB
// boundary inside a region, because FILE_FLAG_NO_BUFFERING wants a sector
// aligned destination and the manifest's aligned run starts at
// `dst + 0`, with the payload at `dst + skew` (design §5.1.2).
//
// The commit limit
// ----------------
// design §5.2 / §9.2.2: on Windows the binding constraint is the system commit
// limit, not the heap size and not the BIOS VGM. `check_commit_available`
// asks GlobalMemoryStatusEx before anything is allocated so the failure mode is
// a sentence pointing at docs/build.md's pagefile section rather than a
// VK_ERROR_OUT_OF_DEVICE_MEMORY 14 GB into a load.
//
// Ownership/threading: PinnedStore owns its regions through the SlabBacking and
// frees them on destruction. `load()` runs once at startup and is not thread
// safe; every lookup afterwards is const and lock free.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/status.h"
#include "core/types.h"
#include "model/manifest.h"
#include "storage/io_engine.h"
#include "store/shard_set.h"
#include "store/slab.h"

namespace deepmoe::store {

// One tensor, and its block-scale plane when it has one. Addresses point at the
// payload, i.e. the skew has already been added.
struct PinnedTensor {
    DeviceAddress data       = kNoDeviceAddress;
    const void*   data_host  = nullptr;
    uint64_t      data_bytes = 0;
    DeviceAddress scale      = kNoDeviceAddress;
    const void*   scale_host = nullptr;
    uint64_t      scale_bytes = 0;
    QuantType     dtype      = QuantType::Unknown;
    std::vector<uint64_t> shape;

    bool valid() const { return data_bytes != 0; }
};

struct PinnedConfig {
    // Below the 2 GiB maxMemoryAllocationSize of design §1.1, with room for the
    // per-tensor 4 KiB padding.
    uint64_t region_bytes = 1ull << 30;
    // P0 BlockingMiss would preempt nothing here -- this runs before the token
    // loop exists -- but it keeps the queue ordering honest if a future caller
    // overlaps a pinned load with decode.
    IoPriority priority = IoPriority::Backfill;
};

// How much of the Windows commit charge is still available, and whether
// `bytes` fits in it. Returns ResourceExhausted with a message naming
// docs/build.md when it does not. Always succeeds off Windows.
Result<void> check_commit_available(uint64_t bytes, const char* what);

class PinnedStore {
public:
    PinnedStore() = default;
    ~PinnedStore() { reset(); }

    PinnedStore(const PinnedStore&) = delete;
    PinnedStore& operator=(const PinnedStore&) = delete;

    Result<void> init(std::unique_ptr<SlabBacking> backing, const PinnedConfig& cfg = {});
    void         reset();

    // Reads `names` (plus each one's scale plane) through the IoEngine and
    // places them. Idempotent per name: a tensor already loaded is skipped, so
    // a caller can ask for overlapping sets.
    Result<void> load(const Manifest& manifest, const ShardSet& shards,
                      storage::IoEngine& io, const std::vector<std::string>& names);

    const PinnedTensor* find(std::string_view name) const;
    Result<const PinnedTensor*> require(std::string_view name) const;

    uint64_t bytes_loaded() const { return bytes_loaded_; }
    uint32_t tensor_count() const { return static_cast<uint32_t>(tensors_.size()); }
    uint32_t region_count() const { return static_cast<uint32_t>(regions_.size()); }
    uint64_t bytes_reserved() const {
        uint64_t n = 0;
        for (const SlabMemory& r : regions_) n += r.bytes;
        return n;
    }

private:
    // Bump-allocates `bytes` at a 4 KiB boundary, opening a new region when the
    // current one cannot hold it.
    Result<SlotAddress> place(uint64_t bytes);

    std::unique_ptr<SlabBacking> backing_;
    PinnedConfig            cfg_{};
    std::vector<SlabMemory> regions_;
    uint64_t                used_in_region_ = 0;
    uint64_t                cur_region_bytes_ = 0;
    uint64_t                bytes_loaded_   = 0;
    std::unordered_map<std::string, PinnedTensor> tensors_;
};

// --- the name lists of design §2.2 -----------------------------------------

// Everything one backbone layer pins: attention (wq_a, wq_b, wkv, wo_a, wo_b,
// attn_sink, the two norms), the shared expert, the router, the mHC
// coefficients, and -- on a source layer -- the compressor and indexer. The
// routed experts are NOT here; they are the streamed set of design §9.3.
std::vector<std::string> pinned_layer_tensors(const Manifest& manifest, uint32_t layer);

// embed, norm and head.
std::vector<std::string> pinned_global_tensors(const Manifest& manifest);

// Sum of `bytes` over a name list, scales included: what `load` will read.
uint64_t pinned_bytes(const Manifest& manifest, const std::vector<std::string>& names);

}  // namespace deepmoe::store
