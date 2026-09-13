// Slab pool (design §5.3).
//
// Vulkan on gfx1151 caps maxMemoryAllocationSize at 2 GiB, so the routed-expert
// cache cannot be one buffer. A slab is a single allocation holding
// `slots_per_slab` fixed-size expert slots:
//
//   slab bytes = slots * 18,808,832 = 1.88 GB at 100 slots
//   slot addr  = slab_base + slot * 18,808,832          (both host and device)
//
// The slot is layout::kExpertSlotBytes, not the 18,800,640 B payload: since
// design §5.1 v0.5 the runtime reads the original safetensors shards, whose
// tensor offsets are not sector-aligned, so a slot holds the two sector-aligned
// *runs* of an expert (17,698,816 + 1,110,016 B) laid back to back. The extra
// 8,192 B is the skew padding at the head of each run.
//
// A SlabBacking supplies the memory. Two exist:
//   HostSlabBacking    plain 4 KiB-aligned host memory. Fully implemented; it
//                      is what the tests, the CPU oracle and bench/ use.
//   VulkanSlabBacking  gpu/vulkan/memory.h; path A allocates from the
//                      DEVICE_LOCAL|HOST_VISIBLE type and maps it, path B
//                      imports host memory with VK_EXT_external_memory_host.
//
// Ownership/threading: SlabPool owns its backings and frees them on destruction.
// Address lookup is const and lock-free. Allocation/release of slots is the
// ExpertStore's job, not the pool's -- the pool only hands out geometry.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/align.h"
#include "core/status.h"
#include "core/types.h"
#include "model/layout.h"

namespace deepmoe::store {

// A contiguous allocation that can be addressed by the CPU, the GPU, or both.
// `host_ptr` is null for a GPU-only allocation; `device_address` is 0 when the
// backing is not visible to Vulkan.
struct SlabMemory {
    void*         host_ptr       = nullptr;
    DeviceAddress device_address = kNoDeviceAddress;
    uint64_t      bytes          = 0;
};

class SlabBacking {
public:
    virtual ~SlabBacking() = default;
    // Allocates one slab of `bytes`, 4 KiB aligned. ResourceExhausted when the
    // heap or the 2 GiB per-allocation cap is hit.
    virtual Result<SlabMemory> allocate(uint64_t bytes) = 0;
    virtual void release(const SlabMemory& mem) = 0;
    virtual const char* name() const = 0;
};

// Host-memory backing: real, used everywhere a GPU is not involved.
class HostSlabBacking final : public SlabBacking {
public:
    Result<SlabMemory> allocate(uint64_t bytes) override;
    void release(const SlabMemory& mem) override;
    const char* name() const override { return "host"; }
private:
    std::vector<std::unique_ptr<AlignedBuffer>> owned_;
};

struct SlabConfig {
    uint32_t slots_per_slab = 100;                        // design §5.3
    uint64_t slot_bytes     = layout::kExpertSlotBytes;   // 18,808,832 (payload + skew)
    uint64_t budget_bytes   = 2ull << 30;                 // total pool budget
};

// Where one slot lives.
struct SlotAddress {
    void*         host_ptr = nullptr;
    DeviceAddress dev_addr = kNoDeviceAddress;
};

class SlabPool {
public:
    SlabPool() = default;
    ~SlabPool() { reset(); }

    SlabPool(const SlabPool&) = delete;
    SlabPool& operator=(const SlabPool&) = delete;

    // Allocates as many whole slabs as `budget_bytes` allows. Returns
    // ResourceExhausted if not even one slab fits.
    Result<void> init(std::unique_ptr<SlabBacking> backing, const SlabConfig& cfg);
    void reset();

    uint32_t slab_count() const { return static_cast<uint32_t>(slabs_.size()); }
    uint32_t slot_count() const { return slot_count_; }
    uint64_t slot_bytes() const { return cfg_.slot_bytes; }
    uint64_t bytes()      const { return uint64_t(slabs_.size()) * slab_bytes_; }
    const SlabConfig& config() const { return cfg_; }
    const char* backing_name() const { return backing_ ? backing_->name() : "none"; }

    // Global slot index -> address. OutOfRange past slot_count().
    Result<SlotAddress> address(uint32_t slot) const;
    // Global slot index -> (slab, index within slab).
    std::pair<uint16_t, uint32_t> decompose(uint32_t slot) const {
        return {static_cast<uint16_t>(slot / cfg_.slots_per_slab), slot % cfg_.slots_per_slab};
    }

private:
    std::unique_ptr<SlabBacking> backing_;
    SlabConfig              cfg_{};
    std::vector<SlabMemory> slabs_;
    uint64_t                slab_bytes_ = 0;
    uint32_t                slot_count_ = 0;
};

}  // namespace deepmoe::store
