// Experimental draft-only weight copy. The target BF16 tensor is borrowed
// read-only during conversion and remains owned by PinnedStore.
#pragma once
#include "gpu/vulkan/memory.h"
#include "store/pinned.h"

namespace cachedmoe::gpu {
class DraftHeadFp8 {
  public:
    DraftHeadFp8() = default;
    ~DraftHeadFp8();
    DraftHeadFp8(const DraftHeadFp8 &) = delete;
    DraftHeadFp8 &operator=(const DraftHeadFp8 &) = delete;
    Result<void> create(MemoryAllocator &, const store::PinnedTensor &, uint32_t vocab, uint32_t dim);
    const GpuBuffer &weights() const { return weights_; }
    const GpuBuffer &scales() const { return scales_; }
    static constexpr uint64_t copy_bytes(uint32_t vocab, uint32_t dim) {
        return uint64_t(vocab) * dim + uint64_t(vocab) * sizeof(float);
    }
  private:
    MemoryAllocator *allocator_ = nullptr;
    GpuBuffer weights_, scales_;
};
} // namespace cachedmoe::gpu
