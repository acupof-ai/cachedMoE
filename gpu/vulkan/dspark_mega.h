#pragma once

#include "core/namespace.h"
#include <array>
#include <cstddef>
#include <span>
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/descriptor.h"
#include "gpu/vulkan/memory.h"
#include "gpu/shaders/dspark_plan_layout.h"
namespace cachedmoe::gpu {
// Immutable schedule: every phase sees its own arguments and addresses.
struct DsparkMegaOp {
    uint32_t kind = 0, gx = 1, gy = 1, reserved = 0;
    std::array<uint32_t, DM_DS_PLAN_PUSH_WORDS> push{};
    std::array<uint64_t, DM_DS_PLAN_PTR_WORDS> ptr{};
};
static_assert(sizeof(DsparkMegaOp) == DM_DS_PLAN_OP_BYTES);
static_assert(offsetof(DsparkMegaOp, push) == DM_DS_PLAN_PUSH_OFFSET);
static_assert(offsetof(DsparkMegaOp, ptr) == DM_DS_PLAN_PTR_OFFSET);
class DsparkMegaRunner {
  public:
    ~DsparkMegaRunner();
    Result<void> create(Device &, MemoryAllocator &, const std::string &);
    Result<void> run(std::span<const DsparkMegaOp>, uint32_t groups = 120);
    bool valid() const {
        return ready_;
    }

  private:
    Device *device_ = nullptr;
    MemoryAllocator *alloc_ = nullptr;
    bool ready_ = false;
    Pipeline pipe_;
    DescriptorPool descriptors_;
    CommandPool pool_;
    GpuBuffer plan_, state_;
#if defined(CACHEDMOE_ENABLE_VULKAN)
    VkDescriptorSet set_ = VK_NULL_HANDLE;
#endif
};
} // namespace cachedmoe::gpu
