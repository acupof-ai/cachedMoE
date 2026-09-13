// Descriptor pool and sets for the compute kernels of design §7.
//
// There is deliberately very little here. The expert weights are NOT
// descriptors: design §5.3 reaches them through buffer_device_address from the
// pointer table, so a MoE kernel binds a handful of small buffers (the pointer
// table, the router ids and weights, the activations) and nothing per expert.
// That is what keeps one pre-recorded command buffer per token possible
// (design §7.1) -- 15,360 possible experts would otherwise be 15,360 descriptors.
//
// Ownership/threading: DescriptorPool owns the VkDescriptorPool and every set
// allocated from it. Sets are written once at setup and bound from the single
// submit thread; nothing is updated while a command buffer is in flight.
#pragma once

#include <cstdint>
#include <vector>

#include "core/status.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/pipeline.h"

namespace deepmoe::gpu {

// One storage-buffer binding. `buffer` may be VK_NULL_HANDLE only if the shader
// never touches that binding (Slang still emits the layout entry).
struct BufferBinding {
    uint32_t binding = 0;
    uint64_t offset  = 0;
    uint64_t range   = 0;      // 0 means VK_WHOLE_SIZE
#if defined(DEEPMOE_ENABLE_VULKAN)
    VkBuffer buffer = VK_NULL_HANDLE;
#endif
};

class DescriptorPool {
public:
    DescriptorPool() = default;
    ~DescriptorPool();

    DescriptorPool(const DescriptorPool&) = delete;
    DescriptorPool& operator=(const DescriptorPool&) = delete;

    Result<void> create(Device& device, uint32_t max_sets, uint32_t max_storage_buffers);
    void         destroy();
    // Frees every set allocated so far. Only legal when nothing is in flight.
    void         reset();

#if defined(DEEPMOE_ENABLE_VULKAN)
    // Allocates a set for `pipeline`'s layout and writes `bindings` into it.
    Result<VkDescriptorSet> allocate(const Pipeline& pipeline,
                                     const std::vector<BufferBinding>& bindings);
    VkDescriptorPool handle() const { return pool_; }
#endif

private:
    Device* device_ = nullptr;
#if defined(DEEPMOE_ENABLE_VULKAN)
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
#endif
};

}  // namespace deepmoe::gpu
