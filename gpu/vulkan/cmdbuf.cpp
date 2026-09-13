#include "gpu/vulkan/cmdbuf.h"

#include <format>

namespace deepmoe::gpu {

CommandPool::~CommandPool() { destroy(); }

#if !defined(DEEPMOE_ENABLE_VULKAN)

Result<void> CommandPool::create(Device&) { return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN"); }
void CommandPool::destroy() {}
void CommandPool::reset() {}
Result<CommandBuffer> CommandPool::acquire() { return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN"); }
Result<void> CommandBuffer::begin() { return fail(Err::Unavailable, "no vulkan"); }
Result<void> CommandBuffer::end()   { return fail(Err::Unavailable, "no vulkan"); }
Result<void> CommandBuffer::dispatch(uint32_t, uint32_t, uint32_t) { return fail(Err::Unavailable, "no vulkan"); }
Result<void> CommandBuffer::barrier() { return fail(Err::Unavailable, "no vulkan"); }
Result<void> submit(Device&, const Submission&) { return fail(Err::Unavailable, "no vulkan"); }

#else

Result<void> CommandPool::create(Device& device) {
    destroy();
    if (!device.valid()) return fail(Err::FailedPrecondition, "device is not created");
    device_ = &device;
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = device.compute_queue_family();
    const VkResult r = vkCreateCommandPool(device.handle(), &ci, nullptr, &pool_);
    if (r != VK_SUCCESS)
        return fail(Err::Internal, std::format("vkCreateCommandPool failed ({})", static_cast<int>(r)));
    return {};
}

void CommandPool::destroy() {
    if (pool_ && device_ && device_->valid()) vkDestroyCommandPool(device_->handle(), pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
    device_ = nullptr;
}

void CommandPool::reset() {
    if (pool_ && device_ && device_->valid()) vkResetCommandPool(device_->handle(), pool_, 0);
}

Result<CommandBuffer> CommandPool::acquire() {
    if (!pool_) return fail(Err::FailedPrecondition, "command pool is not created");
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    const VkResult r = vkAllocateCommandBuffers(device_->handle(), &ai, &cb);
    if (r != VK_SUCCESS)
        return fail(Err::Internal, std::format("vkAllocateCommandBuffers failed ({})", static_cast<int>(r)));
    return CommandBuffer(cb);
}

Result<void> CommandBuffer::begin() {
    if (!cb_) return fail(Err::FailedPrecondition, "command buffer is null");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    const VkResult r = vkBeginCommandBuffer(cb_, &bi);
    if (r != VK_SUCCESS)
        return fail(Err::Internal, std::format("vkBeginCommandBuffer failed ({})", static_cast<int>(r)));
    return {};
}

Result<void> CommandBuffer::end() {
    if (!cb_) return fail(Err::FailedPrecondition, "command buffer is null");
    const VkResult r = vkEndCommandBuffer(cb_);
    if (r != VK_SUCCESS)
        return fail(Err::Internal, std::format("vkEndCommandBuffer failed ({})", static_cast<int>(r)));
    return {};
}

// TODO(design §7): needs gpu/vulkan/pipeline.h bound first; dispatch without a
// pipeline is a validation error, so this stays a stub until P2.
Result<void> CommandBuffer::dispatch(uint32_t, uint32_t, uint32_t) {
    return unimplemented("gpu::CommandBuffer::dispatch (design §7, P2)");
}

Result<void> CommandBuffer::barrier() {
    if (!cb_) return fail(Err::FailedPrecondition, "command buffer is null");
    // The decode chain is strictly serial, so one global shader-write ->
    // shader-read barrier is the whole synchronisation story (design §7.1).
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cb_, &di);
    return {};
}

// TODO(design §7.1): vkQueueSubmit2 with one wait per MoE layer's timeline
// value plus the token-end signal.
Result<void> submit(Device&, const Submission&) {
    return unimplemented("gpu::submit (design §7.1, P2)");
}

#endif  // DEEPMOE_ENABLE_VULKAN

}  // namespace deepmoe::gpu
