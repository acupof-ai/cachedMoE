#include "gpu/vulkan/cmdbuf.h"

#include <format>

namespace deepmoe::gpu {

CommandPool::~CommandPool() { destroy(); }
QueryPool::~QueryPool() { destroy(); }

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
Result<void> submit_and_wait(Device&, const CommandBuffer&) { return fail(Err::Unavailable, "no vulkan"); }
Result<void> QueryPool::create(Device&, uint32_t) { return fail(Err::Unavailable, "no vulkan"); }
void QueryPool::destroy() { count_ = 0; }
Result<std::vector<uint64_t>> QueryPool::read() const { return fail(Err::Unavailable, "no vulkan"); }
Result<double> QueryPool::elapsed_seconds(uint32_t, uint32_t) const { return fail(Err::Unavailable, "no vulkan"); }

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

Result<void> QueryPool::create(Device& device, uint32_t count) {
    destroy();
    if (!device.valid()) return fail(Err::FailedPrecondition, "device is not created");
    if (device.caps().timestamp_valid_bits == 0)
        return fail(Err::Unavailable, "the compute queue family reports timestampValidBits == 0");
    device_ = &device;
    VkQueryPoolCreateInfo ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    ci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = count;
    if (VkResult r = vkCreateQueryPool(device.handle(), &ci, nullptr, &pool_); r != VK_SUCCESS)
        return fail(Err::Internal, std::format("vkCreateQueryPool failed ({})", static_cast<int>(r)));
    count_ = count;
    return {};
}

void QueryPool::destroy() {
    if (pool_ && device_ && device_->valid()) vkDestroyQueryPool(device_->handle(), pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
    device_ = nullptr;
    count_ = 0;
}

Result<std::vector<uint64_t>> QueryPool::read() const {
    if (!pool_) return fail(Err::FailedPrecondition, "query pool is not created");
    std::vector<uint64_t> out(count_, 0);
    const VkResult r = vkGetQueryPoolResults(device_->handle(), pool_, 0, count_,
                                             out.size() * sizeof(uint64_t), out.data(),
                                             sizeof(uint64_t),
                                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (r != VK_SUCCESS)
        return fail(Err::Internal, std::format("vkGetQueryPoolResults failed ({})", static_cast<int>(r)));
    return out;
}

Result<double> QueryPool::elapsed_seconds(uint32_t first, uint32_t last) const {
    auto v = read();
    if (!v) return std::unexpected(v.error());
    if (first >= v->size() || last >= v->size()) return fail(Err::OutOfRange, "query index");
    // timestampValidBits < 64 means the top bits are undefined; mask them off.
    const uint32_t bits = device_->caps().timestamp_valid_bits;
    const uint64_t mask = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
    const uint64_t a = (*v)[first] & mask;
    const uint64_t b = (*v)[last] & mask;
    const uint64_t ticks = b >= a ? b - a : (mask - a) + b + 1;
    return double(ticks) * double(device_->caps().timestamp_period_ns) * 1e-9;
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

Result<void> CommandBuffer::bind(const Pipeline& pipeline, VkDescriptorSet set) {
    if (!cb_) return fail(Err::FailedPrecondition, "command buffer is null");
    if (!pipeline.valid()) return fail(Err::FailedPrecondition, "pipeline is not created");
    vkCmdBindPipeline(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    if (set != VK_NULL_HANDLE)
        vkCmdBindDescriptorSets(cb_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(),
                                0, 1, &set, 0, nullptr);
    return {};
}

Result<void> CommandBuffer::push(const Pipeline& pipeline, const void* data, uint32_t bytes) {
    if (!cb_) return fail(Err::FailedPrecondition, "command buffer is null");
    if (!bytes) return {};
    vkCmdPushConstants(cb_, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes, data);
    return {};
}

Result<void> CommandBuffer::reset_queries(const QueryPool& q, uint32_t first, uint32_t count) {
    if (!cb_) return fail(Err::FailedPrecondition, "command buffer is null");
    if (!q.handle()) return fail(Err::FailedPrecondition, "query pool is not created");
    vkCmdResetQueryPool(cb_, q.handle(), first, count);
    return {};
}

Result<void> CommandBuffer::write_timestamp(const QueryPool& q, uint32_t index, bool bottom) {
    if (!cb_) return fail(Err::FailedPrecondition, "command buffer is null");
    if (!q.handle()) return fail(Err::FailedPrecondition, "query pool is not created");
    vkCmdWriteTimestamp2(cb_, bottom ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                     : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                         q.handle(), index);
    return {};
}

Result<void> CommandBuffer::dispatch(uint32_t gx, uint32_t gy, uint32_t gz) {
    if (!cb_) return fail(Err::FailedPrecondition, "command buffer is null");
    if (gx == 0 || gy == 0 || gz == 0) return fail(Err::InvalidArgument, "zero-sized dispatch");
    vkCmdDispatch(cb_, gx, gy, gz);
    return {};
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

Result<void> submit(Device& device, const Submission& s) {
    if (!device.valid()) return fail(Err::FailedPrecondition, "device is not created");
    if (!s.cmd || !s.cmd->handle()) return fail(Err::InvalidArgument, "submission has no command buffer");

    VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cbi.commandBuffer = s.cmd->handle();

    std::vector<VkSemaphoreSubmitInfo> waits;
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    if (s.timeline && s.timeline->valid()) {
        for (TimelineValue v : s.wait_values) {
            VkSemaphoreSubmitInfo w{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
            w.semaphore = s.timeline->handle();
            w.value     = v;
            w.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            waits.push_back(w);
        }
        signal.semaphore = s.timeline->handle();
        signal.value     = s.signal_value;
        signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    } else if (!s.wait_values.empty() || s.signal_value) {
        return fail(Err::InvalidArgument, "timeline values given without a timeline semaphore");
    }

    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.waitSemaphoreInfoCount = static_cast<uint32_t>(waits.size());
    si.pWaitSemaphoreInfos    = waits.data();
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos    = &cbi;
    if (signal.semaphore != VK_NULL_HANDLE) {
        si.signalSemaphoreInfoCount = 1;
        si.pSignalSemaphoreInfos    = &signal;
    }
    const VkResult r = vkQueueSubmit2(device.compute_queue(), 1, &si, VK_NULL_HANDLE);
    if (r != VK_SUCCESS)
        return fail(Err::Internal, std::format("vkQueueSubmit2 failed ({})", static_cast<int>(r)));
    return {};
}

Result<void> submit_and_wait(Device& device, const CommandBuffer& cmd) {
    Submission s;
    s.cmd = &cmd;
    if (auto r = submit(device, s); !r) return r;
    const VkResult r = vkQueueWaitIdle(device.compute_queue());
    if (r != VK_SUCCESS)
        return fail(Err::Internal, std::format("vkQueueWaitIdle failed ({})", static_cast<int>(r)));
    return {};
}

#endif  // DEEPMOE_ENABLE_VULKAN

}  // namespace deepmoe::gpu
