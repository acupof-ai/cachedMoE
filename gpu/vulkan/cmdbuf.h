// Command buffer recording for the one-command-buffer-per-token model of
// design §7.1.
//
// A decode token is 40 layers x ~11 dispatches plus the head: about 450
// dispatches, pre-recorded once and re-submitted. At 5-20 us of CPU+driver cost
// per dispatch that would be 3-9 ms if re-recorded every token, which is 5-14%
// of the 65 ms floor (design §3.4) -- hence recording once and parameterising
// through push constants and the expert pointer table rather than descriptors.
//
// Expert readiness is expressed as a wait on gpu/vulkan/timeline.h inside the
// submit info, not as a mid-buffer host round trip.
//
// Ownership/threading: a CommandPool is owned by one thread and is NOT
// thread-safe, per the Vulkan spec. deepMoE has a single GPU submit thread, so
// there is one pool; the planner never touches it.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/status.h"
#include "core/types.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/timeline.h"

namespace deepmoe::gpu {

class CommandBuffer;

class CommandPool {
public:
    CommandPool() = default;
    ~CommandPool();

    CommandPool(const CommandPool&) = delete;
    CommandPool& operator=(const CommandPool&) = delete;

    Result<void> create(Device& device);
    void         destroy();

    // A buffer that is recorded once and submitted many times.
    // TODO(design §7.1): implement with VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS?
    // no -- one token at a time, so a plain re-submittable primary buffer.
    Result<CommandBuffer> acquire();
    void                  reset();

#if defined(DEEPMOE_ENABLE_VULKAN)
    VkCommandPool handle() const { return pool_; }
#endif

private:
    Device* device_ = nullptr;
#if defined(DEEPMOE_ENABLE_VULKAN)
    VkCommandPool pool_ = VK_NULL_HANDLE;
#endif
};

class CommandBuffer {
public:
    CommandBuffer() = default;

    Result<void> begin();
    Result<void> end();

    // A compute dispatch with its push constants. The expert addresses come
    // from the pointer table (design §5.3), so nothing is rebound per expert.
    // TODO(design §7).
    Result<void> dispatch(uint32_t gx, uint32_t gy, uint32_t gz);

    // The barrier between two dispatches of design §7.14's list.
    // TODO(design §7.1): a single global memory barrier is enough for the
    // strictly serial decode chain; the kernel bench measures its real cost.
    Result<void> barrier();

#if defined(DEEPMOE_ENABLE_VULKAN)
    VkCommandBuffer handle() const { return cb_; }
    explicit CommandBuffer(VkCommandBuffer cb) : cb_(cb) {}
private:
    VkCommandBuffer cb_ = VK_NULL_HANDLE;
#endif
};

// One submission: wait on `wait` values of `timeline`, signal `signal_value`.
struct Submission {
    const CommandBuffer* cmd = nullptr;
    Timeline*            timeline = nullptr;
    std::span<const TimelineValue> wait_values;
    TimelineValue        signal_value = 0;
};

// TODO(design §7.1): vkQueueSubmit2 with a timeline wait per MoE layer.
Result<void> submit(Device& device, const Submission& s);

}  // namespace deepmoe::gpu
