// Command buffer recording for the one-command-buffer-per-token model of
// design §7.1.
//
// A decode token is 40 layers x ~11 dispatches plus the head: about 450
// dispatches, pre-recorded once and re-submitted. At 5-20 us of CPU+driver cost
// per dispatch that would be 3-9 ms if re-recorded every token, which is 5-14%
// of the 65 ms floor (design §3.4) -- hence recording once and parameterising
// through push constants and the expert pointer table rather than descriptors.
// bench/kernel_bench measures the real per-dispatch cost; see docs/kernel_p1.md.
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
#include "gpu/vulkan/pipeline.h"
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

    // A buffer that is recorded once and submitted many times. One token at a
    // time, so a plain re-submittable primary buffer is enough.
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

// GPU-side timing. `timestampPeriod` nanoseconds per tick; the compute queue's
// timestampValidBits must be non-zero (device.h reports both). Used by
// bench/kernel_bench to separate kernel time from submit overhead.
class QueryPool {
public:
    QueryPool() = default;
    ~QueryPool();

    QueryPool(const QueryPool&) = delete;
    QueryPool& operator=(const QueryPool&) = delete;

    Result<void> create(Device& device, uint32_t count);
    void         destroy();
    uint32_t     count() const { return count_; }

    // Raw tick values, one per query slot.
    Result<std::vector<uint64_t>> read() const;
    // ADDITIVE (Track I): slots [first, first + n) only. `read` waits on every
    // slot of the pool, which hangs on one a command buffer never wrote; a
    // caller that fills a prefix of a large pool per token reads the prefix.
    Result<std::vector<uint64_t>> read_range(uint32_t first, uint32_t n) const;
    // Seconds between two slots, using timestampPeriod.
    Result<double> elapsed_seconds(uint32_t first, uint32_t last) const;

#if defined(DEEPMOE_ENABLE_VULKAN)
    VkQueryPool handle() const { return pool_; }
#endif

private:
    Device*  device_ = nullptr;
    uint32_t count_  = 0;
#if defined(DEEPMOE_ENABLE_VULKAN)
    VkQueryPool pool_ = VK_NULL_HANDLE;
#endif
};

class CommandBuffer {
public:
    CommandBuffer() = default;

    Result<void> begin();
    Result<void> end();

    // The barrier between two dispatches of design §7.14's list: a single
    // global shader-write -> shader-read dependency, which is all a strictly
    // serial decode chain needs.
    Result<void> barrier();

    // A compute dispatch. `bind` and `push` first; the expert addresses come
    // from the pointer table (design §5.3), so nothing is rebound per expert.
    Result<void> dispatch(uint32_t gx, uint32_t gy = 1, uint32_t gz = 1);

#if defined(DEEPMOE_ENABLE_VULKAN)
    Result<void> bind(const Pipeline& pipeline, VkDescriptorSet set = VK_NULL_HANDLE);
    Result<void> push(const Pipeline& pipeline, const void* data, uint32_t bytes);
    Result<void> reset_queries(const QueryPool& q, uint32_t first, uint32_t count);
    Result<void> write_timestamp(const QueryPool& q, uint32_t index, bool bottom = true);

    VkCommandBuffer handle() const { return cb_; }
    explicit CommandBuffer(VkCommandBuffer cb) : cb_(cb) {}

private:
    VkCommandBuffer cb_ = VK_NULL_HANDLE;
#endif
};

// One submission: wait on `wait_values` of `timeline`, signal `signal_value`.
// A null timeline means "no synchronisation", which is what the benchmarks use.
struct Submission {
    const CommandBuffer* cmd = nullptr;
    Timeline*            timeline = nullptr;
    std::span<const TimelineValue> wait_values;
    TimelineValue        signal_value = 0;
    // ADDITIVE (Track I): whether completing the buffer signals `signal_value`.
    // A submission that only WAITS -- the §7.1 residency gate, where the host
    // is the one that signals -- must not also signal, because a timeline
    // cannot be signalled to a value at or below its current one.
    bool                 signal_on_complete = true;
    // ADDITIVE (Track I): signal a DIFFERENT timeline on completion. The token
    // loop waits on the residency timeline, which the host signals, and fences
    // on a second one, which the GPU signals; one semaphore cannot be both
    // without the two writers racing each other's values.
    Timeline*            signal_timeline = nullptr;
};

Result<void> submit(Device& device, const Submission& s);

// Submit and block until the queue is idle. Benchmarks and tests only -- the
// decode loop never waits on the host (design §7.1).
Result<void> submit_and_wait(Device& device, const CommandBuffer& cmd);

}  // namespace deepmoe::gpu
