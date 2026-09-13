// Timeline semaphore: how the CPU tells the GPU "the six experts for layer L
// are resident" without breaking the pre-recorded command buffer (design §7.1,
// §7.8).
//
// One timeline per token stream. The command buffer for token T contains, before
// each MoE dispatch, a wait on `token_base(T) + layer`. The planner host-signals
// that value once every miss for the layer has completed. Layers whose experts
// were already resident are signalled immediately, so the GPU never stalls on a
// hit. The same counter doubles as the eviction guard of design §5.3: a slot
// read by a command buffer that signals V cannot be recycled until the
// semaphore has passed V.
//
// Ownership/threading: Timeline owns one VkSemaphore. `signal` is called from
// the planner thread and the IoEngine dispatcher thread; `wait`/`value` from
// anywhere. Vulkan timeline semaphores are internally synchronised, so no lock
// is needed.
#pragma once

#include <chrono>
#include <cstdint>

#include "core/status.h"
#include "core/types.h"
#include "gpu/vulkan/device.h"

namespace deepmoe::gpu {

// Values are laid out so a token's layer waits are contiguous and monotonic
// across tokens: token T, layer L -> T * kValuesPerToken + L + 1.
inline constexpr uint64_t kValuesPerToken = 64;   // 40 layers + 3 mtp + head + slack

constexpr TimelineValue timeline_value(TokenIndex token, uint32_t layer) {
    return token * kValuesPerToken + layer + 1;
}
constexpr TimelineValue timeline_token_end(TokenIndex token) {
    return (token + 1) * kValuesPerToken;
}

class Timeline {
public:
    Timeline() = default;
    ~Timeline();

    Timeline(const Timeline&) = delete;
    Timeline& operator=(const Timeline&) = delete;

    Result<void> create(Device& device, TimelineValue initial = 0);
    void         destroy();
    bool         valid() const { return valid_; }

    // Current counter. Also what ExpertStore::set_completed_timeline is fed.
    Result<TimelineValue> value() const;

    // Host signal: "everything up to `v` is ready". Monotonic; signalling a
    // value below the current one is InvalidArgument, not a silent no-op.
    Result<void> signal(TimelineValue v);

    // Host wait, mainly for shutdown and for the benchmark harness. The decode
    // loop does not wait on the host -- that is the point of §7.1.
    Result<void> wait(TimelineValue v, std::chrono::nanoseconds timeout);

#if defined(DEEPMOE_ENABLE_VULKAN)
    VkSemaphore handle() const { return semaphore_; }
#endif

private:
    Device* device_ = nullptr;
    bool    valid_  = false;
#if defined(DEEPMOE_ENABLE_VULKAN)
    VkSemaphore semaphore_ = VK_NULL_HANDLE;
#endif
};

}  // namespace deepmoe::gpu
