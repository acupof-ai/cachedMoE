#include "gpu/vulkan/timeline.h"

#include <format>

namespace deepmoe::gpu {

Timeline::~Timeline() { destroy(); }

#if !defined(DEEPMOE_ENABLE_VULKAN)

Result<void> Timeline::create(Device&, TimelineValue) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
void Timeline::destroy() { valid_ = false; }
Result<TimelineValue> Timeline::value() const { return fail(Err::Unavailable, "no vulkan"); }
Result<void> Timeline::signal(TimelineValue) { return fail(Err::Unavailable, "no vulkan"); }
Result<void> Timeline::wait(TimelineValue, std::chrono::nanoseconds) { return fail(Err::Unavailable, "no vulkan"); }

#else

Result<void> Timeline::create(Device& device, TimelineValue initial) {
    destroy();
    if (!device.valid()) return fail(Err::FailedPrecondition, "device is not created");
    if (!device.caps().timeline_semaphore)
        return fail(Err::Unavailable, "VK_KHR_timeline_semaphore is required (design §7.1)");
    device_ = &device;

    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type.initialValue  = initial;
    VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
    const VkResult r = vkCreateSemaphore(device.handle(), &ci, nullptr, &semaphore_);
    if (r != VK_SUCCESS)
        return fail(Err::Internal, std::format("vkCreateSemaphore failed ({})", static_cast<int>(r)));
    valid_ = true;
    return {};
}

void Timeline::destroy() {
    if (semaphore_ && device_ && device_->valid())
        vkDestroySemaphore(device_->handle(), semaphore_, nullptr);
    semaphore_ = VK_NULL_HANDLE;
    device_ = nullptr;
    valid_ = false;
}

Result<TimelineValue> Timeline::value() const {
    if (!valid_) return fail(Err::FailedPrecondition, "timeline is not created");
    uint64_t v = 0;
    const VkResult r = vkGetSemaphoreCounterValue(device_->handle(), semaphore_, &v);
    if (r != VK_SUCCESS)
        return fail(Err::Internal, std::format("vkGetSemaphoreCounterValue failed ({})", static_cast<int>(r)));
    return v;
}

Result<void> Timeline::signal(TimelineValue v) {
    if (!valid_) return fail(Err::FailedPrecondition, "timeline is not created");
    auto cur = value();
    if (!cur) return std::unexpected(cur.error());
    if (v < *cur)
        return fail(Err::InvalidArgument,
                    std::format("timeline signal {} is below the current value {}", v, *cur));
    if (v == *cur) return {};
    VkSemaphoreSignalInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    si.semaphore = semaphore_;
    si.value     = v;
    const VkResult r = vkSignalSemaphore(device_->handle(), &si);
    if (r != VK_SUCCESS)
        return fail(Err::Internal, std::format("vkSignalSemaphore failed ({})", static_cast<int>(r)));
    return {};
}

Result<void> Timeline::wait(TimelineValue v, std::chrono::nanoseconds timeout) {
    if (!valid_) return fail(Err::FailedPrecondition, "timeline is not created");
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores    = &semaphore_;
    wi.pValues        = &v;
    const VkResult r = vkWaitSemaphores(device_->handle(), &wi, static_cast<uint64_t>(timeout.count()));
    if (r == VK_TIMEOUT) return fail(Err::Cancelled, std::format("timeline wait for {} timed out", v));
    if (r != VK_SUCCESS)
        return fail(Err::Internal, std::format("vkWaitSemaphores failed ({})", static_cast<int>(r)));
    return {};
}

#endif  // DEEPMOE_ENABLE_VULKAN

}  // namespace deepmoe::gpu
