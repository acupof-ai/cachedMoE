#include "gpu/vulkan/rawread.h"

#include <algorithm>
#include <chrono>
#include <format>

#include "core/profiler.h"

namespace deepmoe::gpu {

namespace {
// One workgroup iteration is 1024 uint4 = 16 KiB, so a dispatch covers
// groups * 16 KiB at a time.
constexpr uint64_t kBytesPerGroupIter = 1024ull * 16;
}  // namespace

uint64_t RawReadKernel::round_bytes(uint64_t bytes) const {
    if (groups_ == 0) return 0;
    const uint64_t gran = kBytesPerGroupIter * groups_;
    return (bytes / gran) * gran;
}

#if !defined(DEEPMOE_ENABLE_VULKAN)

Result<void> RawReadKernel::create(Device&, MemoryAllocator&, const std::string&, uint32_t) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
void RawReadKernel::destroy() {}
Result<RawReadResult> RawReadKernel::run(const GpuBuffer&, uint64_t, uint32_t) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
Result<RawReadResult> RawReadKernel::run_empty(const GpuBuffer&, uint32_t) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}

#else

Result<void> RawReadKernel::create(Device& device, MemoryAllocator& alloc,
                                   const std::string& shader_dir, uint32_t groups) {
    destroy();
    if (groups == 0) return fail(Err::InvalidArgument, "rawread needs at least one workgroup");
    device_ = &device;
    alloc_  = &alloc;
    groups_ = groups;

    PipelineLayoutSpec lspec;
    lspec.storage_buffers = 2;
    lspec.push_constant_size = 16;
    PipelineSpec spec;   // rawread takes no specialisation constants
    if (auto r = pipeline_.create(device, shader_dir + "/rawread.spv", lspec, spec); !r) {
        destroy();
        return r;
    }
    if (auto r = descriptors_.create(device, 1, 2); !r) { destroy(); return r; }

    auto sink = alloc.allocate(uint64_t(groups) * 256 * 16, /*host_visible=*/false,
                               /*device_address=*/false);
    if (!sink) { destroy(); return std::unexpected(sink.error()); }
    sink_ = *sink;

    if (auto r = pool_.create(device); !r) { destroy(); return r; }
    auto cb = pool_.acquire();
    if (!cb) { destroy(); return std::unexpected(cb.error()); }
    cmd_ = *cb;
    // Timestamps are best-effort: a queue with timestampValidBits == 0 falls
    // back to wall-clock timing around a vkQueueWaitIdle.
    (void)queries_.create(device, 2);
    return {};
}

void RawReadKernel::destroy() {
    queries_.destroy();
    pool_.destroy();
    descriptors_.destroy();
    pipeline_.destroy();
    if (alloc_ && sink_.valid()) alloc_->free(sink_);
    sink_ = GpuBuffer{};
    device_ = nullptr;
    alloc_ = nullptr;
    groups_ = 0;
}

Result<RawReadResult> RawReadKernel::run(const GpuBuffer& src, uint64_t bytes, uint32_t passes) {
    if (!device_ || !pipeline_.valid()) return fail(Err::FailedPrecondition, "rawread is not created");
    if (passes == 0) return fail(Err::InvalidArgument, "passes must be >= 1");
    const uint64_t use = round_bytes(std::min(bytes, src.bytes));
    if (use == 0)
        return fail(Err::InvalidArgument,
                    std::format("{} B is below the {} B granularity of {} workgroups",
                                bytes, kBytesPerGroupIter * groups_, groups_));

    std::vector<BufferBinding> binds(2);
    binds[0].binding = 0; binds[0].buffer = src.buffer;  binds[0].range = use;
    binds[1].binding = 1; binds[1].buffer = sink_.buffer;
    descriptors_.reset();
    auto set = descriptors_.allocate(pipeline_, binds);
    if (!set) return std::unexpected(set.error());
    CommandBuffer* cmd = &cmd_;

    struct Push { uint32_t per_group, r0, r1, r2; } push{};
    push.per_group = static_cast<uint32_t>(use / 16 / groups_);   // uint4 elements

    const bool timed = queries_.count() >= 2;
    if (auto r = cmd->begin(); !r) return std::unexpected(r.error());
    if (timed) {
        (void)cmd->reset_queries(queries_, 0, 2);
        (void)cmd->write_timestamp(queries_, 0, /*bottom=*/false);
    }
    if (auto r = cmd->bind(pipeline_, *set); !r) return std::unexpected(r.error());
    if (auto r = cmd->push(pipeline_, &push, sizeof(push)); !r) return std::unexpected(r.error());
    for (uint32_t p = 0; p < passes; ++p) {
        if (auto r = cmd->dispatch(groups_); !r) return std::unexpected(r.error());
        if (p + 1 < passes) { if (auto r = cmd->barrier(); !r) return std::unexpected(r.error()); }
    }
    if (timed) (void)cmd->write_timestamp(queries_, 1, /*bottom=*/true);
    if (auto r = cmd->end(); !r) return std::unexpected(r.error());

    const auto t0 = Clock::now();
    if (auto r = submit_and_wait(*device_, *cmd); !r) return std::unexpected(r.error());
    const double wall = std::chrono::duration<double>(Clock::now() - t0).count();

    RawReadResult out;
    out.bytes  = use * passes;
    out.groups = groups_;
    out.passes = passes;
    out.seconds = wall;
    if (timed) {
        if (auto s = queries_.elapsed_seconds(0, 1); s && *s > 0.0) {
            out.seconds = *s;
            out.gpu_timed = true;
        }
    }
    out.gbps = out.seconds > 0 ? double(out.bytes) / 1e9 / out.seconds : 0.0;
    return out;
}

#endif  // DEEPMOE_ENABLE_VULKAN

}  // namespace deepmoe::gpu

#if defined(DEEPMOE_ENABLE_VULKAN)
namespace deepmoe::gpu {

Result<RawReadResult> RawReadKernel::run_empty(const GpuBuffer& src, uint32_t dispatches) {
    if (!device_ || !pipeline_.valid()) return fail(Err::FailedPrecondition, "rawread is not created");
    if (dispatches == 0) return fail(Err::InvalidArgument, "dispatches must be >= 1");

    std::vector<BufferBinding> binds(2);
    binds[0].binding = 0; binds[0].buffer = src.buffer; binds[0].range = kBytesPerGroupIter;
    binds[1].binding = 1; binds[1].buffer = sink_.buffer;
    descriptors_.reset();
    auto set = descriptors_.allocate(pipeline_, binds);
    if (!set) return std::unexpected(set.error());

    // per_group = 0: the read loop never runs, so what is left is the launch,
    // the one sink store per lane and the barrier.
    struct Push { uint32_t per_group, r0, r1, r2; } push{};
    const bool timed = queries_.count() >= 2;
    if (auto r = cmd_.begin(); !r) return std::unexpected(r.error());
    if (timed) {
        (void)cmd_.reset_queries(queries_, 0, 2);
        (void)cmd_.write_timestamp(queries_, 0, false);
    }
    if (auto r = cmd_.bind(pipeline_, *set); !r) return std::unexpected(r.error());
    if (auto r = cmd_.push(pipeline_, &push, sizeof(push)); !r) return std::unexpected(r.error());
    for (uint32_t i = 0; i < dispatches; ++i) {
        if (auto r = cmd_.dispatch(1); !r) return std::unexpected(r.error());
        if (i + 1 < dispatches) { if (auto r = cmd_.barrier(); !r) return std::unexpected(r.error()); }
    }
    if (timed) (void)cmd_.write_timestamp(queries_, 1, true);
    if (auto r = cmd_.end(); !r) return std::unexpected(r.error());

    const auto t0 = Clock::now();
    if (auto r = submit_and_wait(*device_, cmd_); !r) return std::unexpected(r.error());
    const double wall = std::chrono::duration<double>(Clock::now() - t0).count();

    RawReadResult out;
    out.passes  = dispatches;
    out.groups  = 1;
    out.seconds = wall;
    if (timed) {
        if (auto s = queries_.elapsed_seconds(0, 1); s && *s > 0.0) {
            out.seconds = *s;
            out.gpu_timed = true;
        }
    }
    return out;
}

}  // namespace deepmoe::gpu
#endif
