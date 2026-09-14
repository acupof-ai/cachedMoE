#include "gpu/vulkan/prefill_kernels.h"

#include <cstring>
#include <format>
#include <future>

#include "core/align.h"
#include "cpu/dequant.h"

namespace deepmoe::gpu {

void pf_act_quant_host(const float* x, uint32_t n, uint32_t k, uint16_t* q16, float* scales) {
    const uint32_t blocks = k / 32;
    uint8_t bytes[32];
    for (uint32_t m = 0; m < n; ++m) {
        for (uint32_t b = 0; b < blocks; ++b) {
            const float* v = x + size_t(m) * k + b * 32;
            const float s = cpu::act_quant_block(v, 32, bytes, nullptr);
            scales[size_t(m) * blocks + b] = s;
            for (uint32_t i = 0; i < 32; ++i)
                q16[size_t(m) * k + b * 32 + i] =
                    cpu::float_to_fp16(cpu::fp8_e4m3_to_float(bytes[i]));
        }
    }
}

Result<PfExpert> pf_load_expert(MemoryAllocator& alloc, const Manifest& manifest,
                                const store::ShardSet& shards, storage::IoEngine& io,
                                ExpertKey key) {
    auto e = manifest.require_expert(key);
    if (!e) return std::unexpected(e.error());
    const ExpertEntry& ent = **e;
    auto b = alloc.allocate(align_up(ent.slot_bytes, kPageSize), true, true);
    if (!b) return std::unexpected(b.error());
    PfExpert out;
    out.buf = *b;
    if (reinterpret_cast<uintptr_t>(out.buf.host_ptr) % kPageSize) {
        out.release(alloc);
        return fail(Err::Internal, "expert buffer is not page aligned");
    }
    std::vector<std::future<storage::IoResult>> futs;
    for (const Run& r : ent.runs) {
        auto f = shards.require(r.file);
        if (!f) { out.release(alloc); return std::unexpected(f.error()); }
        storage::IoRequest req;
        req.key      = key;
        req.priority = IoPriority::BlockingMiss;
        req.file     = *f;
        req.file_off = r.aligned_off;
        req.bytes    = r.aligned_bytes;
        req.dst      = static_cast<std::byte*>(out.buf.host_ptr) + r.slot_offset;
        auto fut = io.submit_future(req);
        if (!fut) { out.release(alloc); return std::unexpected(fut.error()); }
        futs.push_back(std::move(*fut));
    }
    for (auto& f : futs)
        if (!f.get().ok()) { out.release(alloc); return fail(Err::Io, "expert read failed"); }
    for (uint32_t p = 0; p < 6; ++p) {
        out.part_off[p] = ent.offset_of(static_cast<ExpertPart>(p));
        out.addr[p] = out.buf.dev_addr + out.part_off[p];
    }
    return out;
}

#if !defined(DEEPMOE_ENABLE_VULKAN)

Result<void> PrefillRunner::create(Device&, MemoryAllocator&, const std::string&, uint32_t) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
void PrefillRunner::destroy() {}
Result<uint32_t> PrefillRunner::kernel(const PfKernel&) { return fail(Err::Unavailable, "no vulkan"); }
uint64_t* PrefillRunner::slots(uint32_t) { return nullptr; }
Result<void> PrefillRunner::record(CommandBuffer&, uint32_t, const void*, uint32_t, uint32_t,
                                   uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}
Result<void> PrefillRunner::dispatch_now(uint32_t, const void*, uint32_t, uint32_t, uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}

#else

static_assert(sizeof(PfGemmPush) <= kPfPushBytes);
static_assert(sizeof(PfJob) == 64);

Result<void> PrefillRunner::create(Device& device, MemoryAllocator& alloc,
                                   const std::string& shader_dir, uint32_t max_kernels) {
    destroy();
    if (!device.valid()) return fail(Err::FailedPrecondition, "device is not created");
    // Every prefill GEMM reduces a row across the 32 lanes of one Wave32
    // (prefill_gemm.slang), and the cooperative matrices are subgroup-scoped.
    if (device.caps().subgroup_size_control == false)
        return fail(Err::FailedPrecondition,
                    "the prefill kernels need VK_EXT_subgroup_size_control for Wave32");
    device_ = &device;
    alloc_  = &alloc;
    shader_dir_ = shader_dir;
    max_kernels_ = max_kernels;

    auto t = alloc.allocate(uint64_t(max_kernels) * kPfKernelStride, true, false);
    if (!t) { destroy(); return std::unexpected(t.error()); }
    table_ = *t;
    std::memset(table_.host_ptr, 0, static_cast<size_t>(table_.bytes));
    if (auto r = descriptors_.create(device, max_kernels, max_kernels); !r) { destroy(); return r; }
    if (auto r = pool_.create(device); !r) { destroy(); return r; }
    return {};
}

void PrefillRunner::destroy() {
    own_cmd_valid_ = false;
    own_cmd_ = CommandBuffer{};
    pool_.destroy();
    descriptors_.destroy();
    for (auto& p : pipes_) if (p) p->destroy();
    pipes_.clear();
    sets_.clear();
    index_.clear();
    if (alloc_ && table_.valid()) alloc_->free(table_);
    table_ = GpuBuffer{};
    device_ = nullptr;
    alloc_  = nullptr;
}

Result<uint32_t> PrefillRunner::kernel(const PfKernel& k) {
    if (auto it = index_.find(k); it != index_.end()) return it->second;
    if (!device_) return fail(Err::FailedPrecondition, "runner is not created");
    if (pipes_.size() >= max_kernels_)
        return fail(Err::ResourceExhausted,
                    std::format("prefill runner: more than {} distinct kernels", max_kernels_));
    PipelineSpec ps;
    ps.m             = 1;
    ps.lanes_per_row = 32;
    ps.rows_per_wg   = 8;
    ps.subgroup_size = 32;
    ps.extra = {k.stage, k.wfmt, k.xfmt, k.tile, k.extra0, k.extra1};
    PipelineLayoutSpec la;
    la.storage_buffers    = 1;
    la.push_constant_size = kPfPushBytes;
    auto p = std::make_unique<Pipeline>();
    if (auto r = p->create(*device_, shader_dir_ + "/" + k.spv + ".spv", la, ps); !r)
        return fail(r.error().code, std::format("{} stage {} w{} x{} t{}: {}", k.spv, k.stage,
                                                k.wfmt, k.xfmt, k.tile, r.error().message));
    const uint32_t i = static_cast<uint32_t>(pipes_.size());
    std::vector<BufferBinding> b(1);
    b[0] = {0, uint64_t(i) * kPfKernelStride, kPfKernelStride, table_.buffer};
    auto set = descriptors_.allocate(*p, b);
    if (!set) return std::unexpected(set.error());
    pipes_.push_back(std::move(p));
    sets_.push_back(*set);
    index_.emplace(k, i);
    return i;
}

uint64_t* PrefillRunner::slots(uint32_t h) {
    if (!table_.host_ptr || h >= max_kernels_) return nullptr;
    return reinterpret_cast<uint64_t*>(static_cast<std::byte*>(table_.host_ptr) +
                                       uint64_t(h) * kPfKernelStride);
}

Result<void> PrefillRunner::record(CommandBuffer& cmd, uint32_t h, const void* push,
                                   uint32_t bytes, uint32_t gx, uint32_t gy) {
    if (h >= pipes_.size()) return fail(Err::InvalidArgument, "unknown prefill kernel");
    if (bytes > kPfPushBytes) return fail(Err::InvalidArgument, "push constants exceed 64 B");
    if (gx == 0 || gy == 0) return fail(Err::InvalidArgument, "zero workgroups");
    if (gx > 65535 || gy > 65535)
        return fail(Err::InvalidArgument,
                    std::format("{} x {} workgroups is past the 65535 dispatch limit", gx, gy));
    if (auto r = cmd.bind(*pipes_[h], sets_[h]); !r) return r;
    if (bytes) {
        if (auto r = cmd.push(*pipes_[h], push, bytes); !r) return r;
    }
    return cmd.dispatch(gx, gy);
}

Result<void> PrefillRunner::dispatch_now(uint32_t h, const void* push, uint32_t bytes,
                                         uint32_t gx, uint32_t gy) {
    // One buffer, re-begun: the pool is RESET_COMMAND_BUFFER, and acquiring a
    // fresh one per call (DecodeRunner::dispatch_now) leaks one per dispatch.
    if (!own_cmd_valid_) {
        auto cb = pool_.acquire();
        if (!cb) return std::unexpected(cb.error());
        own_cmd_ = *cb;
        own_cmd_valid_ = true;
    }
    CommandBuffer& cmd = own_cmd_;
    if (auto r = cmd.begin(); !r) return r;
    if (auto r = record(cmd, h, push, bytes, gx, gy); !r) return r;
    if (auto r = cmd.end(); !r) return r;
    return submit_and_wait(*device_, cmd);
}

#endif  // DEEPMOE_ENABLE_VULKAN

}  // namespace deepmoe::gpu
