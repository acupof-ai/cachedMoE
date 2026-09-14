#include "gpu/vulkan/attn_kernels.h"

#include <cstring>
#include <format>

#include "core/align.h"

namespace deepmoe::gpu {

const char* attn_stage_name(AttnStage s) {
    switch (s) {
        case AttnStage::MhcPost:     return "mega_mhc.post";
        case AttnStage::MhcMix:      return "mega_mhc.mix";
        case AttnStage::MhcFinal:    return "mega_mhc.final";
        case AttnStage::WqA:         return "wq_a";
        case AttnStage::WqB:         return "wq_b";
        case AttnStage::WkvGemv:     return "wkv.gemv";
        case AttnStage::WkvFinish:   return "wkv.finish";
        case AttnStage::AttnScore:   return "sparse_attn.score";
        case AttnStage::AttnCombine: return "sparse_attn.combine";
        case AttnStage::WoA:         return "wo_a";
        case AttnStage::WoB:         return "wo_b";
        case AttnStage::GateScore:   return "gate.score";
        case AttnStage::GateTopK:    return "gate.topk";
        case AttnStage::Head:        return "head";
        case AttnStage::Count:       break;
    }
    return "?";
}

Result<void> GpuScratch::create(MemoryAllocator& alloc, uint64_t bytes) {
    destroy();
    auto b = alloc.allocate(bytes, /*host_visible=*/true, /*device_address=*/true);
    if (!b) return std::unexpected(b.error());
    if (!b->host_ptr || b->dev_addr == kNoDeviceAddress) {
        alloc.free(*b);
        return fail(Err::Internal,
                    "scratch must be both host-writable and device-addressable");
    }
    alloc_ = &alloc;
    buf_ = *b;
    used_ = 0;
    std::memset(buf_.host_ptr, 0, static_cast<size_t>(bytes));
    return {};
}

void GpuScratch::destroy() {
    if (alloc_ && buf_.valid()) alloc_->free(buf_);
    buf_ = GpuBuffer{};
    alloc_ = nullptr;
    used_ = 0;
}

Result<GpuScratch::View> GpuScratch::alloc(uint64_t bytes, uint64_t align) {
    if (!buf_.valid()) return fail(Err::FailedPrecondition, "scratch is not created");
    const uint64_t off = align_up(used_, align);
    if (off + bytes > buf_.bytes)
        return fail(Err::ResourceExhausted,
                    std::format("scratch: {} B wanted at {}, capacity {}", bytes, off, buf_.bytes));
    used_ = off + bytes;
    View v;
    v.addr  = buf_.dev_addr + off;
    v.host  = static_cast<std::byte*>(buf_.host_ptr) + off;
    v.bytes = bytes;
    return v;
}

#if !defined(DEEPMOE_ENABLE_VULKAN)

Result<void> AttnRunner::create(Device&, MemoryAllocator&, const std::string&, const AttnSpec&) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
void AttnRunner::destroy() {}
uint32_t AttnRunner::rows_per_lane(AttnStage) const { return 1; }
uint64_t* AttnRunner::slots(AttnStage) { return nullptr; }
Result<void> AttnRunner::record(CommandBuffer&, AttnStage, const void*, uint32_t, uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}
Result<void> AttnRunner::dispatch_now(AttnStage, const void*, uint32_t, uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}
Result<void> AttnRunner::make(AttnStage, const std::string&, uint32_t, uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}

#else

namespace {
// Bigger than any of the push structs in the header, so one pipeline layout
// serves every stage and vkCmdPushConstants never runs past the range.
constexpr uint32_t kPushBytes = 64;

struct StageDef {
    AttnStage   stage;
    const char* spv;
    uint32_t    stage_const;
    uint32_t    act_quant;
    uint32_t    rows_cap;     // largest rows_per_lane this shader may use
};

// wo_a is the one entry with act_quant = 0: `Attention.forward` reaches its
// weight through an einsum, not `linear()`, so its activation never makes the
// fp8 round trip (attn_common.slang explains why that matters).
// rows_cap is a workgroup-count question: 40 CUs want a few hundred
// workgroups, so the cap follows the kernel's row count. wq_b has 32768 rows
// and wkv has 512.
constexpr StageDef kStages[] = {
    {AttnStage::MhcPost,     "mega_mhc",    0, 1, 1},
    {AttnStage::MhcMix,      "mega_mhc",    1, 1, 1},
    {AttnStage::MhcFinal,    "mega_mhc",    2, 1, 1},
    {AttnStage::WqA,         "wq_a",        0, 1, 2},   // 1280 rows
    {AttnStage::WqB,         "wq_b",        0, 1, 4},   // 32768
    {AttnStage::WkvGemv,     "wkv",         0, 1, 1},   // 512: workgroup-starved
    {AttnStage::WkvFinish,   "wkv",         1, 1, 1},
    {AttnStage::AttnScore,   "sparse_attn", 0, 1, 1},
    {AttnStage::AttnCombine, "sparse_attn", 1, 1, 1},
    {AttnStage::WoA,         "wo_a",        0, 0, 2},   // 8192, in 1024-row groups
    {AttnStage::WoB,         "wo_b",        0, 1, 2},   // 5120
    {AttnStage::GateScore,   "gate",        0, 1, 1},   // 384: no row blocking
    {AttnStage::GateTopK,    "gate",        1, 1, 1},
    {AttnStage::Head,        "head",        0, 1, 4},   // 129280
};
static_assert(sizeof(kStages) / sizeof(kStages[0]) ==
              static_cast<size_t>(AttnStage::Count));
}  // namespace

uint32_t AttnRunner::rows_per_lane(AttnStage s) const {
    const uint32_t cap = kStages[static_cast<uint32_t>(s)].rows_cap;
    return spec_.rows_per_lane < cap ? spec_.rows_per_lane : cap;
}

Result<void> AttnRunner::make(AttnStage s, const std::string& spv, uint32_t stage_const,
                              uint32_t act_quant) {
    PipelineSpec ps;
    ps.m             = 1;
    ps.lanes_per_row = spec_.lanes_per_row;
    ps.rows_per_wg   = 256 / spec_.lanes_per_row;
    ps.subgroup_size = spec_.subgroup_size;
    ps.extra = {stage_const, act_quant, rows_per_lane(s)};
    PipelineLayoutSpec la;
    la.storage_buffers    = 1;    // the address table slice, and nothing else
    la.push_constant_size = kPushBytes;
    return pipes_[static_cast<uint32_t>(s)].create(*device_, spv, la, ps);
}

Result<void> AttnRunner::create(Device& device, MemoryAllocator& alloc,
                                const std::string& shader_dir, const AttnSpec& spec) {
    destroy();
    if (!device.valid()) return fail(Err::FailedPrecondition, "device is not created");
    if (spec.lanes_per_row != 16 && spec.lanes_per_row != 32 && spec.lanes_per_row != 64)
        return fail(Err::InvalidArgument, "lanes_per_row must be 16, 32 or 64 (design §7.1 rule 3)");
    if (spec.rows_per_lane != 1 && spec.rows_per_lane != 2 && spec.rows_per_lane != 4)
        return fail(Err::InvalidArgument, "rows_per_lane must be 1, 2 or 4");
    // wo_a's block-diagonal groups are o_lora_rank = 1024 rows; a workgroup
    // that straddled two of them would stage the wrong slice of `o`.
    const uint32_t per_wg = (256 / spec.lanes_per_row) * spec.rows_per_lane;
    if (1024 % per_wg != 0)
        return fail(Err::InvalidArgument,
                    std::format("a workgroup retires {} rows, which must divide wo_a's "
                                "1024-row groups", per_wg));
    device_ = &device;
    alloc_  = &alloc;
    spec_   = spec;

    const uint32_t n = static_cast<uint32_t>(AttnStage::Count);
    auto t = alloc.allocate(uint64_t(n) * kAttnStageStride, true, false);
    if (!t) { destroy(); return std::unexpected(t.error()); }
    table_ = *t;
    std::memset(table_.host_ptr, 0, static_cast<size_t>(table_.bytes));

    if (auto r = descriptors_.create(device, n, n); !r) { destroy(); return r; }
    for (const StageDef& d : kStages) {
        if (auto r = make(d.stage, shader_dir + "/" + d.spv + ".spv", d.stage_const,
                          d.act_quant); !r) {
            destroy();
            return fail(r.error().code,
                        std::format("{}: {}", attn_stage_name(d.stage), r.error().message));
        }
        const uint32_t i = static_cast<uint32_t>(d.stage);
        std::vector<BufferBinding> b(1);
        b[0] = {0, uint64_t(i) * kAttnStageStride, kAttnStageStride, table_.buffer};
        auto set = descriptors_.allocate(pipes_[i], b);
        if (!set) { destroy(); return std::unexpected(set.error()); }
        sets_[i] = *set;
    }
    if (auto r = pool_.create(device); !r) { destroy(); return r; }
    return {};
}

void AttnRunner::destroy() {
    pool_.destroy();
    descriptors_.destroy();
    for (Pipeline& p : pipes_) p.destroy();
    if (alloc_ && table_.valid()) alloc_->free(table_);
    table_ = GpuBuffer{};
    for (VkDescriptorSet& s : sets_) s = VK_NULL_HANDLE;
    device_ = nullptr;
    alloc_ = nullptr;
}

uint64_t* AttnRunner::slots(AttnStage s) {
    if (!table_.host_ptr) return nullptr;
    return reinterpret_cast<uint64_t*>(static_cast<std::byte*>(table_.host_ptr) +
                                       uint64_t(static_cast<uint32_t>(s)) * kAttnStageStride);
}

Result<void> AttnRunner::record(CommandBuffer& cmd, AttnStage s, const void* push,
                                uint32_t push_bytes, uint32_t groups) {
    const uint32_t i = static_cast<uint32_t>(s);
    if (!pipes_[i].valid()) return fail(Err::FailedPrecondition, "pipeline is not created");
    if (push_bytes > kPushBytes)
        return fail(Err::InvalidArgument, "push constants exceed the shared 64 B range");
    if (groups == 0) return fail(Err::InvalidArgument, "zero workgroups");
    if (auto r = cmd.bind(pipes_[i], sets_[i]); !r) return r;
    if (push_bytes) {
        if (auto r = cmd.push(pipes_[i], push, push_bytes); !r) return r;
    }
    return cmd.dispatch(groups);
}

Result<void> AttnRunner::dispatch_now(AttnStage s, const void* push, uint32_t push_bytes,
                                      uint32_t groups) {
    auto cb = pool_.acquire();
    if (!cb) return std::unexpected(cb.error());
    CommandBuffer cmd = *cb;
    if (auto r = cmd.begin(); !r) return r;
    if (auto r = record(cmd, s, push, push_bytes, groups); !r) return r;
    if (auto r = cmd.end(); !r) return r;
    return submit_and_wait(*device_, cmd);
}

#endif  // DEEPMOE_ENABLE_VULKAN

}  // namespace deepmoe::gpu
