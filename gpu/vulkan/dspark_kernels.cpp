#include "gpu/vulkan/dspark_kernels.h"

#include <cstring>
#include <format>

namespace deepmoe::gpu {

const char* dspark_stage_name(DsparkStage s) {
    switch (s) {
        case DsparkStage::Gemv:          return "dspark.gemv";
        case DsparkStage::RmsNorm:       return "dspark.rmsnorm";
        case DsparkStage::RopeQuant:     return "dspark.ropequant";
        case DsparkStage::WoA:           return "dspark.wo_a";
        case DsparkStage::AttnScore:     return "dspark.attn.score";
        case DsparkStage::AttnCombine:   return "dspark.attn.combine";
        case DsparkStage::MarkovBias:    return "dspark.markov.bias";
        case DsparkStage::AddBiasArgmax: return "dspark.markov.argmax";
        case DsparkStage::Confidence:    return "dspark.confidence";
        case DsparkStage::VerifyRows:    return "dspark.verify.rows";
        case DsparkStage::Count:         break;
    }
    return "?";
}

std::string DsparkSpec::name() const {
    return std::format("M{} L{} R{} sg{} h{}", m, lanes_per_row, rows_per_lane,
                       subgroup_size, heads_per_wg);
}

#if !defined(DEEPMOE_ENABLE_VULKAN)

Result<void> DsparkRunner::create(Device&, MemoryAllocator&, const std::string&,
                                  const DsparkSpec&) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
void DsparkRunner::destroy() {}
uint64_t* DsparkRunner::slots(DsparkStage) { return nullptr; }
Result<void> DsparkRunner::record(CommandBuffer&, DsparkStage, const void*, uint32_t, uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}
Result<void> DsparkRunner::dispatch_now(DsparkStage, const void*, uint32_t, uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}
Result<void> DsparkRunner::make(DsparkStage, const std::string&, uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}

#else

namespace {
// The same 64 B range attn_kernels.cpp and decode_kernels.cpp use, so a stage
// can be pushed any of the three structs without the layout caring.
constexpr uint32_t kPushBytes = 64;

// gpu/shaders/dspark_common.slang's kMaxDraftM: the LDS and register budgets of
// dspark_gemv.slang are sized against it, so a spec above it would read past
// gXQ rather than merely run slowly.
constexpr uint32_t kMaxDraftM = 6;

struct StageDef {
    DsparkStage stage;
    const char* spv;
    uint32_t    stage_const;
    uint32_t    act_quant;   // 1 = quantise x onto the E4M3 grid before the GEMM
};

// Every stage of the family, in enum order. `act_quant` is the only per-stage
// knob: every fp8 `Linear` of the draft path quantises its activation
// (attn_common.slang's point 1), and wo_a -- reached through `torch.einsum`, its
// module declared bf16 -- does not. The stages that touch no fp8 weight at all
// carry 0 and never read the constant.
constexpr StageDef kStages[] = {
    {DsparkStage::Gemv,          "dspark_gemv", 0, 1},
    {DsparkStage::RmsNorm,       "dspark_gemv", 1, 0},
    {DsparkStage::RopeQuant,     "dspark_gemv", 2, 0},
    {DsparkStage::WoA,           "dspark_gemv", 3, 0},
    {DsparkStage::AttnScore,     "dspark_attn", 0, 0},
    {DsparkStage::AttnCombine,   "dspark_attn", 1, 0},
    {DsparkStage::MarkovBias,    "dspark_head", 0, 0},
    {DsparkStage::AddBiasArgmax, "dspark_head", 1, 0},
    {DsparkStage::Confidence,    "dspark_head", 2, 0},
    {DsparkStage::VerifyRows,    "dspark_verify", 0, 0},
};
static_assert(sizeof(kStages) / sizeof(kStages[0]) ==
              static_cast<size_t>(DsparkStage::Count));
// The three push structs share one 64 B range, so each must fit it.
static_assert(sizeof(DsparkGemvPush) <= kPushBytes);
static_assert(sizeof(DsparkAttnPush) <= kPushBytes);
static_assert(sizeof(DsparkHeadPush) <= kPushBytes);
static_assert(sizeof(DsparkVerifyPush) <= kPushBytes);
}  // namespace

Result<void> DsparkRunner::make(DsparkStage s, const std::string& spv, uint32_t stage_const) {
    uint32_t act_quant = 0;
    for (const StageDef& d : kStages)
        if (d.stage == s) act_quant = d.act_quant;

    PipelineSpec ps;
    ps.m             = spec_.m;
    ps.lanes_per_row = spec_.lanes_per_row;
    ps.rows_per_wg   = 256 / spec_.lanes_per_row;
    ps.subgroup_size = spec_.subgroup_size;
    // Specialisation ids 4..7, the contract attn_common.slang declares:
    // Stage, ActQuant, RowsPerLane, HeadsPerWg.
    ps.extra = {stage_const, act_quant, spec_.rows_per_lane, spec_.heads_per_wg};

    PipelineLayoutSpec la;
    la.storage_buffers    = 1;
    la.push_constant_size = kPushBytes;
    return pipes_[static_cast<uint32_t>(s)].create(*device_, spv, la, ps);
}

Result<void> DsparkRunner::create(Device& device, MemoryAllocator& alloc,
                                  const std::string& shader_dir, const DsparkSpec& spec) {
    // Validated before the device is touched, so a bad spec fails the same way
    // whether or not a GPU is present.
    if (spec.m == 0 || spec.m > kMaxDraftM)
        return fail(Err::InvalidArgument,
                    "m must be 1..6 per design §1.2 (dspark_block_size 5, verify batch k+1)");
    if (spec.lanes_per_row != 16 && spec.lanes_per_row != 32 && spec.lanes_per_row != 64)
        return fail(Err::InvalidArgument, "lanes_per_row must be 16, 32 or 64 (design §7.1 rule 3)");
    if (spec.subgroup_size != 32)
        return fail(Err::InvalidArgument,
                    "subgroup_size must be 32: dspark_attn.slang's score stage reduces one "
                    "512-wide KV row across the 32 lanes of a single Wave32 (design §7.5, §7.12)");
    if (spec.rows_per_lane == 0 || spec.rows_per_lane > 4)
        return fail(Err::InvalidArgument, "rows_per_lane must be 1..4 (design §7.1)");
    if (spec.heads_per_wg == 0 || spec.heads_per_wg > 8)
        return fail(Err::InvalidArgument, "heads_per_wg must be 1..8 (design §7.5)");

    destroy();
    if (!device.valid()) return fail(Err::FailedPrecondition, "device is not created");
    device_ = &device;
    alloc_  = &alloc;
    spec_   = spec;

    const uint32_t n = static_cast<uint32_t>(DsparkStage::Count);
    auto t = alloc.allocate(uint64_t(n) * kAttnStageStride, true, false);
    if (!t) { destroy(); return std::unexpected(t.error()); }
    table_ = *t;
    std::memset(table_.host_ptr, 0, static_cast<size_t>(table_.bytes));

    if (auto r = descriptors_.create(device, n, n); !r) { destroy(); return r; }
    for (const StageDef& d : kStages) {
        if (auto r = make(d.stage, shader_dir + "/" + d.spv + ".spv", d.stage_const); !r) {
            destroy();
            return fail(r.error().code,
                        std::format("{}: {}", dspark_stage_name(d.stage), r.error().message));
        }
        const uint32_t i = static_cast<uint32_t>(d.stage);
        std::vector<BufferBinding> b(1);
        b[0] = {0, uint64_t(i) * kAttnStageStride, kAttnStageStride, table_.buffer};
        auto set = descriptors_.allocate(pipes_[i], b);
        if (!set) {
            destroy();
            return fail(set.error().code,
                        std::format("{}: {}", dspark_stage_name(d.stage), set.error().message));
        }
        sets_[i] = *set;
    }
    if (auto r = pool_.create(device); !r) { destroy(); return r; }
    return {};
}

void DsparkRunner::destroy() {
    pool_.destroy();
    descriptors_.destroy();
    for (Pipeline& p : pipes_) p.destroy();
    if (alloc_ && table_.valid()) alloc_->free(table_);
    table_ = GpuBuffer{};
    for (VkDescriptorSet& s : sets_) s = VK_NULL_HANDLE;
    device_ = nullptr;
    alloc_  = nullptr;
}

uint64_t* DsparkRunner::slots(DsparkStage s) {
    if (!table_.host_ptr) return nullptr;
    return reinterpret_cast<uint64_t*>(static_cast<std::byte*>(table_.host_ptr) +
                                       uint64_t(static_cast<uint32_t>(s)) * kAttnStageStride);
}

Result<void> DsparkRunner::record(CommandBuffer& cmd, DsparkStage s, const void* push,
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

Result<void> DsparkRunner::dispatch_now(DsparkStage s, const void* push, uint32_t push_bytes,
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
