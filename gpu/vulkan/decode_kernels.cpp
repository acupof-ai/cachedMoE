#include "gpu/vulkan/decode_kernels.h"

#include <cstring>
#include <format>

namespace deepmoe::gpu {

const char* decode_stage_name(DecodeStage s) {
    switch (s) {
        case DecodeStage::EngramGemv: return "engram.gemv";
        case DecodeStage::EngramGate: return "engram.gate";
        case DecodeStage::Argmax:     return "head.argmax";
        case DecodeStage::SampleTopK: return "sample_topk";
        case DecodeStage::Count:      break;
    }
    return "?";
}

const char* mgt_stage_name(MgtStage s) {
    switch (s) {
        case MgtStage::WqASplit:       return "mgt1.wq_a.split";
        case MgtStage::WqACombine:     return "mgt1.wq_a.combine";
        case MgtStage::QNorm:          return "mgt1.q_norm";
        case MgtStage::WqBSplit:       return "mgt1.wq_b.split";
        case MgtStage::WqBFinish:      return "mgt1.wq_b.finish";
        case MgtStage::WkvSplit:       return "mgt1.wkv.split";
        case MgtStage::WkvFinish:      return "mgt1.wkv.finish";
        case MgtStage::WoASplit:       return "mgt1.wo_a.split";
        case MgtStage::WoACombine:     return "mgt1.wo_a.combine";
        case MgtStage::WoBSplit:       return "mgt1.wo_b.split";
        case MgtStage::WoBCombine:     return "mgt1.wo_b.combine";
        case MgtStage::IdxQSplit:      return "mgt1.indexer.wq_b.split";
        case MgtStage::IdxQCombine:    return "mgt1.indexer.wq_b.combine";
        case MgtStage::MhcPost:        return "mgt1.mhc.post";
        case MgtStage::MhcMix:         return "mgt1.mhc.mix";
        case MgtStage::MhcFinal:       return "mgt1.mhc.final";
        case MgtStage::MhcPostB:       return "mgt1.mhc.post.ffn";
        case MgtStage::MhcMixB:        return "mgt1.mhc.mix.ffn";
        case MgtStage::MhcFinalB:      return "mgt1.mhc.final.ffn";
        case MgtStage::MhcClose:       return "mgt1.mhc.close";
        case MgtStage::AttnScore:      return "mgt1.attn.score";
        case MgtStage::AttnPv:         return "mgt1.attn.pv";
        case MgtStage::GateScore:      return "mgt1.gate.score";
        case MgtStage::GateTopK:       return "mgt1.gate.topk";
        case MgtStage::CmpKvGemv:      return "mgt1.compressor.wkv";
        case MgtStage::CmpGateGemv:    return "mgt1.compressor.wgate";
        case MgtStage::CmpPool:        return "mgt1.compressor.pool";
        case MgtStage::CmpState:       return "mgt1.compressor.state";
        case MgtStage::CmpStore:       return "mgt1.compressor.store";
        case MgtStage::IdxQFinish:     return "mgt1.indexer.q_finish";
        case MgtStage::IdxKey:         return "mgt1.indexer.key";
        case MgtStage::IdxWeights:     return "mgt1.indexer.weights";
        case MgtStage::IdxScore:       return "mgt1.indexer.score";
        case MgtStage::IdxTopK:        return "mgt1.indexer.topk";
        case MgtStage::IdxBlockKeys:   return "mgt1.indexer.block_keys";
        case MgtStage::IdxBlockSelect: return "mgt1.indexer.block_select";
        case MgtStage::IdxApplyCand:   return "mgt1.indexer.apply_candidates";
        case MgtStage::Head:           return "mgt1.head";
        case MgtStage::HeadArgmax:     return "mgt1.head.argmax";
        case MgtStage::HeadTopK:       return "mgt1.head.topk";
        case MgtStage::EngramGemv:     return "mgt1.engram.gemv";
        case MgtStage::EngramGate:     return "mgt1.engram.gate";
        case MgtStage::Count:          break;
    }
    return "?";
}

#if !defined(DEEPMOE_ENABLE_VULKAN)

Result<void> DecodeRunner::create(Device&, MemoryAllocator&, const std::string&, const AttnSpec&) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
void DecodeRunner::destroy() {}
uint64_t* DecodeRunner::slots(DecodeStage) { return nullptr; }
Result<void> DecodeRunner::record(CommandBuffer&, DecodeStage, const void*, uint32_t, uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}
Result<void> DecodeRunner::dispatch_now(DecodeStage, const void*, uint32_t, uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}
Result<void> DecodeRunner::make(DecodeStage, const std::string&, uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}
Result<void> MgtRunner::create(Device&, MemoryAllocator&, const std::string&, const MgtSpec&) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
void MgtRunner::destroy() {}
Result<void> MgtRunner::ensure(uint32_t) { return fail(Err::Unavailable, "no vulkan"); }
uint64_t* MgtRunner::slots(MgtStage) { return nullptr; }
Result<void> MgtRunner::record(CommandBuffer&, uint32_t, MgtStage, const void*, uint32_t, uint32_t,
                               uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}
Result<void> MgtRunner::dispatch_now(uint32_t, MgtStage, const void*, uint32_t, uint32_t, uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}
Result<void> MgtRunner::make(uint32_t, MgtStage) { return fail(Err::Unavailable, "no vulkan"); }
uint32_t MgtRunner::ksplit(MgtStage) const { return 1; }

#else

namespace {
// The same 64 B range attn_kernels.cpp uses, so a stage can be pushed either
// struct without the layout caring.
constexpr uint32_t kPushBytes = 64;

struct StageDef { DecodeStage stage; const char* spv; uint32_t stage_const; };

constexpr StageDef kStages[] = {
    {DecodeStage::EngramGemv, "engram", 0},
    {DecodeStage::EngramGate, "engram", 1},
    {DecodeStage::Argmax,     "head",   1},
    {DecodeStage::SampleTopK, "sample_topk", 0},
};
static_assert(sizeof(kStages) / sizeof(kStages[0]) ==
              static_cast<size_t>(DecodeStage::Count));
}  // namespace

Result<void> DecodeRunner::make(DecodeStage s, const std::string& spv, uint32_t stage_const) {
    PipelineSpec ps;
    ps.m             = 1;
    ps.lanes_per_row = spec_.lanes_per_row;
    ps.rows_per_wg   = 256 / spec_.lanes_per_row;
    ps.subgroup_size = spec_.subgroup_size;
    // act_quant is on (engram's wkv is an fp8 Linear) and RowsPerLane is 1:
    // neither kernel row-blocks, and head.slang's argmax stage ignores both.
    ps.extra = {stage_const, 1u, 1u};
    PipelineLayoutSpec la;
    la.storage_buffers    = 1;
    la.push_constant_size = kPushBytes;
    return pipes_[static_cast<uint32_t>(s)].create(*device_, spv, la, ps);
}

Result<void> DecodeRunner::create(Device& device, MemoryAllocator& alloc,
                                  const std::string& shader_dir, const AttnSpec& spec) {
    destroy();
    if (!device.valid()) return fail(Err::FailedPrecondition, "device is not created");
    if (spec.lanes_per_row != 16 && spec.lanes_per_row != 32 && spec.lanes_per_row != 64)
        return fail(Err::InvalidArgument, "lanes_per_row must be 16, 32 or 64 (design §7.1 rule 3)");
    device_ = &device;
    alloc_  = &alloc;
    spec_   = spec;
    spec_.rows_per_lane = 1;

    const uint32_t n = static_cast<uint32_t>(DecodeStage::Count);
    auto t = alloc.allocate(uint64_t(n) * kAttnStageStride, true, false);
    if (!t) { destroy(); return std::unexpected(t.error()); }
    table_ = *t;
    std::memset(table_.host_ptr, 0, static_cast<size_t>(table_.bytes));

    if (auto r = descriptors_.create(device, n, n); !r) { destroy(); return r; }
    for (const StageDef& d : kStages) {
        if (auto r = make(d.stage, shader_dir + "/" + d.spv + ".spv", d.stage_const); !r) {
            destroy();
            return fail(r.error().code,
                        std::format("{}: {}", decode_stage_name(d.stage), r.error().message));
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

void DecodeRunner::destroy() {
    pool_.destroy();
    descriptors_.destroy();
    for (Pipeline& p : pipes_) p.destroy();
    if (alloc_ && table_.valid()) alloc_->free(table_);
    table_ = GpuBuffer{};
    for (VkDescriptorSet& s : sets_) s = VK_NULL_HANDLE;
    device_ = nullptr;
    alloc_ = nullptr;
}

uint64_t* DecodeRunner::slots(DecodeStage s) {
    if (!table_.host_ptr) return nullptr;
    return reinterpret_cast<uint64_t*>(static_cast<std::byte*>(table_.host_ptr) +
                                       uint64_t(static_cast<uint32_t>(s)) * kAttnStageStride);
}

Result<void> DecodeRunner::record(CommandBuffer& cmd, DecodeStage s, const void* push,
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

Result<void> DecodeRunner::dispatch_now(DecodeStage s, const void* push, uint32_t push_bytes,
                                        uint32_t groups) {
    auto cb = pool_.acquire();
    if (!cb) return std::unexpected(cb.error());
    CommandBuffer cmd = *cb;
    if (auto r = cmd.begin(); !r) return r;
    if (auto r = record(cmd, s, push, push_bytes, groups); !r) return r;
    if (auto r = cmd.end(); !r) return r;
    return submit_and_wait(*device_, cmd);
}

// --- MgtRunner (Track T) ----------------------------------------------------------

namespace {
struct MgtDef {
    MgtStage    stage;
    const char* spv;
    uint32_t    stage_const;
    uint32_t    act_quant;
    uint32_t    wave;      // row_reduce as one WaveActiveSum (attn_common.slang)
};
// `wave` follows attn_kernels.cpp's kStages for the same GEMV family: the two
// wide re-staging kernels (wq_a, wo_a) measured better with the LDS tree.
constexpr MgtDef kMgt[] = {
    {MgtStage::WqASplit,       "mgt1_gemv",   0, 1, 0},
    {MgtStage::WqACombine,     "mgt1_gemv",   1, 1, 1},
    {MgtStage::QNorm,          "mgt1_gemv",   3, 1, 1},
    {MgtStage::WqBSplit,       "mgt1_gemv",   0, 1, 1},
    {MgtStage::WqBFinish,      "mgt1_gemv",   2, 1, 1},
    {MgtStage::WkvSplit,       "mgt1_gemv",   0, 1, 1},
    {MgtStage::WkvFinish,      "mgt1_gemv",   4, 1, 1},
    {MgtStage::WoASplit,       "mgt1_gemv",   0, 0, 0},
    {MgtStage::WoACombine,     "mgt1_gemv",   1, 0, 1},
    {MgtStage::WoBSplit,       "mgt1_gemv",   0, 1, 1},
    {MgtStage::WoBCombine,     "mgt1_gemv",   1, 1, 1},
    {MgtStage::IdxQSplit,      "mgt1_gemv",   0, 1, 1},
    {MgtStage::IdxQCombine,    "mgt1_gemv",   1, 1, 1},
    {MgtStage::MhcPost,        "mgt1_mhc",    0, 1, 1},
    {MgtStage::MhcMix,         "mgt1_mhc",    1, 1, 1},
    {MgtStage::MhcFinal,       "mgt1_mhc",    2, 1, 1},
    {MgtStage::MhcPostB,       "mgt1_mhc",    0, 1, 1},
    {MgtStage::MhcMixB,        "mgt1_mhc",    1, 1, 1},
    {MgtStage::MhcFinalB,      "mgt1_mhc",    2, 1, 1},
    {MgtStage::MhcClose,       "mgt1_mhc",    0, 1, 1},
    {MgtStage::AttnScore,      "mgt1_attn",   0, 1, 1},
    {MgtStage::AttnPv,         "mgt1_attn",   1, 1, 1},
    {MgtStage::GateScore,      "mgt1_gate",   0, 1, 1},
    {MgtStage::GateTopK,       "mgt1_gate",   1, 1, 1},
    {MgtStage::CmpKvGemv,      "mgt1_cmp",    0, 1, 1},
    {MgtStage::CmpGateGemv,    "mgt1_cmp",    0, 1, 1},
    {MgtStage::CmpPool,        "mgt1_cmp",    1, 1, 1},
    {MgtStage::CmpState,       "mgt1_cmp",    2, 1, 1},
    {MgtStage::CmpStore,       "mgt1_cmp",    3, 1, 1},
    {MgtStage::IdxQFinish,     "mgt1_idx",    1, 1, 1},
    {MgtStage::IdxKey,         "mgt1_idx",    2, 1, 1},
    {MgtStage::IdxWeights,     "mgt1_idx",    3, 1, 1},
    {MgtStage::IdxScore,       "mgt1_idx",    4, 1, 1},
    {MgtStage::IdxTopK,        "mgt1_idx",    5, 1, 1},
    {MgtStage::IdxBlockKeys,   "mgt1_idx",    6, 1, 1},
    {MgtStage::IdxBlockSelect, "mgt1_idx",    7, 1, 1},
    {MgtStage::IdxApplyCand,   "mgt1_idx",    8, 1, 1},
    {MgtStage::Head,           "mgt1_head",   0, 1, 1},
    {MgtStage::HeadArgmax,     "mgt1_head",   1, 1, 1},
    {MgtStage::HeadTopK,       "mgt1_head",   2, 1, 1},
    {MgtStage::EngramGemv,     "mgt1_engram", 0, 1, 1},
    {MgtStage::EngramGate,     "mgt1_engram", 1, 1, 1},
};
static_assert(sizeof(kMgt) / sizeof(kMgt[0]) == static_cast<size_t>(MgtStage::Count));
}  // namespace

uint32_t MgtRunner::ksplit(MgtStage s) const {
    switch (s) {
        case MgtStage::WqASplit:  case MgtStage::WqACombine:  return spec_.ksplit_wq_a;
        case MgtStage::WqBSplit:  case MgtStage::WqBFinish:   return spec_.ksplit_wq_b;
        case MgtStage::WkvSplit:  case MgtStage::WkvFinish:   return spec_.ksplit_wkv;
        case MgtStage::WoASplit:  case MgtStage::WoACombine:  return spec_.ksplit_wo_a;
        case MgtStage::WoBSplit:  case MgtStage::WoBCombine:  return spec_.ksplit_wo_b;
        case MgtStage::IdxQSplit: case MgtStage::IdxQCombine: return spec_.ksplit_idx;
        default: return 1;
    }
}

Result<void> MgtRunner::create(Device& device, MemoryAllocator& alloc,
                               const std::string& shader_dir, const MgtSpec& spec) {
    destroy();
    if (!device.valid()) return fail(Err::FailedPrecondition, "device is not created");
    if (spec.lanes_per_row != 32 || spec.subgroup_size != 32)
        return fail(Err::InvalidArgument, "the M > 1 kernels reduce across one Wave32 a row");
    auto slice_ok = [](uint32_t k, uint32_t f) {
        return f && (f & (f - 1)) == 0 && (k / 32) % f == 0 && k / f <= 1024;
    };
    if (!slice_ok(5120, spec.ksplit_wq_a) || !slice_ok(1280, spec.ksplit_wq_b) ||
        !slice_ok(5120, spec.ksplit_wkv) || !slice_ok(4096, spec.ksplit_wo_a) ||
        !slice_ok(8192, spec.ksplit_wo_b) || !slice_ok(1280, spec.ksplit_idx))
        return fail(Err::InvalidArgument,
                    "an M > 1 K-split factor must be a power of two dividing K/32 with K/f <= 1024");
    if (spec.head_slices == 0 || 5120 % spec.head_slices || 5120 / spec.head_slices > 1024 ||
        (5120 / spec.head_slices) % 32)
        return fail(Err::InvalidArgument, "head_slices must cut 5120 into <= 1024-wide blocks of 32");
    if (spec.engram_slices == 0 || 6144 % spec.engram_slices ||
        6144 / spec.engram_slices > 1024 || (6144 / spec.engram_slices) % 32)
        return fail(Err::InvalidArgument, "engram_slices must cut 6144 into <= 1024-wide blocks of 32");
    device_ = &device;
    alloc_  = &alloc;
    spec_   = spec;
    dir_    = shader_dir;
    const uint32_t n = static_cast<uint32_t>(MgtStage::Count);
    auto t = alloc.allocate(uint64_t(n) * kAttnStageStride, true, false);
    if (!t) { destroy(); return std::unexpected(t.error()); }
    table_ = *t;
    std::memset(table_.host_ptr, 0, static_cast<size_t>(table_.bytes));
    if (auto r = descriptors_.create(device, n * (kMgtMaxM + 1), n * (kMgtMaxM + 1)); !r) {
        destroy();
        return r;
    }
    if (auto r = pool_.create(device); !r) { destroy(); return r; }
    return {};
}

void MgtRunner::destroy() {
    pool_.destroy();
    descriptors_.destroy();
    for (PerM& p : per_m_) {
        p.pipes.clear();
        p.sets.clear();
        p.ready = false;
    }
    if (alloc_ && table_.valid()) alloc_->free(table_);
    table_ = GpuBuffer{};
    device_ = nullptr;
    alloc_ = nullptr;
}

Result<void> MgtRunner::make(uint32_t m, MgtStage s) {
    const MgtDef& d = kMgt[static_cast<uint32_t>(s)];
    PipelineSpec ps;
    ps.m             = m;
    ps.lanes_per_row = spec_.lanes_per_row;
    ps.rows_per_wg   = 256 / spec_.lanes_per_row;
    ps.subgroup_size = spec_.subgroup_size;
    const uint32_t heads = (s == MgtStage::AttnScore) ? spec_.tile_heads_per_wg : 1u;
    ps.extra = {d.stage_const, d.act_quant, 1u, heads, d.wave, ksplit(s), 0u};
    PipelineLayoutSpec la;
    la.storage_buffers    = 1;
    la.push_constant_size = kPushBytes;
    PerM& pm = per_m_[m];
    const uint32_t i = static_cast<uint32_t>(s);
    if (auto r = pm.pipes[i].create(*device_, dir_ + "/" + d.spv + ".spv", la, ps); !r)
        return fail(r.error().code, std::format("{} (M={}): {}", mgt_stage_name(s), m,
                                                r.error().message));
    std::vector<BufferBinding> b(1);
    b[0] = {0, uint64_t(i) * kAttnStageStride, kAttnStageStride, table_.buffer};
    auto set = descriptors_.allocate(pm.pipes[i], b);
    if (!set) return std::unexpected(set.error());
    pm.sets[i] = *set;
    return {};
}

Result<void> MgtRunner::ensure(uint32_t m) {
    if (!device_) return fail(Err::FailedPrecondition, "MgtRunner is not created");
    if (m < 1 || m > kMgtMaxM)
        return fail(Err::InvalidArgument, std::format("M = {} is outside 1..{}", m, kMgtMaxM));
    PerM& pm = per_m_[m];
    if (pm.ready) return {};
    const uint32_t n = static_cast<uint32_t>(MgtStage::Count);
    pm.pipes.clear();
    pm.pipes.resize(n);
    pm.sets.assign(n, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < n; ++i)
        if (auto r = make(m, static_cast<MgtStage>(i)); !r) return r;
    pm.ready = true;
    return {};
}

uint64_t* MgtRunner::slots(MgtStage s) {
    if (!table_.host_ptr) return nullptr;
    return reinterpret_cast<uint64_t*>(static_cast<std::byte*>(table_.host_ptr) +
                                       uint64_t(static_cast<uint32_t>(s)) * kAttnStageStride);
}

Result<void> MgtRunner::record(CommandBuffer& cmd, uint32_t m, MgtStage s, const void* push,
                               uint32_t push_bytes, uint32_t gx, uint32_t gy) {
    if (m < 1 || m > kMgtMaxM || !per_m_[m].ready)
        return fail(Err::FailedPrecondition, std::format("M = {} pipelines are not built", m));
    const uint32_t i = static_cast<uint32_t>(s);
    const PerM& pm = per_m_[m];
    if (push_bytes > kPushBytes)
        return fail(Err::InvalidArgument, "push constants exceed the shared 64 B range");
    if (gx == 0 || gy == 0) return fail(Err::InvalidArgument, "zero workgroups");
    if (gx > 65535 || gy > 65535)
        return fail(Err::InvalidArgument,
                    std::format("{}: {} x {} workgroups exceeds a dispatch", mgt_stage_name(s),
                                gx, gy));
    if (auto r = cmd.bind(pm.pipes[i], pm.sets[i]); !r) return r;
    if (push_bytes)
        if (auto r = cmd.push(pm.pipes[i], push, push_bytes); !r) return r;
    return cmd.dispatch(gx, gy);
}

Result<void> MgtRunner::dispatch_now(uint32_t m, MgtStage s, const void* push,
                                     uint32_t push_bytes, uint32_t gx, uint32_t gy) {
    auto cb = pool_.acquire();
    if (!cb) return std::unexpected(cb.error());
    CommandBuffer cmd = *cb;
    if (auto r = cmd.begin(); !r) return r;
    if (auto r = record(cmd, m, s, push, push_bytes, gx, gy); !r) return r;
    if (auto r = cmd.end(); !r) return r;
    return submit_and_wait(*device_, cmd);
}

#endif  // DEEPMOE_ENABLE_VULKAN

}  // namespace deepmoe::gpu
