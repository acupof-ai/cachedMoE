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
        case AttnStage::MhcPostB:    return "mega_mhc.post.ffn";
        case AttnStage::MhcMixB:     return "mega_mhc.mix.ffn";
        case AttnStage::MhcFinalB:   return "mega_mhc.final.ffn";
        case AttnStage::MhcClose:    return "mega_mhc.close";
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
        case AttnStage::CmpKvGemv:   return "compressor.wkv";
        case AttnStage::CmpGateGemv: return "compressor.wgate";
        case AttnStage::CmpNorm:     return "compressor.norm";
        case AttnStage::CmpStore:    return "compressor.store";
        case AttnStage::IdxQGemv:    return "indexer.wq_b";
        case AttnStage::IdxQFinish:  return "indexer.q_finish";
        case AttnStage::IdxKey:      return "indexer.key";
        case AttnStage::IdxWeights:  return "indexer.weights";
        case AttnStage::IdxScore:    return "indexer.score";
        case AttnStage::IdxTopK:     return "indexer.topk";
        case AttnStage::WqAKSplit:   return "wq_a.ksplit";
        case AttnStage::WqAKCombine: return "wq_a.kcombine";
        case AttnStage::WkvKSplit:   return "wkv.ksplit";
        case AttnStage::WkvKFinish:  return "wkv.kfinish";
        case AttnStage::WoAKSplit:   return "wo_a.ksplit";
        case AttnStage::WoAKCombine: return "wo_a.kcombine";
        case AttnStage::WoBKSplit:   return "wo_b.ksplit";
        case AttnStage::WoBKCombine: return "wo_b.kcombine";
        case AttnStage::AttnScoreT:  return "sparse_attn_t.score";
        case AttnStage::AttnPvT:     return "sparse_attn_t.pv";
        case AttnStage::AttnFinishT: return "sparse_attn_t.finish";
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
Result<void> AttnRunner::make(const StageDef&, const std::string&) {
    return fail(Err::Unavailable, "no vulkan");
}
uint32_t AttnRunner::ksplit(AttnStage) const { return 1; }

#else

namespace {
// Bigger than any of the push structs in the header, so one pipeline layout
// serves every stage and vkCmdPushConstants never runs past the range.
constexpr uint32_t kPushBytes = 64;

// wo_a is the one entry with act_quant = 0: `Attention.forward` reaches its
// weight through an einsum, not `linear()`, so its activation never makes the
// fp8 round trip (attn_common.slang explains why that matters).
// rows_cap is per stage because row blocking trades workgroups and register
// pressure for activation reuse, and which side wins is not predictable from
// the shape alone. The numbers in the comments are bench/attn_bench at
// --layers 8, one runner per layer, idle machine: the two tall-and-narrow
// kernels gain from it and everything else loses, including the head, whose
// activation is only 20 KiB and already in LDS.
using StageDef = AttnRunner::StageDef;
constexpr StageDef kStages[] = {
    {AttnStage::MhcPost,     "mega_mhc",    0, 1, 1, 1},
    {AttnStage::MhcMix,      "mega_mhc",    1, 1, 1, 1},
    {AttnStage::MhcFinal,    "mega_mhc",    2, 1, 1, 1},
    {AttnStage::MhcPostB,    "mega_mhc",    0, 1, 1, 1},
    {AttnStage::MhcMixB,     "mega_mhc",    1, 1, 1, 1},
    {AttnStage::MhcFinalB,   "mega_mhc",    2, 1, 1, 1},
    {AttnStage::MhcClose,    "mega_mhc",    0, 1, 1, 1},
    {AttnStage::WqA,         "wq_a",        0, 1, 2, 0},   // 1280 rows: 125 -> 147 GB/s
    {AttnStage::WqB,         "wq_b",        0, 1, 2, 1},   // 32768:      114 -> 160
    {AttnStage::WkvGemv,     "wkv",         0, 1, 1, 1},   // 512: workgroup-starved
    {AttnStage::WkvFinish,   "wkv",         1, 1, 1, 1},
    {AttnStage::AttnScore,   "sparse_attn", 0, 1, 1, 1},
    {AttnStage::AttnCombine, "sparse_attn", 1, 1, 1, 1},
    {AttnStage::WoA,         "wo_a",        0, 0, 1, 0},   // 8192:  166 -> 132, so 1
    {AttnStage::WoB,         "wo_b",        0, 1, 1, 1},   // 5120:  124 -> 114, so 1
    {AttnStage::GateScore,   "gate",        0, 1, 1, 1},   // 384: no row blocking
    {AttnStage::GateTopK,    "gate",        1, 1, 1, 1},
    {AttnStage::Head,        "head",        0, 1, 1, 1},   // 129280: 235 -> 208, so 1
    // §7.4. The compressor's two projections are bf16 [512 x 5120], the same
    // shape wkv is fp8, and workgroup-starved for the same reason: 512 rows at
    // eight a workgroup is 64 of them on 40 CUs, and row blocking would make it
    // worse, not better.
    {AttnStage::CmpKvGemv,   "compressor",  0, 1, 1, 1},
    {AttnStage::CmpGateGemv, "compressor",  0, 1, 1, 1},
    {AttnStage::CmpNorm,     "compressor",  1, 1, 1, 1},
    {AttnStage::CmpStore,    "compressor",  2, 1, 1, 1},
    // The indexer's wq_b is [4096 x 1280] -- tall and narrow like the main
    // wq_b, so the same rows_per_lane cap of 2.
    {AttnStage::IdxQGemv,    "indexer",     0, 1, 2, 1},
    {AttnStage::IdxQFinish,  "indexer",     1, 1, 1, 1},
    {AttnStage::IdxKey,      "indexer",     2, 1, 1, 1},
    {AttnStage::IdxWeights,  "indexer",     3, 1, 1, 1},
    {AttnStage::IdxScore,    "indexer",     4, 1, 1, 1},
    {AttnStage::IdxTopK,     "indexer",     5, 1, 1, 1},
    // --- P3 ------------------------------------------------------------------
    // The split half of a K-split GEMV is the generic gemv_ksplit.spv; what
    // differs per stage is the slice count, whether the activation is
    // quantised, and the row blocking. The combine half is the same .spv's
    // stage 1 and has no knobs at all. `ksplit` comes from AttnSpec at create
    // time; docs/p2_attention.md §13 has the sweep behind each number.
    {AttnStage::WqAKSplit,   "gemv_ksplit", 0, 1, 1, 0},
    {AttnStage::WqAKCombine, "gemv_ksplit", 1, 1, 1, 1},
    {AttnStage::WkvKSplit,   "gemv_ksplit", 0, 1, 1, 1},
    {AttnStage::WkvKFinish,  "wkv",         2, 1, 1, 1},
    {AttnStage::WoAKSplit,   "gemv_ksplit", 0, 0, 1, 0},
    {AttnStage::WoAKCombine, "gemv_ksplit", 1, 1, 1, 1},
    {AttnStage::WoBKSplit,   "gemv_ksplit", 0, 1, 1, 1},
    {AttnStage::WoBKCombine, "gemv_ksplit", 1, 1, 1, 1},
    {AttnStage::AttnScoreT,  "sparse_attn_t", 0, 1, 1, 1},
    {AttnStage::AttnPvT,     "sparse_attn_t", 1, 1, 1, 1},
    {AttnStage::AttnFinishT, "sparse_attn_t", 2, 1, 1, 1},
};
static_assert(sizeof(kStages) / sizeof(kStages[0]) ==
              static_cast<size_t>(AttnStage::Count));
}  // namespace

uint32_t AttnRunner::rows_per_lane(AttnStage s) const {
    const uint32_t idx = static_cast<uint32_t>(s);
    const uint32_t cap = ((spec_.sweep_rows_cap4 >> idx) & 1u) ? 4u : kStages[idx].rows_cap;
    const bool split = s == AttnStage::WqAKSplit || s == AttnStage::WkvKSplit ||
                       s == AttnStage::WoAKSplit || s == AttnStage::WoBKSplit;
    const uint32_t want = (split && spec_.rows_per_lane_ksplit)
                              ? spec_.rows_per_lane_ksplit : spec_.rows_per_lane;
    return want < cap ? want : cap;
}

// The K-split factor a stage's pipeline was compiled with. A split stage and
// its combine MUST agree: the combine's loop bound is the same specialisation
// constant.
uint32_t AttnRunner::ksplit(AttnStage s) const {
    switch (s) {
        case AttnStage::WqAKSplit:   case AttnStage::WqAKCombine: return spec_.ksplit_wq_a;
        case AttnStage::WkvKSplit:   case AttnStage::WkvKFinish:  return spec_.ksplit_wkv;
        case AttnStage::WoAKSplit:   case AttnStage::WoAKCombine: return spec_.ksplit_wo_a;
        case AttnStage::WoBKSplit:   case AttnStage::WoBKCombine: return spec_.ksplit_wo_b;
        default: return 1;
    }
}

Result<void> AttnRunner::make(const StageDef& d, const std::string& spv) {
    const AttnStage s = d.stage;
    // The tiled attention wants its own head grouping: 8, where the untiled
    // kernel measured 1 as the winner, because the KV tiling is what gives the
    // workgroup count back (docs/p2_attention.md §10.2 against §13).
    const uint32_t heads = (s == AttnStage::AttnScoreT || s == AttnStage::AttnFinishT)
                               ? spec_.tile_heads_per_wg
                               : (s == AttnStage::AttnPvT ? spec_.pv_heads_per_wg
                                                          : spec_.heads_per_wg);
    const uint32_t idx = static_cast<uint32_t>(s);
    uint32_t wave = d.wave_reduce;
    if ((spec_.sweep_wave_on >> idx) & 1u)  wave = 1;
    if ((spec_.sweep_wave_off >> idx) & 1u) wave = 0;
    PipelineSpec ps;
    ps.m             = 1;
    ps.lanes_per_row = spec_.lanes_per_row;
    ps.rows_per_wg   = 256 / spec_.lanes_per_row;
    ps.subgroup_size = spec_.subgroup_size;
    ps.extra = {d.stage_const, d.act_quant, rows_per_lane(s),
                heads,
                wave, ksplit(s), spec_.fp8_arith_decode};
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
    // sparse_attn.slang sizes its staged-q LDS at kMaxHeadsPerWg * 512, and its
    // stage 1 gives a thread head_dim * heads_per_wg / 256 output dims against
    // a 16-wide register array.
    if (spec.heads_per_wg < 1 || spec.heads_per_wg > 8)
        return fail(Err::InvalidArgument, "heads_per_wg must be 1..8 (design §7.5)");
    if (spec.tile_heads_per_wg < 1 || spec.tile_heads_per_wg > 8 ||
        spec.pv_heads_per_wg < 1 || spec.pv_heads_per_wg > 8)
        return fail(Err::InvalidArgument,
                    "tile_heads_per_wg and pv_heads_per_wg must be 1..8 (design §7.5)");
    // A K-split factor is the combine's loop bound and the divisor of the slice
    // width; anything that is not a power of two either fails to divide K/32 or
    // leaves a ragged last slice gemv_ksplit.slang does not test for.
    for (uint32_t f : {spec.ksplit_wq_a, spec.ksplit_wkv, spec.ksplit_wo_a, spec.ksplit_wo_b})
        if (f == 0 || (f & (f - 1)) != 0 || f > 16)
            return fail(Err::InvalidArgument, "a K-split factor must be a power of two, 1..16");
    // The indexer's score stage (§7.4) puts the 32 heads of one position in the
    // 32 lanes of one wave, and sparse_attn's score stage does the same for the
    // 32 lanes of a KV row.
    if (spec.subgroup_size != 32)
        return fail(Err::InvalidArgument,
                    "sparse_attn and indexer.score both reduce across one Wave32");
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
        if (auto r = make(d, shader_dir + "/" + d.spv + ".spv"); !r) {
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
