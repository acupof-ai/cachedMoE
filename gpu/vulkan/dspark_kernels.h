// Host side of the DSpark draft kernels (design §7.12): the three shaders of
// gpu/shaders/dspark_*.slang, nine pipelines, one address table.
//
// What a draft cycle needs from this runner
// -----------------------------------------
// `main_proj` + `main_norm` once, then three `mtp` stages of Q / KV / window
// attention / O, then the head-side tail -- five sequential Markov steps and one
// confidence projection. The Mega-mHC, hc_pre/hc_post, gate and FP4 MoE halves
// of a draft stage are NOT here: mega_mhc.slang, gate.slang, moe_gateup.slang
// and moe_down.slang already compute them, and they take `M` as a
// specialisation constant, so the draft path drives them through AttnRunner and
// MoeRunner with `m = 5` rather than through a second copy.
//
// Why DSpark gets its own runner rather than more stages on AttnRunner
// --------------------------------------------------------------------
// A stage owns one 32-slot slice of a shared `uint64_t` address table, and a
// runner owns one `kStages` array and one enum. `AttnRunner` is the attention
// track's file -- its enum is explicitly append-only because runtime/ names
// stages by enumerator -- and `DecodeRunner` is where the two kernels that
// belong to neither the attention chain nor the MoE path already went, for
// exactly this reason. `DsparkRunner` is developed alongside the speculative
// decode loop of design §10, on a different clock from the attention chain, and
// two tracks that share an enum share a merge conflict on every append and, far
// worse, share an address table: two dispatches of the same stage in one command
// buffer both see whatever was written to that slice last. So: a third runner,
// the same shape as the other two.
//
// It is otherwise deliberately identical to DecodeRunner. One descriptor per
// pipeline, holding a 32-slot slice of a shared table that the host writes and
// the shader dereferences (design §5.3's argument for the expert pointer table,
// applied to weights): the pinned set is 17.7 GB across many sub-2-GiB slabs and
// one descriptor per tensor per stage would be thousands of them. Write
// addresses into `slots(stage)`, push a small struct of dimensions, dispatch.
// Nothing is rebound between stages.
//
// `M` is a pipeline constant, `m` is a push constant
// --------------------------------------------------
// Parts of a draft cycle run at M = 1 -- `main_proj` + `main_norm`, and the
// `main_kv` half of the draft attention, which sees only the committed main
// positions -- and the rest at M = 5. `M` is a specialisation constant baked
// into a pipeline, so honouring that literally would mean a second pipeline for
// every affected stage: eighteen instead of nine, eighteen descriptor sets,
// eighteen table slices. Instead every shader loop over the batch is bounded by
// `min(M, pc.m)` and the caller pushes the live count. The M = 1 dispatches lose
// nothing measurable -- the loop simply stops after one column, and the weight
// read, which is what a GEMV costs, is identical either way -- and the pipeline
// count halves.
//
// Ownership/threading: created, recorded and submitted from the single GPU
// submit thread. Owns its pipelines, descriptor pool, address table and command
// pool; the weights belong to store::PinnedStore and the activations to whoever
// allocated them.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/status.h"
#include "gpu/vulkan/attn_kernels.h"
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/descriptor.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/pipeline.h"

namespace deepmoe::gpu {

enum class DsparkStage : uint32_t {
    // --- dspark_gemv.slang ---------------------------------------------------
    Gemv = 0,       // §7.12: y[m][row] = sum_k W[row][k] * x[m][k], fp8 E4M3
    RmsNorm,        // §7.12: y[m][i] = w[i] * x[m][i] * rsqrt(mean + eps), bf16 out
    RopeQuant,      // §7.12: RoPE over the last rope_dim dims (+ the KV act_quant)
    WoA,            // §7.12: einsum("bsgd,grd->bsgr"), fp8 weight, no act_quant
    // --- dspark_attn.slang ---------------------------------------------------
    AttnScore,      // §7.12: every (m, head, t) score of the M = 5 window attention
    AttnCombine,    // §7.12: the max, bf16 p, P.V, the sink, the divide
    // --- dspark_head.slang ---------------------------------------------------
    MarkovBias,     // §7.12: bias[v] = sum_d head_m[v][d] * embed_m[tok][d], 66.2 MB
    AddBiasArgmax,  // §7.12: logits[pos] += bias, then argmax over 129,280
    Confidence,     // §7.12: c[m] = dot(cat(x[m], markov_embed[m]), proj[5376])
    Count,
};

const char* dspark_stage_name(DsparkStage s);

// design §7.1's knobs, as far as the draft path has them. `act_quant` is not
// here: it is a property of the stage (Gemv always quantises because every fp8
// `Linear` does; WoA never does, because `Attention.forward` reaches wo_a
// through an einsum), not a sweep dimension.
struct DsparkSpec {
    // Draft positions the pipelines are specialised to. design §1.2 puts M at
    // 1..6 -- dspark_block_size is 5, a verify batch is k + 1 -- and
    // gpu/shaders/dspark_common.slang's `kMaxDraftM` sizes the LDS and register
    // budgets against 6, so anything higher is rejected rather than silently
    // truncated.
    uint32_t m             = 5;
    uint32_t lanes_per_row = 32;   // kernel_p1.md §3.2's winner; {16, 32, 64}
    uint32_t subgroup_size = 32;   // Wave32, and see create()
    // Weight rows per lane. Left at 1: the draft path's GEMVs are the same tall
    // thin shapes the decode path's are (wq_b is 32768 x 1280), but the M
    // dimension already amortises the staged activation five ways, which is what
    // row blocking buys at M = 1. Raising it is a sweep, not a default.
    uint32_t rows_per_lane = 1;
    // Attention heads one dspark_attn workgroup covers. Fixed at 1 and NOT read
    // by the shader: gpu/vulkan/attn_kernels.h records that grouping heads
    // measured a loss for the M = 1 attention (72 us at 1, 288 at 8) because 64
    // heads at 8 a workgroup is eight workgroups on forty CUs, and the draft
    // attention has a third of the KV and the same head count.
    uint32_t heads_per_wg  = 1;
    std::string name() const;
};

// --- push constants, mirroring the shaders exactly --------------------------

// gpu/shaders/dspark_gemv.slang, all four stages. One struct for four stages
// because a stage is one pipeline and a pipeline has one push range; the
// per-stage meaning of each field is in the trailing comment.
struct DsparkGemvPush {
    uint32_t m;              // live draft columns: 1 (main_proj, main_kv) or 5
    uint32_t rows;           // 0: 5120/1280/512/32768. 2: rows per position. 3: 8192
    uint32_t k;              // 0: 15360/5120/1280/8192. 1: 5120/1280/512. 2: 512. 3: 4096
    uint32_t scale_cols;     // 0, 3: k / 32, the UE8M0 plane's row stride (wo_a: 128)
    uint32_t row_base;       // 0, 3: first weight row of the dispatch
    uint32_t x_stride;       // elements between draft columns of the input
    uint32_t y_stride;       // elements between draft columns of the output
    uint32_t rows_per_group; // 3: o_lora_rank = 1024
    uint32_t rope_dim;       // 2: rope_head_dim = 64
    uint32_t inverse;        // 2: 1 = conjugate (the attention output's un-rope)
    uint32_t quant;          // 2: 1 = the act_quant round trip the KV ring stores
    uint32_t flags;          // kDsFlagInBf16 | kDsFlagOutBf16 | kDsFlagRoundIn
    float    eps;            // 1: rms_norm_eps = 1e-20
};

// gpu/shaders/dspark_attn.slang, both stages.
struct DsparkAttnPush {
    uint32_t m;              // live draft positions = dspark_block_size = 5
    uint32_t n_kv;           // 128 ring slots + 5 in-block rows = 133
    uint32_t n_heads;        // 64
    uint32_t head_dim;       // 512
    uint32_t score_stride;   // floats between score rows, >= n_kv
    uint32_t q_stride;       // elements between draft columns of q and o = 32768
    float    softmax_scale;  // head_dim ** -0.5 = 512 ** -0.5
};

// gpu/shaders/dspark_head.slang, all three stages.
struct DsparkHeadPush {
    uint32_t m;             // 2: live draft positions = 5
    uint32_t rows;          // 0, 1: vocab_size = 129280
    uint32_t k;             // 0: dspark_markov_rank = 256. 2: dim = 5120
    uint32_t rank;          // 2: dspark_markov_rank = 256
    uint32_t row_base;      // 0: first vocabulary row of the dispatch
    uint32_t pos;           // 0, 1: which of the five draft positions (0..4)
    uint32_t token;         // 0: the previous token id, unless kDsFlagTokenFromBuf
    uint32_t x_stride;      // 2: elements between draft columns of x = 5120
    uint32_t logit_stride;  // 1: floats between logit rows = 129280
    uint32_t flags;         // kDsFlagInBf16 | kDsFlagTokenFromBuf
};

// The `flags` bits of all three push structs, mirroring dspark_common.slang.
inline constexpr uint32_t kDsFlagInBf16       = 1u;  // the input activation is bf16
inline constexpr uint32_t kDsFlagOutBf16      = 2u;  // write the output as bf16
// RmsNorm only: round an fp32 input onto the bf16 grid before the RMS statistic,
// which is the reference's order (`linear()` returns bf16). Off by default,
// matching every other GEMV in gpu/shaders.
inline constexpr uint32_t kDsFlagRoundIn      = 4u;
// MarkovBias only: take the token id from `kMkTokenIn[pos]` rather than
// `DsparkHeadPush::token`, so the five sequential Markov steps chain on the GPU
// with no host readback between them.
inline constexpr uint32_t kDsFlagTokenFromBuf = 8u;

// What AddBiasArgmax writes at `kAbSample`: the token, its logit, the runner-up
// (design §12 L3 wants the margin at a divergence) and the row count it scanned,
// as a self-check that the dispatch saw the whole vocabulary. Layout-identical
// to DecodeRunner's SampleOut, deliberately, so one reader serves both.
struct DsparkSampleOut {
    uint32_t token = 0;
    float    top1  = 0.0f;
    float    top2  = 0.0f;
    uint32_t rows  = 0;
    float margin() const { return top1 - top2; }
};

// Slot indices inside a stage's address table. They are the `static const uint`
// names at the top of each shader; keeping both lists in one place is the only
// coupling between the two sides. Every stage owns its own 32-slot slice, so the
// indices are free to repeat across stages.
namespace dkslot {
// Gemv (stage 0) and WoA (stage 3): the fp8 E4M3 weight, its UE8M0 scale plane
// (wo_a's is [256][128], so WoA pushes scale_cols = 128), the activation, the
// output.
enum : uint32_t { kW = 0, kS = 1, kX = 2, kY = 3 };
// RmsNorm (stage 1)
enum : uint32_t { kNormX = 0, kNormW = 1, kNormY = 2 };
// RopeQuant (stage 2). kRopeTab is [positions][rope_dim / 2] of (cos, sin) fp32.
enum : uint32_t { kRopeX = 0, kRopeTab = 1, kRopeY = 2 };
// AttnScore / AttnCombine. kAttnKv is the flat `cat([window_kv_cache, kv])`
// the reference builds -- 128 ring rows then the five in-block rows -- so a
// topk index IS a row index into it.
enum : uint32_t { kAttnQ = 0, kAttnKv = 1, kAttnTopIdx = 2, kAttnSink = 3,
                  kAttnScore = 4, kAttnO = 5 };
// MarkovBias
enum : uint32_t { kMkW = 0, kMkEmbed = 1, kMkTokenIn = 2, kMkBias = 3, kMkEmbedOut = 4 };
// AddBiasArgmax
enum : uint32_t { kAbLogits = 0, kAbBias = 1, kAbSample = 2, kAbOutIds = 3 };
// Confidence. kCfEmbed is what MarkovBias published at kMkEmbedOut.
enum : uint32_t { kCfX = 0, kCfEmbed = 1, kCfProj = 2, kCfOut = 3 };
}  // namespace dkslot

class DsparkRunner {
public:
    DsparkRunner() = default;
    ~DsparkRunner() { destroy(); }

    DsparkRunner(const DsparkRunner&) = delete;
    DsparkRunner& operator=(const DsparkRunner&) = delete;

    Result<void> create(Device& device, MemoryAllocator& alloc,
                        const std::string& shader_dir, const DsparkSpec& spec = {});
    void destroy();

    const DsparkSpec& spec() const { return spec_; }

    // The address table for one stage: 32 slots, host-visible, written directly.
    // Valid until destroy().
    uint64_t* slots(DsparkStage s);

    // Records bind + push + dispatch. No barrier: the caller decides, because a
    // draft cycle is a strict chain and a benchmark is not.
    Result<void> record(CommandBuffer& cmd, DsparkStage s, const void* push,
                        uint32_t push_bytes, uint32_t groups);

    // Records one dispatch into a private command buffer, submits it and waits.
    // Tests and one-shot use only; the draft loop never waits on the host.
    Result<void> dispatch_now(DsparkStage s, const void* push, uint32_t push_bytes,
                              uint32_t groups);

    // --- workgroup counts, so callers never open-code the division -----------

    // Gemv, WoA and MarkovBias: one group per `(256 / lanes) * rows_per_lane`
    // consecutive weight rows, covering every live draft column.
    uint32_t gemv_groups(uint32_t rows) const {
        const uint32_t per = (256 / spec_.lanes_per_row) * spec_.rows_per_lane;
        return (rows + per - 1) / per;
    }

    // AttnScore and AttnCombine: one group per head, covering every live draft
    // position and every KV row. `heads_per_wg` is 1, so this is `n_heads`.
    uint32_t attn_groups(uint32_t n_heads) const {
        const uint32_t g = spec_.heads_per_wg ? spec_.heads_per_wg : 1u;
        return (n_heads + g - 1) / g;
    }

    // RmsNorm: one group per live draft column, each covering all `n`.
    uint32_t norm_groups(uint32_t m) const { return m ? m : 1u; }

    // RopeQuant: one THREAD per 32-element UE8M0 block of an
    // `[m][rows_per_pos][row_dim]` tensor.
    static uint32_t rope_groups(uint32_t m, uint32_t rows_per_pos, uint32_t row_dim) {
        const uint64_t blocks = uint64_t(m) * rows_per_pos * (row_dim / 32);
        return static_cast<uint32_t>((blocks + 255) / 256);
    }

    // AddBiasArgmax and Confidence are one workgroup each: the argmax is a
    // whole-vocabulary reduction and the confidence head is a 5376-wide dot
    // product per position.
    static uint32_t single_group() { return 1u; }

private:
    Result<void> make(DsparkStage s, const std::string& spv, uint32_t stage_const);

    Device*          device_ = nullptr;
    MemoryAllocator* alloc_  = nullptr;
    DsparkSpec       spec_{};
    Pipeline         pipes_[static_cast<uint32_t>(DsparkStage::Count)];
    DescriptorPool   descriptors_;
    CommandPool      pool_;
    GpuBuffer        table_{};
#if defined(DEEPMOE_ENABLE_VULKAN)
    VkDescriptorSet  sets_[static_cast<uint32_t>(DsparkStage::Count)]{};
#endif
};

}  // namespace deepmoe::gpu
