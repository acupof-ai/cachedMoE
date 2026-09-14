// Host side of the non-MoE decode path: dispatches 1-9 and the head of design
// §7.14.
//
// The shape of the thing
// ----------------------
// Nine shaders, fourteen pipelines (mega_mhc, wkv, sparse_attn and gate each
// run more than one stage of the same .spv, selected by a specialisation
// constant). Every one of them binds exactly ONE descriptor: a slice of a
// shared `uint64_t` table holding the device addresses it needs. Weights and
// activations both arrive that way, for the same reason the MoE kernels reach
// their experts through a pointer table (design §5.3): the pinned set is 17.7
// GB spread over many sub-2-GiB regions, and one descriptor per tensor per
// layer would be thousands of them, rewritten every token.
//
// So the whole binding story is: write addresses into `slots(stage)`, push a
// small struct of dimensions, dispatch. Nothing is rebound between layers.
//
// Ownership/threading: one AttnRunner is created, recorded and submitted from a
// single thread. It owns its pipelines, its descriptor pool, the slot table and
// its command pool; the weights belong to store::PinnedStore and the
// activations to whoever allocated them.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/status.h"
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/descriptor.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/pipeline.h"

namespace deepmoe::gpu {

// The dispatch list of design §7.14, expanded to one entry per pipeline.
//
// mega_mhc appears three times over because a layer runs it three times with
// DIFFERENT weights and buffers -- the attention half, the FFN half, and the
// hc_post that closes the block -- and a stage owns one slice of the shared
// address table. Two dispatches of the same stage in one command buffer would
// both see whatever was written to that slice last. (The same argument scaled
// up is why a single PRE-RECORDED command buffer per token, which design §7.1
// wants, needs the table indexed by layer inside the shader; P2 step 1 records
// one command buffer per layer instead. See docs/p2_attention.md.)
enum class AttnStage : uint32_t {
    MhcPost = 0,   // §7.7 + §7.2: hc_post, hc_pre and the two RMS partials
    MhcMix,        // §7.2: the 24 hc_fn dot products
    MhcFinal,      // §7.2: Sinkhorn, and the RMSNorm of the sublayer input
    MhcPostB,      // the same three, for the FFN half of the layer
    MhcMixB,
    MhcFinalB,
    MhcClose,      // hc_post only: folds the MoE output back into the stream
    WqA,           // §7.3
    WqB,           // §7.3: + q_norm + RoPE
    WkvGemv,       // §7.4
    WkvFinish,     // §7.4: + kv_norm + RoPE + the fp8 ring write
    AttnScore,     // §7.5: every score
    AttnCombine,   // §7.5: softmax, P.V, inverse RoPE
    WoA,           // §7.6: grouped, and the one fp8 GEMV with no act_quant
    WoB,           // §7.6
    GateScore,     // §7.8
    GateTopK,      // §7.8: noaux_tc selection into host-coherent memory
    Head,          // §7.11
    // --- §7.4's compressor and indexer, source layers only -------------------
    // Appended, never inserted: runtime/ names these by enumerator and the
    // table above is what the decode path indexes. A layer that is not a
    // kv_source_layer dispatches none of the Cmp* stages, and one that is not
    // an index_source_layer dispatches none of the Idx*.
    CmpKvGemv,     // §7.4: compressor.wkv,   bf16 [512 x 5120]
    CmpGateGemv,   // §7.4: compressor.wgate, bf16 [512 x 5120], ratio > 1 only
    CmpNorm,       // §7.4: the state write, the softmax pooling, compressor.norm
    CmpStore,      // §7.4: RoPE, FP4 block-16/E4M3, the compressed-KV write
    IdxQGemv,      // §7.4: indexer.wq_b, fp8 [4096 x 1280] + q_norm
    IdxQFinish,    // §7.4: RoPE + FP4 block-32/UE8M0 on the index queries
    IdxKey,        // §7.4: k_norm(wk(latent)) -> the index-key cache
    IdxWeights,    // §7.4: indexer.weights_proj, bf16 [32 x 5120]
    IdxScore,      // §7.4: sum_h relu(q[h].k[t]) * w[h]
    IdxTopK,       // §7.4: the index_topk best positions, in index order
    Count,
};

const char* attn_stage_name(AttnStage s);

// design §7.1's knobs, as far as this path has them. `act_quant` is not here:
// it is a property of the kernel (wo_a never quantises, everything else
// always does), not a sweep dimension -- except in bench/attn_bench, which
// creates its own pipelines to measure what the round trip costs.
struct AttnSpec {
    uint32_t lanes_per_row = 32;   // kernel_p1.md §3.2's winner at M=1
    uint32_t subgroup_size = 32;   // Wave32
    // Weight rows per lane. The attention GEMVs are tall and thin -- wq_b is
    // 32768 x 1280 -- so at RowsPerLane = 1 a workgroup stages a 1280-wide
    // activation to retire eight rows, and the staging is a third of its L2
    // traffic. 4 makes that 32 rows. Must divide the row count of every kernel
    // it is applied to, including wo_a's 1024-row groups.
    uint32_t rows_per_lane = 4;
    // Heads one sparse_attn workgroup covers (§7.5). 1..8, and only sparse_attn
    // reads it, and only when the caller pushes AttnPush::n_heads.
    //
    // It is 1, i.e. the KV is still read once per head, and that is a MEASURED
    // choice rather than a default nobody touched. Grouping heads is what makes
    // the 320 KiB of KV cross L2 8x instead of 64x, and it costs more than it
    // saves, because 64 heads at 8 a workgroup is eight workgroups on forty CUs
    // (us, score + combine, --layers 8): 1 -> 72, 2 -> 100, 4 -> 169, 8 -> 288.
    // The traffic and the occupancy are traded against each other along the
    // wrong axis; what design §7.5 actually wants is a head-group x KV-tile
    // split with a partial-softmax combine, which keeps 64 workgroups AND reads
    // the KV eight times. See docs/p2_attention.md §7.
    uint32_t heads_per_wg = 1;
};

// --- push constants, mirroring the shaders exactly --------------------------

struct MhcPush {
    uint32_t dim, hc, mix_rows, n_wg0, sinkhorn_iters, flags;
    float    norm_eps, hc_eps;
};
inline constexpr uint32_t kMhcFlagPost = 1u;          // apply hc_post
inline constexpr uint32_t kMhcFlagSkipSinkhorn = 2u;  // final collapse before the head

struct GemvPush { uint32_t rows, k, scale_cols, row_base; };
struct WqbPush  { uint32_t rows, k, scale_cols, head_dim, rope_dim; float norm_eps; };
struct WkvPush  { uint32_t rows, k, scale_cols, rope_dim, slot, n_wg0; float norm_eps; };
struct WoaPush  { uint32_t rows, k, scale_cols, rows_per_group; };
// `n_heads` is ADDITIVE and optional: a caller that leaves it out (aggregate
// initialisation zero-fills it) gets the pre-existing geometry, one workgroup
// per head. Passing the real head count lets sparse_attn.slang put
// AttnSpec::heads_per_wg heads in a workgroup, which divides both the
// workgroup count -- see `attn_groups` -- and the 41 MB of L2 traffic
// docs/p2_attention.md §7 item 2 is about.
struct AttnPush {
    uint32_t n_kv, n_win, head_dim, rope_dim, score_stride;
    float    softmax_scale;
    uint32_t n_heads = 0;
};
struct GatePush { uint32_t n_experts, k, topk, record; float gate_temp, route_scale; };
struct HeadPush { uint32_t rows, k, row_base; };
// §7.4. `complete` is `(start_pos + 1) % ratio == 0`: at ratio 1 it is always
// true and at ratio 2 it is true every other token, which is the whole reason
// the compressor carries state.
struct CmpPush {
    uint32_t rows, k, ratio, slot, complete, rope_dim, cmp_row;
    float    norm_eps;
};
// §7.4. One struct for all six indexer stages; each fills the fields its stage
// reads, because the dimensions differ per stage (stage 0 is [4096 x 1280],
// stage 2 is [128 x 512], stage 3 is [32 x 5120]).
struct IdxPush {
    uint32_t rows, k, scale_cols, n_heads, head_dim, rope_dim;
    uint32_t n_pos, topk, offset, k_row;
    float    norm_eps, wscale;
};

// Slot indices inside a stage's address table. They are the `static const uint`
// names at the top of each shader; keeping both lists in one place is the only
// coupling between the two sides.
namespace slot {
// mega_mhc
enum : uint32_t { kX = 0, kA = 1, kPostIn = 2, kCombIn = 3, kPreMix = 4, kHcFn = 5,
                  kHcBase = 6, kHcScale = 7, kNormW = 8, kXout = 9, kUtmp = 10,
                  kScratch = 11, kMixRaw = 12, kMixOut = 13, kU = 14 };
// wq_a / wo_b: W, S, X, Y
enum : uint32_t { kGemvW = 0, kGemvS = 1, kGemvX = 2, kGemvY = 3 };
// wq_b: W, S, Qr, NormW, Rope, Q
enum : uint32_t { kWqbW = 0, kWqbS = 1, kWqbQr = 2, kWqbNormW = 3, kWqbRope = 4, kWqbQ = 5 };
// wkv
enum : uint32_t { kWkvW = 0, kWkvS = 1, kWkvX = 2, kWkvNormW = 3, kWkvRope = 4,
                  kWkvRaw = 5, kWkvScratch = 6, kWkvVal = 7, kWkvScale = 8, kWkvKv = 9 };
// sparse_attn
enum : uint32_t { kAttnQ = 0, kAttnWinVal = 1, kAttnWinScale = 2, kAttnCmpKv = 3,
                  kAttnTopIdx = 4, kAttnSink = 5, kAttnRope = 6, kAttnScore = 7, kAttnO = 8 };
// wo_a: W, S, O, Y
enum : uint32_t { kWoaW = 0, kWoaS = 1, kWoaO = 2, kWoaY = 3 };
// gate
enum : uint32_t { kGateW = 0, kGateBias = 1, kGateX = 2, kGateScores = 3,
                  kGateIds = 4, kGateWeights = 5, kGateLayerDone = 6 };
// head
enum : uint32_t { kHeadW = 0, kHeadX = 1, kHeadLogits = 2 };
// compressor (§7.4). kCmpY is the wkv projection and kCmpG the wgate one, so
// the two stage-0 dispatches differ only in kCmpW and kCmpY.
enum : uint32_t { kCmpW = 0, kCmpX = 1, kCmpY = 2, kCmpG = 3, kCmpKvState = 4,
                  kCmpScoreState = 5, kCmpNormW = 6, kCmpLatent = 7, kCmpRope = 8,
                  kCmpVal = 9, kCmpFp4 = 10, kCmpScaleB = 11, kCmpLatentQ = 12 };
// indexer (§7.4)
enum : uint32_t { kIdxW = 0, kIdxS = 1, kIdxQr = 2, kIdxQNormW = 3, kIdxRope = 4,
                  kIdxQRaw = 5, kIdxQ = 6, kIdxWk = 7, kIdxKNormW = 8,
                  kIdxLatent = 9, kIdxKRaw = 10, kIdxKCache = 11, kIdxKFp4 = 12,
                  kIdxKScale = 13, kIdxWProjW = 14, kIdxX = 15, kIdxWeights = 16,
                  kIdxScore = 17, kIdxOut = 18, kIdxQFp4 = 19, kIdxQScale = 20 };
}  // namespace slot

// Positions the §7.4 indexer's score stage covers in one workgroup, and the
// head count its wave reduction assumes. `AttnRunner::create` rejects a spec
// whose lanes_per_row is not the head count, because stage 4 puts the 32 heads
// of one position in the 32 lanes of one wave.
inline constexpr uint32_t kIdxScoreTile = 8;

// A bump allocator over one host-visible, device-addressable buffer: every
// activation of design §7.14 for one token fits in a couple of MB, so there is
// no reason for them to be separate allocations.
class GpuScratch {
public:
    struct View {
        uint64_t addr  = 0;
        void*    host  = nullptr;
        uint64_t bytes = 0;
        bool valid() const { return host != nullptr; }
    };

    Result<void> create(MemoryAllocator& alloc, uint64_t bytes);
    void         destroy();

    // 256-byte aligned by default, which covers every `uint4` load in the
    // shaders and the descriptor offset alignment of the slot table.
    Result<View> alloc(uint64_t bytes, uint64_t align = 256);
    void         rewind() { used_ = 0; }
    uint64_t     used() const { return used_; }
    uint64_t     capacity() const { return buf_.bytes; }

private:
    MemoryAllocator* alloc_ = nullptr;
    GpuBuffer        buf_{};
    uint64_t         used_ = 0;
};

class AttnRunner {
public:
    AttnRunner() = default;
    ~AttnRunner() { destroy(); }

    AttnRunner(const AttnRunner&) = delete;
    AttnRunner& operator=(const AttnRunner&) = delete;

    Result<void> create(Device& device, MemoryAllocator& alloc,
                        const std::string& shader_dir, const AttnSpec& spec = {});
    void destroy();

    const AttnSpec& spec() const { return spec_; }

    // The address table for one stage: 32 slots, host-visible, written
    // directly. Valid until destroy().
    uint64_t* slots(AttnStage s);

    // Records bind + push + dispatch. No barrier: the caller decides, because a
    // decode layer is a strict chain and a benchmark is not.
    Result<void> record(CommandBuffer& cmd, AttnStage s, const void* push,
                        uint32_t push_bytes, uint32_t groups);

    // Records one dispatch into a private command buffer, submits it and waits.
    // Tests and one-shot use only; the decode loop never waits on the host.
    Result<void> dispatch_now(AttnStage s, const void* push, uint32_t push_bytes,
                              uint32_t groups);

    // Rows per lane actually compiled into one stage's pipeline. Not the spec
    // value everywhere: row blocking trades workgroups for reuse, and a kernel
    // with few rows runs out of workgroups first. wkv has 512 rows, which at
    // rows_per_lane 4 is sixteen workgroups on forty CUs -- measured at 22 GB/s
    // against 122 at rows_per_lane 2 -- and the gate's 384 rows are worse
    // still, so both are capped at 1.
    uint32_t rows_per_lane(AttnStage s) const;

    // Workgroups for one sparse_attn stage over `n_heads` heads. A caller that
    // pushes AttnPush::n_heads MUST dispatch this many; one that does not must
    // dispatch `n_heads`, which is what this returns when heads_per_wg is 1.
    uint32_t attn_groups(uint32_t n_heads) const {
        const uint32_t g = spec_.heads_per_wg ? spec_.heads_per_wg : 1u;
        return (n_heads + g - 1) / g;
    }

    // Workgroups for a `rows`-tall GEMV of that stage.
    uint32_t gemv_groups(AttnStage s, uint32_t rows) const {
        const uint32_t per = (256 / spec_.lanes_per_row) * rows_per_lane(s);
        return (rows + per - 1) / per;
    }

private:
    Result<void> make(AttnStage s, const std::string& spv, uint32_t stage_const,
                      uint32_t act_quant);

    Device*          device_ = nullptr;
    MemoryAllocator* alloc_  = nullptr;
    AttnSpec         spec_{};
    Pipeline         pipes_[static_cast<uint32_t>(AttnStage::Count)];
    DescriptorPool   descriptors_;
    CommandPool      pool_;
    GpuBuffer        table_{};
#if defined(DEEPMOE_ENABLE_VULKAN)
    VkDescriptorSet  sets_[static_cast<uint32_t>(AttnStage::Count)]{};
#endif
};

// Slots per stage in the shared table, and the resulting per-stage byte
// stride. 32 slots is well above the 15 mega_mhc needs and keeps the stride a
// round 256 B, which satisfies every plausible
// minStorageBufferOffsetAlignment.
inline constexpr uint32_t kAttnSlotsPerStage = 32;
inline constexpr uint32_t kAttnStageStride   = kAttnSlotsPerStage * sizeof(uint64_t);

}  // namespace deepmoe::gpu
