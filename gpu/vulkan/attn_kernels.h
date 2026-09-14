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
    // --- P3: the K-split GEMVs and the tiled attention -----------------------
    // Appended, never inserted, for the same reason the §7.4 block above was:
    // runtime/ names stages by enumerator. Every stage listed before this point
    // behaves exactly as it did, so a caller that adopts none of these is
    // unaffected. See docs/p2_attention.md §13.
    //
    // A K-split GEMV is TWO dispatches: `*KSplit` over
    // `gemv_groups(stage, rows) * ksplit(stage)` workgroups writing fp32
    // partials, then `*KCombine` over `combine_groups(rows)` adding them.
    // wkv is the exception -- its combine folds in the whole kv_norm / RoPE /
    // ring-write tail, so WkvKSplit + WkvKFinish REPLACE WkvGemv + WkvFinish
    // rather than adding a dispatch.
    WqAKSplit,     // §7.3, gemv_ksplit stage 0
    WqAKCombine,   // §7.3, gemv_ksplit stage 1
    WkvKSplit,     // §7.4, gemv_ksplit stage 0
    WkvKFinish,    // §7.4, wkv stage 2: combine + kv_norm + RoPE + ring write
    WoAKSplit,     // §7.6, grouped, ActQuant = 0
    WoAKCombine,
    WoBKSplit,     // §7.6, the 42 MB one
    WoBKCombine,
    // §7.5 on a head-group x KV-tile grid with a partial-softmax combine.
    // THREE dispatches, replacing AttnScore + AttnCombine.
    AttnScoreT,    // scores + one tile max per head
    AttnPvT,       // p against the final max, and P.V over one tile; at
                   // pv_tiles == 1 also the divide + inverse RoPE, i.e. done
    AttnFinishT,   // pv_tiles > 1 only: add the tiles, add the sink, divide,
                   // inverse RoPE
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
    // The same thing for the TILED attention stages (AttnScoreT/PvT), where it
    // is 8 rather than 1 because the tiling gives back the occupancy that made
    // grouping a loss for the untiled kernel: 8 heads a workgroup and n_tiles
    // KV tiles is 8 * n_tiles workgroups, not 8. 1..8.
    uint32_t tile_heads_per_wg = 8;
    // Heads per workgroup for AttnPvT alone. Its default is 1, not 8: P.V
    // cannot share a KV read across heads the way the scores can (see
    // sparse_attn_t.slang), so grouping only costs it workgroups.
    uint32_t pv_heads_per_wg = 1;
    // Rows per lane for the *KSplit stages only, because the K-split changes
    // the trade: the slice is narrower, so the LDS a workgroup holds is
    // smaller, so it can afford to retire more rows against one staged
    // activation than the unsplit kernel could. 0 = follow `rows_per_lane`.
    uint32_t rows_per_lane_ksplit = 0;
    // Decode E4M3 arithmetically instead of through the 256-entry LDS table
    // (attn_common.slang `fp8_tbl`). Applies to every fp8 kernel in the family.
    uint32_t fp8_arith_decode = 0;
    // Sweep hooks, bit i = AttnStage i. Zero in every real caller; they exist
    // so bench/attn_bench can re-measure kStages' per-stage choices (which
    // docs/p2_attention.md §13 shows moved once the LDS conflicts were gone)
    // without a rebuild between points.
    uint64_t sweep_rows_cap4   = 0;   // let these stages use rows_per_lane up to 4
    uint64_t sweep_wave_on     = 0;   // force WaveReduce = 1
    uint64_t sweep_wave_off    = 0;   // force WaveReduce = 0
    // K-split factors, per family, for the *KSplit stages. Each must divide
    // K/32 and leave K / ksplit <= kGemvKSplitMaxSlice.
    uint32_t ksplit_wq_a = 2;
    uint32_t ksplit_wkv  = 4;
    uint32_t ksplit_wo_a = 1;
    uint32_t ksplit_wo_b = 2;
};

// The widest K slice gemv_ksplit.slang stages, i.e. its `DEEPMOE_GEMV_MAX_K`.
// A caller must keep `k / ksplit(stage)` at or under it.
inline constexpr uint32_t kGemvKSplitMaxSlice = 4096;

// --- push constants, mirroring the shaders exactly --------------------------

struct MhcPush {
    uint32_t dim, hc, mix_rows, n_wg0, sinkhorn_iters, flags;
    float    norm_eps, hc_eps;
};
inline constexpr uint32_t kMhcFlagPost = 1u;          // apply hc_post
inline constexpr uint32_t kMhcFlagSkipSinkhorn = 2u;  // final collapse before the head

struct GemvPush { uint32_t rows, k, scale_cols, row_base; };
struct WqbPush  { uint32_t rows, k, scale_cols, head_dim, rope_dim; float norm_eps; };
// `part_stride` is trailing and defaulted: it is read only by WkvKFinish, so a
// caller that aggregate-initialises the first seven fields is unchanged.
struct WkvPush  {
    uint32_t rows, k, scale_cols, rope_dim, slot, n_wg0;
    float    norm_eps;
    uint32_t part_stride = 0;
};
// gemv_ksplit.slang. `rows_per_group` is 0 for an ungrouped GEMV and wo_a's
// o_lora_rank (1024) for the block-diagonal one; `part_stride` is the floats
// between K slices in the partial plane and must be >= rows.
struct KSplitPush {
    uint32_t rows, k, scale_cols, rows_per_group, part_stride, row_base;
};
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
// sparse_attn_t.slang. A superset of AttnPush with the tiling fields; it is a
// separate struct because the tiled stages are separate pipelines and mixing
// the two would let a caller push one at the other.
//
// Two grids. AttnScoreT runs `(n_heads / tile_heads_per_wg) * n_tiles`
// workgroups over tiles of `tile_len`; AttnPvT runs
// `(n_heads / pv_heads_per_wg) * pv_tiles` over tiles of `pv_tile_len`, and the
// kPartO / kPartD planes are laid out by `pv_tiles`. kTileMax is always
// [n_heads][n_tiles].
struct AttnTPush {
    uint32_t n_kv, n_win, head_dim, rope_dim, score_stride;
    float    softmax_scale;
    uint32_t n_heads, n_tiles, tile_len, part_stride;
    uint32_t pv_tiles, pv_tile_len;
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
// wkv. kWkvPart is the fp32 partial plane WkvKSplit writes and WkvKFinish adds.
enum : uint32_t { kWkvW = 0, kWkvS = 1, kWkvX = 2, kWkvNormW = 3, kWkvRope = 4,
                  kWkvRaw = 5, kWkvScratch = 6, kWkvVal = 7, kWkvScale = 8, kWkvKv = 9,
                  kWkvPart = 10 };
// gemv_ksplit: W, S, X, Y, P. The first four are the same four slots the
// unsplit GEMV uses, so a caller can fill them the same way and add the
// partial plane.
enum : uint32_t { kKspW = 0, kKspS = 1, kKspX = 2, kKspY = 3, kKspPart = 4 };
// sparse_attn
enum : uint32_t { kAttnQ = 0, kAttnWinVal = 1, kAttnWinScale = 2, kAttnCmpKv = 3,
                  kAttnTopIdx = 4, kAttnSink = 5, kAttnRope = 6, kAttnScore = 7, kAttnO = 8 };
// sparse_attn_t: the same nine, plus the three planes the tiling needs --
// [n_heads][n_tiles] maxima, [n_tiles][n_heads][head_dim] fp32 output partials
// and [n_tiles][n_heads] denominator partials.
enum : uint32_t { kAttnTileMax = 9, kAttnPartO = 10, kAttnPartD = 11 };
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
    // One row of the stage table: which .spv, which Stage specialisation, and
    // the per-stage knobs measured rather than reasoned about. Public because
    // the table itself lives in the .cpp's anonymous namespace.
    struct StageDef {
        AttnStage   stage;
        const char* spv;
        uint32_t    stage_const;
        uint32_t    act_quant;
        uint32_t    rows_cap;     // largest rows_per_lane this shader may use
        // Whether row_reduce is one WaveActiveSum (1) or the LDS tree (0). Not
        // a free win either way -- attn_common.slang's row_reduce has the sweep.
        uint32_t    wave_reduce;
    };

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

    // The K-split factor compiled into a *KSplit stage (1 for everything else).
    uint32_t ksplit(AttnStage s) const;

    // Workgroups for the split half of a K-split GEMV: one per (row group,
    // slice), flattened as `rg * ksplit + slice`.
    uint32_t ksplit_groups(AttnStage s, uint32_t rows) const {
        return gemv_groups(s, rows) * ksplit(s);
    }
    // Workgroups for the combine half: one thread a row.
    static uint32_t combine_groups(uint32_t rows) { return (rows + 255) / 256; }

    // Workgroups for AttnScoreT: head groups x KV tiles, flattened as
    // `head_group * n_tiles + tile`.
    uint32_t attn_tile_groups(uint32_t n_heads, uint32_t n_tiles) const {
        const uint32_t g = spec_.tile_heads_per_wg ? spec_.tile_heads_per_wg : 1u;
        return ((n_heads + g - 1) / g) * n_tiles;
    }
    // Workgroups for AttnPvT, on its own grid.
    uint32_t attn_pv_groups(uint32_t n_heads, uint32_t pv_tiles) const {
        const uint32_t g = spec_.pv_heads_per_wg ? spec_.pv_heads_per_wg : 1u;
        return ((n_heads + g - 1) / g) * pv_tiles;
    }
    // AttnFinishT exists to add the P.V tiles; with one tile AttnPvT writes the
    // final output itself and AttnFinishT must NOT be dispatched.
    static bool attn_tiled_finish_needed(uint32_t pv_tiles) { return pv_tiles > 1; }
    // Workgroups for AttnFinishT: one thread per adjacent output pair.
    static uint32_t attn_finish_groups(uint32_t n_heads, uint32_t head_dim) {
        const uint32_t pairs = n_heads * head_dim / 2;
        return (pairs + 255) / 256;
    }

private:
    Result<void> make(const StageDef& d, const std::string& spv);

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
