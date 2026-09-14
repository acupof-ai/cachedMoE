// The FP8 E4M3 GEMV shared by wq_a, wq_b, wkv, wo_a and wo_b (design §7.3,
// §7.4, §7.6), following the recipe kernel_p1.md §3.2 measured on the FP4 path:
// 32 lanes to a row, Wave32, a constant decode table, fp32 accumulation.
//
// Weight layout, from the manifest (model/manifest.h): `[out, K]` E4M3 bytes
// plus a `[out/32, K/32]` UE8M0 scale plane -- a 32x32 tile scale, not the
// per-row-block scale the FP4 experts use. A row's K walk therefore reads one
// scale byte per 32 K elements from row `out >> 5` of the scale plane.
//
// The activation
// --------------
// x is staged once per workgroup into LDS *already quantised*, as BF16 holding
// the E4M3-rounded value plus the block's power-of-two scale. Three things
// fall out of that:
//
//   * the `act_quant` round trip design §6 does not mention but every fp8
//     `Linear` performs (see attn_common.slang) happens once per workgroup
//     instead of once per row group;
//   * the inner loop becomes two LDS reads, a table lookup and an FMA, which is
//     the shape kernel_p1.md §3.2 found to be 1.4x the arithmetic decode;
//   * the block scale is a power of two and rejoins the accumulator through
//     `ldexp`, so folding it out of the inner loop is exact.
//
// x is staged as BF16, not as the E4M3 byte. The byte would be half the LDS but
// costs a second LDS access per element -- one for the byte's word, one to
// index the decode table -- and LDS ports are what this kernel runs out of.
// Measured: the byte form ran wq_b at 165 GB/s and wo_b at 141, while wo_a,
// which stages bf16 because its activation is not quantised at all, hit 260 on
// the same recipe. Storing the quantised VALUE as bf16 is exact, an E4M3
// number having three mantissa bits.
//
// What the P2 bandwidth pass changed (docs/p2_attention.md §7 item 1)
// ------------------------------------------------------------------
// wq_b and wo_b sat at 157 and 121 GB/s against a 217 GB/s ceiling while wo_a,
// the same kernel with the quantisation switched off, reached 167. The gap was
// never the weight stream; it was three things in the staging and the
// reduction, none of which had anything to do with reading weights:
//
//   1. **The LDS budget was wo_b's for everyone.** `gXQ` was sized for
//      K = 8192 unconditionally, so wq_b -- whose K is 1280 -- reserved 16 KiB
//      to use 2.5 KiB and ran at a third of the occupancy it could have.
//      `DEEPMOE_GEMV_MAX_K` is now defined by the including shader. It is NOT
//      simply "make it as small as K": wq_a and wo_a both measured slower with
//      a tight budget (152 against 166 GB/s for wq_a), because more resident
//      workgroups each re-staging a 4-5K-wide activation is more L2 traffic,
//      not less. The kernels that gain are the ones whose activation is small
//      to begin with -- wq_b and the indexer's wq_b at K = 1280 -- where the
//      occupancy is free. Both sides are in the per-shader #define.
//   2. **Only K/32 of the 256 threads staged anything.** One thread owned one
//      32-element `act_quant` block, so wq_b staged 1280 activations on 40
//      threads while 216 idled through the most expensive arithmetic in the
//      kernel. Threads per block is now a compile-time function of the same
//      K, and the block amax becomes a reduction over those threads.
//   3. **`fp8_round` was branchy.** See attn_common.slang: it is now nine
//      instructions and no branches, which matters because wo_b runs it 32
//      times per thread before the first weight byte is read.
//
// Plus `row_reduce` collapsing to one `WaveActiveSum` (attn_common.slang). The
// row loop was also tried unrolled against `kMaxRowsPerLane` with a
// specialisation-constant predicate, on the theory that a loop bounded by
// RowsPerLane leaves `acc[]` dynamically indexed and therefore in scratch --
// p2_attention.md §4.4's trap, one step removed. It measured 1% SLOWER on
// every one of the five, so the driver evidently folds the specialisation
// constant before it decides where `acc` lives, and the plain loop stays.

#include "attn_common.slang"

// The widest activation this .spv will ever stage, which sets the LDS budget.
// Each including shader defines it before the include; 8192 (wo_b's K, the
// widest in the model) is the safe default. A dispatch whose pushed `k`
// exceeds it would walk off gXQ, so the host side keeps the two in step:
// gpu/vulkan/attn_kernels.cpp's kStages table carries the same number.
#ifndef DEEPMOE_GEMV_MAX_K
#define DEEPMOE_GEMV_MAX_K 8192
#endif

static const uint kMaxK      = DEEPMOE_GEMV_MAX_K;
static const uint kMaxBlocks = kMaxK / kBlockK;

// **A 32-element block is 17 words in LDS, not 16.** Sixteen would be the
// natural packing -- two bf16 to a word -- and it puts a 16-WAY BANK CONFLICT
// on every single read of the staged activation, which was the largest single
// cost in wo_b and wq_b both.
//
// The inner loop gives lane `sublane` the blocks `sublane, sublane + 32, ...`,
// so at a 16-word stride the 32 lanes of a row group read word `i` of blocks
// `l` at LDS index `l * 16 + i`. LDS has 32 banks of one dword, so the bank is
// `(16 l + i) mod 32`, which takes exactly TWO values across the 32 lanes:
// every even lane hits bank `i` and every odd lane bank `i + 16`. Sixteen lanes
// to a bank is sixteen LDS cycles where one would do, and `read_x_block` issues
// this sixteen times per block.
//
// 17 is odd, so `17 l mod 32` is a bijection over l = 0..31 and every lane
// lands in its own bank. The cost is one word of LDS per 32 elements -- wo_b's
// gXQ goes from 16 KiB to 17 KiB -- and the gain is measured in
// docs/p2_attention.md §13.
//
// Anything that indexes gXQ must go through `gxq_word`.
static const uint kXStride = kBlockK / 2 + 1;          // 17
uint gxq_word(uint blk, uint w) { return blk * kXStride + w; }

groupshared uint  gXQ[kMaxBlocks * kXStride];   // staged activation, two bf16 per word
groupshared float gXS[kMaxBlocks];    // one power-of-two scale per 32 elements

// Threads that cooperate on one 32-element act_quant block, and the elements
// each of them owns. A power of two <= 8 so the amax reduction is a handful of
// LDS reads, and chosen so the whole workgroup is busy whenever K is small
// enough for that: K = 1280 gives 40 blocks and 4 threads each (160 of 256
// busy instead of 40), K = 4096 gives 2, K >= 8192 gives 1 with all 256 busy
// on one block apiece.
//
// `DEEPMOE_GEMV_STAGE_TPB_MAX` caps it, for the sweep in docs/p2_attention.md
// §13: sharing a block costs two workgroup barriers in `stage_amax`, and a
// K-split multiplies the number of workgroups that pay them.
#ifndef DEEPMOE_GEMV_STAGE_TPB_MAX
#define DEEPMOE_GEMV_STAGE_TPB_MAX 8
#endif
static const uint kStageRaw0 = kWg / kMaxBlocks;
static const uint kStageRaw  = (kStageRaw0 < DEEPMOE_GEMV_STAGE_TPB_MAX)
                             ? kStageRaw0 : DEEPMOE_GEMV_STAGE_TPB_MAX;
static const uint kStageTPB = (kStageRaw >= 8u) ? 8u
                            : ((kStageRaw >= 4u) ? 4u
                            : ((kStageRaw >= 2u) ? 2u : 1u));
static const uint kStageE   = kBlockK / kStageTPB;    // 32, 16, 8 or 4

// amax over the kStageTPB threads of one block, broadcast to all of them.
// Whole workgroup, uniform control flow.
float stage_amax(float a, uint tid, uint sub) {
    if (kStageTPB == 1) return a;
    gRed[tid] = a;
    GroupMemoryBarrierWithGroupSync();
    float m = 0.0f;
    [unroll] for (uint j = 0; j < kStageTPB; ++j) m = max(m, gRed[tid - sub + j]);
    GroupMemoryBarrierWithGroupSync();
    return m;
}

// `DEEPMOE_GEMV_FOLD_SCALE`: fold the weight's UE8M0 scale into the staged
// block factor, so the inner loop multiplies by one LDS float per block instead
// of loading a scale byte and calling `ldexp` per lane per block per row.
//
// It is exact, not an approximation: the weight scale is 2^(e - 127) and the
// activation scale is a power of two too, so `p * ldexp(xs, e - 127)` and
// `ldexp(p, e - 127) * xs` are the same multiplication by one power of two.
// It is only possible because every row a workgroup retires lies in the same
// 32-row band of the [rows/32][K/32] scale plane -- `(256 / LanesPerRow) *
// RowsPerLane` divides 32 at 32 lanes a row -- so one scale row per workgroup is
// all the inner loop ever reads, and `srow` is its address (offset by the
// slice's first block for a K-split).
#ifndef DEEPMOE_GEMV_FOLD_SCALE
#define DEEPMOE_GEMV_FOLD_SCALE 0
#endif
float fold_scale(float s, uint64_t srow, uint blk) {
#if DEEPMOE_GEMV_FOLD_SCALE
    return ldexp(s, (int)load_u8(srow + blk) - 127);
#else
    return s;
#endif
}

// This thread's slice of one block, into LDS as bf16.
void stage_store(uint blk, uint sub, const float v[kStageE]) {
    [unroll] for (uint i = 0; i < kStageE / 2; ++i)
        gXQ[gxq_word(blk, sub * (kStageE / 2) + i)] =
            bf16_encode(v[i * 2]) | (bf16_encode(v[i * 2 + 1]) << 16);
}

// Stage `k` activations from `xaddr` (fp32) into gXQ/gXS. Whole workgroup,
// uniform control flow, barrier afterwards.
void stage_x_fp8(uint64_t xaddr, uint k, uint tid, uint64_t srow) {
    const uint blocks = k / kBlockK;
    // wo_a's path: nothing to quantise, so nothing to share a block for, and
    // splitting one anyway measured 144 GB/s against 167. kStageTPB exists to
    // spread the `act_quant` arithmetic; with no arithmetic it is pure cost.
    if (ActQuant == 0) {
        if (tid < blocks) {
            float w[32];
            [unroll] for (uint i = 0; i < 8; ++i) {
                const uint4 q = load16(xaddr + (uint64_t)tid * 128 + (uint64_t)i * 16);
                [unroll] for (uint j = 0; j < 4; ++j) w[i * 4 + j] = asfloat(q[j]);
            }
            [unroll] for (uint i = 0; i < 16; ++i)
                gXQ[gxq_word(tid, i)] =
                    bf16_encode(w[i * 2]) | (bf16_encode(w[i * 2 + 1]) << 16);
            gXS[tid] = fold_scale(1.0f, srow, tid);
        }
        return;
    }

    const uint blk = tid / kStageTPB, sub = tid % kStageTPB;
    const bool active = blk < blocks;
    float v[kStageE];
    float amax = 0.0f;
    [unroll] for (uint z = 0; z < kStageE; ++z) v[z] = 0.0f;
    if (active) {
        const uint64_t base = xaddr + ((uint64_t)blk * kBlockK + sub * kStageE) * 4;
        [unroll] for (uint i = 0; i < kStageE / 4; ++i) {
            const uint4 w = load16(base + (uint64_t)i * 16);
            [unroll] for (uint j = 0; j < 4; ++j) {
                v[i * 4 + j] = asfloat(w[j]);
                amax = max(amax, abs(v[i * 4 + j]));
            }
        }
    }
    // Every thread reaches the reduction -- it has barriers in it whenever
    // kStageTPB > 1 -- but only the ones with a block to quantise pay for the
    // rounding. K = 5120 leaves 96 of 256 threads with nothing, and letting
    // them run 32 `fp8_round`s each is 8% off wq_a.
    const float a = stage_amax(amax, tid, sub);
    if (active) {
        const float s = act_scale(a);
        const float inv = 1.0f / s;
        // The stored value is q, not q * s: bf16 holds every E4M3 number
        // exactly, and the scale rejoins the accumulator through ldexp.
        [unroll] for (uint i = 0; i < kStageE; ++i)
            v[i] = fp8_round(clamp(v[i] * inv, -kFp8MaxF, kFp8MaxF));
        stage_store(blk, sub, v);
        if (sub == 0) gXS[blk] = fold_scale(s, srow, blk);
    }
}

// The same staging with an RMSNorm folded in, which is how design §7.3's
// `q_norm` disappears into wq_b: the norm needs a statistic over all of x, and
// the staging pass already has all of x in registers. `normw` is a bf16 vector.
// All 256 threads must call this.
void stage_x_fp8_rmsnorm(uint64_t xaddr, uint64_t normw, uint k, uint tid, float eps,
                         uint64_t srow) {
    const uint blocks = k / kBlockK;
    const uint blk = tid / kStageTPB, sub = tid % kStageTPB;
    const bool active = blk < blocks;
    const uint d0 = blk * kBlockK + sub * kStageE;

    float v[kStageE];
    [unroll] for (uint z = 0; z < kStageE; ++z) v[z] = 0.0f;
    float sq = 0.0f;
    if (active) {
        const uint64_t base = xaddr + (uint64_t)d0 * 4;
        [unroll] for (uint i = 0; i < kStageE / 4; ++i) {
            const uint4 w = load16(base + (uint64_t)i * 16);
            [unroll] for (uint j = 0; j < 4; ++j) {
                v[i * 4 + j] = asfloat(w[j]);
                sq += v[i * 4 + j] * v[i * 4 + j];
            }
        }
    }
    const float rs = rsqrt(wg_reduce(sq, tid) / float(k) + eps);

    float amax = 0.0f;
    if (active) {
        [unroll] for (uint i = 0; i < kStageE; ++i) {
            // bf16_round: RMSNorm.forward ends in `.to(dtype)` and the default
            // dtype is bf16. See attn_common.slang for why keeping fp32 here
            // costs agreement rather than buying precision.
            v[i] = bf16_round(v[i] * rs *
                              bf16_decode(load_u16(normw + (uint64_t)(d0 + i) * 2)));
            amax = max(amax, abs(v[i]));
        }
    }
    float s = 1.0f;
    if (ActQuant != 0) {
        const float a = stage_amax(amax, tid, sub);
        if (active) {
            s = act_scale(a);
            const float inv = 1.0f / s;
            [unroll] for (uint i = 0; i < kStageE; ++i)
                v[i] = fp8_round(clamp(v[i] * inv, -kFp8MaxF, kFp8MaxF));
        }
    }
    if (active) {
        stage_store(blk, sub, v);
        if (sub == 0) gXS[blk] = fold_scale(s, srow, blk);
    }
}

// The 32 staged activations of one block, into registers: eight LDS words
// rather than 32 separate indexed reads.
void read_x_block(uint blk, out float v[32]) {
    [unroll] for (uint i = 0; i < 16; ++i) {
        const uint w = gXQ[gxq_word(blk, i)];
        v[i * 2]     = bf16_decode(w & 0xFFFFu);
        v[i * 2 + 1] = bf16_decode(w >> 16);
    }
}

// One 32-element block: the weight's two 16-byte loads against the staged x.
//
// `DEEPMOE_GEMV_ACC` is how many accumulators the 32 multiply-adds are dealt
// across. At 1 -- the default, and what every number before §13 was measured
// with -- it is one chain of 32 dependent FMAs, in this exact order. At 4 each
// of the eight byte-words' four elements feeds its own accumulator, the chain
// is eight deep, and the four meet in a balanced tree: the same sum,
// re-associated. wo_b walks eight of these per lane per row, so the chain is
// the one place its latency cannot be hidden by occupancy.
#ifndef DEEPMOE_GEMV_ACC
#define DEEPMOE_GEMV_ACC 1
#endif
float block_dot(uint4 w0, uint4 w1, float x[32]) {
#if DEEPMOE_GEMV_ACC >= 4
    float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
    [unroll] for (uint j = 0; j < 4; ++j) {
        const uint sh = 8 * j;
        q0 += fp8_tbl((w0[0] >> sh) & 0xFFu) * x[j];
        q1 += fp8_tbl((w0[1] >> sh) & 0xFFu) * x[4 + j];
        q2 += fp8_tbl((w0[2] >> sh) & 0xFFu) * x[8 + j];
        q3 += fp8_tbl((w0[3] >> sh) & 0xFFu) * x[12 + j];
        q0 += fp8_tbl((w1[0] >> sh) & 0xFFu) * x[16 + j];
        q1 += fp8_tbl((w1[1] >> sh) & 0xFFu) * x[20 + j];
        q2 += fp8_tbl((w1[2] >> sh) & 0xFFu) * x[24 + j];
        q3 += fp8_tbl((w1[3] >> sh) & 0xFFu) * x[28 + j];
    }
    return (q0 + q1) + (q2 + q3);
#else
    float p = 0.0f;
    [unroll] for (uint i = 0; i < 4; ++i) {
        const uint a = w0[i], b = w1[i];
        [unroll] for (uint j = 0; j < 4; ++j) {
            p += fp8_tbl((a >> (8 * j)) & 0xFFu) * x[i * 4 + j];
            p += fp8_tbl((b >> (8 * j)) & 0xFFu) * x[16 + i * 4 + j];
        }
    }
    return p;
#endif
}

// The dot products of `RowsPerLane` consecutive weight rows starting at
// `rowBase` with the staged x, over the lanes of one row group. Returns the
// per-lane partials; the caller reduces each.
//
// The rows are walked together so their weight loads are independent streams
// in flight at once -- the same reason kernel_p1.md §3.3 found RowsPerLane
// worth 13% on the FP4 path at M=6 -- and so the LDS read of x is shared
// across them instead of repeated.
void gemv8_rows(uint64_t wbase, uint64_t sbase, uint rowBase, uint k,
                uint scale_cols, uint sublane, out float acc[kMaxRowsPerLane]) {
    [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) acc[r] = 0.0f;
    const uint blocks = k / kBlockK;
    for (uint blk = sublane; blk < blocks; blk += LanesPerRow) {
        float x[32];
        read_x_block(blk, x);
        const float xs = gXS[blk];
        // Unrolled against the compile-time bound with a specialisation
        // constant predicate: a loop bounded by RowsPerLane itself would leave
        // acc[] dynamically indexed, and p2_attention.md §4.4 is the record of
        // what that costs when the compiler then spills it to scratch.
        for (uint r = 0; r < RowsPerLane; ++r) {
            const uint     row    = rowBase + r;
            const uint64_t rowoff = (uint64_t)row * (uint64_t)k;
            const uint4 w0 = load16(wbase + rowoff + (uint64_t)blk * 32);
            const uint4 w1 = load16(wbase + rowoff + (uint64_t)blk * 32 + 16);
#if DEEPMOE_GEMV_FOLD_SCALE
            acc[r] += block_dot(w0, w1, x) * xs;
#else
            const int   e  = (int)load_u8(sbase + (uint64_t)(row >> 5) * scale_cols + blk) - 127;
            acc[r] += ldexp(block_dot(w0, w1, x), e) * xs;
#endif
        }
    }
}

// The same dot products over a SLICE of the contraction axis, `kc` elements
// starting at `k0` of a row whose full length is `kfull` (docs/p2_attention.md
// §13's K-split). The staged activation in gXQ/gXS is the slice, so `blk`
// indexes from zero there while the weight and scale planes are indexed from
// `k0` -- the scale plane's column for a slice block is `k0/32 + blk`.
//
// Why this exists at all, given that a K-split reads exactly the same weight
// bytes: the LDS budget a workgroup reserves is the SLICE width, not K. wo_b at
// K = 8192 reserved 19 KiB and got three workgroups to a CU; at two slices it
// reserves 10.5 KiB and gets six, and the number of rows a staged activation is
// amortised over is unchanged because each slice still retires the same rows.
void gemv8_rows_slice(uint64_t wbase, uint64_t sbase, uint rowBase, uint kfull,
                      uint k0, uint kc, uint scale_cols, uint sublane,
                      out float acc[kMaxRowsPerLane]) {
    [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) acc[r] = 0.0f;
    const uint blocks = kc / kBlockK;
    const uint blk0   = k0 / kBlockK;
    for (uint blk = sublane; blk < blocks; blk += LanesPerRow) {
        float x[32];
        read_x_block(blk, x);
        const float xs = gXS[blk];
        for (uint r = 0; r < RowsPerLane; ++r) {
            const uint     row    = rowBase + r;
            const uint64_t rowoff = (uint64_t)row * (uint64_t)kfull + (uint64_t)k0;
            const uint4 w0 = load16(wbase + rowoff + (uint64_t)blk * 32);
            const uint4 w1 = load16(wbase + rowoff + (uint64_t)blk * 32 + 16);
#if DEEPMOE_GEMV_FOLD_SCALE
            acc[r] += block_dot(w0, w1, x) * xs;
#else
            const int   e  = (int)load_u8(sbase + (uint64_t)(row >> 5) * scale_cols
                                          + blk0 + blk) - 127;
            acc[r] += ldexp(block_dot(w0, w1, x), e) * xs;
#endif
        }
    }
}

// Rows one workgroup retires, and the first of them.
uint gemv_rows_per_wg()          { return (kWg / LanesPerRow) * RowsPerLane; }
uint gemv_row_base(uint gid, uint tid) {
    return gid * gemv_rows_per_wg() + (tid / LanesPerRow) * RowsPerLane;
}
