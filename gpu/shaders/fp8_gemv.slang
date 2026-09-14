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
// x is staged once per workgroup into LDS *already quantised*, as the E4M3
// byte plus the block's power-of-two scale. Three things fall out of that:
//
//   * the `act_quant` round trip design §6 does not mention but every fp8
//     `Linear` performs (see attn_common.slang) happens once per workgroup
//     instead of once per row group -- with 8 row groups in a 256-thread
//     workgroup that is an 8x saving on the only expensive arithmetic here;
//   * x costs 1 B + 1/32 float per element in LDS instead of 4 B, so the
//     largest case (wo_b, K = 8192) needs 9 KiB and still leaves room for the
//     decode table inside the 32 KiB of design §1.1;
//   * the inner loop becomes two table lookups and an FMA, which is exactly
//     the shape kernel_p1.md §3.2 found to be 1.4x the arithmetic decode.
//
// The block scale is a power of two and the accumulator is scaled with `ldexp`,
// so folding it out of the inner loop is exact, not an approximation.

#include "attn_common.slang"

// K <= 8192 (wo_b): 16 KiB of staged x, plus the 1 KiB decode table and the
// 1 KiB reduction scratch, inside the 32 KiB of design §1.1.
//
// x is staged as BF16, not as the E4M3 byte, even on the quantised path. The
// byte would be half the LDS, but it costs a second LDS access per element --
// one to fetch the byte's word, one to index the decode table -- and LDS ports
// are what this kernel runs out of. Measured: the byte form ran wq_b at 165
// GB/s and wo_b at 141, while wo_a, which stages bf16 because its activation
// is not quantised at all, hit 260 on the same recipe. Storing the quantised
// VALUE as bf16 is exact (an E4M3 number has three mantissa bits) and makes
// both paths one shift and one table lookup.
//
// The power-of-two block scale stays out of the stored value and is folded
// into the accumulator with `ldexp`, which keeps the staged number in a safe
// range whatever the block's magnitude.
static const uint kMaxK      = 8192;
static const uint kMaxBlocks = kMaxK / kBlockK;      // 256, = one block per thread

groupshared uint  gXQ[kMaxK / 2];     // the staged activation, two bf16 per word
groupshared float gXS[kMaxBlocks];    // one power-of-two scale per 32 elements

// Write one 32-element block of already-scaled activation into LDS as bf16.
void store_x_block(uint blk, const float v[32]) {
    [unroll] for (uint i = 0; i < 16; ++i)
        gXQ[blk * 16 + i] = bf16_encode(v[i * 2]) | (bf16_encode(v[i * 2 + 1]) << 16);
}

// `act_quant` one block in place and return its scale; 1.0 when ActQuant is off.
float quant_x_block(inout float v[32]) {
    if (ActQuant == 0) return 1.0f;
    float amax = 0.0f;
    [unroll] for (uint i = 0; i < 32; ++i) amax = max(amax, abs(v[i]));
    const float s = act_scale(amax);
    const float inv = 1.0f / s;
    // The stored value is q, not q * s: bf16 holds every E4M3 number exactly,
    // and the scale rejoins the accumulator through ldexp.
    [unroll] for (uint i = 0; i < 32; ++i)
        v[i] = fp8_round(clamp(v[i] * inv, -kFp8MaxF, kFp8MaxF));
    return s;
}

// Stage `k` activations from `xaddr` (fp32) into gXQ/gXS. Whole workgroup,
// uniform control flow, barrier afterwards. One thread owns one 32-element
// block, which is also exactly the granularity `act_quant` works at.
void stage_x_fp8(uint64_t xaddr, uint k, uint tid) {
    const uint blocks = k / kBlockK;
    if (tid < blocks) {
        float v[32];
        [unroll] for (uint i = 0; i < 8; ++i) {
            const uint4 w = load16(xaddr + (uint64_t)tid * 128 + (uint64_t)i * 16);
            [unroll] for (uint j = 0; j < 4; ++j) v[i * 4 + j] = asfloat(w[j]);
        }
        gXS[tid] = quant_x_block(v);
        store_x_block(tid, v);
    }
}

// The same staging with an RMSNorm folded in, which is how design §7.3's
// `q_norm` disappears into wq_b: the norm needs a statistic over all of x, and
// the staging pass already has all of x in registers one block per thread.
// `normw` is a bf16 vector. All 256 threads must call this.
void stage_x_fp8_rmsnorm(uint64_t xaddr, uint64_t normw, uint k, uint tid, float eps) {
    const uint blocks = k / kBlockK;
    float v[32];
    [unroll] for (uint z = 0; z < 32; ++z) v[z] = 0.0f;
    float sq = 0.0f;
    if (tid < blocks) {
        [unroll] for (uint i = 0; i < 8; ++i) {
            const uint4 w = load16(xaddr + (uint64_t)tid * 128 + (uint64_t)i * 16);
            [unroll] for (uint j = 0; j < 4; ++j) {
                v[i * 4 + j] = asfloat(w[j]);
                sq += v[i * 4 + j] * v[i * 4 + j];
            }
        }
    }
    const float rs = rsqrt(wg_reduce(sq, tid) / float(k) + eps);
    if (tid < blocks) {
        [unroll] for (uint i = 0; i < 32; ++i) {
            // bf16_round: RMSNorm.forward ends in `.to(dtype)` and the default
            // dtype is bf16. See attn_common.slang for why keeping fp32 here
            // costs agreement rather than buying precision.
            v[i] = bf16_round(v[i] * rs *
                              bf16_decode(load_u16(normw + (uint64_t)(tid * 32 + i) * 2)));
        }
        gXS[tid] = quant_x_block(v);
        store_x_block(tid, v);
    }
}

// The 32 staged activations of one block, into registers: eight LDS words
// rather than 32 separate indexed reads.
void read_x_block(uint blk, out float v[32]) {
    [unroll] for (uint i = 0; i < 16; ++i) {
        const uint w = gXQ[blk * 16 + i];
        v[i * 2]     = bf16_decode(w & 0xFFFFu);
        v[i * 2 + 1] = bf16_decode(w >> 16);
    }
}

// The dot products of `RowsPerLane` consecutive weight rows starting at
// `rowBase` with the staged x, over the lanes of one row group. Returns the
// per-lane partials; the caller reduces each.
//
// The rows are walked together so their weight loads are independent streams
// in flight at once -- the same reason kernel_p1.md ss3.3 found RowsPerLane
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
        for (uint r = 0; r < RowsPerLane; ++r) {
            const uint     row    = rowBase + r;
            const uint64_t rowoff = (uint64_t)row * (uint64_t)k;
            const uint4 w0 = load16(wbase + rowoff + (uint64_t)blk * 32);
            const uint4 w1 = load16(wbase + rowoff + (uint64_t)blk * 32 + 16);
            const int   e  = (int)load_u8(sbase + (uint64_t)(row >> 5) * scale_cols + blk) - 127;
            float p = 0.0f;
            [unroll] for (uint i = 0; i < 4; ++i) {
                const uint a = w0[i], b = w1[i];
                [unroll] for (uint j = 0; j < 4; ++j) {
                    p += gTbl[(a >> (8 * j)) & 0xFFu] * x[i * 4 + j];
                    p += gTbl[(b >> (8 * j)) & 0xFFu] * x[16 + i * 4 + j];
                }
            }
            acc[r] += ldexp(p, e) * xs;
        }
    }
}

// Rows one workgroup retires, and the first of them.
uint gemv_rows_per_wg()          { return (kWg / LanesPerRow) * RowsPerLane; }
uint gemv_row_base(uint gid, uint tid) {
    return gid * gemv_rows_per_wg() + (tid / LanesPerRow) * RowsPerLane;
}
