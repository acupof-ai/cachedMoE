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

// K <= 8192 (wo_b). With the quantisation on, a block is 32 B + one scale, so
// the whole staged x is 9 KiB; with it off (wo_a) the values stay bf16 and it
// is 16 KiB. Both leave room for the 1 KiB decode table and the 1 KiB
// reduction scratch inside the 32 KiB of design §1.1.
static const uint kMaxK      = 8192;
static const uint kMaxBlocks = kMaxK / kBlockK;      // 256, = one block per thread
static const uint kXWords    = (ActQuant != 0) ? (kMaxK / 4) : (kMaxK / 2);
static const uint kXPerBlk   = (ActQuant != 0) ? 8u : 16u;

groupshared uint  gXQ[kXWords];       // E4M3 bytes (4/word), or bf16 (2/word)
groupshared float gXS[kMaxBlocks];    // one power-of-two scale per 32 elements

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
        if (ActQuant != 0) {
            float amax = 0.0f;
            [unroll] for (uint i = 0; i < 32; ++i) amax = max(amax, abs(v[i]));
            const float s = act_scale(amax);
            const float inv = 1.0f / s;
            gXS[tid] = s;
            [unroll] for (uint i = 0; i < 8; ++i) {
                uint packed = 0;
                [unroll] for (uint j = 0; j < 4; ++j)
                    packed |= fp8_encode_rn(clamp(v[i * 4 + j] * inv,
                                                  -kFp8MaxF, kFp8MaxF)) << (8 * j);
                gXQ[tid * 8 + i] = packed;
            }
        } else {
            // wo_a: the reference multiplies bf16 by bf16 here, so bf16 is not
            // a compromise, it is the operand the reference used.
            gXS[tid] = 1.0f;
            [unroll] for (uint i = 0; i < 16; ++i)
                gXQ[tid * 16 + i] = bf16_encode(v[i * 2]) | (bf16_encode(v[i * 2 + 1]) << 16);
        }
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
        float amax = 0.0f;
        [unroll] for (uint i = 0; i < 32; ++i) {
            // bf16_round: RMSNorm.forward ends in `.to(dtype)` and the default
            // dtype is bf16. See attn_common.slang for why keeping fp32 here
            // costs agreement rather than buying precision.
            v[i] = bf16_round(v[i] * rs *
                              bf16_decode(load_u16(normw + (uint64_t)(tid * 32 + i) * 2)));
            amax = max(amax, abs(v[i]));
        }
        const float s = (ActQuant != 0) ? act_scale(amax) : 1.0f;
        gXS[tid] = s;
        if (ActQuant != 0) {
            const float inv = 1.0f / s;
            [unroll] for (uint i = 0; i < 8; ++i) {
                uint packed = 0;
                [unroll] for (uint j = 0; j < 4; ++j)
                    packed |= fp8_encode_rn(clamp(v[i * 4 + j] * inv,
                                                  -kFp8MaxF, kFp8MaxF)) << (8 * j);
                gXQ[tid * 8 + i] = packed;
            }
        } else {
            [unroll] for (uint i = 0; i < 16; ++i)
                gXQ[tid * 16 + i] = bf16_encode(v[i * 2]) | (bf16_encode(v[i * 2 + 1]) << 16);
        }
    }
}

// One staged activation element.
float staged_x(uint blk, uint j) {
    if (ActQuant != 0)
        return gTbl[(gXQ[blk * 8 + (j >> 2)] >> (8 * (j & 3))) & 0xFFu];
    const uint w = gXQ[blk * 16 + (j >> 1)];
    return bf16_decode(((j & 1) != 0) ? (w >> 16) : (w & 0xFFFFu));
}

// The dot product of weight row `row` with the staged x, over the lanes of one
// row group. Returns the per-lane partial; the caller reduces it.
float gemv8_partial(uint64_t wbase, uint64_t sbase, uint row, uint k,
                    uint scale_cols, uint sublane) {
    const uint blocks = k / kBlockK;
    const uint64_t rowoff = (uint64_t)row * (uint64_t)k;
    const uint64_t soff = (uint64_t)(row >> 5) * (uint64_t)scale_cols;
    float acc = 0.0f;
    for (uint blk = sublane; blk < blocks; blk += LanesPerRow) {
        const uint4 w0 = load16(wbase + rowoff + (uint64_t)blk * 32);
        const uint4 w1 = load16(wbase + rowoff + (uint64_t)blk * 32 + 16);
        const int   e  = (int)load_u8(sbase + soff + blk) - 127;
        float p = 0.0f;
        [unroll] for (uint i = 0; i < 4; ++i) {
            const uint a = w0[i], b = w1[i];
            [unroll] for (uint j = 0; j < 4; ++j) {
                p += gTbl[(a >> (8 * j)) & 0xFFu] * staged_x(blk, i * 4 + j);
                p += gTbl[(b >> (8 * j)) & 0xFFu] * staged_x(blk, 16 + i * 4 + j);
            }
        }
        acc += ldexp(p, e) * gXS[blk];
    }
    return acc;
}
