// mgt1_gemv -- the fp8 E4M3 GEMVs of the attention path for a verify batch of
// M = k + 1 <= 6 tokens (Track T, docs/p4_mgt1.md): wq_a, wq_b, wkv, wo_a, wo_b
// and the indexer's wq_b, plus the per-token tails that ride on them (q_norm,
// q's RoPE, kv_norm / RoPE / fp8 ring write).
//
// What changes at M > 1, and what does not
// ----------------------------------------
// The weight is read ONCE whatever M is; only the arithmetic is per token. So
// the geometry is gemv_ksplit.slang's -- one row a lane, the contraction axis
// cut into KSplit slices, a partial per (slice, row) -- with the batch as the
// innermost dimension, weight-outer / batch-inner exactly as moe_gateup.slang
// and dspark_gemv.slang do it: a 32-byte weight block is loaded, decoded into
// registers once, and dotted against the M staged activation blocks of the
// same K range.
//
// The staged activation is the slice of all M columns. Its LDS layout keeps
// fp8_gemv.slang's odd 17-word block (a lane owns one bank), and the COLUMN
// stride is a whole number of 32-block lanes' worth (`kColBlocks`, a multiple
// of 32), so the M reads one lane makes all land in that lane's own bank:
// `(m * kColBlocks + blk) * 17 + w` mod 32 does not depend on m. A slice of at
// most 1024 elements keeps the stride at 32 blocks, which is why the M > 1
// K-split factors are chosen so that K / KSplit <= 1024 (runtime/decode_layer).
//
// Stages (`Stage`):
//   0  split    one K slice of every row for all M columns -> fp32 partials
//               [slice][M][rows]. ActQuant 1 = every fp8 Linear, 0 = wo_a.
//   1  combine  y[m][row] = sum of the KSplit partials, fp32.
//   2  q finish combine + RoPE of q at token m's position + bf16 write of
//               [M][n_heads][head_dim] (wq_b's tail, docs/p2_attention.md §2.2's
//               adjacent pairs).
//   3  rmsnorm  y[m][i] = bf16(w[i] x[m][i] rsqrt(mean(x[m]^2) + eps)), one
//               workgroup per token (q_norm; its output feeds wq_b AND the
//               indexer's wq_b, which re-normalised the same qr at M = 1).
//   4  kv finish combine + kv_norm + RoPE + act_quant of token m's KV, the
//               write into ring slot (p0 + m) % window, and -- first -- the copy
//               of what that slot held into overflow row m - 1: the window
//               position p0 + m - window that the batch's EARLIER queries still
//               attend to (docs/p3_dspark.md §4.9). One workgroup per token.
//
// Dispatch geometry: stage 0 is `row_groups * KSplit` workgroups (gid.x = rg *
// KSplit + slice); stages 1, 2 are `ceil(rows / 256)` x M (gid.y = m); stages 3
// and 4 are 1 x M.

#define DEEPMOE_MGT1_MAX_K 1024
#include "attn_common.slang"

static const uint kMaxM       = 6;
static const uint kMaxBlocksS = DEEPMOE_MGT1_MAX_K / kBlockK;            // 32
static const uint kColBlocks  = ((kMaxBlocksS + 31u) / 32u) * 32u;        // 32
static const uint kXStride    = kBlockK / 2 + 1;                          // 17

// Slots.
static const uint kW = 0, kS = 1, kX = 2, kY = 3, kP = 4;
static const uint kNormW = 5, kRope = 6, kRingVal = 7, kRingScale = 8;
static const uint kOvfVal = 9, kOvfScale = 10, kKvOut = 11;
// pc.flags bits. `X` is bf16 rather than fp32: the DSpark draft chain's
// activations (`main_hidden`, `main_x`) are bf16 in the reference, and reading
// them as fp32 would be reading garbage. Only the staged load changes.
static const uint kGemmFlagInBf16 = 1u;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct Mgt1GemvPush {
    uint  rows;            // output rows of the whole GEMV
    uint  k;               // the FULL contraction length
    uint  scale_cols;      // k / 32
    uint  rows_per_group;  // 0, or wo_a's 1024
    uint  part_stride;     // floats between (slice, m) runs in the partial plane (>= rows)
    uint  x_stride;        // elements between token columns of X
    uint  y_stride;        // elements between token columns of Y (or of Q / KvOut)
    uint  head_dim;        // 2: 512.  4: 512 (the KV row).  3: vector length
    uint  rope_dim;        // 2, 4: 64
    uint  p0;              // 4: position of token 0
    uint  window;          // 4: 128
    float eps;             // 3, 4: rms_norm_eps
    uint  flags;           // kGemmFlag* below
};
[[vk::push_constant]] Mgt1GemvPush pc;

// --- staging ------------------------------------------------------------------
// Threads that cooperate on one staged 32-element block: a power of two <= 4,
// chosen so M * 32 blocks fill the workgroup. A function of the specialisation
// constant, evaluated in the function body (Slang cannot emit a `select` over a
// specialisation constant at global scope); the driver folds it.
uint stage_tpb() {
    const uint raw = kWg / (M * kMaxBlocksS);
    return (raw >= 4u) ? 4u : ((raw >= 2u) ? 2u : 1u);
}

groupshared uint  gXQ[M * kColBlocks * kXStride];
groupshared float gXS[M * kColBlocks];

uint xq_word(uint m, uint blk, uint w) { return (m * kColBlocks + blk) * kXStride + w; }

float stage_amax(float a, uint tid, uint sub, uint tpb) {
    if (tpb == 1u) return a;
    gRed[tid] = a;
    GroupMemoryBarrierWithGroupSync();
    float mx = 0.0f;
    [unroll] for (uint j = 0; j < 4; ++j) if (j < tpb) mx = max(mx, gRed[tid - sub + j]);
    GroupMemoryBarrierWithGroupSync();
    return mx;
}

// Stage `kc` elements of every token column, starting at element `k0` of the
// column (plus `xoff`, wo_a's group). Whole workgroup, uniform control flow.
//
// `kGemmFlagInBf16` (push constant `flags`) says the activation is bf16 rather
// than fp32: DSpark's draft chain (docs/p4_dspark_runtime.md §2.3) feeds
// `main_proj` the bf16 `main_hidden` the reference keeps, and `wkv(main_x)`
// where `main_x` is `main_norm`'s bf16 output. Reading it here is the only
// difference -- the amax, the ActQuant round trip and the LDS layout are the
// same, and the reference quantises exactly this bf16 value.
void stage_cols(uint64_t xaddr, uint xoff, uint k0, uint kc, uint tid) {
    const uint tpb = stage_tpb();
    const uint E = kBlockK / tpb;                 // elements this thread owns
    const uint blocks = kc / kBlockK;
    const uint slot = tid / tpb, sub = tid % tpb;
    const uint m = slot / kMaxBlocksS, blk = slot % kMaxBlocksS;
    const bool active = (m < M) && (blk < blocks);
    float v[32];
    float amax = 0.0f;
    [unroll] for (uint z = 0; z < 32; ++z) v[z] = 0.0f;
    if (active) {
        const uint elem = xoff + k0 + blk * kBlockK + sub * E;
        if ((pc.flags & kGemmFlagInBf16) != 0u) {
            const uint64_t base = xaddr + ((uint64_t)m * pc.x_stride + elem) * 2;
            [unroll] for (uint i = 0; i < 8; ++i) {
                if (i * 4 >= E) continue;
                // Four bf16 a load pair: two halves a word, low half first
                // (gpu/shaders/dspark_gemv.slang reads its activations the same
                // way).
                const uint64_t a = base + (uint64_t)i * 8;
                const uint w0 = load_u32(a), w1 = load_u32(a + 4);
                v[i * 4 + 0] = bf16_decode(w0 & 0xFFFFu);
                v[i * 4 + 1] = bf16_decode(w0 >> 16);
                v[i * 4 + 2] = bf16_decode(w1 & 0xFFFFu);
                v[i * 4 + 3] = bf16_decode(w1 >> 16);
                [unroll] for (uint j = 0; j < 4; ++j) amax = max(amax, abs(v[i * 4 + j]));
            }
        } else {
            const uint64_t base = xaddr + ((uint64_t)m * pc.x_stride + elem) * 4;
            [unroll] for (uint i = 0; i < 8; ++i) {
                if (i * 4 >= E) continue;
                const uint4 w = load16(base + (uint64_t)i * 16);
                [unroll] for (uint j = 0; j < 4; ++j) {
                    v[i * 4 + j] = asfloat(w[j]);
                    amax = max(amax, abs(v[i * 4 + j]));
                }
            }
        }
    }
    const float a = (ActQuant != 0) ? stage_amax(amax, tid, sub, tpb) : 0.0f;
    if (!active) return;
    float s = 1.0f;
    if (ActQuant != 0) {
        s = act_scale(a);
        const float inv = 1.0f / s;
        [unroll] for (uint i = 0; i < 32; ++i) {
            if (i >= E) continue;
            v[i] = fp8_round(clamp(v[i] * inv, -kFp8MaxF, kFp8MaxF));
        }
    }
    [unroll] for (uint i = 0; i < 16; ++i) {
        if (i * 2 >= E) continue;
        gXQ[xq_word(m, blk, sub * (E / 2) + i)] =
            bf16_encode(v[i * 2]) | (bf16_encode(v[i * 2 + 1]) << 16);
    }
    if (sub == 0) gXS[m * kColBlocks + blk] = s;
}

// --- stage 0 ------------------------------------------------------------------
void stage_split(uint gid, uint tid) {
    const uint rowsPerWg = kWg / LanesPerRow;
    const uint rg = gid / KSplit;
    const uint sl = gid % KSplit;
    const uint kc = (pc.k / (kBlockK * KSplit)) * kBlockK;
    const uint k0 = sl * kc;
    const uint base  = rg * rowsPerWg;
    const uint group = (pc.rows_per_group != 0u) ? (base / pc.rows_per_group) : 0u;

    fp8_table_fill(tid);
    stage_cols(Ptr[kX], group * pc.k, k0, kc, tid);
    GroupMemoryBarrierWithGroupSync();

    const uint sublane = tid % LanesPerRow;
    const uint row     = base + tid / LanesPerRow;
    const uint blocks  = kc / kBlockK;
    const uint blk0    = k0 / kBlockK;

    float acc[M];
    [unroll] for (uint m = 0; m < kMaxM; ++m) if (m < M) acc[m] = 0.0f;
    if (row < pc.rows) {
        const uint64_t roff = (uint64_t)row * (uint64_t)pc.k + (uint64_t)k0;
        const uint64_t srow = Ptr[kS] + (uint64_t)(row >> 5) * pc.scale_cols + blk0;
        for (uint blk = sublane; blk < blocks; blk += LanesPerRow) {
            const uint4 w0 = load16(Ptr[kW] + roff + (uint64_t)blk * 32);
            const uint4 w1 = load16(Ptr[kW] + roff + (uint64_t)blk * 32 + 16);
            const float we = ldexp(1.0f, (int)load_u8(srow + blk) - 127);
            float wv[32];
            [unroll] for (uint i = 0; i < 4; ++i) {
                const uint a = w0[i], b = w1[i];
                [unroll] for (uint j = 0; j < 4; ++j) {
                    wv[i * 4 + j]      = fp8_tbl((a >> (8 * j)) & 0xFFu);
                    wv[16 + i * 4 + j] = fp8_tbl((b >> (8 * j)) & 0xFFu);
                }
            }
            [unroll] for (uint m = 0; m < kMaxM; ++m) {
                if (m >= M) continue;
                float p = 0.0f;
                [unroll] for (uint i = 0; i < 16; ++i) {
                    const uint xw = gXQ[xq_word(m, blk, i)];
                    p += wv[i * 2]     * bf16_decode(xw & 0xFFFFu);
                    p += wv[i * 2 + 1] * bf16_decode(xw >> 16);
                }
                // Both factors are powers of two, so this is ldexp(p, e) * xs.
                acc[m] += p * (we * gXS[m * kColBlocks + blk]);
            }
        }
    }
    [unroll] for (uint m = 0; m < kMaxM; ++m) {
        if (m >= M) continue;
        const float y = row_reduce(acc[m], tid, sublane);
        if (sublane == 0 && row < pc.rows)
            *(float*)(Ptr[kP] + (((uint64_t)sl * M + m) * pc.part_stride + row) * 4) = y;
    }
}

float part_sum(uint m, uint row) {
    float y = 0.0f;
    for (uint s = 0; s < KSplit; ++s)
        y += load_f32(Ptr[kP] + (((uint64_t)s * M + m) * pc.part_stride + row) * 4);
    return y;
}

// --- stage 1 ------------------------------------------------------------------
void stage_combine(uint gx, uint m, uint tid) {
    const uint row = gx * kWg + tid;
    if (row >= pc.rows) return;
    *(float*)(Ptr[kY] + ((uint64_t)m * pc.y_stride + row) * 4) = part_sum(m, row);
}

// --- stage 2: q ---------------------------------------------------------------
void stage_qfinish(uint gx, uint m, uint tid) {
    const uint r = gx * kWg + tid;
    if (r >= pc.rows) return;
    const uint d = r % pc.head_dim;
    const uint rope_lo = pc.head_dim - pc.rope_dim;
    const uint64_t q = Ptr[kY] + (uint64_t)m * pc.y_stride * 2;
    if (d < rope_lo) {
        *(uint16_t*)(q + (uint64_t)r * 2) = (uint16_t)bf16_encode(part_sum(m, r));
    } else if (((d - rope_lo) & 1u) == 0u) {
        const uint j = (d - rope_lo) >> 1;
        const uint64_t tab = Ptr[kRope] + ((uint64_t)m * (pc.rope_dim / 2) + j) * 8;
        float re = part_sum(m, r), im = part_sum(m, r + 1);
        rope_pair(load_f32(tab), load_f32(tab + 4), re, im, false);
        *(uint16_t*)(q + (uint64_t)r * 2) = (uint16_t)bf16_encode(re);
        *(uint16_t*)(q + (uint64_t)(r + 1) * 2) = (uint16_t)bf16_encode(im);
    }
}

// --- stage 3: RMSNorm, one workgroup a token ------------------------------------
void stage_rmsnorm(uint m, uint tid) {
    const uint n = pc.head_dim;
    float sq = 0.0f;
    for (uint i = tid; i < n; i += kWg) {
        const float v = load_f32(Ptr[kX] + ((uint64_t)m * pc.x_stride + i) * 4);
        sq += v * v;
    }
    const float rs = rsqrt(wg_reduce(sq, tid) / float(n) + pc.eps);
    for (uint i = tid; i < n; i += kWg) {
        const float v = load_f32(Ptr[kX] + ((uint64_t)m * pc.x_stride + i) * 4);
        const float w = bf16_decode(load_u16(Ptr[kNormW] + (uint64_t)i * 2));
        *(float*)(Ptr[kY] + ((uint64_t)m * pc.y_stride + i) * 4) = bf16_round(v * rs * w);
    }
}

// --- stage 4: kv, one workgroup a token -----------------------------------------
void stage_kvfinish(uint m, uint tid) {
    const uint rows = pc.head_dim;                      // 512
    float sq = 0.0f;
    for (uint r = tid; r < rows; r += kWg) {
        const float y = part_sum(m, r);
        sq += y * y;
    }
    const float rs = rsqrt(wg_reduce(sq, tid) / float(rows) + pc.eps);

    const uint blocks = rows / kBlockK;
    if (tid >= blocks) return;
    const uint d0 = tid * kBlockK;
    const uint rope_lo = rows - pc.rope_dim;
    float v[32];
    [unroll] for (uint i = 0; i < 32; ++i)
        v[i] = bf16_round(part_sum(m, d0 + i) * rs
                          * bf16_decode(load_u16(Ptr[kNormW] + (uint64_t)(d0 + i) * 2)));
    [unroll] for (uint i = 0; i < 32; i += 2) {
        const uint d = d0 + i;
        if (d < rope_lo) continue;
        const uint j = (d - rope_lo) >> 1;
        const uint64_t tab = Ptr[kRope] + ((uint64_t)m * (pc.rope_dim / 2) + j) * 8;
        rope_pair(load_f32(tab), load_f32(tab + 4), v[i], v[i + 1], false);
        v[i] = bf16_round(v[i]);
        v[i + 1] = bf16_round(v[i + 1]);
    }
    float amax = 0.0f;
    [unroll] for (uint i = 0; i < 32; ++i) amax = max(amax, abs(v[i]));
    const float sc = act_scale(amax);
    const float inv = 1.0f / sc;

    const uint slot = (pc.p0 + m) % pc.window;
    const uint64_t val = Ptr[kRingVal] + (uint64_t)slot * rows + d0;
    const uint64_t scl = Ptr[kRingScale] + (uint64_t)slot * blocks + tid;
    if (m > 0) {
        // The overflow row first: what the slot held before this token wrote it.
        const uint64_t ov = Ptr[kOvfVal] + (uint64_t)(m - 1) * rows + d0;
        const uint4 a = load16(val), b = load16(val + 16);
        *(uint4*)ov = a;
        *(uint4*)(ov + 16) = b;
        *(uint8_t*)(Ptr[kOvfScale] + (uint64_t)(m - 1) * blocks + tid) = (uint8_t)load_u8(scl);
    }
    [unroll] for (uint i = 0; i < 32; ++i) {
        const uint b = fp8_encode_rn(clamp(v[i] * inv, -kFp8MaxF, kFp8MaxF));
        *(uint8_t*)(val + i) = (uint8_t)b;
        *(float*)(Ptr[kKvOut] + ((uint64_t)m * rows + d0 + i) * 4) = fp8_decode(b) * sc;
    }
    *(uint8_t*)scl = (uint8_t)((asuint(sc) >> 23) & 0xFFu);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0)      stage_split(gid.x, tid);
    else if (Stage == 1) stage_combine(gid.x, gid.y, tid);
    else if (Stage == 2) stage_qfinish(gid.x, gid.y, tid);
    else if (Stage == 3) stage_rmsnorm(gid.y, tid);
    else                 stage_kvfinish(gid.y, tid);
}
