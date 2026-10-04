// decode_attn_cm -- the M = 1 decode attention (design §7.5) as two small
// GEMMs on cooperative-matrix tiles.
//
// Why a new kernel (docs/STATUS.md §7 0j, bench/results/linux/gpu_model/)
// ----------------------------------------------------------------------
// The attention is MQA: 64 heads, ONE shared 512-wide KV row per position, and
// that row is both the key and the value. sparse_attn.slang gives every head
// its own workgroup, so every head reads, decodes and dots every KV row on its
// own -- 64 copies of the same work through L2. attn_bench measured it on this
// machine at the decode list length a real conversation has (window 128 +
// index_topk 512 = 640 entries): score 85 us + combine 63 us a layer, 5.9 ms a
// token; per position 0.13 + 0.08 us. The tiled kernel (sparse_attn_t) is
// 52 + 68. The per-dispatch model (tools/gpu_model.py) puts the work -- 0.6 MB
// of cached KV and 2 x 42 MFLOP -- at a few microseconds.
//
// Here the heads are the rows of a matrix:
//   S[h][t] = sum_d Q[h][d] G[t][d]          (64 x E) = (64 x 512)(512 x E)
//   O[h][d] = sum_t P[h][t] G[t][d]          (64 x 512) = (64 x E)(E x 512)
// with G the gathered KV rows, so each KV element is loaded and converted ONCE
// and used by all 64 heads.
//
// Numerics. The tiles are fp16 in, fp32 accumulated (prefill_coopmat's
// choice). The window KV is E4M3 x a power-of-two scale and the compressed KV
// bf16: both are exact in fp16 over their range. q is bf16 and exact in fp16
// down to fp16's normal minimum (6.1e-5); p is bf16 in (0, 1] and exact above
// the same bound. What changes against sparse_attn is the summation order and
// the few bits those two sub-6e-5 tails keep -- measured by the drift gates.
// Everything the reference specifies is kept: the max over the KV scores only
// (the sink never competes), clamped at -1e30; p = bf16(exp(s - max)) AFTER the
// final max; the sink in the denominator; the division; the inverse RoPE.
//
// Stages (all 256 threads; a matrix stage gives each of the 8 waves a tile):
//   0 gather    rows 0..E-1 of G from the index list (-1 and the padding past
//               n_kv are zero rows), rows E..E+63 are q -> Q16. 16 rows a
//               workgroup, 32 dims (one UE8M0 block) a thread.
//   1 score     S tiles: (head tile, entry tile, d slice) per wave. s_split > 1
//               writes the slices to kPart and the softmax adds them.
//   2 softmax   one workgroup a head: scale, mask, max, p -> P16, 1/denominator.
//   3 pv        O tiles: (head tile, dim tile, list slice) per wave; the raw
//               sums go to kO, or at pv_split > 1 to kPart.
//   4 finish    (add the slices,) *= 1/denominator, inverse RoPE on the last
//               rope_dim dims.
//
// Measured (attn_bench --kv 640, RADV 8060S, us): gather 3.1, score 6.9,
// softmax 3.4, pv 18.5 unsplit / 8.0 at pv_split 4, finish 2.1 -- 23.5 against
// sparse_attn's 85 + 64. A split P.V has four times the waves over the same
// 40 k-steps; splitting the score's 32 k-steps moves the time into the
// softmax's extra reads instead (5.9 + 4.5 at 2).

#include "attn_common.slang"

typealias TileA = linalg.CoopMat<half,  MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA>;
typealias TileB = linalg.CoopMat<half,  MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB>;
typealias TileC = linalg.CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixAccumulator>;

// p is in (0, 1] and bf16, so it is exact in fp16 only down to fp16's normal
// minimum, 2^-14; below that fp16 is subnormal and drops bits. P is therefore
// stored as p * 2^14 -- a power of two, so exact -- which keeps every p down to
// 2^-28 exact, and the 2^-14 comes back in the 1/denominator.
static const float kPScale = 16384.0f;

// The first nine slots are sparse_attn's (gpu::slot::kAttn*), so the host binds
// one table for both kernels; 12..15 are this kernel's planes.
static const uint kQ = 0, kWinVal = 1, kWinScale = 2, kCmpKv = 3, kTopIdx = 4;
static const uint kSink = 5, kRope = 6, kScore = 7, kO = 8;
static const uint kG16 = 12, kQ16 = 13, kP16 = 14, kInv = 15, kPart = 16;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct CmPush {
    uint  n_kv;            // entries in the index list
    uint  e;               // n_kv rounded up to 16: the G / P row count
    uint  n_win;           // an index below it is a window ring slot
    uint  head_dim;        // 512
    uint  rope_dim;        // 64
    uint  n_heads;         // 64
    uint  score_stride;    // row stride of S
    float softmax_scale;   // head_dim ** -0.5
    uint  s_split;         // score: slices of head_dim, partial S planes summed by softmax
    uint  pv_split;        // pv: slices of the list, partial O planes summed by finish
#if defined(DEEPMOE_CM_BATCH)
    uint n_ovf, list_stride, part_stride; // verify overflow and per-query planes
#endif
};
[[vk::push_constant]] CmPush pc;

#if defined(DEEPMOE_CM_BATCH)
static uint cm_query;
#endif
uint64_t cm_ptr(uint slot) {
    const uint64_t base = Ptr[slot];
#if defined(DEEPMOE_CM_BATCH)
    const uint64_t m = cm_query;
    if (slot == kQ) return base + m * pc.n_heads * pc.head_dim * 2;
    if (slot == kTopIdx) return base + m * pc.list_stride * 4;
    if (slot == kRope) return base + m * pc.rope_dim * 4;
    if (slot == kScore) return base + m * pc.n_heads * pc.score_stride * 4;
    if (slot == kO) return base + m * pc.n_heads * pc.head_dim * 4;
    if (slot == kG16) return base + m * pc.e * pc.head_dim * 2;
    if (slot == kQ16) return base + m * pc.n_heads * pc.head_dim * 2;
    if (slot == kP16) return base + m * pc.n_heads * pc.e * 2;
    if (slot == kInv) return base + m * pc.n_heads * 4;
    if (slot == kPart) return base + m * pc.part_stride * 4;
#endif
    return base;
}
groupshared float gRedF[kWg];

int top_idx(uint t) {
    return (t < pc.n_kv) ? (int)load_u32(cm_ptr(kTopIdx) + (uint64_t)t * 4) : -1;
}

// --- stage 0 ------------------------------------------------------------------
// 32 values -> 32 fp16 at `dst`, as four 16-byte stores. Every value here is
// exact in fp16 (the header), so the conversion's rounding never acts.
void store_h32(uint64_t dst, float v[32]) {
    [unroll] for (uint w = 0; w < 4; ++w) {
        uint4 o;
        [unroll] for (uint j = 0; j < 4; ++j)
            o[j] = f32tof16(v[w * 8 + j * 2]) | (f32tof16(v[w * 8 + j * 2 + 1]) << 16);
        *(uint4*)(dst + (uint64_t)w * 16) = o;
    }
}

// 32 bf16 at `src` (64 bytes) -> floats.
void load_bf16x32(uint64_t src, out float v[32]) {
    [unroll] for (uint w = 0; w < 4; ++w) {
        const uint4 b = load16(src + (uint64_t)w * 16);
        [unroll] for (uint j = 0; j < 4; ++j) {
            v[w * 8 + j * 2]     = bf16_decode(b[j] & 0xFFFFu);
            v[w * 8 + j * 2 + 1] = bf16_decode(b[j] >> 16);
        }
    }
}

// 16 threads a row, 32 dims (one UE8M0 block) a thread, 16 rows a workgroup.
void stage_gather(uint gid, uint tid) {
    const uint row = gid * 16 + tid / 16;
    const uint d0  = (tid % 16) * 32;
    if (row >= pc.e + pc.n_heads) return;
    float v[32];
    if (row < pc.e) {
        const int idx = top_idx(row);
        [unroll] for (uint i = 0; i < 32; ++i) v[i] = 0.0f;
        if (idx >= 0 && uint(idx) < pc.n_win) {
            const uint64_t base = cm_ptr(kWinVal) + (uint64_t)idx * pc.head_dim + d0;
            const uint e = load_u8(cm_ptr(kWinScale) + (uint64_t)idx * (pc.head_dim / kBlockK)
                                   + d0 / kBlockK);
            const float sc = ldexp(1.0f, int(e) - 127);
            [unroll] for (uint w = 0; w < 2; ++w) {
                const uint4 b = load16(base + (uint64_t)w * 16);
                [unroll] for (uint j = 0; j < 16; ++j)
                    v[w * 16 + j] = fp8_decode((b[j / 4] >> (8 * (j % 4))) & 0xFFu) * sc;
            }
#if defined(DEEPMOE_CM_BATCH)
        } else if (idx >= 0 && uint(idx) < pc.n_win + pc.n_ovf) {
            const uint row = uint(idx) - pc.n_win;
            const uint64_t base = cm_ptr(10) + (uint64_t)row * pc.head_dim + d0;
            const uint e = load_u8(cm_ptr(11) + (uint64_t)row * (pc.head_dim / kBlockK) + d0 / kBlockK);
            const float sc = ldexp(1.0f, int(e) - 127);
            [unroll] for (uint w = 0; w < 2; ++w) {
                const uint4 b = load16(base + (uint64_t)w * 16);
                [unroll] for (uint j = 0; j < 16; ++j)
                    v[w * 16 + j] = fp8_decode((b[j / 4] >> (8 * (j % 4))) & 0xFFu) * sc;
            }
        } else if (idx >= 0) {
            load_bf16x32(cm_ptr(kCmpKv) + ((uint64_t)(uint(idx) - pc.n_win - pc.n_ovf) * pc.head_dim + d0) * 2, v);
#else
        } else if (idx >= 0) {
            load_bf16x32(cm_ptr(kCmpKv) + ((uint64_t)(uint(idx) - pc.n_win) * pc.head_dim + d0) * 2, v);
#endif
        }
        store_h32(cm_ptr(kG16) + ((uint64_t)row * pc.head_dim + d0) * 2, v);
    } else {
        const uint h = row - pc.e;
        load_bf16x32(cm_ptr(kQ) + ((uint64_t)h * pc.head_dim + d0) * 2, v);
        store_h32(cm_ptr(kQ16) + ((uint64_t)h * pc.head_dim + d0) * 2, v);
    }
}

// --- stage 1 ------------------------------------------------------------------
// A wave a (head tile, entry tile, d slice). At s_split 1 the tile goes straight
// to S; otherwise slice c goes to partial plane c of kPart, [c][head][stride].
void stage_score(uint gid, uint tid) {
    const uint task = gid * (kWg / 32) + tid / 32;
    const uint ht = pc.n_heads / 16;
    const uint tiles = ht * (pc.e / 16);
    if (task >= tiles * pc.s_split) return;
    const uint tile = task % tiles, c = task / tiles;
    const uint h0 = (tile % ht) * 16, t0 = (tile / ht) * 16;
    const uint dn = pc.head_dim / pc.s_split, d0 = c * dn;
    half* Q = (half*)cm_ptr(kQ16);
    half* G = (half*)cm_ptr(kG16);
    TileC acc = TileC(0.0f);
    for (uint d = d0; d < d0 + dn; d += 16) {
        const TileA a = TileA.Load<linalg.CoopMatMatrixLayout.RowMajor>(Q, h0 * pc.head_dim + d, pc.head_dim);
        const TileB b = TileB.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(G, t0 * pc.head_dim + d, pc.head_dim);
        acc = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, b, acc);
    }
    float* S = (pc.s_split == 1u) ? (float*)cm_ptr(kScore) : (float*)cm_ptr(kPart);
    acc.Store<linalg.CoopMatMatrixLayout.RowMajor>(S, (c * pc.n_heads + h0) * pc.score_stride + t0,
                                                   pc.score_stride);
}

// --- stage 2 ------------------------------------------------------------------
float wg_sum_f(float v, uint tid) {
    gRedF[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = kWg / 2; s > 0; s >>= 1) {
        if (tid < s) gRedF[tid] += gRedF[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = gRedF[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

float wg_max_f(float v, uint tid) {
    gRedF[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = kWg / 2; s > 0; s >>= 1) {
        if (tid < s) gRedF[tid] = max(gRedF[tid], gRedF[tid + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = gRedF[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// The raw dot product of head h with entry t: S itself, or the sum of the
// s_split partial planes in slice order.
float score_at(uint h, uint t) {
    if (pc.s_split == 1u) return load_f32(cm_ptr(kScore) + ((uint64_t)h * pc.score_stride + t) * 4);
    float v = 0.0f;
    for (uint c = 0; c < pc.s_split; ++c)
        v += load_f32(cm_ptr(kPart) + (((uint64_t)c * pc.n_heads + h) * pc.score_stride + t) * 4);
    return v;
}

void stage_softmax(uint h, uint tid) {
    half* P = (half*)cm_ptr(kP16) + (uint64_t)h * pc.e;
    float local = -3.0e38f;
    for (uint t = tid; t < pc.n_kv; t += kWg)
        if (top_idx(t) >= 0) local = max(local, score_at(h, t) * pc.softmax_scale);
    // The reference clamps the row max at a finite -1e30 so an all-empty row
    // comes out zero instead of NaN.
    const float mx = max(wg_max_f(local, tid), -1.0e30f);
    float lsum = 0.0f;
    for (uint t = tid; t < pc.e; t += kWg) {
        float p = 0.0f;
        if (t < pc.n_kv && top_idx(t) >= 0)
            p = bf16_round(exp(score_at(h, t) * pc.softmax_scale - mx));
        P[t] = half(p * kPScale);
        lsum += p;
    }
    const float denom = wg_sum_f(lsum, tid) + exp(load_f32(cm_ptr(kSink) + (uint64_t)h * 4) - mx);
    // 1/denominator, and the 2^-14 that undoes kPScale: both exact scalings of
    // the one division the reference does.
    if (tid == 0) *(float*)(cm_ptr(kInv) + (uint64_t)h * 4) = (1.0f / denom) / kPScale;
}

// --- stage 3 ------------------------------------------------------------------
// A wave a (head tile, dim tile, list slice). At pv_split 1 the tile goes to
// kO; otherwise slice c goes to partial plane c of kPart, [c][head][head_dim].
void stage_pv(uint gid, uint tid) {
    const uint task = gid * (kWg / 32) + tid / 32;
    const uint ht = pc.n_heads / 16;
    const uint tiles = ht * (pc.head_dim / 16);
    if (task >= tiles * pc.pv_split) return;
    const uint tile = task % tiles, c = task / tiles;
    const uint h0 = (tile % ht) * 16, d0 = (tile / ht) * 16;
    // Slices of whole 16-row tiles; the last may be short or empty.
    const uint steps = pc.e / 16, per = (steps + pc.pv_split - 1) / pc.pv_split;
    const uint t_lo = min(c * per, steps) * 16, t_hi = min(c * per + per, steps) * 16;
    half* P = (half*)cm_ptr(kP16);
    half* G = (half*)cm_ptr(kG16);
    TileC acc = TileC(0.0f);
    for (uint t = t_lo; t < t_hi; t += 16) {
        const TileA a = TileA.Load<linalg.CoopMatMatrixLayout.RowMajor>(P, h0 * pc.e + t, pc.e);
        const TileB b = TileB.Load<linalg.CoopMatMatrixLayout.RowMajor>(G, t * pc.head_dim + d0, pc.head_dim);
        acc = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, b, acc);
    }
    float* O = (pc.pv_split == 1u) ? (float*)cm_ptr(kO) : (float*)cm_ptr(kPart);
    acc.Store<linalg.CoopMatMatrixLayout.RowMajor>(O, (c * pc.n_heads + h0) * pc.head_dim + d0,
                                                   pc.head_dim);
}

// --- stage 4 ------------------------------------------------------------------
// One thread a (head, dim pair): the pair is what the inverse rotation needs.
void stage_finish(uint gid, uint tid) {
    const uint pair = gid * kWg + tid;
    const uint half_dim = pc.head_dim / 2;
    if (pair >= pc.n_heads * half_dim) return;
    const uint h = pair / half_dim, d = (pair % half_dim) * 2;
    const uint64_t o = cm_ptr(kO) + ((uint64_t)h * pc.head_dim + d) * 4;
    const float inv = load_f32(cm_ptr(kInv) + (uint64_t)h * 4);
    float re, im;
    if (pc.pv_split == 1u) {
        re = load_f32(o);
        im = load_f32(o + 4);
    } else {
        re = 0.0f;
        im = 0.0f;
        for (uint c = 0; c < pc.pv_split; ++c) {
            const uint64_t q = cm_ptr(kPart) + (((uint64_t)c * pc.n_heads + h) * pc.head_dim + d) * 4;
            re += load_f32(q);
            im += load_f32(q + 4);
        }
    }
    re *= inv;
    im *= inv;
    const uint rope_lo = pc.head_dim - pc.rope_dim;
    if (d >= rope_lo) {
        const uint j = (d - rope_lo) >> 1;
        rope_pair(load_f32(cm_ptr(kRope) + (uint64_t)j * 8), load_f32(cm_ptr(kRope) + (uint64_t)j * 8 + 4),
                  re, im, true);     // inverse: conjugate
    }
    *(float*)o = re;
    *(float*)(o + 4) = im;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
#if defined(DEEPMOE_CM_BATCH)
    cm_query = gid.y;
#endif
    if (Stage == 0)      stage_gather(gid.x, tid);
    else if (Stage == 1) stage_score(gid.x, tid);
    else if (Stage == 2) stage_softmax(gid.x, tid);
    else if (Stage == 3) stage_pv(gid.x, tid);
    else                 stage_finish(gid.x, tid);
}
