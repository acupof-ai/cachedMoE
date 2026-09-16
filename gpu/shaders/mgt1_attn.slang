// mgt1_attn -- sparse_attn_t.slang for a verify batch of M queries (Track T,
// docs/p4_mgt1.md), with the causal mask inside the batch.
//
// Each query m has its OWN index row (`[M][list_stride]` int32): the window
// ring slots holding positions <= p0 + m, the overflow rows of later batch
// tokens (below), and its own indexer's compressed picks. So causality is a
// property of the lists the host and the indexer write, not of this kernel:
// query i never has an index to the ring slot token j > i wrote. Everything
// else -- the tile maxima, `p = bf16(exp(s - mx))` rounded against the FINAL
// row max, the sink entering the denominator once, the inverse RoPE -- is
// sparse_attn_t.slang's arithmetic, token by token.
//
// The overflow region. Writing token j's KV into ring slot (p0 + j) % window
// destroys position p0 + j - window, which queries i < j still attend to once
// the ring has wrapped. mgt1_gemv.slang stage 4 copies each such row into
// overflow row j - 1 before the write, in the ring's own format (E4M3 bytes
// plus a UE8M0 scale per 32 dims), so the index space is
//
//     [0, n_win)                      ring slot
//     [n_win, n_win + n_ovf)          overflow row idx - n_win
//     [n_win + n_ovf, ...)            compressed row idx - n_win - n_ovf (bf16)
//
// and the compressed half's offset is `window + M - 1`. The overflow rows are
// also exactly what a rejected draft must restore (docs/p3_dspark.md §3.5).
//
// Stages: 0 scores + tile maxima, gid = (head_group * n_tiles + tile, m);
// 1 P.V against the final max, divide, inverse RoPE -- one P.V tile, one head a
// workgroup, gid = (head, m).

#include "attn_common.slang"

static const uint kQ = 0, kWinVal = 1, kWinScale = 2, kCmpKv = 3, kTopIdx = 4;
static const uint kSink = 5, kRope = 6, kScore = 7, kO = 8, kTileMax = 9;
static const uint kOvfVal = 10, kOvfScale = 11;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct Mgt1AttnPush {
    uint  n_kv;            // entries a query's row is walked for
    uint  n_win;           // ring slots
    uint  n_ovf;           // overflow rows (M - 1)
    uint  head_dim;        // 512
    uint  rope_dim;        // 64
    uint  score_stride;    // >= n_kv
    float softmax_scale;
    uint  n_heads;         // 64
    uint  n_tiles;
    uint  tile_len;
    uint  list_stride;     // int32 entries between query rows of kTopIdx
};
[[vk::push_constant]] Mgt1AttnPush pc;

static const uint kMaxSpan = 16;
static const uint kMaxHeadsPerWg = 8;
static const uint kMaxTiles = 64;
static const uint kPvStride = kMaxSpan + 1;
groupshared float gPool[kWg * (kMaxSpan + 1)];

void kv_span(uint idx, uint d0, uint span, out float v[kMaxSpan]) {
    [unroll] for (uint z = 0; z < kMaxSpan; ++z) v[z] = 0.0f;
    const uint nfp8 = pc.n_win + pc.n_ovf;
    if (idx < nfp8) {
        const bool ovf = idx >= pc.n_win;
        const uint row = ovf ? (idx - pc.n_win) : idx;
        const uint64_t vb = ovf ? Ptr[kOvfVal] : Ptr[kWinVal];
        const uint64_t sb = ovf ? Ptr[kOvfScale] : Ptr[kWinScale];
        const uint64_t base = vb + (uint64_t)row * pc.head_dim + d0;
        const uint e = load_u8(sb + (uint64_t)row * (pc.head_dim / kBlockK) + (d0 / kBlockK));
        const float sc = ldexp(1.0f, int(e) - 127);
        [unroll] for (uint w = 0; w < kMaxSpan / 4; ++w) {
            if (w * 4 >= span) continue;
            const uint b = load_u32(base + (uint64_t)w * 4);
            [unroll] for (uint j = 0; j < 4; ++j)
                v[w * 4 + j] = fp8_tbl((b >> (8 * j)) & 0xFFu) * sc;
        }
        return;
    }
    const uint64_t base = Ptr[kCmpKv] + ((uint64_t)(idx - nfp8) * pc.head_dim + d0) * 2;
    [unroll] for (uint w = 0; w < kMaxSpan / 2; ++w) {
        if (w * 2 >= span) continue;
        const uint b = load_u32(base + (uint64_t)w * 4);
        v[w * 2]     = bf16_decode(b & 0xFFFFu);
        v[w * 2 + 1] = bf16_decode(b >> 16);
    }
}

float row_max(uint m, uint head) {
    const uint64_t b = Ptr[kTileMax] + (((uint64_t)m * pc.n_heads + head) * pc.n_tiles) * 4;
    float m0 = -3.0e38f, m1 = -3.0e38f, m2 = -3.0e38f, m3 = -3.0e38f;
    [unroll] for (uint z = 0; z < kMaxTiles; z += 4) {
        if (z     < pc.n_tiles) m0 = max(m0, load_f32(b + (uint64_t)z * 4));
        if (z + 1 < pc.n_tiles) m1 = max(m1, load_f32(b + (uint64_t)(z + 1) * 4));
        if (z + 2 < pc.n_tiles) m2 = max(m2, load_f32(b + (uint64_t)(z + 2) * 4));
        if (z + 3 < pc.n_tiles) m3 = max(m3, load_f32(b + (uint64_t)(z + 3) * 4));
    }
    return max(max(max(m0, m1), max(m2, m3)), -1.0e30f);
}

void stage_score(uint gid, uint m, uint tid) {
    fp8_table_fill(tid);
    const uint hg = gid / pc.n_tiles, tl = gid % pc.n_tiles;
    const uint h0 = hg * HeadsPerWg;
    const uint hn = min(h0 + HeadsPerWg, pc.n_heads);
    const uint g  = hn - h0;
    const uint span = pc.head_dim / 32;
    const uint64_t qbase = Ptr[kQ] + (uint64_t)m * pc.n_heads * 512 * 2;
    [unroll] for (uint w = 0; w < 16; ++w) {
        const uint k  = tid * 16 + w;
        const uint hh = k >> 9;
        if (hh >= g) continue;
        const uint d  = ((k & 31u) << 4) | ((k >> 5) & 15u);
        gPool[k] = bf16_decode(load_u16(qbase + ((uint64_t)(h0 + hh) * 512 + d) * 2));
    }
    GroupMemoryBarrierWithGroupSync();

    const uint t0 = tl * pc.tile_len;
    const uint t1 = min(t0 + pc.tile_len, pc.n_kv);
    const uint wave = tid / 32, lane = tid % 32, waves = kWg / 32;
    const uint d0 = lane * span;
    const uint64_t list = Ptr[kTopIdx] + (uint64_t)m * pc.list_stride * 4;
    const uint64_t srow = Ptr[kScore] + (uint64_t)m * pc.n_heads * pc.score_stride * 4;

    float hmax[kMaxHeadsPerWg];
    [unroll] for (uint z = 0; z < kMaxHeadsPerWg; ++z) hmax[z] = -3.0e38f;
    for (uint t = t0 + wave; t < t1; t += waves) {
        const int idx = (int)load_u32(list + (uint64_t)t * 4);
        float v[kMaxSpan];
        [unroll] for (uint z = 0; z < kMaxSpan; ++z) v[z] = 0.0f;
        if (idx >= 0) kv_span((uint)idx, d0, span, v);
        [unroll] for (uint hh = 0; hh < kMaxHeadsPerWg; ++hh) {
            if (hh >= g) continue;
            float p = 0.0f;
            if (idx >= 0) {
                [unroll] for (uint i = 0; i < kMaxSpan; ++i) {
                    if (i >= span) continue;
                    p += gPool[hh * pc.head_dim + i * 32 + lane] * v[i];
                }
            }
            const float s0 = WaveActiveSum(p);
            const float s  = (idx >= 0) ? (s0 * pc.softmax_scale) : -3.0e38f;
            hmax[hh] = max(hmax[hh], s);
            if (lane == 0)
                *(float*)(srow + ((uint64_t)(h0 + hh) * pc.score_stride + t) * 4) = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0) {
        [unroll] for (uint hh = 0; hh < kMaxHeadsPerWg; ++hh)
            gRed[wave * kMaxHeadsPerWg + hh] = hmax[hh];
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid < g) {
        float mx = -3.0e38f;
        for (uint w = 0; w < waves; ++w) mx = max(mx, gRed[w * kMaxHeadsPerWg + tid]);
        *(float*)(Ptr[kTileMax] + ((((uint64_t)m * pc.n_heads + h0 + tid) * pc.n_tiles) + tl) * 4) = mx;
    }
}

// One head a workgroup (pv_heads_per_wg = 1), one P.V tile: 8 waves each own a
// disjoint position subset, and the workgroup finishes the head itself.
void stage_pv(uint head, uint m, uint tid) {
    fp8_table_fill(tid);
    const uint ngrp = kWg / 32;
    const uint grp  = tid / 32;
    const uint dl   = tid % 32;
    const uint span = pc.head_dim / 32;
    const uint d0   = dl * span;
    GroupMemoryBarrierWithGroupSync();

    const float mx = row_max(m, head);
    const uint64_t sc = Ptr[kScore] + ((uint64_t)m * pc.n_heads + head) * pc.score_stride * 4;
    const uint64_t list = Ptr[kTopIdx] + (uint64_t)m * pc.list_stride * 4;

    float acc[kMaxSpan];
    [unroll] for (uint i = 0; i < kMaxSpan; ++i) acc[i] = 0.0f;
    float dsum = 0.0f;
    for (uint t = grp; t < pc.n_kv; t += ngrp) {
        const int idx = (int)load_u32(list + (uint64_t)t * 4);
        if (idx < 0) continue;
        const float p = bf16_decode(bf16_encode(exp(load_f32(sc + (uint64_t)t * 4) - mx)));
        dsum += p;
        if (p == 0.0f) continue;
        float v[kMaxSpan];
        kv_span((uint)idx, d0, span, v);
        [unroll] for (uint i = 0; i < kMaxSpan; ++i) {
            if (i >= span) continue;
            acc[i] += p * v[i];
        }
    }
    [unroll] for (uint i = 0; i < kMaxSpan; ++i) gPool[tid * kPvStride + i] = acc[i];
    gPool[tid * kPvStride + kMaxSpan] = dsum;
    GroupMemoryBarrierWithGroupSync();
    if (grp != 0) return;
    for (uint gg = 1; gg < ngrp; ++gg) {
        const uint b = (tid + gg * 32) * kPvStride;
        [unroll] for (uint i = 0; i < kMaxSpan; ++i) acc[i] += gPool[b + i];
        dsum += gPool[b + kMaxSpan];
    }
    const float inv = 1.0f / (dsum + exp(load_f32(Ptr[kSink] + (uint64_t)head * 4) - mx));
    float o[kMaxSpan];
    [unroll] for (uint i = 0; i < kMaxSpan; ++i) o[i] = acc[i] * inv;
    const uint rope_lo = pc.head_dim - pc.rope_dim;
    [unroll] for (uint i = 0; i + 1 < kMaxSpan; i += 2) {
        if (i + 1 >= span) continue;
        const uint d = d0 + i;
        if (d < rope_lo) continue;
        const uint j = (d - rope_lo) >> 1;
        const uint64_t tab = Ptr[kRope] + ((uint64_t)m * (pc.rope_dim / 2) + j) * 8;
        rope_pair(load_f32(tab), load_f32(tab + 4), o[i], o[i + 1], true);
    }
    const uint64_t ob = Ptr[kO] + ((uint64_t)m * pc.n_heads * pc.head_dim
                                   + (uint64_t)head * pc.head_dim + d0) * 4;
    [unroll] for (uint i = 0; i < kMaxSpan; ++i) {
        if (i >= span) continue;
        *(float*)(ob + (uint64_t)i * 4) = o[i];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0) stage_score(gid.x, gid.y, tid);
    else            stage_pv(gid.x, gid.y, tid);
}
