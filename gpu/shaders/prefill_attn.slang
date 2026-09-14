// prefill_attn -- the attention of the prefill branch, for a block of B queries
// at once (docs/p3_prefill.md §1, §6).
//
// `sparse_attn(q, kv, sink, topk_idxs, scale)` with q [B][H][hd], kv the
// concatenation `cat([kv rows of this layer's chunk, compressed rows])`, and
// topk_idxs [B][n_idx] of which a query's window band comes first and its
// top-k compressed picks after (offset by the chunk length). The same two
// passes as the decode kernel (sparse_attn.slang), for the same reason: the
// reference rounds p onto bf16 AFTER subtracting the final row max, and that
// max is over the KV scores only, so an online softmax computes a different
// number.
//
// Stage 0  scores. One workgroup per (head, query): its 256 threads are eight
//          waves; wave w walks index entries t = w, w+8, ... and its 32 lanes
//          split a 512-wide KV row sixteen dims apiece, so a score is one
//          WaveActiveSum. The query row sits in LDS, read once.
// Stage 1  combine. The row max, p = bf16(exp(s - max)) written back over the
//          scores, the denominator with the sink, P.V per wave into LDS, and
//          wave 0 adds the eight partials and divides. bf16 out.
// Stage 2  the indexer's scores [B][G]: sum_h bf16(relu(bf16(q[h].k[t])) w[h]),
//          one workgroup per (8-key tile, query), heads on the lanes of a wave;
//          then the prefill branch's completion mask, `-inf` where key t is not
//          yet complete for this query: t >= (pos + 1) / ratio.
//
// The KV rows are fp32 VALUES (the window KV already on its E4M3 grid, the
// compressed KV on its FP4 grid), so nothing here decodes a byte.

#include "prefill_common.slang"

static const uint kQ = 0, kKv = 1, kCmp = 2, kIdx = 3, kSink = 4, kScore = 5, kO = 6, kW = 7;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct AttnPush {
    uint  b;             // queries in this dispatch
    uint  n_idx;         // index entries per query
    uint  n_heads;       // 64 (attention) or 32 (indexer)
    uint  head_dim;      // 512 or 128
    uint  n_win;         // rows of kKv; an index >= n_win is compressed row idx - n_win
    uint  g;             // stage 2: keys
    float scale;         // stage 0: head_dim ** -0.5
    uint  ratio;         // stage 2
    uint  pos0;          // stage 2: absolute position of query 0
    uint  flags;         // stage 1: kFlagRound
};
[[vk::push_constant]] AttnPush pc;

static const uint kMaxIdx = 1024;           // 128 window + 512 top-k, with room
groupshared int   gIdx[kMaxIdx];
groupshared float gPool[5120];
groupshared float gW[8];

void kv_span(int idx, uint d0, out float v[16]) {
    const uint64_t base = (uint(idx) < pc.n_win)
        ? Ptr[kKv]  + ((uint64_t)uint(idx) * pc.head_dim + d0) * 4
        : Ptr[kCmp] + ((uint64_t)(uint(idx) - pc.n_win) * pc.head_dim + d0) * 4;
    [unroll] for (uint i = 0; i < 4; ++i) {
        const uint4 a = load16(base + (uint64_t)i * 16);
        [unroll] for (uint j = 0; j < 4; ++j) v[i * 4 + j] = asfloat(a[j]);
    }
}

void load_idx_row(uint b, uint tid) {
    for (uint t = tid; t < pc.n_idx; t += kWg)
        gIdx[t] = load_i32(Ptr[kIdx] + ((uint64_t)b * pc.n_idx + t) * 4);
}

// --- stage 0 ------------------------------------------------------------------
void s_score(uint3 gid, uint tid) {
    const uint h = gid.x, b = gid.y;
    const uint hd = pc.head_dim;
    for (uint i = tid; i < hd; i += kWg)
        gPool[i] = load_f32(Ptr[kQ] + (((uint64_t)b * pc.n_heads + h) * hd + i) * 4);
    load_idx_row(b, tid);
    GroupMemoryBarrierWithGroupSync();

    const uint wave = tid / 32, lane = tid % 32;
    const uint d0 = lane * (hd / 32);
    const uint64_t srow = Ptr[kScore] + ((uint64_t)b * pc.n_heads + h) * pc.n_idx * 4;
    for (uint t = wave; t < pc.n_idx; t += 8) {
        const int idx = gIdx[t];
        float p = 0.0f;
        if (idx >= 0) {
            float v[16];
            kv_span(idx, d0, v);
            [unroll] for (uint i = 0; i < 16; ++i) p += gPool[d0 + i] * v[i];
        }
        const float s = WaveActiveSum(p);
        if (lane == 0) store_f32(srow + (uint64_t)t * 4, (idx >= 0) ? s * pc.scale : -3.0e38f);
    }
}

// --- stage 1 ------------------------------------------------------------------
void s_combine(uint3 gid, uint tid) {
    const uint h = gid.x, b = gid.y;
    const uint hd = pc.head_dim;
    load_idx_row(b, tid);
    GroupMemoryBarrierWithGroupSync();
    const uint wave = tid / 32, lane = tid % 32;
    const uint64_t srow = Ptr[kScore] + ((uint64_t)b * pc.n_heads + h) * pc.n_idx * 4;

    // the row max over this wave's index subset, then over the eight waves
    float local = -3.0e38f;
    if (lane == 0)
        for (uint t = wave; t < pc.n_idx; t += 8) local = max(local, load_f32(srow + (uint64_t)t * 4));
    if (lane == 0) gW[wave] = local;
    GroupMemoryBarrierWithGroupSync();
    float mx = -3.0e38f;
    [unroll] for (uint w = 0; w < 8; ++w) mx = max(mx, gW[w]);
    mx = max(mx, -1.0e30f);     // the reference's finite floor: an empty row is 0, not NaN

    // p onto bf16 against that max, back over the scores; the partial sums
    float lsum = 0.0f;
    if (lane == 0) {
        for (uint t = wave; t < pc.n_idx; t += 8) {
            float p = 0.0f;
            if (gIdx[t] >= 0) p = bf16_round(exp(load_f32(srow + (uint64_t)t * 4) - mx));
            store_f32(srow + (uint64_t)t * 4, p);
            lsum += p;
        }
        gW[wave] = lsum;
    }
    AllMemoryBarrierWithGroupSync();
    float denom = exp(load_f32(Ptr[kSink] + (uint64_t)h * 4) - mx);
    [unroll] for (uint w = 0; w < 8; ++w) denom += gW[w];

    // P.V over this wave's subset, sixteen output dims a lane
    const uint d0 = lane * (hd / 32);
    float acc[16];
    [unroll] for (uint i = 0; i < 16; ++i) acc[i] = 0.0f;
    for (uint t = wave; t < pc.n_idx; t += 8) {
        const int idx = gIdx[t];
        if (idx < 0) continue;
        const float p = load_f32(srow + (uint64_t)t * 4);
        if (p == 0.0f) continue;
        float v[16];
        kv_span(idx, d0, v);
        [unroll] for (uint i = 0; i < 16; ++i) acc[i] += p * v[i];
    }
    [unroll] for (uint i = 0; i < 16; ++i) gPool[tid * 16 + i] = acc[i];
    GroupMemoryBarrierWithGroupSync();
    if (wave != 0) return;
    const float inv = 1.0f / denom;
    const uint64_t orow = Ptr[kO] + (((uint64_t)b * pc.n_heads + h) * hd + d0) * 4;
    [unroll] for (uint i = 0; i < 16; ++i) {
        float s = 0.0f;
        [unroll] for (uint w = 0; w < 8; ++w) s += gPool[(tid + w * 32) * 16 + i];
        store_f32(orow + (uint64_t)i * 4, maybe_round(s * inv, pc.flags));
    }
}

// --- stage 2 ------------------------------------------------------------------
// kQ [B][32][128] post-fp4 values, kKv = keys [G][128], kW = weights [B][32]
// (already bf16 and scaled), kScore = out [B][G].
void s_index_score(uint3 gid, uint tid) {
    const uint b = gid.y, t0 = gid.x * 8;
    const uint nh = pc.n_heads, hd = pc.head_dim;
    for (uint i = tid; i < nh * hd; i += kWg)
        gPool[i] = load_f32(Ptr[kQ] + (((uint64_t)b * nh) * hd + i) * 4);
    for (uint i = tid; i < 8 * hd; i += kWg) {
        const uint t = t0 + i / hd;
        gPool[nh * hd + i] = (t < pc.g) ? load_f32(Ptr[kKv] + ((uint64_t)t * hd + i % hd) * 4) : 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint p = tid / 32, hh = tid % 32;
    float dot = 0.0f;
    for (uint d = 0; d < hd; ++d) dot += gPool[hh * hd + d] * gPool[nh * hd + p * hd + d];
    const float w = (hh < nh) ? load_f32(Ptr[kW] + ((uint64_t)b * nh + hh) * 4) : 0.0f;
    const float contrib = bf16_round(max(bf16_round(dot), 0.0f) * w);
    const float s = bf16_round(WaveActiveSum(contrib));
    const uint t = t0 + p;
    if (hh == 0 && t < pc.g) {
        const uint visible = (pc.pos0 + b + 1) / pc.ratio;
        store_f32(Ptr[kScore] + ((uint64_t)b * pc.g + t) * 4,
                  (t < visible) ? s : asfloat(0xFF800000u));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0)      s_score(gid, tid);
    else if (Stage == 1) s_combine(gid, tid);
    else                 s_index_score(gid, tid);
}
