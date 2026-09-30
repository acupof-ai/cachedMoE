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
//          one workgroup per (256-key block, query), heads on the lanes of a
//          wave, head_dim 128; then the prefill branch's completion mask,
//          `-inf` where key t is not yet complete for this query:
//          t >= (pos + 1) / ratio.
//
// Stages 3-5 are the same attention with the two dot-product passes moved onto
// cooperative-matrix tiles (prefill_coopmat.slang stages 3 and 4,
// docs/p4_prefill_speed.md §3):
//
// Stage 3  gather: G[b][t] = the KV row idx[b][t] names, as fp16 (zeros for
//          -1). One thread per 8 dims of a (query, entry). pc.n_idx = E, the entry stride
//          of G / S / P (a multiple of 16); pc.g = the index list's own stride
//          (<= E): entries past it are -1.
// Stage 4  softmax: from the tile scores S[b][h][t] (q.k, fp32 sums of fp16
//          products), per (query, head): s = S * scale, the row max over the
//          valid entries clamped at -1e30, p = bf16(exp(s - mx)) written as
//          fp16 into P [b][h][t] (0 for -1), and 1 / (sum p + exp(sink - mx))
//          into kW [b][h]. One wave per (query, head): the lanes read 32
//          consecutive entries at a time (one thread a row walked E-strided
//          rows and was 6.7 of a 4K prefill's 19 s of attention, STATUS §7
//          0t); the max is order-free, and the sum still adds p in entry order
//          (each lane folds in the other lanes' p one by one), so P and inv
//          are bit for bit the one-thread version's.
// Stage 5  finish: O[b][h][d] = bf16(O * inv[b][h]) in place, sixteen dims a
//          thread; or, with kFlagOutF16, straight into wo_a's staged fp16 plane
//          with the inverse RoPE applied (STATUS §7 0be).
//
// The KV rows are fp32 VALUES (the window KV already on its E4M3 grid, the
// compressed KV on its FP4 grid), so nothing here decodes a byte.

#include "prefill_common.slang"

static const uint kQ = 0, kKv = 1, kCmp = 2, kIdx = 3, kSink = 4, kScore = 5, kO = 6, kW = 7;
static const uint kX16 = 8, kTab = 9;       // stage 5 with kFlagOutF16

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
    uint  flags;         // stage 1: kFlagRound; stage 5: kFlagOutF16 (+ kFlagInverse)
    uint  rope_dim;      // stage 5 with kFlagOutF16: the inverse RoPE folded in (prefill_coopmat stage 2's)
    uint  rope_pos0;     //   rows of the table at kTab
    uint  x16_n32;       //   the fp16 plane at kX16 is [groups][x16_n32][hg * head_dim]
    uint  hg;            //   heads a group
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
// One workgroup per (256-key block, query). The query's heads live in
// registers, a head per lane, and the keys pass through gPool 32 at a time,
// four to a wave; every lane reads the same key word, a broadcast, where one
// key per wave with q in LDS had all 32 lanes on one bank. Per (head, key) the
// dot product still runs d = 0..127 in order and the heads are still summed by
// one WaveActiveSum, so the scores are bit for bit the old kernel's; keys past
// the completion mask are written -inf without being scored (STATUS §7 0w).
static const uint kIdxHd = 128, kIdxBlock = 256, kIdxSub = 32;
void s_index_score(uint3 gid, uint tid) {
    const uint b = gid.y, t_blk = gid.x * kIdxBlock;
    const uint nh = pc.n_heads, wave = tid / 32, hh = tid % 32;
    const uint t_end = min(t_blk + kIdxBlock, pc.g);
    const uint t_live = min(t_end, max(min((pc.pos0 + b + 1) / pc.ratio, pc.g), t_blk));
    const uint64_t srow = Ptr[kScore] + (uint64_t)b * pc.g * 4;
    for (uint t = t_live + tid; t < t_end; t += kWg) store_f32(srow + (uint64_t)t * 4, asfloat(0xFF800000u));
    if (t_live == t_blk) return;                  // the same for the whole workgroup

    const float w = (hh < nh) ? load_f32(Ptr[kW] + ((uint64_t)b * nh + hh) * 4) : 0.0f;
    float q[kIdxHd];
    const uint64_t qrow = Ptr[kQ] + ((uint64_t)b * nh + hh) * kIdxHd * 4;
    [unroll] for (uint i = 0; i < kIdxHd / 4; ++i) {
        const uint4 a = (hh < nh) ? load16(qrow + (uint64_t)i * 16) : uint4(0, 0, 0, 0);
        [unroll] for (uint j = 0; j < 4; ++j) q[i * 4 + j] = asfloat(a[j]);
    }
    for (uint s0 = t_blk; s0 < t_live; s0 += kIdxSub) {
        GroupMemoryBarrierWithGroupSync();        // the last sub-block is read
        for (uint i = tid; i < kIdxSub * kIdxHd / 4; i += kWg) {
            const uint t = s0 + i / (kIdxHd / 4);
            const uint4 a = (t < t_live) ? load16(Ptr[kKv] + (uint64_t)t * kIdxHd * 4 + (uint64_t)(i % (kIdxHd / 4)) * 16)
                                         : uint4(0, 0, 0, 0);
            [unroll] for (uint j = 0; j < 4; ++j) gPool[i * 4 + j] = asfloat(a[j]);
        }
        GroupMemoryBarrierWithGroupSync();
        const uint k0 = wave * 4;
        float dot[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        [unroll] for (uint d = 0; d < kIdxHd; ++d)
            [unroll] for (uint j = 0; j < 4; ++j) dot[j] += q[d] * gPool[(k0 + j) * kIdxHd + d];
        [unroll] for (uint j = 0; j < 4; ++j) {
            const float contrib = bf16_round(max(bf16_round(dot[j]), 0.0f) * w);
            const float sc = bf16_round(WaveActiveSum(contrib));
            const uint t = s0 + k0 + j;
            if (hh == 0 && t < t_live) store_f32(srow + (uint64_t)t * 4, sc);
        }
    }
}

// --- stage 3: gather ------------------------------------------------------------
// kIdx [B][g] int32, kKv [n_win][hd] f32, kCmp [..][hd] f32, kO = G16 [B][E][hd].
// A thread per 8 dims of a (query, entry) row: 32 bytes in, one 16-byte store
// out, a wave covering a quarter of a row contiguously. A thread per row with
// 2-byte stores was 10.5 s of a 17K prefill (STATUS §7 0x).
void s_gather(uint3 gid, uint tid) {
    const uint hd = pc.head_dim, chunks = hd / 8;
    const uint i = gid_linear(gid) * kWg + tid;
    const uint row = i / chunks, d0 = (i % chunks) * 8;
    const uint b = row / pc.n_idx, t = row % pc.n_idx;
    if (b >= pc.b) return;
    const int idx = (t < pc.g) ? load_i32(Ptr[kIdx] + ((uint64_t)b * pc.g + t) * 4) : -1;
    const uint64_t dst = Ptr[kO] + ((uint64_t)row * hd + d0) * 2;
    if (idx < 0) { *(uint4*)dst = uint4(0, 0, 0, 0); return; }
    const uint64_t src = ((uint(idx) < pc.n_win)
        ? Ptr[kKv]  + (uint64_t)uint(idx) * hd * 4
        : Ptr[kCmp] + (uint64_t)(uint(idx) - pc.n_win) * hd * 4) + (uint64_t)d0 * 4;
    const uint4 a = load16(src), c = load16(src + 16);
    float v[8];
    [unroll] for (uint j = 0; j < 4; ++j) { v[j] = asfloat(a[j]); v[4 + j] = asfloat(c[j]); }
    *(uint4*)dst = f16x8(v);
}

// --- stage 4: softmax -----------------------------------------------------------
// kScore = S [B][H][E] f32 (q.k), kIdx [B][g], kSink [H] f32, kO = P16
// [B][H][E], kW = inv [B][H] f32. A wave is a row; a lane takes four
// consecutive entries (one float4 of S, one 8-byte store of P), and the
// denominator is still summed in entry order -- lane j's four in turn -- so
// P and inv are bit for bit the one-entry-a-lane version's (STATUS §7 0al).
// E is a multiple of 16, the index list's length g anything: its four
// entries are scalar loads (the same 64 x 4 bytes for every head, L1 hits).
void s_softmax(uint3 gid, uint tid) {
    const uint i = gid_linear(gid) * kWg + tid;
    const uint row = i / 32, lane = i % 32;     // SubgroupSize is 32: a wave is a row
    const uint b = row / pc.n_heads, h = row % pc.n_heads;
    if (b >= pc.b) return;
    const uint E = pc.n_idx;
    const uint64_t srow = Ptr[kScore] + ((uint64_t)b * pc.n_heads + h) * E * 4;
    const uint64_t irow = Ptr[kIdx] + (uint64_t)b * pc.g * 4;
    const uint64_t prow = Ptr[kO] + ((uint64_t)b * pc.n_heads + h) * E * 2;
    // The lane's entries (at most kMaxIdx / 128 float4s) and their validity
    // stay in registers between the max pass and the exp pass: S and the
    // index list are read once, not twice (STATUS §7 0bf).
    static const uint kIt = kMaxIdx / 128;
    float4 sv[kIt];
    uint   ok[kIt];
    float mx = -3.0e38f;
    [unroll] for (uint it = 0; it < kIt; ++it) {
        const uint t = it * 128 + lane * 4;
        sv[it] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        ok[it] = 0u;
        if (t < E) {
            sv[it] = load_f32x4(srow + (uint64_t)t * 4);
            [unroll] for (uint k = 0; k < 4; ++k)
                if (t + k < pc.g && load_i32(irow + (uint64_t)(t + k) * 4) >= 0) {
                    ok[it] |= 1u << k;
                    mx = max(mx, sv[it][k] * pc.scale);
                }
        }
    }
    mx = max(WaveActiveMax(mx), -1.0e30f);
    float den = 0.0f;
    [unroll] for (uint it = 0; it < kIt; ++it) {
        const uint t0 = it * 128;
        if (t0 >= E) break;
        const uint t = t0 + lane * 4;
        float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (t < E) {
            [unroll] for (uint k = 0; k < 4; ++k)
                if ((ok[it] & (1u << k)) != 0u) p[k] = bf16_round(exp(sv[it][k] * pc.scale - mx));
            *(uint2*)(prow + (uint64_t)t * 2) =
                uint2(f32tof16(p[0]) | (f32tof16(p[1]) << 16), f32tof16(p[2]) | (f32tof16(p[3]) << 16));
        }
        [unroll] for (uint j = 0; j < 32; ++j)
            [unroll] for (uint k = 0; k < 4; ++k) den += WaveReadLaneAt(p[k], j);
    }
    if (lane != 0) return;
    den += exp(load_f32(Ptr[kSink] + (uint64_t)h * 4) - mx);
    store_f32(Ptr[kW] + ((uint64_t)b * pc.n_heads + h) * 4, 1.0f / den);
}

// --- stage 5: finish ------------------------------------------------------------
// kO [B][H][hd] f32 in place, kW = inv [B][H].
void s_finish(uint3 gid, uint tid) {
    const uint i = gid_linear(gid) * kWg + tid;
    const uint chunks = pc.head_dim / 16;
    const uint bh = i / chunks, c0 = (i % chunks) * 16;
    const uint b = bh / pc.n_heads;
    if (b >= pc.b) return;
    const float inv = load_f32(Ptr[kW] + (uint64_t)bh * 4);
    const uint64_t o = Ptr[kO] + ((uint64_t)bh * pc.head_dim + c0) * 4;
    if ((pc.flags & kFlagOutF16) == 0u) {
        [unroll] for (uint q = 0; q < 16; ++q)
            store_f32(o + (uint64_t)q * 4, maybe_round(load_f32(o + (uint64_t)q * 4) * inv, pc.flags));
        return;
    }
    // kFlagOutF16: the values wo_a's staging would have made of the fp32 plane
    // (prefill_coopmat stage 2 with its RoPE: the pair rotation, the rounding,
    // then fp16), written straight into that staging's [groups][n32][K] plane
    // and nowhere else (STATUS §7 0be). Same operations in the same order on
    // every element, so the plane is bit for bit the staged one.
    float v[16];
    [unroll] for (uint q = 0; q < 16; ++q) v[q] = maybe_round(load_f32(o + (uint64_t)q * 4) * inv, pc.flags);
    const uint lo = pc.head_dim - pc.rope_dim;
    if (c0 + 16 > lo) {
        const uint64_t tab = Ptr[kTab] + (uint64_t)(pc.rope_pos0 + b) * pc.rope_dim * 4;
        const bool inverse = (pc.flags & kFlagInverse) != 0u;
        [unroll] for (uint i = 0; i < 16; i += 2) {
            const uint rel = c0 + i;
            if (rel < lo) continue;
            const uint j = (rel - lo) >> 1;
            const float c = load_f32(tab + (uint64_t)j * 8);
            const float sn = load_f32(tab + (uint64_t)j * 8 + 4);
            rope_pair(c, sn, v[i], v[i + 1], inverse);
            v[i] = maybe_round(v[i], pc.flags);
            v[i + 1] = maybe_round(v[i + 1], pc.flags);
        }
    }
    const uint h = bh % pc.n_heads, K = pc.hg * pc.head_dim;
    const uint64_t dst = Ptr[kX16] + (((uint64_t)(h / pc.hg) * pc.x16_n32 + b) * K + (h % pc.hg) * pc.head_dim + c0) * 2;
    float a[8], d[8];
    [unroll] for (uint q = 0; q < 8; ++q) { a[q] = v[q]; d[q] = v[8 + q]; }
    *(uint4*)dst = f16x8(a);
    *(uint4*)(dst + 16) = f16x8(d);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0)      s_score(gid, tid);
    else if (Stage == 1) s_combine(gid, tid);
    else if (Stage == 2) s_index_score(gid, tid);
    else if (Stage == 3) s_gather(gid, tid);
    else if (Stage == 4) s_softmax(gid, tid);
    else                 s_finish(gid, tid);
}
