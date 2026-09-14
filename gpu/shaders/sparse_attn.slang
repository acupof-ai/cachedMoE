// sparse_attn -- MQA over the sliding window plus the indexer's compressed
// picks, with the inverse RoPE fused into the write-out (design §7.5,
// dispatch #5 of the §7.14 list).
//
// One decode query, 64 heads, one shared 512-wide KV row per position: about
// 640 positions, 320 KiB of KV and 42 MFLOP. A latency kernel, not a bandwidth
// one.
//
// Why two stages rather than design §7.5's single fused pass
// ----------------------------------------------------------
// `sparse_attn_kernel` rounds the probabilities to bf16 *after* subtracting the
// final row max (`acc_s_cast`), and takes that max over the KV scores ONLY --
// the per-head `attn_sink` logit enters the denominator as `exp(sink - mx)` but
// never competes for the max. An online softmax that rescaled as it went would
// round `p` against a running max instead, so it would not be the same number.
// Reproducing the reference costs one extra pass over 320 KiB.
//
//   Stage 0  every score, into a [head][slot] plane.
//   Stage 1  the max, `p = bf16(exp(s - mx))`, the P.V accumulation, the
//            divide, and the inverse rotation.
//
// A slot whose index is -1 holds nothing (`get_window_topk_idxs` marks a ring
// slot the sequence has not reached yet, and the indexer marks an unreachable
// compressed block); it scores -inf and contributes nothing, which is why the
// running max starts at a finite -1e30 in the reference and here.
//
// Index space: `topk_idxs` addresses the CONCATENATION the reference builds,
// `cat([window_kv, compress_kv], dim=1)`. Values below `n_win` are ring slots
// in the fp8 window cache; the rest are compressed rows, offset by `n_win`.
//
// What the P2 bandwidth pass changed (docs/p2_attention.md §7 item 2)
// ------------------------------------------------------------------
// The two stages cost 142 us a layer -- 5.7 ms a token -- for 320 KiB of data,
// and the reason was never bandwidth. It was that **every KV element was a
// scalar byte load**. Stage 0 gave one thread one (head, position) pair and
// walked 512 dims serially: per pair, 512 loads of q, 512 of the E4M3 byte and
// 512 of its UE8M0 scale. Across 64 heads and 640 positions that is 63 million
// load instructions for 320 KiB of data, which no cache hierarchy can rescue.
//
// Now a KV row is covered by the 32 lanes of one wave, sixteen dims each, so
// the E4M3 half is ONE 16-byte load and one scale byte per lane and the
// compressed half is two. The per-head dot product is a `WaveActiveSum` over
// those 32 lanes. Stage 1 does the same for P.V, with the thread that owns an
// output span also owning the inverse-RoPE pair inside it.
//
// `HeadsPerWg` then attacks the traffic rather than the instruction count: one
// workgroup per head meant all 64 of them pulled the whole 320 KiB through L2
// twice, 41 MB for 320 KiB of unique data. A workgroup covering G heads reads
// it once for all G. This is **opt-in through `n_heads`**, which is a trailing
// field of AttnPush and therefore zero in a caller that predates it: at
// `n_heads == 0` the kernel is one head a workgroup, exactly the dispatch
// geometry it has always had, and only the vectorisation applies. Passing the
// real head count divides both the workgroup count and the L2 traffic by G.

#include "attn_common.slang"

static const uint kQ = 0, kWinVal = 1, kWinScale = 2, kCmpKv = 3, kTopIdx = 4;
static const uint kSink = 5, kRope = 6, kScore = 7, kO = 8;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct AttnPush {
    uint  n_kv;            // entries in topk_idxs
    uint  n_win;           // window_size, and the offset of the compressed half
    uint  head_dim;        // 512
    uint  rope_dim;        // 64
    uint  score_stride;    // row stride of the score plane
    float softmax_scale;   // head_dim ** -0.5
    uint  n_heads;         // 0 = one head per workgroup (pre-HeadsPerWg caller)
};
[[vk::push_constant]] AttnPush pc;

// 32 lanes to a KV row, so a lane owns head_dim / 32 dims: 16 at head_dim 512.
// Stage 1's output spans are the same width when HeadsPerWg is 8.
static const uint kMaxSpan = 16;
static const uint kMaxKv   = 1024;        // window 128 + index_topk 512, with room
static const uint kMaxHeadsPerWg = 8;

// One pool, two uses, because LDS is the union over a file's pipelines: stage 0
// stages q for its head group (HeadsPerWg * head_dim floats) and stage 1 the
// per-thread P.V partials (256 threads * head_dim / 32 dims). At head_dim 512
// and HeadsPerWg 8 those are the same 4096 floats.
groupshared int   gIdx[kMaxKv];           // topk_idxs, read once instead of per t
groupshared float gPool[kMaxHeadsPerWg * 512];

// This workgroup's head range. `hn - h0` is 1 for a caller that does not push
// n_heads, which is what keeps the old dispatch geometry correct.
uint head_lo(uint gid) { return (pc.n_heads == 0u) ? gid : gid * HeadsPerWg; }
uint head_hi(uint gid) {
    return (pc.n_heads == 0u) ? (gid + 1u) : min(gid * HeadsPerWg + HeadsPerWg, pc.n_heads);
}

// `span` consecutive KV elements starting at `d0`, dequantised. The window half
// is fp8 E4M3 with a UE8M0 scale per 32 dims, exactly as wkv.slang wrote it; the
// compressed half is bf16, holding the values the compressor already put on the
// FP4 grid (design §6: the fp4 byte packing of §11.3 is a storage form, not a
// different number).
//
// `d0` is a multiple of `span` and `span` divides 32, so a span never straddles
// two UE8M0 blocks: one scale byte covers it.
void kv_span(uint idx, uint d0, uint span, out float v[kMaxSpan]) {
    [unroll] for (uint z = 0; z < kMaxSpan; ++z) v[z] = 0.0f;
    if (idx < pc.n_win) {
        const uint64_t base = Ptr[kWinVal] + (uint64_t)idx * pc.head_dim + d0;
        const uint e = load_u8(Ptr[kWinScale] + (uint64_t)idx * (pc.head_dim / kBlockK)
                               + (d0 / kBlockK));
        const float sc = ldexp(1.0f, int(e) - 127);
        if (span >= 4u) {
            // Four E4M3 bytes a load, so a 16-wide span is four dwords (one
            // dwordx4 after the compiler merges them) instead of sixteen byte
            // loads. `span` is 2, 4, 8 or 16 here: head_dim / (256 / heads).
            [unroll] for (uint w = 0; w < kMaxSpan / 4; ++w) {
                if (w * 4 >= span) continue;
                const uint b = load_u32(base + (uint64_t)w * 4);
                [unroll] for (uint j = 0; j < 4; ++j)
                    v[w * 4 + j] = gTbl[(b >> (8 * j)) & 0xFFu] * sc;
            }
        } else {
            const uint b = load_u16(base);
            v[0] = gTbl[b & 0xFFu] * sc;
            if (span > 1u) v[1] = gTbl[b >> 8] * sc;
        }
        return;
    }
    const uint64_t base = Ptr[kCmpKv] + ((uint64_t)(idx - pc.n_win) * pc.head_dim + d0) * 2;
    // The compressed half is bf16: two elements a dword.
    [unroll] for (uint w = 0; w < kMaxSpan / 2; ++w) {
        if (w * 2 >= span) continue;
        const uint b = load_u32(base + (uint64_t)w * 4);
        v[w * 2]     = bf16_decode(b & 0xFFFFu);
        v[w * 2 + 1] = bf16_decode(b >> 16);
    }
}

// Reduce over the `n` consecutive lanes of one group. `n` is 32 whenever the
// group is a wave, which is the case stage 0 always hits.
float grp_sum(float v, uint tid, uint n) {
    if (n == SubgroupSize) return WaveActiveSum(v);
    gRed[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = n >> 1; s > 0; s >>= 1) {
        if ((tid % n) < s) gRed[tid] = gRed[tid] + gRed[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = gRed[tid - (tid % n)];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

float grp_max(float v, uint tid, uint n) {
    if (n == SubgroupSize) return WaveActiveMax(v);
    gRed[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = n >> 1; s > 0; s >>= 1) {
        if ((tid % n) < s) gRed[tid] = max(gRed[tid], gRed[tid + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = gRed[tid - (tid % n)];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

void load_idx(uint tid) {
    for (uint t = tid; t < pc.n_kv; t += kWg) gIdx[t] = (int)load_u32(Ptr[kTopIdx] + (uint64_t)t * 4);
}

// --- stage 0: the scores ----------------------------------------------------
// 256 threads = eight waves; wave `w` walks positions t = w, w + 8, ... and its
// 32 lanes split the 512-dim row sixteen dims apiece. The staged q is read once
// per head per position from LDS, and the KV row once for all `hn - h0` heads.
void stage_score(uint gid, uint tid) {
    fp8_table_fill(tid);
    const uint h0 = head_lo(gid), hn = head_hi(gid);
    const uint span = pc.head_dim / 32;             // 16
    for (uint i = tid; i < (hn - h0) * pc.head_dim; i += kWg)
        gPool[i] = bf16_decode(load_u16(Ptr[kQ] + ((uint64_t)h0 * pc.head_dim + i) * 2));
    load_idx(tid);
    GroupMemoryBarrierWithGroupSync();

    const uint wave = tid / 32, lane = tid % 32;
    const uint waves = kWg / 32;
    const uint d0 = lane * span;
    for (uint t = wave; t < pc.n_kv; t += waves) {
        const int idx = gIdx[t];
        float v[kMaxSpan];
        [unroll] for (uint z = 0; z < kMaxSpan; ++z) v[z] = 0.0f;
        if (idx >= 0) kv_span((uint)idx, d0, span, v);
        for (uint h = h0; h < hn; ++h) {
            float p = 0.0f;
            if (idx >= 0) {
                [unroll] for (uint i = 0; i < kMaxSpan; ++i) {
                    if (i >= span) continue;
                    p += gPool[(h - h0) * pc.head_dim + d0 + i] * v[i];
                }
            }
            // A KV row is one wave's worth of lanes, so this is one
            // WaveActiveSum -- and it must be, because the t loop is not
            // uniform across the workgroup and an LDS tree here would deadlock.
            const float s = WaveActiveSum(p);
            if (lane == 0)
                *(float*)(Ptr[kScore] + ((uint64_t)h * pc.score_stride + t) * 4) =
                    (idx >= 0) ? (s * pc.softmax_scale) : -3.0e38f;
        }
    }
}

// --- stage 1: softmax, P.V, inverse RoPE ------------------------------------
// A head is owned by `kWg / (hn - h0)` threads and each of them by
// `head_dim / that` output dims -- 32 threads and sixteen dims at
// HeadsPerWg 8, 256 threads and two dims for a caller that pushes no head
// count. The probabilities go back into the score plane, which is what makes
// the P.V pass a single scalar load per position.
void stage_combine(uint gid, uint tid) {
    fp8_table_fill(tid);
    const uint h0 = head_lo(gid), hn = head_hi(gid);
    const uint g    = hn - h0;
    const uint tph  = kWg / g;                    // threads on one head, >= 32
    const uint sub  = tid % tph;                  // lane inside the head
    const uint ngrp = tph / 32;                   // t subsets inside the head
    const uint grp  = sub / 32;                   // which subset this lane walks
    const uint dl   = sub % 32;                   // which span of the output row
    const uint span = pc.head_dim / 32;           // 16 dims a lane, always
    const uint d0   = dl * span;
    const uint head = h0 + tid / tph;
    load_idx(tid);
    GroupMemoryBarrierWithGroupSync();

    const uint64_t sc = Ptr[kScore] + (uint64_t)head * pc.score_stride * 4;

    float local = -3.0e38f;
    for (uint t = sub; t < pc.n_kv; t += tph) local = max(local, load_f32(sc + (uint64_t)t * 4));
    // The reference clamps the row max at a finite -1e30 so an all-empty row
    // comes out zero instead of NaN.
    const float mx = max(grp_max(local, tid, tph), -1.0e30f);

    // p, rounded to bf16 exactly where `acc_s_cast` does it, written back over
    // the scores so the P.V pass reads them from the same plane.
    float lsum = 0.0f;
    for (uint t = sub; t < pc.n_kv; t += tph) {
        float p = 0.0f;
        if (gIdx[t] >= 0) p = bf16_decode(bf16_encode(exp(load_f32(sc + (uint64_t)t * 4) - mx)));
        *(float*)(sc + (uint64_t)t * 4) = p;
        lsum += p;
    }
    const float denom = grp_sum(lsum, tid, tph)
                      + exp(load_f32(Ptr[kSink] + (uint64_t)head * 4) - mx);
    // The p values just written went to DEVICE memory, and the P.V loop below
    // reads all of them, not just this thread's: the barrier has to cover the
    // buffer, not only LDS.
    AllMemoryBarrierWithGroupSync();

    // P.V. A lane owns sixteen output dims whatever HeadsPerWg is -- 32 lanes
    // cover the 512-wide row -- so the KV read is always the vectorised path of
    // kv_span. What HeadsPerWg changes is how many DISJOINT subsets of the KV
    // list the head's lanes split into: one at 8 heads a workgroup, eight at
    // one. Their partial sums then meet in LDS. Giving a thread only two dims
    // instead (256 lanes over one head's row, which is what this did before)
    // turns every KV element into a byte load again and measured 106 us against
    // 89 -- the span has to stay wide even when the workgroup owns one head.
    float acc[kMaxSpan];
    [unroll] for (uint i = 0; i < kMaxSpan; ++i) acc[i] = 0.0f;
    for (uint t = grp; t < pc.n_kv; t += ngrp) {
        const int idx = gIdx[t];
        if (idx < 0) continue;
        const float p = load_f32(sc + (uint64_t)t * 4);
        if (p == 0.0f) continue;
        float v[kMaxSpan];
        kv_span((uint)idx, d0, span, v);
        [unroll] for (uint i = 0; i < kMaxSpan; ++i) {
            if (i >= span) continue;
            acc[i] += p * v[i];
        }
    }
    if (ngrp > 1) {
        [unroll] for (uint i = 0; i < kMaxSpan; ++i) {
            if (i >= span) continue;
            gPool[tid * span + i] = acc[i];
        }
        GroupMemoryBarrierWithGroupSync();
        if (grp == 0) {
            for (uint gg = 1; gg < ngrp; ++gg) {
                [unroll] for (uint i = 0; i < kMaxSpan; ++i) {
                    if (i >= span) continue;
                    acc[i] += gPool[(tid + gg * 32) * span + i];
                }
            }
        }
    }
    if (grp != 0) return;

    const float inv = 1.0f / denom;
    float o[kMaxSpan];
    [unroll] for (uint i = 0; i < kMaxSpan; ++i) o[i] = acc[i] * inv;
    const uint rope_lo = pc.head_dim - pc.rope_dim;
    // `span` and `d0` are both even, so a rotation pair is owned by one thread.
    [unroll] for (uint i = 0; i + 1 < kMaxSpan; i += 2) {
        if (i + 1 >= span) continue;
        const uint d = d0 + i;
        if (d < rope_lo) continue;
        const uint j = (d - rope_lo) >> 1;
        const float c = load_f32(Ptr[kRope] + (uint64_t)j * 8);
        const float s = load_f32(Ptr[kRope] + (uint64_t)j * 8 + 4);
        rope_pair(c, s, o[i], o[i + 1], true);     // inverse: conjugate
    }
    [unroll] for (uint i = 0; i < kMaxSpan; ++i) {
        if (i >= span) continue;
        *(float*)(Ptr[kO] + ((uint64_t)head * pc.head_dim + d0 + i) * 4) = o[i];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0) stage_score(gid.x, tid);
    else            stage_combine(gid.x, tid);
}
