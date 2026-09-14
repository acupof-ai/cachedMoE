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
//   Stage 0  one workgroup per head: every score, into a [head][slot] plane.
//   Stage 1  one workgroup per head: the max, `p = bf16(exp(s - mx))`, the
//            P.V accumulation, the divide, and the inverse rotation.
//
// A slot whose index is -1 holds nothing (`get_window_topk_idxs` marks a ring
// slot the sequence has not reached yet, and the indexer marks an unreachable
// compressed block); it scores -inf and contributes nothing, which is why the
// running max starts at a finite -1e30 in the reference and here.
//
// Index space: `topk_idxs` addresses the CONCATENATION the reference builds,
// `cat([window_kv, compress_kv], dim=1)`. Values below `n_win` are ring slots
// in the fp8 window cache; the rest are compressed rows, offset by `n_win`.

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
};
[[vk::push_constant]] AttnPush pc;

// One KV element. The window half is fp8 E4M3 with a UE8M0 scale per 32 dims,
// exactly as wkv.slang wrote it; the compressed half is bf16, holding the
// values the compressor already put on the FP4 grid (design §6: the fp4 byte
// packing of §11.3 is a storage form, not a different number).
float kv_at(uint idx, uint d) {
    if (idx < pc.n_win) {
        const uint b = load_u8(Ptr[kWinVal] + (uint64_t)idx * pc.head_dim + d);
        const uint e = load_u8(Ptr[kWinScale] + (uint64_t)idx * (pc.head_dim / kBlockK)
                               + (d / kBlockK));
        return gTbl[b] * ldexp(1.0f, int(e) - 127);
    }
    return bf16_decode(load_u16(Ptr[kCmpKv]
                                + ((uint64_t)(idx - pc.n_win) * pc.head_dim + d) * 2));
}

int top_idx(uint t) { return (int)load_u32(Ptr[kTopIdx] + (uint64_t)t * 4); }

void stage_score(uint head, uint tid) {
    fp8_table_fill(tid);
    GroupMemoryBarrierWithGroupSync();
    const uint64_t q = Ptr[kQ] + (uint64_t)head * pc.head_dim * 2;
    for (uint t = tid; t < pc.n_kv; t += kWg) {
        const int idx = top_idx(t);
        float s = -3.0e38f;
        if (idx >= 0) {
            float acc = 0.0f;
            for (uint d = 0; d < pc.head_dim; ++d)
                acc += bf16_decode(load_u16(q + (uint64_t)d * 2)) * kv_at((uint)idx, d);
            s = acc * pc.softmax_scale;
        }
        *(float*)(Ptr[kScore] + ((uint64_t)head * pc.score_stride + t) * 4) = s;
    }
}

void stage_combine(uint head, uint tid) {
    fp8_table_fill(tid);
    GroupMemoryBarrierWithGroupSync();
    const uint64_t sc = Ptr[kScore] + (uint64_t)head * pc.score_stride * 4;

    float local = -3.0e38f;
    for (uint t = tid; t < pc.n_kv; t += kWg) local = max(local, load_f32(sc + (uint64_t)t * 4));
    // The reference clamps the row max at a finite -1e30 so an all-empty row
    // comes out zero instead of NaN.
    const float mx = max(wg_max(local, tid), -1.0e30f);

    // p, rounded to bf16 exactly where `acc_s_cast` does it, written back over
    // the scores so the P.V pass reads them from the same plane.
    float lsum = 0.0f;
    for (uint t = tid; t < pc.n_kv; t += kWg) {
        const int idx = top_idx(t);
        float p = 0.0f;
        if (idx >= 0) p = bf16_decode(bf16_encode(exp(load_f32(sc + (uint64_t)t * 4) - mx)));
        *(float*)(sc + (uint64_t)t * 4) = p;
        lsum += p;
    }
    const float denom = wg_reduce(lsum, tid)
                      + exp(load_f32(Ptr[kSink] + (uint64_t)head * 4) - mx);

    // Two output dims per thread, adjacent, so a RoPE pair is owned by one
    // thread and the loads across a wave cover a whole KV row.
    const uint per = pc.head_dim / kWg;
    float acc[8];
    [unroll] for (uint i = 0; i < 8; ++i) acc[i] = 0.0f;
    for (uint t = 0; t < pc.n_kv; ++t) {
        const int idx = top_idx(t);
        if (idx < 0) continue;
        const float p = load_f32(sc + (uint64_t)t * 4);
        if (p == 0.0f) continue;
        for (uint i = 0; i < per; ++i) acc[i] += p * kv_at((uint)idx, tid * per + i);
    }

    const float inv = 1.0f / denom;
    float o[8];
    for (uint i = 0; i < per; ++i) o[i] = acc[i] * inv;
    const uint rope_lo = pc.head_dim - pc.rope_dim;
    for (uint i = 0; i + 1 < per; i += 2) {
        const uint d = tid * per + i;
        if (d >= rope_lo) {
            const uint j = (d - rope_lo) >> 1;
            const float c = load_f32(Ptr[kRope] + (uint64_t)j * 8);
            const float s = load_f32(Ptr[kRope] + (uint64_t)j * 8 + 4);
            rope_pair(c, s, o[i], o[i + 1], true);     // inverse: conjugate
        }
    }
    for (uint i = 0; i < per; ++i)
        *(float*)(Ptr[kO] + ((uint64_t)head * pc.head_dim + tid * per + i) * 4) = o[i];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0) stage_score(gid.x, tid);
    else            stage_combine(gid.x, tid);
}
