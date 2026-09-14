// The DSpark draft window attention (design §7.12), M = 5 queries over the
// 128-slot ring plus the five in-block positions.
//
// What it computes
// ----------------
// `DSparkAttention.forward` with `compress_ratio == 0`, so plain sliding-window
// attention and no indexer, no compressed half:
//
//     for each draft position m and head h:
//         s_t   = softmax_scale * dot(q[m][h], kv[t])      for every VALID t
//         mx    = max_t s_t                                (over the KV only)
//         p_t   = bf16(exp(s_t - mx))
//         denom = sum_t p_t + exp(attn_sink[h] - mx)
//         o[m][h] = (sum_t p_t * kv[t]) / denom
//
// `kv` is the reference's `cat([window_kv_cache, kv], dim=1)`: 128 ring rows
// then the five in-block rows, 133 x 512 bf16 = 136 KB. A topk index IS a row
// index into that concatenation, which is why the host passes one flat buffer.
//
// Three things the reference does that are easy to get wrong
// ----------------------------------------------------------
// 1. **Every one of the five queries attends to the SAME index set.** The
//    reference builds `matrix = cat(arange(min(win, start_pos + seqlen)),
//    win + arange(block_size))` and expands it over the block dimension, so
//    `topk_idxs` does not vary with m. In particular **there is no causal mask
//    inside the five-position block**: draft position 0 sees draft positions
//    1..4. That is not an oversight to be helpfully fixed; it is what produces
//    the numbers the oracle exports, and a causal mask here would silently
//    change every draft token.
// 2. **The row max is taken over the KV scores only.** `attn_sink[h]` enters the
//    denominator as `exp(sink - mx)` but never competes for the max, and
//    `sparse_attn_kernel` rounds `p` to bf16 AFTER subtracting the FINAL max
//    (`acc_s_cast`). An online softmax that rescaled as it went would round `p`
//    against a running max instead and compute a different number -- which is
//    why this is two dispatches over 136 KB rather than one, exactly as
//    sparse_attn.slang is for the M = 1 path.
// 3. **The running max starts at a finite -1e30, not -inf.** A topk index of -1
//    means "this slot holds nothing" -- `get_window_topk_idxs` marks a ring slot
//    the sequence has not reached yet -- and scores -3e38 here; clamping the row
//    max at -1e30 is what makes an all-empty row come out zero instead of NaN,
//    and the reference clamps in the same place.
//
// Stages
// ------
//   Stage 0  Score    every (m, h, t) score into an [M][H][score_stride] plane
//   Stage 1  Combine  the max, p = bf16(exp(s - mx)), P.V, the divide
//
// **The inverse RoPE is NOT here.** `apply_rotary_emb(o[..., -64:], freqs_cis,
// True)` runs as dspark_gemv.slang's stage 2 with `inverse = 1` and `quant = 0`
// over the `[M][64][512]` output. sparse_attn.slang fuses it into its combine
// stage; the draft path does not, because dspark_gemv already owns the family's
// single RoPE implementation and one more 320 KB dispatch is not what §7.12's
// 11 ms budget is spent on.
//
// Dispatch geometry
// -----------------
// One workgroup per head, all M positions, all t: 64 workgroups a stage.
// `HeadsPerWg` is deliberately NOT read. For the M = 1 decode path grouping
// heads is a way to stop the 320 KiB of KV crossing L2 64 times, and
// gpu/vulkan/attn_kernels.h records that it measured a LOSS anyway (72 us at 1
// head a workgroup against 288 at 8) because 64 heads at 8 a workgroup is eight
// workgroups on forty CUs. The draft attention has a third of the KV and the
// same 64 heads, so the same argument applies with less to gain, and one head a
// workgroup keeps the kernel simple enough to read.
//
// Where the M dimension pays and where it does not
// ------------------------------------------------
// Stage 0 gets the reuse for free: a wave loads one KV row's span into registers
// and dots it against all `mlive` staged queries, so the KV is read ONCE per
// head however many draft positions there are. Stage 1 does not: its P.V pass
// walks the whole KV per draft position, because holding `mlive` accumulators of
// sixteen dims each would be 96 VGPRs of partials on top of the KV span. At
// 136 KB, L2-resident, read five times per head, that is a deliberate and
// measured-later trade; the M-fold reuse that actually matters in a draft cycle
// is the 41.9 MB weight reads in dspark_gemv.slang, not this.

#include "dspark_common.slang"

static const uint kQ = 0, kKv = 1, kTopIdx = 2, kSink = 3, kScore = 4, kO = 5;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct DsparkAttnPush {
    uint  m;              // live draft positions; the loops run min(M, m)
    uint  n_kv;           // entries in topk_idxs: 128 ring + 5 in-block = 133
    uint  n_heads;        // 64
    uint  head_dim;       // 512
    uint  score_stride;   // floats between score rows, >= n_kv
    uint  q_stride;       // elements between draft columns of q and o = 64 * 512
    float softmax_scale;  // head_dim ** -0.5
};
[[vk::push_constant]] DsparkAttnPush pc;

// 32 lanes to a KV row, so a lane owns head_dim / 32 = 16 dims. Stage 1's output
// spans are the same width.
static const uint kMaxSpan = 16;
// Ring 128 + the five in-block rows, rounded up with room for a verify batch.
static const uint kMaxKv   = 256;

groupshared int gIdx[kMaxKv];      // topk_idxs, read once instead of per t

// One pool, two uses, because LDS is the union over a file's pipelines: stage 0
// stages q for every live draft position (`kMaxDraftM * head_dim` floats) and
// stage 1 the per-thread P.V partials (256 threads x span). Both are padded or
// transposed against bank conflicts, for the reason fp8_gemv.slang pads gXQ:
//
//   * the staged q is [dim-within-span][lane], so lane `l` reads index
//     `m * head_dim + i * 32 + l` -- bank `l`, one lane per bank -- instead of
//     the natural `l * 16 + i`, whose bank takes two values across 32 lanes and
//     costs a 16-way conflict on every read;
//   * a thread's P.V partial has stride span + 1 = 17, which is odd and
//     therefore a bijection on the banks.
static const uint kPvStride = kMaxSpan + 1;                     // 17, odd on purpose
// max(kMaxDraftM * 512, kWg * kPvStride) = max(3072, 4352)
groupshared float gPool[kWg * kPvStride];

// Where element `d` of draft column `m` lives, given `span` dims to a lane.
uint q_slot(uint m, uint head_dim, uint d, uint span) {
    return m * head_dim + (d % span) * 32 + (d / span);
}

// `span` consecutive KV elements of row `idx`, dequantised. The ring is plain
// bf16 holding fp8-rounded values -- `act_quant(..., inplace=True)` on a bf16
// tensor, see dspark_gemv.slang stage 2 -- so there is no E4M3 decode and no
// scale plane here, unlike sparse_attn.slang's window half.
//
// `d0` is a multiple of `span` and `span` is even, so this is `span / 2` dword
// loads, which the compiler merges into dwordx4s.
void kv_span(uint idx, uint d0, uint span, out float v[kMaxSpan]) {
    [unroll] for (uint z = 0; z < kMaxSpan; ++z) v[z] = 0.0f;
    const uint64_t base = Ptr[kKv] + ((uint64_t)idx * pc.head_dim + d0) * 2;
    [unroll] for (uint w = 0; w < kMaxSpan / 2; ++w) {
        if (w * 2 >= span) continue;
        const uint b = load_u32(base + (uint64_t)w * 4);
        v[w * 2]     = bf16_decode(b & 0xFFFFu);
        v[w * 2 + 1] = bf16_decode(b >> 16);
    }
}

void load_idx(uint tid) {
    for (uint t = tid; t < pc.n_kv; t += kWg)
        gIdx[t] = (int)load_u32(Ptr[kTopIdx] + (uint64_t)t * 4);
}

// --- stage 0: the scores ----------------------------------------------------
// 256 threads = eight waves; wave `w` walks positions t = w, w + 8, ... and its
// 32 lanes split the 512-dim row sixteen dims apiece. A KV row is therefore
// exactly one wave's worth of lanes, so the dot product is one `WaveActiveSum`
// -- and it MUST be, because the t loop is not uniform across the workgroup and
// an LDS tree here would deadlock.
void stage_score(uint gid, uint tid) {
    const uint head = gid;
    const uint ml   = mlive(pc.m);
    const uint span = pc.head_dim / 32;              // 16

    for (uint i = tid; i < ml * pc.head_dim; i += kWg) {
        const uint m = i / pc.head_dim, d = i % pc.head_dim;
        gPool[q_slot(m, pc.head_dim, d, span)] =
            bf16_decode(load_u16(Ptr[kQ] +
                        (act_index(m, pc.q_stride, head * pc.head_dim + d)) * 2));
    }
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
        // The KV row is in registers; every draft position reuses it.
        for (uint m = 0; m < ml; ++m) {
            float p = 0.0f;
            if (idx >= 0) {
                [unroll] for (uint i = 0; i < kMaxSpan; ++i) {
                    if (i >= span) continue;
                    p += gPool[m * pc.head_dim + i * 32 + lane] * v[i];
                }
            }
            const float s = WaveActiveSum(p);
            if (lane == 0)
                *(float*)(Ptr[kScore] +
                          (((uint64_t)m * pc.n_heads + head) * pc.score_stride + t) * 4) =
                    (idx >= 0) ? (s * pc.softmax_scale) : -3.0e38f;
        }
    }
}

// --- stage 1: softmax, P.V, the divide --------------------------------------
// All 256 threads on one head, one draft position at a time. The probabilities
// go back into the score plane, which is what makes the P.V pass a single scalar
// load per position; the eight 32-lane groups then walk disjoint subsets of t
// and their partials meet in gPool.
//
// The whole m loop is uniform -- `ml` is a push constant -- so every barrier in
// it, including the ones inside `wg_max` / `wg_reduce`, is reached by all 256
// threads the same number of times.
void stage_combine(uint gid, uint tid) {
    const uint head = gid;
    const uint ml   = mlive(pc.m);
    const uint span = pc.head_dim / 32;              // 16
    const uint ngrp = kWg / 32;                      // 8 disjoint t subsets
    const uint grp  = tid / 32;
    const uint dl   = tid % 32;
    const uint d0   = dl * span;

    load_idx(tid);
    GroupMemoryBarrierWithGroupSync();

    const float sink = load_f32(Ptr[kSink] + (uint64_t)head * 4);

    for (uint m = 0; m < ml; ++m) {
        const uint64_t sc = Ptr[kScore] +
                            ((uint64_t)m * pc.n_heads + head) * pc.score_stride * 4;

        float local = -3.0e38f;
        for (uint t = tid; t < pc.n_kv; t += kWg)
            local = max(local, load_f32(sc + (uint64_t)t * 4));
        // Clamped at a finite -1e30 so an all-empty row comes out zero, not NaN.
        const float mx = max(wg_max(local, tid), -1.0e30f);

        // p, rounded to bf16 exactly where `acc_s_cast` does it, written back
        // over the scores so the P.V pass reads them from the same plane.
        float lsum = 0.0f;
        for (uint t = tid; t < pc.n_kv; t += kWg) {
            float p = 0.0f;
            if (gIdx[t] >= 0)
                p = bf16_decode(bf16_encode(exp(load_f32(sc + (uint64_t)t * 4) - mx)));
            *(float*)(sc + (uint64_t)t * 4) = p;
            lsum += p;
        }
        // The sink logit enters the denominator but never competed for the max.
        const float denom = wg_reduce(lsum, tid) + exp(sink - mx);
        // The p values just written went to DEVICE memory and the P.V loop below
        // reads all of them, not just this thread's: the barrier has to cover
        // the buffer, not only LDS.
        AllMemoryBarrierWithGroupSync();

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
        [unroll] for (uint i = 0; i < kMaxSpan; ++i) {
            if (i >= span) continue;
            gPool[tid * kPvStride + i] = acc[i];
        }
        GroupMemoryBarrierWithGroupSync();
        if (grp == 0) {
            for (uint gg = 1; gg < ngrp; ++gg) {
                [unroll] for (uint i = 0; i < kMaxSpan; ++i) {
                    if (i >= span) continue;
                    acc[i] += gPool[(tid + gg * 32) * kPvStride + i];
                }
            }
            const float inv = 1.0f / denom;
            const uint64_t o = Ptr[kO] +
                               act_index(m, pc.q_stride, head * pc.head_dim + d0) * 2;
            [unroll] for (uint i = 0; i < kMaxSpan; ++i) {
                if (i >= span) continue;
                *(uint16_t*)(o + (uint64_t)i * 2) = (uint16_t)bf16_encode(acc[i] * inv);
            }
        }
        // gPool is reused by the next draft position, and by the next m's own
        // wg_max before that, so the readers above must be done first.
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0) stage_score(gid.x, tid);
    else            stage_combine(gid.x, tid);
}
