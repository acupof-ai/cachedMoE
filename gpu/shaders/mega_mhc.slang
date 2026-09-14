// mega_mhc -- fused hc_post + hc_mixes + Sinkhorn + hc_pre + RMSNorm
// (design §7.2 and §7.7; dispatches #1 and #8 of the §7.14 list).
//
// This is `Block.forward`'s scaffolding, everything between one sublayer's
// output and the next one's input:
//
//     x'      = post[j] * a + sum_i comb[i][j] * x[i]          hc_post
//     mixes   = (hc_fn @ flatten(x')) * rsqrt(mean(x'^2))      hc_mixes
//     pre, post', comb' = sinkhorn(mixes)
//     u       = sum_j pre_mix[j] * x'[j]                       hc_pre
//     out     = norm_w * u * rsqrt(mean(u^2))                  RMSNorm
//
// `pre_mix` is the coefficient set the *previous* sublayer produced, and the
// `pre` this one produces is for the next: design §2.4's handoff. `post` and
// `comb` in the hc_post above likewise come from the previous sublayer, which
// is why hc_post fuses here and not at the end of the sublayer that made `a`.
//
// Three corrections the reference forced on design §7.2
// ----------------------------------------------------
// 1. `hc_post` sums over comb's FIRST index: `sum(comb.unsqueeze(-1) *
//    residual.unsqueeze(-2), dim=2)` contracts comb[i][j] against residual[i],
//    so it is comb transposed against the obvious reading.
// 2. Sinkhorn ends on a COLUMN normalise and its result is not doubly
//    stochastic (design §2.4, route_trace.md §11.1). The loop below is
//    transcribed from `hc_split_sinkhorn_kernel` and must not gain a trailing
//    row normalise.
// 3. The 1.97 MB of hc_fn cannot go through one workgroup, and the two RMS
//    statistics are whole-vector reductions, so this is three dispatches of
//    one pipeline rather than the single dispatch design §7.2 assumed. At the
//    0.66 us per dispatch kernel_p1.md §3.4 measured that is 1.3 us a layer.
//
// Stage 0  dim/256 workgroups: hc_post, hc_pre, and the two partial sums of
//          squares. Writes x' and the un-normalised u.
// Stage 1  mix_rows workgroups, one per row of hc_fn: one 20480-wide dot
//          product against x', scaled by the first RMS statistic.
// Stage 2  one workgroup: Sinkhorn in registers, and the RMSNorm of u.

#include "attn_common.slang"

// Slots in the address table. The host points several of these into the same
// allocation -- MixOut is [pre 4][post 4][comb 16], which is exactly the
// PostIn / CombIn / PreMix the next sublayer's dispatch reads.
static const uint kX = 0, kA = 1, kPostIn = 2, kCombIn = 3, kPreMix = 4;
static const uint kHcFn = 5, kHcBase = 6, kHcScale = 7, kNormW = 8;
static const uint kXout = 9, kUtmp = 10, kScratch = 11, kMixRaw = 12;
static const uint kMixOut = 13, kU = 14;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct MhcPush {
    uint  dim;          // 5120
    uint  hc;           // 4
    uint  mix_rows;     // (2 + hc) * hc = 24
    uint  n_wg0;        // workgroups stage 0 used, = dim / 256
    uint  sinkhorn_iters;
    // bit 0: apply hc_post (0 = x' is x, which is the very first sublayer)
    // bit 1: skip Sinkhorn -- the final collapse before the head (§7.11) wants
    //        hc_pre + RMSNorm and nothing else, and its caller does not
    //        dispatch stage 1 at all, so MixRaw would be stale.
    uint  flags;
    float norm_eps;     // 1e-20
    float hc_eps;       // 1e-6
};
[[vk::push_constant]] MhcPush pc;

static const uint kMaxHc = 4;

void stage_post(uint gid, uint tid) {
    const uint d = gid * kWg + tid;
    const bool live = d < pc.dim;
    const uint64_t X = Ptr[kX], Xout = Ptr[kXout];

    float xp[kMaxHc];
    float sq = 0.0f, u = 0.0f;
    if (live) {
        const bool post_on = (pc.flags & 1u) != 0u;
        const float a = post_on ? load_f32(Ptr[kA] + (uint64_t)d * 4) : 0.0f;
        float xin[kMaxHc];
        [unroll] for (uint j = 0; j < kMaxHc; ++j)
            xin[j] = load_f32(X + ((uint64_t)j * pc.dim + d) * 4);
        [unroll] for (uint j = 0; j < kMaxHc; ++j) {
            float v = xin[j];
            if (post_on) {
                v = load_f32(Ptr[kPostIn] + (uint64_t)j * 4) * a;
                // comb[i][j]: the contraction is over comb's FIRST index
                [unroll] for (uint i = 0; i < kMaxHc; ++i)
                    v += load_f32(Ptr[kCombIn] + ((uint64_t)i * kMaxHc + j) * 4) * xin[i];
            }
            xp[j] = v;
            sq += v * v;
            u += load_f32(Ptr[kPreMix] + (uint64_t)j * 4) * v;
        }
        u = bf16_round(u);      // hc_pre returns `.to(x.dtype)`, i.e. bf16
        [unroll] for (uint j = 0; j < kMaxHc; ++j)
            *(float*)(Xout + ((uint64_t)j * pc.dim + d) * 4) = xp[j];
        *(float*)(Ptr[kUtmp] + (uint64_t)d * 4) = u;
    }
    const float tot_sq = wg_reduce(sq, tid);
    const float tot_u  = wg_reduce(u * u, tid);
    if (tid == 0) {
        *(float*)(Ptr[kScratch] + (uint64_t)gid * 4) = tot_sq;
        *(float*)(Ptr[kScratch] + (uint64_t)(gid + 256) * 4) = tot_u;
    }
}

// The two whole-vector statistics stage 0 left as per-workgroup partials.
float scratch_sum(uint base) {
    float s = 0.0f;
    for (uint i = 0; i < pc.n_wg0; ++i) s += load_f32(Ptr[kScratch] + (uint64_t)(base + i) * 4);
    return s;
}

void stage_mix(uint row, uint tid) {
    const float rs = rsqrt(scratch_sum(0) / float(pc.dim * pc.hc) + pc.norm_eps);
    const uint  n  = pc.dim * pc.hc;                       // 20480
    const uint64_t fn = Ptr[kHcFn] + (uint64_t)row * n * 4;
    const uint64_t xf = Ptr[kXout];
    float acc = 0.0f;
    // Four consecutive floats per thread per step: one 16 B load each side.
    for (uint base = tid * 4; base < n; base += kWg * 4) {
        const uint4 a = load16(fn + (uint64_t)base * 4);
        const uint4 b = load16(xf + (uint64_t)base * 4);
        [unroll] for (uint j = 0; j < 4; ++j) acc += asfloat(a[j]) * asfloat(b[j]);
    }
    const float dot = wg_reduce(acc, tid);
    if (tid == 0) *(float*)(Ptr[kMixRaw] + (uint64_t)row * 4) = dot * rs;
}

// `hc_split_sinkhorn_kernel`, transcribed. The loop ends on a column
// normalise; see the header.
void sinkhorn(uint tid) {
    if (tid != 0) return;
    const uint hc = pc.hc;
    const uint64_t mr = Ptr[kMixRaw], bs = Ptr[kHcBase], sc = Ptr[kHcScale], out = Ptr[kMixOut];
    const float s0 = load_f32(sc), s1 = load_f32(sc + 4), s2 = load_f32(sc + 8);
    const float eps = pc.hc_eps;

    for (uint j = 0; j < hc; ++j) {
        const float m = load_f32(mr + (uint64_t)j * 4) * s0 + load_f32(bs + (uint64_t)j * 4);
        *(float*)(out + (uint64_t)j * 4) = 1.0f / (1.0f + exp(-m)) + eps;
        const uint k = hc + j;
        const float p = load_f32(mr + (uint64_t)k * 4) * s1 + load_f32(bs + (uint64_t)k * 4);
        *(float*)(out + (uint64_t)k * 4) = 2.0f / (1.0f + exp(-p));
    }

    float c[kMaxHc][kMaxHc];
    for (uint j = 0; j < hc; ++j) {
        float mx = -3.0e38f;
        for (uint k = 0; k < hc; ++k) {
            const uint i = 2 * hc + j * hc + k;
            c[j][k] = load_f32(mr + (uint64_t)i * 4) * s2 + load_f32(bs + (uint64_t)i * 4);
            mx = max(mx, c[j][k]);
        }
        float sum = 0.0f;
        for (uint k = 0; k < hc; ++k) { c[j][k] = exp(c[j][k] - mx); sum += c[j][k]; }
        for (uint k = 0; k < hc; ++k) c[j][k] = c[j][k] / sum + eps;   // softmax(-1) + eps
    }
    for (uint k = 0; k < hc; ++k) {                                    // /= colsum + eps
        float s = 0.0f;
        for (uint j = 0; j < hc; ++j) s += c[j][k];
        s += eps;
        for (uint j = 0; j < hc; ++j) c[j][k] /= s;
    }
    for (uint it = 1; it < pc.sinkhorn_iters; ++it) {
        for (uint j = 0; j < hc; ++j) {                                // /= rowsum + eps
            float s = 0.0f;
            for (uint k = 0; k < hc; ++k) s += c[j][k];
            s += eps;
            for (uint k = 0; k < hc; ++k) c[j][k] /= s;
        }
        for (uint k = 0; k < hc; ++k) {                                // /= colsum + eps
            float s = 0.0f;
            for (uint j = 0; j < hc; ++j) s += c[j][k];
            s += eps;
            for (uint j = 0; j < hc; ++j) c[j][k] /= s;
        }
    }
    for (uint j = 0; j < hc; ++j)
        for (uint k = 0; k < hc; ++k)
            *(float*)(out + (uint64_t)(2 * hc + j * hc + k) * 4) = c[j][k];
}

void stage_final(uint tid) {
    const float rs = rsqrt(scratch_sum(256) / float(pc.dim) + pc.norm_eps);
    const uint64_t nw = Ptr[kNormW], ut = Ptr[kUtmp], u = Ptr[kU];
    for (uint d = tid; d < pc.dim; d += kWg) {
        const float w = bf16_decode(load_u16(nw + (uint64_t)d * 2));
        // bf16_round because RMSNorm.forward ends in `.to(dtype)`; this value
        // is act_quant'd by wq_a and wkv, where the extra fp32 bits only move
        // the block scale away from the reference's.
        *(float*)(u + (uint64_t)d * 4) =
            bf16_round(w * load_f32(ut + (uint64_t)d * 4) * rs);
    }
    if ((pc.flags & 2u) == 0u) sinkhorn(tid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0)      stage_post(gid.x, tid);
    else if (Stage == 1) stage_mix(gid.x, tid);
    else                 stage_final(tid);
}
