// mgt1_mhc -- mega_mhc.slang for a verify batch of M tokens (Track T,
// docs/p4_mgt1.md): hc_post, hc_mixes, Sinkhorn, hc_pre and the RMSNorm, per
// token, with the token as the workgroup's second dispatch coordinate.
//
// The arithmetic is mega_mhc.slang's, line for line -- every correction its
// header records (comb contracted over its FIRST index, Sinkhorn ending on a
// column normalise, the bf16 rounding of hc_pre and of the norm) applies
// unchanged. What M adds is only addressing: every buffer is per token, laid out
// [M][...] with the stride a push constant, and a workgroup's `gid.y` is the
// token. The three stages keep their dispatch geometry in x:
//
//   Stage 0  (n_wg0 x M) hc_post, hc_pre, the two partial sums of squares
//   Stage 1  (mix_rows x M) the 24 hc_fn dot products
//   Stage 2  (n_wg0 x M) the RMSNorm of u, Sinkhorn on token m's workgroup 0
//
// Buffers, per token m: X / Xout [hc][dim] at m * hc * dim; A, Utmp, U [dim] at
// m * dim; the mix triple [pre 4][post 4][comb 16] and MixRaw [24] at m * 32
// floats; the partial plane [512] at m * 512.

#include "attn_common.slang"

static const uint kX = 0, kA = 1, kPostIn = 2, kCombIn = 3, kPreMix = 4;
static const uint kHcFn = 5, kHcBase = 6, kHcScale = 7, kNormW = 8;
static const uint kXout = 9, kUtmp = 10, kScratch = 11, kMixRaw = 12;
static const uint kMixOut = 13, kU = 14;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct MhcPush {
    uint  dim;
    uint  hc;
    uint  mix_rows;
    uint  n_wg0;
    uint  sinkhorn_iters;
    uint  flags;         // bit 0 hc_post, bit 1 skip Sinkhorn (the final collapse)
    float norm_eps;
    float hc_eps;
};
[[vk::push_constant]] MhcPush pc;

static const uint kMaxHc  = 4;
static const uint kMixStride = 32;      // floats per token in the mix buffers
static const uint kScrStride = 512;     // floats per token in the partial plane

uint64_t tok_vec(uint slot, uint m) { return Ptr[slot] + (uint64_t)m * pc.dim * 4; }
uint64_t tok_mix(uint slot, uint m) { return Ptr[slot] + (uint64_t)m * kMixStride * 4; }

void stage_post(uint gid, uint m, uint tid) {
    const uint d = gid * kWg + tid;
    const bool live = d < pc.dim;
    const uint64_t X = Ptr[kX] + (uint64_t)m * kMaxHc * pc.dim * 4;
    const uint64_t Xout = Ptr[kXout] + (uint64_t)m * kMaxHc * pc.dim * 4;
    const uint64_t mix = Ptr[kPreMix] + (uint64_t)m * kMixStride * 4;   // pre 0..3, post 4..7, comb 8..

    float xp[kMaxHc];
    float sq = 0.0f, u = 0.0f;
    if (live) {
        const bool post_on = (pc.flags & 1u) != 0u;
        const float a = post_on ? load_f32(tok_vec(kA, m) + (uint64_t)d * 4) : 0.0f;
        float xin[kMaxHc];
        [unroll] for (uint j = 0; j < kMaxHc; ++j)
            xin[j] = load_f32(X + ((uint64_t)j * pc.dim + d) * 4);
        [unroll] for (uint j = 0; j < kMaxHc; ++j) {
            float v = xin[j];
            if (post_on) {
                v = load_f32(mix + (uint64_t)(4 + j) * 4) * a;
                [unroll] for (uint i = 0; i < kMaxHc; ++i)
                    v += load_f32(mix + (uint64_t)(8 + i * kMaxHc + j) * 4) * xin[i];
            }
            xp[j] = v;
            sq += v * v;
            u += load_f32(mix + (uint64_t)j * 4) * v;
        }
        u = bf16_round(u);
        [unroll] for (uint j = 0; j < kMaxHc; ++j)
            *(float*)(Xout + ((uint64_t)j * pc.dim + d) * 4) = xp[j];
        *(float*)(tok_vec(kUtmp, m) + (uint64_t)d * 4) = u;
    }
    const float tot_sq = wg_reduce(sq, tid);
    const float tot_u  = wg_reduce(u * u, tid);
    if (tid == 0) {
        const uint64_t scr = Ptr[kScratch] + (uint64_t)m * kScrStride * 4;
        *(float*)(scr + (uint64_t)gid * 4) = tot_sq;
        *(float*)(scr + (uint64_t)(gid + 256) * 4) = tot_u;
    }
}

float scratch_sum(uint m, uint base, uint tid) {
    const uint64_t scr = Ptr[kScratch] + (uint64_t)m * kScrStride * 4;
    const float v = (tid < pc.n_wg0) ? load_f32(scr + (uint64_t)(base + tid) * 4) : 0.0f;
    return wg_reduce(v, tid);
}

void stage_mix(uint row, uint m, uint tid) {
    const float rs = rsqrt(scratch_sum(m, 0, tid) / float(pc.dim * pc.hc) + pc.norm_eps);
    const uint  n  = pc.dim * pc.hc;
    const uint64_t fn = Ptr[kHcFn] + (uint64_t)row * n * 4;
    const uint64_t xf = Ptr[kXout] + (uint64_t)m * n * 4;
    float acc = 0.0f;
    for (uint base = tid * 4; base < n; base += kWg * 4) {
        const uint4 a = load16(fn + (uint64_t)base * 4);
        const uint4 b = load16(xf + (uint64_t)base * 4);
        [unroll] for (uint j = 0; j < 4; ++j) acc += asfloat(a[j]) * asfloat(b[j]);
    }
    const float dot = wg_reduce(acc, tid);
    if (tid == 0) *(float*)(tok_mix(kMixRaw, m) + (uint64_t)row * 4) = dot * rs;
}

void sinkhorn(uint m, uint tid) {
    if (tid != 0) return;
    const uint64_t mr = tok_mix(kMixRaw, m), bs = Ptr[kHcBase], sc = Ptr[kHcScale];
    const uint64_t out = tok_mix(kMixOut, m);
    const float s0 = load_f32(sc), s1 = load_f32(sc + 4), s2 = load_f32(sc + 8);
    const float eps = pc.hc_eps;
    [unroll] for (uint j = 0; j < kMaxHc; ++j) {
        const float mm = load_f32(mr + (uint64_t)j * 4) * s0 + load_f32(bs + (uint64_t)j * 4);
        *(float*)(out + (uint64_t)j * 4) = 1.0f / (1.0f + exp(-mm)) + eps;
        const uint k = kMaxHc + j;
        const float p = load_f32(mr + (uint64_t)k * 4) * s1 + load_f32(bs + (uint64_t)k * 4);
        *(float*)(out + (uint64_t)k * 4) = 2.0f / (1.0f + exp(-p));
    }
    float c[kMaxHc][kMaxHc];
    [unroll] for (uint j = 0; j < kMaxHc; ++j) {
        float mx = -3.0e38f;
        [unroll] for (uint k = 0; k < kMaxHc; ++k) {
            const uint i = 2 * kMaxHc + j * kMaxHc + k;
            c[j][k] = load_f32(mr + (uint64_t)i * 4) * s2 + load_f32(bs + (uint64_t)i * 4);
            mx = max(mx, c[j][k]);
        }
        float sum = 0.0f;
        [unroll] for (uint k = 0; k < kMaxHc; ++k) { c[j][k] = exp(c[j][k] - mx); sum += c[j][k]; }
        [unroll] for (uint k = 0; k < kMaxHc; ++k) c[j][k] = c[j][k] / sum + eps;
    }
    [unroll] for (uint k = 0; k < kMaxHc; ++k) {
        float s = eps;
        [unroll] for (uint j = 0; j < kMaxHc; ++j) s += c[j][k];
        [unroll] for (uint j = 0; j < kMaxHc; ++j) c[j][k] /= s;
    }
    for (uint it = 1; it < pc.sinkhorn_iters; ++it) {
        [unroll] for (uint j = 0; j < kMaxHc; ++j) {
            float s = eps;
            [unroll] for (uint k = 0; k < kMaxHc; ++k) s += c[j][k];
            [unroll] for (uint k = 0; k < kMaxHc; ++k) c[j][k] /= s;
        }
        [unroll] for (uint k = 0; k < kMaxHc; ++k) {
            float s = eps;
            [unroll] for (uint j = 0; j < kMaxHc; ++j) s += c[j][k];
            [unroll] for (uint j = 0; j < kMaxHc; ++j) c[j][k] /= s;
        }
    }
    [unroll] for (uint j = 0; j < kMaxHc; ++j)
        [unroll] for (uint k = 0; k < kMaxHc; ++k)
            *(float*)(out + (uint64_t)(2 * kMaxHc + j * kMaxHc + k) * 4) = c[j][k];
}

void stage_final(uint gid, uint m, uint tid) {
    const float rs = rsqrt(scratch_sum(m, 256, tid) / float(pc.dim) + pc.norm_eps);
    const uint d = gid * kWg + tid;
    if (d < pc.dim) {
        const float w = bf16_decode(load_u16(Ptr[kNormW] + (uint64_t)d * 2));
        *(float*)(tok_vec(kU, m) + (uint64_t)d * 4) =
            bf16_round(w * load_f32(tok_vec(kUtmp, m) + (uint64_t)d * 4) * rs);
    }
    if (gid == 0 && (pc.flags & 2u) == 0u) sinkhorn(m, tid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0)      stage_post(gid.x, gid.y, tid);
    else if (Stage == 1) stage_mix(gid.x, gid.y, tid);
    else                 stage_final(gid.x, gid.y, tid);
}
