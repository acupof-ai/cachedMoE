// mgt1_cmp -- compressor.slang for a verify batch of M tokens (Track T,
// docs/p4_mgt1.md), with compression groups completing INSIDE the batch.
//
// The reference's M > 1 generalisation (tools/oracle_dspark.py
// `compressor_forward`) walks the batch in order: token j writes slot
// (p0 + j) % ratio of kv_state / score_state, and whenever (p0 + j + 1) % ratio
// == 0 the group pools. At ratio 2 and M = 6 that is up to three pooled
// latents a batch. The same thing, without a sequential walk:
//
//   * completion c is token m_c = first + c * ratio, first = (ratio - 1 -
//     p0 % ratio) % ratio; its slot s member is token m_c - (ratio - 1 - s),
//     or -- when that is before the batch -- the carried state's slot s, which
//     no token of THIS batch has written yet when the group pools, because the
//     only in-batch writer of slot s before m_c is that member itself;
//   * the carried state is written AFTER every pooling, once per slot, from the
//     last token of the batch that holds that slot.
//
// Stages:
//   0  bf16 [512 x 5120] GEMV of `wkv` (or `wgate`), (64 x M) workgroups, one
//      token a workgroup (compressor.slang stage 0 with gid.y = m).
//   1  pooling + `kv.to(bf16)` + RMSNorm, one workgroup a completion
//      (gid.y = c). Ratio 1: every token completes and nothing pools.
//   2  the carried-state write, one workgroup (ratio > 1 only).
//   3  RoPE at the latent's own position, FP4 block-16 / E4M3, the
//      compressed-KV row (p0 + m_c) / ratio, one workgroup a completion.
//
// Buffers: X [M][k]; Y, G [M][rows] fp32; Latent [C][rows] fp32 (pre-RoPE, what
// the indexer's key needs); RopeLat [C][rope_dim/2][2]; Val the bf16 cache;
// Fp4 [C][rows/2]; ScaleB [C][rows/16]; LatentQ [C][rows].

#include "attn_common.slang"

static const uint kW = 0, kX = 1, kY = 2, kG = 3;
static const uint kKvState = 4, kScoreState = 5, kNormW = 6, kLatent = 7;
static const uint kRope = 8, kVal = 9, kFp4 = 10, kScaleB = 11, kLatentQ = 12;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct Mgt1CmpPush {
    uint  rows;        // 512
    uint  k;           // 5120
    uint  ratio;       // 1 or 2
    uint  p0;          // position of token 0
    uint  rope_dim;    // 64
    float norm_eps;
};
[[vk::push_constant]] Mgt1CmpPush pc;

static const uint kMaxDim = 5120;
static const uint kMaxRatio = 8;
static const uint kMaxPer = 4;
groupshared float gXf[kMaxDim];

uint first_completion() { return (pc.ratio - 1u - pc.p0 % pc.ratio) % pc.ratio; }

// --- stage 0 --------------------------------------------------------------------
void stage_gemv(uint gid, uint m, uint tid) {
    const uint64_t X = Ptr[kX] + (uint64_t)m * pc.k * 4;
    for (uint i = tid; i < pc.k; i += kWg) gXf[i] = load_f32(X + (uint64_t)i * 4);
    GroupMemoryBarrierWithGroupSync();

    const uint sublane = tid % LanesPerRow;
    const uint row = gid * (kWg / LanesPerRow) + tid / LanesPerRow;
    float acc = 0.0f;
    if (row < pc.rows) {
        const uint64_t w = Ptr[kW] + (uint64_t)row * pc.k * 2;
        for (uint base = sublane * 8; base < pc.k; base += LanesPerRow * 8) {
            const uint4 v = load16(w + (uint64_t)base * 2);
            [unroll] for (uint j = 0; j < 4; ++j) {
                acc += bf16_decode(v[j] & 0xFFFFu) * gXf[base + j * 2];
                acc += bf16_decode(v[j] >> 16)     * gXf[base + j * 2 + 1];
            }
        }
    }
    const float dot = row_reduce(acc, tid, sublane);
    if (sublane == 0 && row < pc.rows)
        *(float*)(Ptr[kY] + ((uint64_t)m * pc.rows + row) * 4) = dot;
}

// --- stage 1 --------------------------------------------------------------------
void stage_pool(uint c, uint tid) {
    const uint mc  = first_completion() + c * pc.ratio;
    const uint per = (pc.rows + kWg - 1) / kWg;
    float pre[kMaxPer];
    float sq = 0.0f;
    [unroll] for (uint i = 0; i < kMaxPer; ++i) {
        const uint d = tid * per + i;
        pre[i] = 0.0f;
        if (i >= per || d >= pc.rows) continue;
        float v;
        if (pc.ratio <= 1) {
            v = bf16_round(load_f32(Ptr[kY] + ((uint64_t)mc * pc.rows + d) * 4));
        } else {
            float s[kMaxRatio];
            float kv[kMaxRatio];
            float mx = -3.0e38f;
            [unroll] for (uint j = 0; j < kMaxRatio; ++j) {
                s[j] = 0.0f; kv[j] = 0.0f;
                if (j >= pc.ratio) continue;
                const uint back = pc.ratio - 1u - j;           // slot j's member
                if (mc >= back) {
                    const uint t = mc - back;
                    s[j]  = load_f32(Ptr[kG] + ((uint64_t)t * pc.rows + d) * 4);
                    kv[j] = load_f32(Ptr[kY] + ((uint64_t)t * pc.rows + d) * 4);
                } else {
                    s[j]  = load_f32(Ptr[kScoreState] + ((uint64_t)j * pc.rows + d) * 4);
                    kv[j] = load_f32(Ptr[kKvState] + ((uint64_t)j * pc.rows + d) * 4);
                }
                mx = max(mx, s[j]);
            }
            float den = 0.0f, num = 0.0f;
            [unroll] for (uint j = 0; j < kMaxRatio; ++j) {
                if (j >= pc.ratio) continue;
                const float e = exp(s[j] - mx);
                den += e;
                num += e * kv[j];
            }
            v = bf16_round(num / den);
        }
        pre[i] = v;
        sq += v * v;
    }
    const float rs = rsqrt(wg_reduce(sq, tid) / float(pc.rows) + pc.norm_eps);
    [unroll] for (uint i = 0; i < kMaxPer; ++i) {
        const uint d = tid * per + i;
        if (i >= per || d >= pc.rows) continue;
        *(float*)(Ptr[kLatent] + ((uint64_t)c * pc.rows + d) * 4) =
            bf16_round(pre[i] * rs * bf16_decode(load_u16(Ptr[kNormW] + (uint64_t)d * 2)));
    }
}

// --- stage 2 --------------------------------------------------------------------
void stage_state(uint tid) {
    const uint per = (pc.rows + kWg - 1) / kWg;
    [unroll] for (uint j = 0; j < kMaxRatio; ++j) {
        if (j >= pc.ratio) continue;
        // The last token of the batch sitting in slot j, if any.
        const uint off = (j + pc.ratio - pc.p0 % pc.ratio) % pc.ratio;   // first token with slot j
        if (off >= M) continue;
        const uint t = off + ((M - 1u - off) / pc.ratio) * pc.ratio;
        [unroll] for (uint i = 0; i < kMaxPer; ++i) {
            const uint d = tid * per + i;
            if (i >= per || d >= pc.rows) continue;
            *(float*)(Ptr[kKvState] + ((uint64_t)j * pc.rows + d) * 4) =
                load_f32(Ptr[kY] + ((uint64_t)t * pc.rows + d) * 4);
            *(float*)(Ptr[kScoreState] + ((uint64_t)j * pc.rows + d) * 4) =
                load_f32(Ptr[kG] + ((uint64_t)t * pc.rows + d) * 4);
        }
    }
}

// --- stage 3 --------------------------------------------------------------------
static const uint kCmpBlock = 16;
static const float kFp4Max  = 6.0f;
static const float kFp4Floor = 6.0f * 0.001953125f;

void stage_store(uint c, uint tid) {
    const uint blocks = pc.rows / kCmpBlock;
    if (tid >= blocks) return;
    const uint mc = first_completion() + c * pc.ratio;
    const uint cmp_row = (pc.p0 + mc) / pc.ratio;
    const uint d0 = tid * kCmpBlock;
    const uint rope_lo = pc.rows - pc.rope_dim;

    float v[kCmpBlock];
    [unroll] for (uint i = 0; i < kCmpBlock; ++i)
        v[i] = load_f32(Ptr[kLatent] + ((uint64_t)c * pc.rows + d0 + i) * 4);
    [unroll] for (uint i = 0; i < kCmpBlock; i += 2) {
        const uint d = d0 + i;
        if (d < rope_lo) continue;
        const uint j = (d - rope_lo) >> 1;
        const uint64_t tab = Ptr[kRope] + ((uint64_t)c * (pc.rope_dim / 2) + j) * 8;
        rope_pair(load_f32(tab), load_f32(tab + 4), v[i], v[i + 1], false);
        v[i] = bf16_round(v[i]);
        v[i + 1] = bf16_round(v[i + 1]);
    }
    float amax = 0.0f;
    [unroll] for (uint i = 0; i < kCmpBlock; ++i) amax = max(amax, abs(v[i]));
    const float sc  = fp8_decode(fp8_encode_rn(clamp(max(amax, kFp4Floor) / kFp4Max, 0.0f, kFp8MaxF)));
    const float inv = 1.0f / sc;
    uint packed[kCmpBlock / 2];
    [unroll] for (uint z = 0; z < kCmpBlock / 2; ++z) packed[z] = 0u;
    [unroll] for (uint i = 0; i < kCmpBlock; ++i) {
        const float q = fp4_round(clamp(v[i] * inv, -kFp4Max, kFp4Max));
        const uint  code = fp4_code(q);
        if ((i & 1) == 0) packed[i >> 1] = code;
        else              packed[i >> 1] |= code << 4;
        const float out = q * sc;
        *(uint16_t*)(Ptr[kVal] + ((uint64_t)cmp_row * pc.rows + d0 + i) * 2) =
            (uint16_t)bf16_encode(out);
        *(float*)(Ptr[kLatentQ] + ((uint64_t)c * pc.rows + d0 + i) * 4) = out;
    }
    [unroll] for (uint i = 0; i < kCmpBlock / 2; ++i)
        *(uint8_t*)(Ptr[kFp4] + ((uint64_t)c * (pc.rows / 2) + d0 / 2 + i)) = (uint8_t)packed[i];
    *(uint8_t*)(Ptr[kScaleB] + ((uint64_t)c * blocks + tid)) = (uint8_t)fp8_encode_rn(sc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0)      stage_gemv(gid.x, gid.y, tid);
    else if (Stage == 1) stage_pool(gid.y, tid);
    else if (Stage == 2) stage_state(tid);
    else                 stage_store(gid.y, tid);
}
