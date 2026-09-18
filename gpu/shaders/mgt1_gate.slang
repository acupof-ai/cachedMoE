// mgt1_gate -- gate.slang for a verify batch of M tokens (Track T,
// docs/p4_mgt1.md): `sqrt(softplus(x . W))`, then noaux_tc top-k, per token.
//
// The arithmetic is gate.slang's exactly (bf16 weight promoted to fp32 against
// an fp32 activation, the bias picks and the unbiased score weights, the 1e-20
// floor, route_scale). What M adds:
//
//   Stage 0  ((rows + 7) / 8 x M): token m's column staged in LDS, 384 scores.
//            A workgroup owns ONE token: the column is 20 KiB of LDS already,
//            and M columns in one workgroup would be 120 KiB.
//   Stage 1  (1 x M): the sixteen-round workgroup argmax, one workgroup a
//            token -- they are independent, so M of them run side by side.
//
// Buffers per token m: X [dim] at m * dim, Scores [n_experts] at m * n_experts,
// Ids and Weights [16] at m * 16. The progress counter is bumped once per
// dispatch, by token 0.

#include "attn_common.slang"

static const uint kW = 0, kBias = 1, kX = 2, kScores = 3;
static const uint kIds = 4, kWeights = 5, kLayerDone = 6;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct GatePush {
    uint  n_experts;
    uint  k;
    uint  topk;
    uint  record;
    float gate_temp;
    float route_scale;
};
[[vk::push_constant]] GatePush pc;

static const uint kMaxRecord  = 16;
static const uint kMaxExperts = 384;

groupshared float gXf[5120];

void stage_scores(uint gid, uint m, uint tid) {
    const uint64_t X = Ptr[kX] + (uint64_t)m * pc.k * 4;
    for (uint i = tid; i < pc.k; i += kWg) gXf[i] = load_f32(X + (uint64_t)i * 4);
    GroupMemoryBarrierWithGroupSync();

    const uint sublane = tid % LanesPerRow;
    const uint row = gid * (kWg / LanesPerRow) + tid / LanesPerRow;
    const bool live = row < pc.n_experts;
    float acc = 0.0f;
    if (live) {
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
    if (live && sublane == 0) {
        const float z = dot / pc.gate_temp;
        const float sp = softplus_stable(z);        // attn_common.slang
        *(float*)(Ptr[kScores] + ((uint64_t)m * pc.n_experts + row) * 4) = sqrt(sp);
    }
}

groupshared float gScore[kMaxExperts];
groupshared float gBiased[kMaxExperts];
groupshared float gArgV[kWg];
groupshared uint  gArgI[kWg];

void stage_topk(uint m, uint tid) {
    const uint64_t sc = Ptr[kScores] + (uint64_t)m * pc.n_experts * 4;
    const uint64_t bi = Ptr[kBias];
    const uint64_t ids = Ptr[kIds] + (uint64_t)m * kMaxRecord * 4;
    const uint64_t wts = Ptr[kWeights] + (uint64_t)m * kMaxRecord * 4;
    for (uint e = tid; e < pc.n_experts; e += kWg) {
        gScore[e]  = load_f32(sc + (uint64_t)e * 4);
        gBiased[e] = gScore[e] + load_f32(bi + (uint64_t)e * 4);
    }
    GroupMemoryBarrierWithGroupSync();

    const uint record = min(pc.record, kMaxRecord);
    float sum = 0.0f;
    float raw[kMaxRecord];
    [unroll] for (uint z = 0; z < kMaxRecord; ++z) raw[z] = 0.0f;
    [unroll] for (uint r = 0; r < kMaxRecord; ++r) {
        if (r >= record) break;
        float best = -3.0e38f;
        uint  arg = 0;
        for (uint e = tid; e < pc.n_experts; e += kWg)
            if (gBiased[e] > best) { best = gBiased[e]; arg = e; }
        gArgV[tid] = best;
        gArgI[tid] = arg;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = kWg >> 1; s > 0; s >>= 1) {
            if (tid < s && gArgV[tid + s] > gArgV[tid]) {
                gArgV[tid] = gArgV[tid + s];
                gArgI[tid] = gArgI[tid + s];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        const uint pick = gArgI[0];
        raw[r] = gScore[pick];
        if (r < pc.topk) sum += raw[r];
        if (tid == 0) {
            gBiased[pick] = -3.0e38f;
            *(uint*)(ids + (uint64_t)r * 4) = pick;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid != 0) return;
    const float inv = pc.route_scale / (sum + 1e-20f);
    [unroll] for (uint r = 0; r < kMaxRecord; ++r) {
        if (r >= record) break;
        *(float*)(wts + (uint64_t)r * 4) = (r < pc.topk) ? raw[r] * inv : raw[r];
    }
    if (m == 0) {
        const uint64_t done = Ptr[kLayerDone];
        *(uint*)done = load_u32(done) + 1u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0) stage_scores(gid.x, gid.y, tid);
    else            stage_topk(gid.y, tid);
}
