// gate -- the router: sqrt(softplus(x . W / temp)), then noaux_tc top-k
// (design §7.8, dispatch #9 of the §7.14 list). W is bf16 [384 x 5120],
// 3.9 MB.
//
// No `act_quant` here, and no fp16 anywhere: `Gate.forward` is
// `linear(x.float(), self.weight.float())`, i.e. a bf16 weight promoted to
// fp32 against an fp32 activation, and the score, the bias comparison and the
// normalisation are all fp32 with a 1e-20 floor that is deliberately NOT
// `norm_eps` (it "matches training"). cpu/gate.cpp is the CPU oracle for
// exactly this arithmetic.
//
// The semantics design §2.4 insists on: the bias picks the experts, the
// unbiased score weights them, the weights are normalised and multiplied by
// `route_scale`.
//
//   Stage 0  rows / (256 / LanesPerRow) workgroups: the 384 scores.
//   Stage 1  one workgroup: top-k of (score + bias) by repeated
//            WaveActiveMax-style masking, then the weights.
//
// Stage 1 also writes `top16_ids` / `top16_scores` for the planner's heat
// statistics and bumps a host-coherent counter the CPU spins on rather than
// waiting on a fence (design §7.8). The top-16 is OURS, not the reference's:
// `Gate.forward` only ever takes top-6, but all 384 scores exist anyway
// (route_trace.md §11 item 7).

#include "attn_common.slang"

static const uint kW = 0, kBias = 1, kX = 2, kScores = 3;
static const uint kIds = 4, kWeights = 5, kLayerDone = 6;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct GatePush {
    uint  n_experts;     // 384
    uint  k;             // 5120
    uint  topk;          // 6
    uint  record;        // how many ids/scores to record (>= topk), 16
    float gate_temp;     // 1.0
    float route_scale;   // 1.5
};
[[vk::push_constant]] GatePush pc;

static const uint kMaxRecord = 16;

groupshared float gXf[5120];      // the fp32 activation, staged once

void stage_scores(uint gid, uint tid) {
    for (uint i = tid; i < pc.k; i += kWg) gXf[i] = load_f32(Ptr[kX] + (uint64_t)i * 4);
    GroupMemoryBarrierWithGroupSync();

    const uint sublane = tid % LanesPerRow;
    const uint row = gid * (kWg / LanesPerRow) + tid / LanesPerRow;
    const bool live = row < pc.n_experts;

    float acc = 0.0f;
    if (live) {
        const uint64_t w = Ptr[kW] + (uint64_t)row * pc.k * 2;
        // Eight bf16 per 16 B load; the lane stride keeps a wave's loads
        // contiguous across the row.
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
        // sqrtsoftplus: softplus(z) = log1p(exp(z)), stable for large z.
        const float z = dot / pc.gate_temp;
        const float sp = (z > 20.0f) ? z : log(1.0f + exp(z));
        *(float*)(Ptr[kScores] + (uint64_t)row * 4) = sqrt(sp);
    }
}

void stage_topk(uint tid) {
    if (tid != 0) return;
    const uint64_t sc = Ptr[kScores], bi = Ptr[kBias];
    const uint64_t ids = Ptr[kIds], wts = Ptr[kWeights];

    uint  chosen[kMaxRecord];
    float raw[kMaxRecord];
    [unroll] for (uint z = 0; z < kMaxRecord; ++z) { chosen[z] = 0xFFFFFFFFu; raw[z] = 0.0f; }
    const uint record = min(pc.record, kMaxRecord);
    for (uint r = 0; r < record; ++r) {
        float best = -3.0e38f;
        uint  arg = 0;
        for (uint e = 0; e < pc.n_experts; ++e) {
            bool taken = false;
            for (uint p = 0; p < r; ++p) taken = taken || (chosen[p] == e);
            if (taken) continue;
            const float v = load_f32(sc + (uint64_t)e * 4) + load_f32(bi + (uint64_t)e * 4);
            if (v > best) { best = v; arg = e; }
        }
        chosen[r] = arg;
        raw[r] = load_f32(sc + (uint64_t)arg * 4);
        *(uint*)(ids + (uint64_t)r * 4) = arg;
    }

    float sum = 0.0f;
    for (uint r = 0; r < pc.topk; ++r) sum += raw[r];
    const float inv = pc.route_scale / (sum + 1e-20f);   // 1e-20 matches training
    for (uint r = 0; r < record; ++r)
        *(float*)(wts + (uint64_t)r * 4) = (r < pc.topk) ? raw[r] * inv : raw[r];

    // Host-coherent progress counter: the planner spins on this instead of
    // waiting on a fence, so the command buffer never pauses (design §7.8).
    const uint64_t done = Ptr[kLayerDone];
    *(uint*)done = load_u32(done) + 1u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0) stage_scores(gid.x, tid);
    else            stage_topk(tid);
}
