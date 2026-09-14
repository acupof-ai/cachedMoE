// head -- the LM head, bf16 [129280 x 5120], 1.32 GB (design §7.11).
//
// The single largest resident read of a decode token, about 10% of the §2.3
// budget, and therefore the place speculative decoding pays best: at M = 5 the
// same 1.32 GB yields logits for five positions. This is the M = 1 form.
//
// `ParallelHead.forward` is `F.linear(x.float(), weight)` with the weight held
// as fp32 (it is bf16 in the checkpoint and "kept as fp32 here so the logits
// come out in fp32 directly"), so there is no activation quantisation and no
// fp16 anywhere: bf16 weights promoted to fp32 against an fp32 activation.
//
// Its input is `norm(hc_pre(h, pre_mix))`, which mega_mhc already computes --
// stage 0 and stage 2 with the `kFlagSkipSinkhorn` bit and no stage 1, since
// the final collapse needs neither the mixes nor hc_post.
//
// 129280 rows over (256 / LanesPerRow) * RowsPerLane rows per workgroup is
// 4,040 workgroups at the default spec, which still balances across 40 CUs
// while staging the 20 KiB activation four times less often.

// Stage 1 is the greedy sampler (design §7.11's temperature-0 half): one
// workgroup reduces the 129,280 logits to the argmax and the runner-up, so the
// only thing that crosses PCIe is four words -- never the logit vector. The
// runner-up is there because design §12 L3 asks for the top1-top2 margin at any
// disagreement with the oracle: a divergence at a margin of 1e-6 is a different
// finding from one at 0.5. Ties go to the lower id, which is what
// `torch.argmax` does and therefore what the oracle's tokens assume.

#include "attn_common.slang"

static const uint kW = 0, kX = 1, kLogits = 2, kSample = 3;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct HeadPush {
    uint rows;      // vocab_size = 129280
    uint k;         // dim = 5120
    uint row_base;  // for splitting the vocab across several dispatches
};
[[vk::push_constant]] HeadPush pc;

groupshared float gXf[5120];
groupshared float gBest[kWg];
groupshared uint  gBestId[kWg];
groupshared float gSecond[kWg];

void stage_head(uint3 gid, uint tid) {
    for (uint i = tid; i < pc.k; i += kWg) gXf[i] = load_f32(Ptr[kX] + (uint64_t)i * 4);
    GroupMemoryBarrierWithGroupSync();

    const uint sublane = tid % LanesPerRow;
    const uint rowBase = pc.row_base +
                         gid.x * ((kWg / LanesPerRow) * RowsPerLane) +
                         (tid / LanesPerRow) * RowsPerLane;

    // RowsPerLane rows walked together: the activation is read from LDS once
    // for all of them and their weight loads are independent streams.
    float acc[kMaxRowsPerLane];
    [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) acc[r] = 0.0f;
    for (uint base = sublane * 8; base < pc.k; base += LanesPerRow * 8) {
        float x[8];
        [unroll] for (uint j = 0; j < 8; ++j) x[j] = gXf[base + j];
        [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) {
            if (r >= RowsPerLane || rowBase + r >= pc.rows) continue;
            const uint4 v = load16(Ptr[kW] + ((uint64_t)(rowBase + r) * pc.k + base) * 2);
            [unroll] for (uint j = 0; j < 4; ++j) {
                acc[r] += bf16_decode(v[j] & 0xFFFFu) * x[j * 2];
                acc[r] += bf16_decode(v[j] >> 16)     * x[j * 2 + 1];
            }
        }
    }
    // Unrolled against the compile-time bound, predicated on the
    // specialisation constant: see gemv8_rows in fp8_gemv.slang.
    [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) {
        if (r >= RowsPerLane) continue;
        const float dot = row_reduce(acc[r], tid, sublane);
        if (sublane == 0 && rowBase + r < pc.rows)
            *(float*)(Ptr[kLogits] + (uint64_t)(rowBase + r) * 4) = dot;
    }
}

// Merge two (best, id, second) candidates. `a` wins ties, so reducing with the
// lower id on the left keeps argmax first-wins.
void top2_merge(inout float ab, inout uint ai, inout float a2,
                float bb, uint bi, float b2) {
    if (ab >= bb) { a2 = max(a2, bb); }
    else          { a2 = max(ab, b2); ab = bb; ai = bi; }
}

void stage_argmax(uint tid) {
    float best = -3.0e38f, second = -3.0e38f;
    uint  bid = 0;
    const uint64_t L = Ptr[kLogits];
    // Strided so consecutive lanes read consecutive words. Within a thread a
    // later equal value never displaces an earlier one, and the tree below
    // breaks ties towards the lower thread, so the winner is the first maximum
    // in id order -- `torch.argmax`'s rule, which the oracle's tokens assume.
    for (uint i = tid; i < pc.rows; i += kWg) {
        const float v = load_f32(L + (uint64_t)i * 4);
        if (v > best)        { second = best; best = v; bid = i; }
        else if (v > second) { second = v; }
    }
    gBest[tid] = best; gBestId[tid] = bid; gSecond[tid] = second;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = kWg >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float ab = gBest[tid], a2 = gSecond[tid];
            uint  ai = gBestId[tid];
            top2_merge(ab, ai, a2, gBest[tid + s], gBestId[tid + s], gSecond[tid + s]);
            gBest[tid] = ab; gBestId[tid] = ai; gSecond[tid] = a2;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0) {
        uint* out = (uint*)Ptr[kSample];
        out[0] = gBestId[0];
        out[1] = asuint(gBest[0]);
        out[2] = asuint(gSecond[0]);
        out[3] = pc.rows;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0) stage_head(gid, tid);
    else            stage_argmax(tid);
}
