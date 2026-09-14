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

#include "attn_common.slang"

static const uint kW = 0, kX = 1, kLogits = 2;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct HeadPush {
    uint rows;      // vocab_size = 129280
    uint k;         // dim = 5120
    uint row_base;  // for splitting the vocab across several dispatches
};
[[vk::push_constant]] HeadPush pc;

groupshared float gXf[5120];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
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
        for (uint r = 0; r < RowsPerLane; ++r) {
            if (rowBase + r >= pc.rows) continue;
            const uint4 v = load16(Ptr[kW] + ((uint64_t)(rowBase + r) * pc.k + base) * 2);
            [unroll] for (uint j = 0; j < 4; ++j) {
                acc[r] += bf16_decode(v[j] & 0xFFFFu) * x[j * 2];
                acc[r] += bf16_decode(v[j] >> 16)     * x[j * 2 + 1];
            }
        }
    }
    for (uint r = 0; r < RowsPerLane; ++r) {
        const float dot = row_reduce(acc[r], tid, sublane);
        if (sublane == 0 && rowBase + r < pc.rows)
            *(float*)(Ptr[kLogits] + (uint64_t)(rowBase + r) * 4) = dot;
    }
}
