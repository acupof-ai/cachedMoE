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
// 129280 rows over 256 / LanesPerRow rows per workgroup is 16,160 workgroups,
// which balances across 40 CUs without any tail handling.

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
    const uint row = pc.row_base + gid.x * (kWg / LanesPerRow) + tid / LanesPerRow;
    const bool live = row < pc.rows;

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
    if (live && sublane == 0) *(float*)(Ptr[kLogits] + (uint64_t)row * 4) = dot;
}
