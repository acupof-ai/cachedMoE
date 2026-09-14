// moe_xquant -- x [M][5120] fp16 -> int8 with a per-block-32 scale, once per
// token (docs/kernel_p2_moe.md §3.6 / §8 item 2).
//
// design §7.9's int8 dot4 path turns 32 fp32 FMA + 32 f16tof32 per (column,
// block, weight row) into 8 dot4add_i8packed. P2 measured it as a loss on
// dispatch A (XMode = 5, 77 GB/s against 117) because every one of the
// 2304/rows_per_wg row groups quantises the same 32 activations again: the
// amax + round costs about as much as the FMAs it removes, and only the weight
// rows of one lane amortise it.
//
// Hoisting the quantisation here makes that cost 1/(2304/rows_per_wg) of what
// it was -- and once per *token*, not once per layer, because x for the MoE is
// the same 5120 values for all 40 layers. 160 threads at M=1 read 10 KB and
// write 5 KB + 640 B; the dispatch is launch-bound.
//
// The int8 plane and the scales live past the fp16 x inside the same
// allocation (moe_common.slang xq_value_words), so there is no new descriptor
// and dispatch A reads both through one extra read-only alias of the x buffer.
// Writes start at word M*k/2 and reads end there, so quantising in place is
// safe here in a way it is not for h (see moe_hquant.slang).
//
// This is an approximation of x and therefore never on by default: XMode = 6
// has to be asked for. docs/kernel_p2_moe.md §8 item 2 has the measured error.

#include "moe_common.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> XU;   // the x allocation, raw words

struct XQuantPush {
    uint k;   // 5120
};
[[vk::push_constant]] XQuantPush pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint blocks = pc.k / kBlockK;                 // 160
    const uint idx    = gid.x * kWgSize + tid;
    if (idx >= M * blocks) return;
    const uint m   = idx / blocks;
    const uint blk = idx % blocks;

    float v[kBlockK];
    const uint w0 = (m * pc.k + blk * kBlockK) >> 1;
    [unroll] for (uint i = 0; i < kBlockK / 2; ++i) {
        const uint w = XU[w0 + i];
        v[2 * i + 0] = f16tof32(w & 0xFFFFu);
        v[2 * i + 1] = f16tof32(w >> 16);
    }
    float amax = 0.0f;
    [unroll] for (uint j = 0; j < kBlockK; ++j) amax = max(amax, abs(v[j]));
    const float s   = i8_block_scale(amax);
    const float inv = 1.0f / s;

    const uint elems = M * pc.k;
    XU[xq_scale_words(elems) + m * blocks + blk] = asuint(s);
    const uint o = xq_value_words(elems) + (m * pc.k + blk * kBlockK) / 4;
    [unroll] for (uint u = 0; u < kBlockK / 4; ++u)
        XU[o + u] = i8_pack4(v[u * 4 + 0], v[u * 4 + 1], v[u * 4 + 2], v[u * 4 + 3], inv);
}
