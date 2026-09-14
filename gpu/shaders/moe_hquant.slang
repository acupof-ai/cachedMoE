// moe_hquant -- the fp8 E4M3 + UE8M0 block-32 quantisation of `h`, as a third
// dispatch between the two of design §7.9 (docs/kernel_p2_moe.md §8 item 1).
//
// Why it exists. design §7.9 v0.6 requires `silu(gate)*up` to make an fp8 round
// trip before w2, and P2 measured the two obvious places to do it:
//
//   HQuant = 1  inside dispatch B: costs a second read of h and one amax per
//               (lane, weight row) that all recompute the same number -- +49%
//               at M=1 on the L32 R1 decode winner.
//   HQuant = 2  inside dispatch A's write-out: nearly free per byte, but a
//               block is 32 consecutive rows of h and a workgroup only owns
//               (256/LanesPerRow) * RowsPerLane of them, so it forces
//               L16 R2 / L32 R4. That reshape -- not the quantisation -- is the
//               +27% M=1 regression of §5.3.
//
// This kernel is the third option §5.3 recommended: leave dispatch A alone and
// quantise h afterwards. The whole of h is M x slots x 2304 values -- 32 KB at
// M=1, 194 KB at M=6 -- so one thread per 32-element block reads 64 B, writes
// 32 B + one scale word, and the dispatch is dominated by its own launch.
//
// It cannot quantise in place: thread t reads fp16 bytes [64t, 64t+64) and
// would write fp8 bytes [32t, 32t+32), which is inside thread t/2's read. So
// the fp8 plane lives past the fp16 one (moe_common.slang hq_value_words) and
// the h allocation carries 3.125 B/element. That is the entire price.
//
// Only the slots in SlotList are touched, so design §7.9's "compute the experts
// that arrived first" splits this dispatch exactly like the other two.

#include "moe_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<uint>   SlotList;  // compute index -> slot
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> H;         // the h allocation, raw words

struct HQuantPush {
    uint num_slots;    // stride of the h slot dimension
    uint list_count;   // how many entries of SlotList to quantise
    uint k;            // 2304, the intermediate dimension h is blocked along
};
[[vk::push_constant]] HQuantPush pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint blocks = pc.k / kBlockK;                 // 72
    const uint idx    = gid.x * kWgSize + tid;
    if (idx >= M * pc.list_count * blocks) return;

    const uint blk  = idx % blocks;
    const uint li   = (idx / blocks) % pc.list_count;
    const uint m    = idx / (blocks * pc.list_count);
    const uint row  = m * pc.num_slots + SlotList[li];

    // The fp16 h dispatch A wrote. HPrecision == 1 is rejected host-side: the
    // offsets below are the fp16 layout.
    float v[kBlockK];
    const uint w0 = (row * pc.k + blk * kBlockK) >> 1;
    [unroll] for (uint i = 0; i < kBlockK / 2; ++i) {
        const uint w = H[w0 + i];
        v[2 * i + 0] = f16tof32(w & 0xFFFFu);
        v[2 * i + 1] = f16tof32(w >> 16);
    }
    float amax = 0.0f;
    [unroll] for (uint j = 0; j < kBlockK; ++j) amax = max(amax, abs(v[j]));

    // inference/kernel.py act_quant(x, 32, scale_fmt="ue8m0"): amax floored at
    // 1e-4, the scale rounded up to a power of two. Bit for bit what dispatch
    // A does under HQuant == 2 and dispatch B under HQuant == 1.
    const float s   = fast_round_scale(max(amax, kFp8ActMin), 1.0f / kFp8ActMax);
    const float inv = 1.0f / s;

    const uint elems = M * pc.num_slots * pc.k;
    H[hq_scale_words(elems) + row * blocks + blk] = asuint(s);
    const uint o = hq_value_words(elems) + (row * pc.k + blk * kBlockK) / 4;
    [unroll] for (uint u = 0; u < kBlockK / 4; ++u) {
        uint packed = 0;
        [unroll] for (uint j = 0; j < 4; ++j) {
            const float c = clamp(v[u * 4 + j] * inv, -kFp8ActMax, kFp8ActMax);
            packed |= fp8_e4m3_pack(fp8_e4m3_round(c)) << (8 * j);
        }
        H[o + u] = packed;
    }
}
