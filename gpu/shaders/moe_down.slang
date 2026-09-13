// moe_down -- FP4 w2 projection with the expert reduction folded in
// (design §7.9, dispatch B; #11 of the §7.14 list).
//
// One workgroup owns a row block of y and loops over every entry of SlotList
// internally, so the sum over experts happens in the lane's own fp32
// accumulator: no atomics, no third kernel. Because the reduction across a row
// is linear, the per-lane partials of all slots are accumulated first and
// reduced across the row exactly once at the end.
//
// h comes from moe_gateup; HPrecision selects whether it is fp16 or fp32. Both
// are read through a raw uint view so one binding covers the two cases.

#include "moe_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> ExpertAddr;  // pointer table (design §5.3)
[[vk::binding(1, 0)]] StructuredBuffer<uint>     Ids;
[[vk::binding(2, 0)]] StructuredBuffer<uint>     SlotList;
[[vk::binding(3, 0)]] StructuredBuffer<uint>     H;   // h[M][slots][K] as raw words
[[vk::binding(4, 0)]] RWStructuredBuffer<float>  Y;   // y[M][N] fp32 (design §6)

struct DownPush {
    uint layer;
    uint experts_per_layer;
    uint num_slots;    // stride of the h slot dimension
    uint list_count;   // how many entries of SlotList to sum (design §7.9)
    uint n_rows;       // 5120
    uint k;            // 2304
};
[[vk::push_constant]] DownPush pc;

groupshared float sRed[kWgSize];

float row_reduce(float v, uint tid, uint sublane) {
    sRed[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = LanesPerRow >> 1; s > 0; s >>= 1) {
        if (sublane < s) sRed[tid] = sRed[tid] + sRed[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float r = sRed[tid - sublane];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Eight consecutive h values starting at element `e` of the flat h array.
void load_h8(uint e, out float v[8]) {
    if (HPrecision == 0) {
        const uint w0 = e >> 1;                       // two halves per word
        [unroll] for (uint i = 0; i < 4; ++i) {
            const uint w = H[w0 + i];
            v[2 * i + 0] = f16tof32(w & 0xFFFFu);
            v[2 * i + 1] = f16tof32(w >> 16);
        }
    } else {
        [unroll] for (uint i = 0; i < 8; ++i) v[i] = asfloat(H[e + i]);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint rowsPerWg = kWgSize / LanesPerRow;
    const uint sublane   = tid % LanesPerRow;
    const uint rowInWg   = tid / LanesPerRow;
    const uint row       = gid.x * rowsPerWg + rowInWg;

    const uint blocks   = pc.k / kBlockK;   // 72
    const uint rowBytes = pc.k / 2;         // 1152
    const bool live     = row < pc.n_rows;

    float acc[kMaxM];
    [unroll] for (uint i = 0; i < kMaxM; ++i) acc[i] = 0.0f;

    if (live) {
        for (uint ci = 0; ci < pc.list_count; ++ci) {
            const uint slot   = SlotList[ci];
            const uint expert = Ids[slot];
            const uint tbase  = (pc.layer * pc.experts_per_layer + expert) * 6;
            const uint64_t w2Base = ExpertAddr[tbase + 2];
            const uint64_t s2Base = ExpertAddr[tbase + 3];
            if (w2Base == 0) continue;

            const uint64_t w2Row = w2Base + (uint64_t)row * rowBytes;
            const uint64_t s2Row = s2Base + (uint64_t)row * blocks;

            for (uint blk = sublane; blk < blocks; blk += LanesPerRow) {
                const uint4 w2q = load_fp4_block(w2Row + (uint64_t)blk * 16);
                const int   e2  = (int)load_scale_byte(s2Row + blk) - 127;

                float p[kMaxM];
                [unroll] for (uint i = 0; i < kMaxM; ++i) p[i] = 0.0f;

                [unroll] for (uint w = 0; w < 4; ++w) {
                    const uint pw = w2q[w];
                    float d[8];
                    [unroll] for (uint j = 0; j < 8; ++j)
                        d[j] = fp4_decode((pw >> (4 * j)) & 0xFu);
                    for (uint m = 0; m < M; ++m) {
                        float hv[8];
                        load_h8((m * pc.num_slots + slot) * pc.k + blk * kBlockK + w * 8, hv);
                        [unroll] for (uint j = 0; j < 8; ++j) p[m] += d[j] * hv[j];
                    }
                }
                for (uint m = 0; m < M; ++m) acc[m] += ldexp(p[m], e2);
            }
        }
    }

    for (uint m = 0; m < M; ++m) {
        const float s = row_reduce(acc[m], tid, sublane);
        if (live && sublane == 0) Y[m * pc.n_rows + row] = s;
    }
}
