// moe_gateup -- fused FP4 gate/up + SwiGLU for the routed experts of a layer
// (design §7.9, dispatch A; #10 of the §7.14 list).
//
// One workgroup owns (expert slot, row block). A row block is
// kWgSize / LanesPerRow rows of w1 and w3; the LanesPerRow lanes of a row walk
// that row's 5120 K-elements in 32-element UE8M0 blocks, 16 B of packed FP4 per
// lane per block, and the same lane reuses those weights for all M activation
// columns (design §7.1 rule 7: weights read once, M accumulators).
//
// The set of experts to compute is an indirection list, not 0..6: design §7.9's
// "compute the three that arrived first" only needs a shorter SlotList and a
// smaller gid.y, with no change to the kernel.
//
// x and h are small enough to stay in L2 across the whole dispatch (10 KiB and
// 4.5 KiB per column), so they are read straight from global memory rather than
// staged in LDS. That keeps LDS at 1 KiB -- the cross-lane reduction scratch --
// which is what lets M go to 6 without the 60 KiB x tile design §7.1 rule 6
// would have needed.

#include "moe_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> ExpertAddr;  // pointer table (design §5.3)
[[vk::binding(1, 0)]] StructuredBuffer<uint>     Ids;         // Ids[slot] = expert id
[[vk::binding(2, 0)]] StructuredBuffer<uint>     SlotList;    // compute index -> slot
[[vk::binding(3, 0)]] StructuredBuffer<float>    RouteW;      // [M][slots] routing weights
[[vk::binding(4, 0)]] StructuredBuffer<uint4>    X;           // x[M][K] fp16, 8 per uint4
[[vk::binding(5, 0)]] RWStructuredBuffer<half>   H16;         // h[M][slots][N] fp16
[[vk::binding(6, 0)]] RWStructuredBuffer<float>  H32;         // h[M][slots][N] fp32

struct GateUpPush {
    uint layer;
    uint experts_per_layer;
    uint num_slots;      // stride of the h/RouteW slot dimension
    uint n_rows;         // 2304
    uint k;              // 5120
    float swiglu_limit;  // 10.0 (design §2.4)
};
[[vk::push_constant]] GateUpPush pc;

groupshared float sRed[kWgSize];

// Sum `v` across the LanesPerRow lanes of one row and broadcast the result.
// An LDS tree rather than WaveActiveSum: LanesPerRow is independent of the
// subgroup size (a 64-lane row spans two wave32 subgroups), and at one call per
// row per workgroup the barriers do not show up in the bandwidth measurement.
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

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint rowsPerWg = kWgSize / LanesPerRow;
    const uint sublane   = tid % LanesPerRow;
    const uint rowInWg   = tid / LanesPerRow;
    const uint row       = gid.x * rowsPerWg + rowInWg;

    const uint slot   = SlotList[gid.y];
    const uint expert = Ids[slot];
    const uint tbase  = (pc.layer * pc.experts_per_layer + expert) * 6;

    const uint64_t w1Base = ExpertAddr[tbase + 0];
    const uint64_t s1Base = ExpertAddr[tbase + 1];
    const uint64_t w3Base = ExpertAddr[tbase + 4];
    const uint64_t s3Base = ExpertAddr[tbase + 5];

    const uint blocks  = pc.k / kBlockK;        // 160
    const uint rowBytes = pc.k / 2;             // 2560 packed bytes per row
    const bool live    = (row < pc.n_rows) && (w1Base != 0) && (w3Base != 0);

    float accGate[kMaxM];
    float accUp[kMaxM];
    [unroll] for (uint i = 0; i < kMaxM; ++i) { accGate[i] = 0.0f; accUp[i] = 0.0f; }

    if (live) {
        const uint64_t w1Row = w1Base + (uint64_t)row * rowBytes;
        const uint64_t w3Row = w3Base + (uint64_t)row * rowBytes;
        const uint64_t s1Row = s1Base + (uint64_t)row * blocks;
        const uint64_t s3Row = s3Base + (uint64_t)row * blocks;

        for (uint blk = sublane; blk < blocks; blk += LanesPerRow) {
            {
                const uint4 w1q = load_fp4_block(w1Row + (uint64_t)blk * 16);
                const uint4 w3q = load_fp4_block(w3Row + (uint64_t)blk * 16);
                const int  e1   = (int)load_scale_byte(s1Row + blk) - 127;
                const int  e3   = (int)load_scale_byte(s3Row + blk) - 127;

                float p1[kMaxM];
                float p3[kMaxM];
                [unroll] for (uint i = 0; i < kMaxM; ++i) { p1[i] = 0.0f; p3[i] = 0.0f; }

                [unroll] for (uint w = 0; w < 4; ++w) {
                    const uint pw1 = w1q[w];
                    const uint pw3 = w3q[w];
                    float d1[8];
                    float d3[8];
                    [unroll] for (uint j = 0; j < 8; ++j) {
                        d1[j] = fp4_decode((pw1 >> (4 * j)) & 0xFu);
                        d3[j] = fp4_decode((pw3 >> (4 * j)) & 0xFu);
                    }
                    for (uint m = 0; m < M; ++m) {
                        const uint4 xr = X[m * (pc.k / 8) + blk * 4 + w];
                        [unroll] for (uint j = 0; j < 8; ++j) {
                            const float xv = unpack_half(xr, j);
                            p1[m] += d1[j] * xv;
                            p3[m] += d3[j] * xv;
                        }
                    }
                }
                for (uint m = 0; m < M; ++m) {
                    accGate[m] += ldexp(p1[m], e1);
                    accUp[m]   += ldexp(p3[m], e3);
                }
            }
        }
    }

    for (uint m = 0; m < M; ++m) {
        const float g = row_reduce(accGate[m], tid, sublane);
        const float u = row_reduce(accUp[m], tid, sublane);
        if (live && sublane == 0) {
            const float rw = RouteW[m * pc.num_slots + slot];
            const float h  = swiglu(g, u, pc.swiglu_limit) * rw;
            const uint  oi = (m * pc.num_slots + slot) * pc.n_rows + row;
            if (HPrecision == 0) H16[oi] = (half)h;
            else                 H32[oi] = h;
        }
    }
}
