// wq_b -- Q up-projection with q_norm and RoPE fused (design §7.3, dispatch #3
// of the §7.14 list). fp8 [32768 x 1280], 41.9 MB: one of the two biggest
// attention reads of a decode token.
//
// `q_norm` folds into the staging pass, not into a separate dispatch: the
// staging already holds all 1280 of qr one 32-element block per thread, and
// the RMS statistic is one workgroup reduction over exactly those registers
// (fp8_gemv.slang `stage_x_fp8_rmsnorm`). The `act_quant` round trip then runs
// on the normalised value, which is the order `linear()` imposes.
//
// **RoPE pairs adjacent elements.** design §7.3 says rows (d, d+32) rotate
// together; `apply_rotary_emb` actually does
// `view_as_complex(x.unflatten(-1, (-1, 2)))`, so within the last 64 dims of a
// head the pairs are (448 + 2j, 448 + 2j + 1) and frequency j drives pair j.
// Following the design text would rotate the wrong pairs and every layer's
// attention would be subtly wrong in a way only L2 could catch.
//
// Because a workgroup owns `256 / LanesPerRow` CONSECUTIVE rows and 512
// divides that block size, a rotation pair never straddles two workgroups: the
// partner value is always in this workgroup's row results.

// The LDS budget for the staged activation: wq_b's K is q_lora_rank = 1280, and a
// workgroup never stages more (fp8_gemv.slang).
#define DEEPMOE_GEMV_MAX_K 1280

#include "fp8_gemv.slang"

static const uint kW = 0, kS = 1, kQr = 2, kNormW = 3, kRope = 4, kQ = 5;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct WqbPush {
    uint  rows;         // n_heads * head_dim = 32768
    uint  k;            // q_lora_rank = 1280
    uint  scale_cols;   // k / 32 = 40
    uint  head_dim;     // 512
    uint  rope_dim;     // 64
    float norm_eps;     // 1e-20
};
[[vk::push_constant]] WqbPush pc;

// (256 / LanesPerRow) * RowsPerLane, at the widest setting of each.
static const uint kMaxRowsPerWg = (kWg / 16) * kMaxRowsPerLane;
groupshared float gYrow[kMaxRowsPerWg];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    fp8_table_fill(tid);
    stage_x_fp8_rmsnorm(Ptr[kQr], Ptr[kNormW], pc.k, tid, pc.norm_eps,
                        Ptr[kS] + (uint64_t)((gid.x * gemv_rows_per_wg()) >> 5) * pc.scale_cols);
    GroupMemoryBarrierWithGroupSync();

    const uint rowsPerWg = gemv_rows_per_wg();
    const uint sublane = tid % LanesPerRow;
    const uint base = gid.x * rowsPerWg;
    const uint rowBase = base + (tid / LanesPerRow) * RowsPerLane;

    float acc[kMaxRowsPerLane];
    gemv8_rows(Ptr[kW], Ptr[kS], rowBase, pc.k, pc.scale_cols, sublane, acc);
    // Unrolled against the compile-time bound, predicated on the
    // specialisation constant: see gemv8_rows in fp8_gemv.slang.
    [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) {
        if (r >= RowsPerLane) continue;
        const float y = row_reduce(acc[r], tid, sublane);
        if (sublane == 0) gYrow[rowBase - base + r] = y;
    }
    GroupMemoryBarrierWithGroupSync();

    // One thread per row for the tail, so a rotation pair is owned by the
    // thread holding its even half.
    if (tid < rowsPerWg && base + tid < pc.rows) {
        const uint r = base + tid;
        const uint d = r % pc.head_dim;
        const uint rope_lo = pc.head_dim - pc.rope_dim;
        if (d < rope_lo) {
            *(uint16_t*)(Ptr[kQ] + (uint64_t)r * 2) = (uint16_t)bf16_encode(gYrow[tid]);
        } else if (((d - rope_lo) & 1u) == 0u) {
            const uint j = (d - rope_lo) >> 1;
            const float c = load_f32(Ptr[kRope] + (uint64_t)j * 8);
            const float s = load_f32(Ptr[kRope] + (uint64_t)j * 8 + 4);
            float re = gYrow[tid], im = gYrow[tid + 1];
            rope_pair(c, s, re, im, false);
            *(uint16_t*)(Ptr[kQ] + (uint64_t)r * 2) = (uint16_t)bf16_encode(re);
            *(uint16_t*)(Ptr[kQ] + (uint64_t)(r + 1) * 2) = (uint16_t)bf16_encode(im);
        }
    }
}
