// wo_b -- the output up-projection (design §7.6, dispatch #7 of the §7.14
// list). fp8 [5120 x 8192], 41.9 MB, the other big attention read.
//
// Its input is wo_a's 8192-wide output, and unlike wo_a this one DOES go
// through `linear()`, so the `act_quant` round trip applies (compiled with
// ActQuant = 1). K = 8192 is the widest activation in the model and sets the
// LDS budget in fp8_gemv.slang: 8 KiB of E4M3 bytes plus 1 KiB of block
// scales.
//
// The output `a[5120]` is the `hc_post` input of the next mega_mhc (§7.7), so
// it is written fp32 -- the residual stream never leaves fp32 on our side even
// though the reference keeps it in bf16.

// The LDS budget for the staged activation: wo_b's K is the widest activation in the model = 8192, and a
// workgroup never stages more (fp8_gemv.slang).
#define DEEPMOE_GEMV_MAX_K 8192

#include "fp8_gemv.slang"

static const uint kW = 0, kS = 1, kX = 2, kY = 3;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct GemvPush {
    uint rows;          // 5120
    uint k;             // 8192
    uint scale_cols;    // k / 32 = 256
    uint row_base;
};
[[vk::push_constant]] GemvPush pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    fp8_table_fill(tid);
    stage_x_fp8(Ptr[kX], pc.k, tid,
                Ptr[kS] + (uint64_t)((pc.row_base + gid.x * gemv_rows_per_wg()) >> 5) * pc.scale_cols);
    GroupMemoryBarrierWithGroupSync();

    const uint sublane = tid % LanesPerRow;
    const uint rowBase = pc.row_base + gemv_row_base(gid.x, tid);
    float acc[kMaxRowsPerLane];
    gemv8_rows(Ptr[kW], Ptr[kS], rowBase, pc.k, pc.scale_cols, sublane, acc);
    // Unrolled against the compile-time bound, predicated on the
    // specialisation constant: see gemv8_rows in fp8_gemv.slang.
    [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) {
        if (r >= RowsPerLane) continue;
        const float y = row_reduce(acc[r], tid, sublane);
        if (sublane == 0 && rowBase + r < pc.rows)
            *(float*)(Ptr[kY] + (uint64_t)(rowBase + r) * 4) = y;
    }
}
