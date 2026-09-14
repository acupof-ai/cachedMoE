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
    stage_x_fp8(Ptr[kX], pc.k, tid);
    GroupMemoryBarrierWithGroupSync();

    const uint sublane = tid % LanesPerRow;
    const uint row = pc.row_base + gid.x * (kWg / LanesPerRow) + tid / LanesPerRow;
    const bool live = row < pc.rows;
    const float part = live ? gemv8_partial(Ptr[kW], Ptr[kS], row, pc.k,
                                            pc.scale_cols, sublane)
                            : 0.0f;
    const float y = row_reduce(part, tid, sublane);
    if (live && sublane == 0) *(float*)(Ptr[kY] + (uint64_t)row * 4) = y;
}
