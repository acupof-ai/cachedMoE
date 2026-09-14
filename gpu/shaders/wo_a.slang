// wo_a -- the block-diagonal output down-projection (design §7.6, dispatch #6
// of the §7.14 list). fp8 [8192 x 4096], 33.6 MB, read as 8 groups of
// [1024 x 4096]: group g projects only the 4096 dims of heads 8g..8g+7.
//
// Two things separate it from the other fp8 GEMVs:
//
// 1. **No activation quantisation.** `Attention.forward` reaches this weight
//    through `torch.einsum`, not `linear()`, and the module is declared bf16
//    (convert.py dequantises it; "an fp8 grouped GEMM would halve the
//    memory"). So the checkpoint stores fp8 bytes but the arithmetic is
//    bf16 x bf16, with no `act_quant` round trip on `o`. Compiled with
//    ActQuant = 0, which also switches the LDS staging to bf16.
//
// 2. **The activation is a slice, not the whole vector.** A workgroup covers
//    `256 / LanesPerRow` consecutive rows and a group is 1024 rows, so a
//    workgroup never straddles two groups and can stage the one 4096-wide
//    slice of `o` its group needs.

#include "fp8_gemv.slang"

static const uint kW = 0, kS = 1, kO = 2, kY = 3;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct WoaPush {
    uint rows;            // n_groups * o_lora_rank = 8192
    uint k;               // n_heads * head_dim / n_groups = 4096
    uint scale_cols;      // k / 32 = 128
    uint rows_per_group;  // o_lora_rank = 1024
};
[[vk::push_constant]] WoaPush pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint base  = gid.x * gemv_rows_per_wg();
    const uint group = base / pc.rows_per_group;

    fp8_table_fill(tid);
    stage_x_fp8(Ptr[kO] + (uint64_t)group * pc.k * 4, pc.k, tid);
    GroupMemoryBarrierWithGroupSync();

    const uint sublane = tid % LanesPerRow;
    const uint rowBase = gemv_row_base(gid.x, tid);
    float acc[kMaxRowsPerLane];
    gemv8_rows(Ptr[kW], Ptr[kS], rowBase, pc.k, pc.scale_cols, sublane, acc);
    for (uint r = 0; r < RowsPerLane; ++r) {
        const float y = row_reduce(acc[r], tid, sublane);
        if (sublane == 0 && rowBase + r < pc.rows)
            *(float*)(Ptr[kY] + (uint64_t)(rowBase + r) * 4) = y;
    }
}
