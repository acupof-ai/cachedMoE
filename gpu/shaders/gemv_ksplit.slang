// gemv_ksplit -- the fp8 E4M3 GEMV of design §7.3/§7.4/§7.6, cut along the
// CONTRACTION axis (docs/p2_attention.md §11 item 1, §13).
//
// What the split is for
// ---------------------
// docs/p2_attention.md §10.1 ruled out the two obvious explanations for wo_b
// sitting at 135 GB/s against a 217 GB/s memory system: it is not the `act_quant`
// round trip (compiling it away measured 150.2 GB/s against 151.7) and it is not
// DRAM (42 MB in 312 us). What was left was the LDS budget and the single
// accumulator per lane. wo_b's K is 8192, so `gXQ` alone is 16 KiB and a
// workgroup reserves 19 KiB -- three of them to a CU -- and each of its lanes
// walks 8 blocks of 32 elements into ONE accumulator.
//
// Splitting K into `KSplit` slices attacks exactly that, and nothing else:
//
//   * the staged activation is the SLICE, so wo_b at KSplit = 2 reserves 10.5
//     KiB and gets six workgroups to a CU, at KSplit = 4 reserves 6.5 KiB;
//   * the workgroup count is multiplied by KSplit, which is what the two
//     workgroup-starved GEMVs need -- `wkv` is 512 rows, i.e. 64 workgroups on
//     40 CUs at RowsPerLane 1, and it has been the reason wkv reads 154 GB/s
//     for two documents running;
//   * it costs NO extra weight traffic. A slice reads its own columns of the
//     same rows; the union over slices is the same 42 MB, read once.
//
// The activation IS re-staged per slice, so the staging traffic is unchanged
// as well: KSplit times as many workgroups each staging 1/KSplit of x.
//
// The cost is one fp32 partial per (slice, row) and a combine dispatch of
// ceil(rows / 256) workgroups: 1.0-1.4 us measured. What it buys, idle,
// docs/p2_attention.md §13.3: wo_b 260 -> 204 us at 8 slices, wo_a 195 -> 162
// at 4, wq_a 38 -> 35 at 2, wkv (with its finish folded in) 19 -> 17 at 2 --
// and only at ONE row a lane; at two rows a lane the split loses.
//
// Re-association
// --------------
// The partials are summed slice by slice in index order, and inside a slice a
// lane accumulates its blocks in the same order it always did. What moves is
// where the `WaveActiveSum` across the 32 lanes of a row group falls: it now
// happens once per slice instead of once per row, so the result is the same sum
// re-associated, not a different sum. `tests/test_gpu_attn.cpp` measures the
// difference against the unsplit kernel and prints it: relative L2 5e-8 to 7e-8
// on every layer, i.e. fp32 round-off, four orders of magnitude under the
// 1.6e-3 bf16 floor the reference itself sits on (§3). Agreement with the
// oracle is unchanged to nine digits.
//
// Geometry
// --------
// One dispatch, `row_groups * KSplit` workgroups, `gid = rg * KSplit + slice`.
// Consecutive workgroups therefore cover the SAME rows in adjacent column
// slices, which keeps a row's 8192 bytes a contiguous read across the pair
// rather than two streams 4 KiB apart.
//
// `rows_per_group` carries wo_a's block-diagonal structure (design §7.6): a
// workgroup never straddles two 1024-row groups, so it stages the one
// `k`-wide slice of `o` its group needs, offset by the slice.

// The widest SLICE this .spv stages. Every caller keeps K / KSplit at or under
// it: wo_b 8192/2, wq_a 5120/2, wkv 5120/2 or /4, wo_a 4096/1.
// AttnRunner::create checks it.
#define DEEPMOE_GEMV_MAX_K 4096

#include "fp8_gemv.slang"

static const uint kW = 0, kS = 1, kX = 2, kY = 3, kP = 4;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct KSplitPush {
    uint rows;             // total output rows
    uint k;                // the FULL contraction length
    uint scale_cols;       // k / 32, the UE8M0 plane's row stride
    uint rows_per_group;   // 0 = one activation for all rows; wo_a passes 1024
    uint part_stride;      // floats between slices in the partial plane (>= rows)
    uint row_base;         // first row this dispatch owns
};
[[vk::push_constant]] KSplitPush pc;

// --- stage 0: one K slice of every row --------------------------------------
void stage_split(uint gid, uint tid) {
    const uint rowsPerWg = gemv_rows_per_wg();
    const uint rg = gid / KSplit;
    const uint sl = gid % KSplit;
    // KSplit divides k / 32 -- AttnRunner::create rejects anything else -- so
    // every slice is the same width and no lane needs a tail test.
    const uint kc = (pc.k / (kBlockK * KSplit)) * kBlockK;
    const uint k0 = sl * kc;

    const uint base  = rg * rowsPerWg;
    const uint group = (pc.rows_per_group != 0u) ? (base / pc.rows_per_group) : 0u;

    fp8_table_fill(tid);
    stage_x_fp8(Ptr[kX] + ((uint64_t)group * pc.k + k0) * 4, kc, tid,
                Ptr[kS] + (uint64_t)((pc.row_base + base) >> 5) * pc.scale_cols + k0 / kBlockK);
    GroupMemoryBarrierWithGroupSync();

    const uint sublane = tid % LanesPerRow;
    const uint rowBase = pc.row_base + base + (tid / LanesPerRow) * RowsPerLane;
    float acc[kMaxRowsPerLane];
    gemv8_rows_slice(Ptr[kW], Ptr[kS], rowBase, pc.k, k0, kc, pc.scale_cols, sublane, acc);
    [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) {
        if (r >= RowsPerLane) continue;
        const float y = row_reduce(acc[r], tid, sublane);
        if (sublane == 0 && rowBase + r < pc.rows)
            *(float*)(Ptr[kP] + ((uint64_t)sl * pc.part_stride + rowBase + r) * 4) = y;
    }
}

// --- stage 1: add the KSplit partials ---------------------------------------
// One thread a row, `ceil(rows / 256)` workgroups. No LDS, no reduction: the
// whole point of the split is that the cross-slice sum is KSplit scalars.
void stage_combine(uint gid, uint tid) {
    const uint row = gid * kWg + tid;
    if (row >= pc.rows) return;
    float y = 0.0f;
    for (uint s = 0; s < KSplit; ++s)
        y += load_f32(Ptr[kP] + ((uint64_t)s * pc.part_stride + row) * 4);
    *(float*)(Ptr[kY] + (uint64_t)row * 4) = y;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0) stage_split(gid.x, tid);
    else            stage_combine(gid.x, tid);
}
