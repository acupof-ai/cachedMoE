// wq_a -- the plain FP8 E4M3 GEMV (design §7.3, dispatch #2 of §7.14).
//
// `wq_a` itself is fp8 [1280 x 5120], 6.6 MB, and this shader is written to be
// the generic one: rows, K and the scale-plane stride are push constants, so
// the same .spv also runs `wkv`'s projection and any other
// `y[rows] = W[rows][K] @ x[K]` in the attention path. What it does NOT do is
// any of the fused tails -- q_norm, RoPE, the cache write -- which is why
// wq_b.slang and wkv.slang exist separately.
//
// The activation round trip is the one attn_common.slang describes: x is
// staged once per workgroup, already on the FP8 E4M3 grid with a per-32
// power-of-two scale, because that is what `linear()` feeds an fp8 GEMM.
//
// 256 threads, LanesPerRow lanes to a row and RowsPerLane rows to a lane, so
// a workgroup retires (256 / LanesPerRow) * RowsPerLane rows and stages x once
// for all of them.

// The LDS budget for the staged activation. wq_a's own K is the model dim,
// 5120, but this is deliberately left at the 8192 maximum: sizing it down to
// 5120 raised occupancy and measured 152 GB/s against 166: more resident
// workgroups each re-staging a 5120-wide activation is more L2 traffic, not
// less. wq_a is also the generic fp8 GEMV of the family, so 8192 keeps it able
// to run any K in the model (fp8_gemv.slang).
#define DEEPMOE_GEMV_MAX_K 8192

#include "fp8_gemv.slang"

static const uint kW = 0, kS = 1, kX = 2, kY = 3;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct GemvPush {
    uint rows;          // 1280
    uint k;             // 5120
    uint scale_cols;    // k / 32
    uint row_base;      // first row this dispatch owns (grouped callers use it)
};
[[vk::push_constant]] GemvPush pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    fp8_table_fill(tid);
    stage_x_fp8(Ptr[kX], pc.k, tid);
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
