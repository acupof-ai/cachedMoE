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
// 256 threads, LanesPerRow lanes to a row, so a workgroup retires
// 256 / LanesPerRow rows: 160 workgroups for wq_a's 1280 rows at the
// LanesPerRow = 32 of kernel_p1.md §3.2.

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
    const uint row = pc.row_base + gid.x * (kWg / LanesPerRow) + tid / LanesPerRow;
    const bool live = row < pc.rows;
    const float part = live ? gemv8_partial(Ptr[kW], Ptr[kS], row, pc.k,
                                            pc.scale_cols, sublane)
                            : 0.0f;
    const float y = row_reduce(part, tid, sublane);
    if (live && sublane == 0) *(float*)(Ptr[kY] + (uint64_t)row * 4) = y;
}
