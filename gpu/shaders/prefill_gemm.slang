// prefill_gemm -- the batched GEMM of design §7.13, as design §7.1 rule 7's
// GEMV generalised to many tokens ("option (b)" of docs/p3_prefill.md §5).
//
// The shape
// ---------
// One workgroup owns (row block, token tile): gid.x selects 256 / LanesPerRow
// = 8 consecutive weight rows, gid.y a tile of TileM consecutive tokens. The
// 32 lanes of a row walk the row's K elements in 32-element blocks -- lane l
// owns blocks l, l+32, l+64, ... -- and for every block decode the weight ONCE
// and multiply-add it against all TileM token columns. So the weight is read
// n / TileM times, never n times, and nothing is decoded into a transit buffer:
// the difference from option (a), the cooperative-matrix GEMM of
// prefill_coopmat.slang, which needs fp16 weights in memory (4x an FP4 expert's
// bytes) and re-loads them per 32-token wave.
//
// A row's dot product for every tile column is then one WaveActiveSum per
// column across the row's 32 lanes -- LanesPerRow == SubgroupSize is enforced
// on the host -- exactly as attn_common.slang's `row_reduce` does at M = 1.
//
// Stages
// ------
//   0  dense GEMM: y[m][r] = sum_k W[r][k] x[m][k] for any of the four weight
//      formats and both activation formats, with optional gather (x row from
//      an index list), scatter (y row from the same list), accumulation, an
//      output scale, a per-row scale, and wo_a's block-diagonal grouping.
//   1  expert gate/up + SwiGLU x route weight -> h, weights from a JOB TABLE.
//      One command buffer holds many experts because every dispatch reads its
//      six addresses from job[pc.job] instead of from this stage's slot slice
//      (a stage owns one slice; p2_attention.md §4.1). Rows are gathered from
//      the layer's FFN input through the CSR token list.
//   2  expert down from the fp8-quantised h, scatter-ADDED into y through the
//      same token list. Top-6 ids are distinct per token, so no two rows of one
//      dispatch write the same y row.
//   3  weight decode to fp16, [R][K] -> [R][K] half: the transit buffer option
//      (a) needs. One thread per (row, 32-element block).
//
// Rounding: `kFlagRound` rounds every written value onto bf16, which is what
// the reference holds after every Linear (prefill_common.slang). Stage 1 also
// rounds gate and up before SwiGLU (`self.w1(x).float()` is a bf16 GEMM output
// widened) and h after the route weight (`self.w2(x.to(dtype))`).

#include "prefill_common.slang"

static const uint kW = 0, kS = 1, kX = 2, kXS = 3, kY = 4, kIdx = 5, kRW = 6;
static const uint kJob = 7, kRowScale = 8;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct GemmPush {
    uint  rows;            // R, output rows per token
    uint  k;               // K, input elements per token (per group for wo_a)
    uint  scale_cols;      // stride of the UE8M0 plane: ceil(K / 32)
    uint  n;               // tokens (stage 1/2: the job's row count)
    uint  x_stride;        // elements between x rows
    uint  y_stride;        // elements between y rows
    uint  rows_per_group;  // 0, or wo_a's o_lora_rank: row r reads group r / rpg
    uint  flags;           // kFlag*
    uint  idx_off;         // offset into Idx / RouteW
    uint  job;             // job table entry (stages 1, 2)
    uint  row_base;        // first weight row of gid.x == 0
    float out_scale;       // multiplies the result (weights_proj's scale); 1.0
    float swiglu_limit;    // 10.0
    uint  tile_base;       // first token of gid.y == 0
};
[[vk::push_constant]] GemmPush pc;

// design §5.3's pointer table, one level up: everything one expert needs.
struct Job {
    uint64_t w1, s1, w3, s3, w2, s2;
    uint     rows_off;     // into Idx / RouteW
    uint     n;            // rows routed to this expert
    uint     h_off;        // first row of this expert in the h plane
    uint     fmt;          // 0 fp8 E4M3 (the shared expert), 3 FP4 E2M1
};
Job load_job(uint j) {
    const uint64_t a = Ptr[kJob] + (uint64_t)j * 64;
    Job o;
    o.w1 = *(uint64_t*)(a);       o.s1 = *(uint64_t*)(a + 8);
    o.w3 = *(uint64_t*)(a + 16);  o.s3 = *(uint64_t*)(a + 24);
    o.w2 = *(uint64_t*)(a + 32);  o.s2 = *(uint64_t*)(a + 40);
    o.rows_off = load_u32(a + 48); o.n = load_u32(a + 52);
    o.h_off = load_u32(a + 56);    o.fmt = load_u32(a + 60);
    return o;
}

groupshared float gTbl[256];   // FP8 E4M3 decode table (design §7.1 rule 5)

// One 32-element weight block of row `row`, decoded, and its UE8M0 exponent.
void load_wblock(uint fmt, uint64_t W, uint64_t S, uint row, uint k, uint scale_cols,
                 uint blk, out float w[32], out int e) {
    e = 0;
    if (fmt == 0) {
        const uint64_t o = W + (uint64_t)row * k + (uint64_t)blk * 32;
        const uint4 a = load16(o), b = load16(o + 16);
        [unroll] for (uint i = 0; i < 4; ++i)
            [unroll] for (uint j = 0; j < 4; ++j) {
                w[i * 4 + j]      = gTbl[(a[i] >> (8 * j)) & 0xFFu];
                w[16 + i * 4 + j] = gTbl[(b[i] >> (8 * j)) & 0xFFu];
            }
        e = int(load_u8(S + (uint64_t)(row >> 5) * scale_cols + blk)) - 127;
    } else if (fmt == 3) {
        const uint4 a = load16(W + (uint64_t)row * (k / 2) + (uint64_t)blk * 16);
        [unroll] for (uint i = 0; i < 4; ++i)
            [unroll] for (uint j = 0; j < 8; ++j)
                w[i * 8 + j] = kE2M1[(a[i] >> (4 * j)) & 0xFu];
        e = int(load_u8(S + (uint64_t)row * scale_cols + blk)) - 127;
    } else if (fmt == 1) {
        const uint64_t o = W + ((uint64_t)row * k + (uint64_t)blk * 32) * 2;
        [unroll] for (uint i = 0; i < 4; ++i) {
            const uint4 a = load16(o + (uint64_t)i * 16);
            [unroll] for (uint j = 0; j < 4; ++j) {
                w[i * 8 + j * 2]     = bf16_decode(a[j] & 0xFFFFu);
                w[i * 8 + j * 2 + 1] = bf16_decode(a[j] >> 16);
            }
        }
    } else {
        const uint64_t o = W + ((uint64_t)row * k + (uint64_t)blk * 32) * 4;
        [unroll] for (uint i = 0; i < 8; ++i) {
            const uint4 a = load16(o + (uint64_t)i * 16);
            [unroll] for (uint j = 0; j < 4; ++j) w[i * 4 + j] = asfloat(a[j]);
        }
    }
}

// sum_i w[i] * x[i] over one block of token row `xrow`, times the activation's
// block scale when it is pre-quantised.
float dot_block(uint64_t X, uint64_t XS, uint xrow, uint col0, uint blk, uint x_stride,
                const float w[32]) {
    float p = 0.0f;
    if (XFmt == 0) {
        const uint64_t o = X + ((uint64_t)xrow * x_stride + col0 + (uint64_t)blk * 32) * 4;
        [unroll] for (uint i = 0; i < 8; ++i) {
            const uint4 a = load16(o + (uint64_t)i * 16);
            [unroll] for (uint j = 0; j < 4; ++j) p += w[i * 4 + j] * asfloat(a[j]);
        }
        return p;
    }
    const uint64_t o = X + ((uint64_t)xrow * x_stride + col0 + (uint64_t)blk * 32) * 2;
    [unroll] for (uint i = 0; i < 4; ++i) {
        const uint4 a = load16(o + (uint64_t)i * 16);
        [unroll] for (uint j = 0; j < 4; ++j) {
            p += w[i * 8 + j * 2]     * f16tof32(a[j] & 0xFFFFu);
            p += w[i * 8 + j * 2 + 1] * f16tof32(a[j] >> 16);
        }
    }
    return p * load_f32(XS + ((uint64_t)xrow * (x_stride / 32) + (col0 / 32) + blk) * 4);
}

// --- stage 0 ------------------------------------------------------------------
void stage_dense(uint3 gid, uint tid) {
    if (WFmt == 0) {
        gTbl[tid] = fp8_decode(tid);
        GroupMemoryBarrierWithGroupSync();
    }
    const uint lane = tid % LanesPerRow;
    const uint row  = pc.row_base + gid.x * (kWg / LanesPerRow) + tid / LanesPerRow;
    const uint t0   = pc.tile_base + gid.y * TileM;
    const uint blocks = pc.k / kBlockK;
    const bool live_row = row < pc.row_base + pc.rows;
    const uint col0 = (pc.rows_per_group != 0u) ? (row / pc.rows_per_group) * pc.k : 0u;

    float acc[TileM];
    for (uint j = 0; j < TileM; ++j) acc[j] = 0.0f;
    for (uint blk = lane; blk < blocks; blk += LanesPerRow) {
        float w[32];
        int e;
        load_wblock(WFmt, Ptr[kW], Ptr[kS], row, pc.k, pc.scale_cols, blk, w, e);
        // No early exit on a short tail tile, so the loop trip count is the
        // specialisation constant: a dead column reads token 0 and is dropped
        // at the write.
        for (uint j = 0; j < TileM; ++j) {
            const uint m = t0 + j;
            const uint mm = (m < pc.n) ? m : 0u;
            const uint xrow = ((pc.flags & kFlagGather) != 0u)
                                  ? load_u32(Ptr[kIdx] + (uint64_t)(pc.idx_off + mm) * 4) : mm;
            acc[j] += ldexp(dot_block(Ptr[kX], Ptr[kXS], xrow, col0, blk, pc.x_stride, w), e);
        }
    }
    for (uint j = 0; j < TileM; ++j) {
        const float s = WaveActiveSum(acc[j]);
        const uint m = t0 + j;
        if (lane != 0 || m >= pc.n || !live_row) continue;
        const uint yrow = ((pc.flags & kFlagScatter) != 0u)
                              ? load_u32(Ptr[kIdx] + (uint64_t)(pc.idx_off + m) * 4) : m;
        // kFlagRoundPre: the GEMM's own bf16 output is scaled by a bf16 constant
        // (`weights_proj(x) * scale`), so the reference rounds twice.
        float v = (((pc.flags & kFlagRoundPre) != 0u) ? bf16_round(s) : s) * pc.out_scale;
        if ((pc.flags & kFlagRowScale) != 0u) v *= load_f32(Ptr[kRowScale] + (uint64_t)m * 4);
        const uint64_t ya = Ptr[kY] + ((uint64_t)yrow * pc.y_stride + (row - pc.row_base)) * 4;
        if ((pc.flags & kFlagAccumulate) != 0u) v += load_f32(ya);
        store_f32(ya, maybe_round(v, pc.flags));
    }
}

// --- stage 1 ------------------------------------------------------------------
void stage_gateup(uint3 gid, uint tid) {
    gTbl[tid] = fp8_decode(tid);
    GroupMemoryBarrierWithGroupSync();
    const Job jb = load_job(pc.job);
    const uint lane = tid % LanesPerRow;
    const uint row  = gid.x * (kWg / LanesPerRow) + tid / LanesPerRow;
    const uint t0   = gid.y * TileM;
    const uint blocks = pc.k / kBlockK;

    float a1[TileM], a3[TileM];
    for (uint j = 0; j < TileM; ++j) { a1[j] = 0.0f; a3[j] = 0.0f; }
    if (row < pc.rows) {
        for (uint blk = lane; blk < blocks; blk += LanesPerRow) {
            float w1[32], w3[32];
            int e1, e3;
            load_wblock(jb.fmt, jb.w1, jb.s1, row, pc.k, pc.scale_cols, blk, w1, e1);
            load_wblock(jb.fmt, jb.w3, jb.s3, row, pc.k, pc.scale_cols, blk, w3, e3);
            for (uint j = 0; j < TileM; ++j) {
                const uint m = t0 + j;
                if (m >= jb.n) break;
                const uint xrow = load_u32(Ptr[kIdx] + (uint64_t)(jb.rows_off + m) * 4);
                // One block of x, both dot products: the activation load is shared.
                const uint64_t o = Ptr[kX] + ((uint64_t)xrow * pc.x_stride + (uint64_t)blk * 32) * 2;
                float p1 = 0.0f, p3 = 0.0f;
                [unroll] for (uint i = 0; i < 4; ++i) {
                    const uint4 av = load16(o + (uint64_t)i * 16);
                    [unroll] for (uint q = 0; q < 4; ++q) {
                        const float xa = f16tof32(av[q] & 0xFFFFu);
                        const float xb = f16tof32(av[q] >> 16);
                        p1 += w1[i * 8 + q * 2] * xa + w1[i * 8 + q * 2 + 1] * xb;
                        p3 += w3[i * 8 + q * 2] * xa + w3[i * 8 + q * 2 + 1] * xb;
                    }
                }
                const float xs = load_f32(Ptr[kXS] + ((uint64_t)xrow * (pc.x_stride / 32) + blk) * 4);
                a1[j] += ldexp(p1, e1) * xs;
                a3[j] += ldexp(p3, e3) * xs;
            }
        }
    }
    for (uint j = 0; j < TileM; ++j) {
        const float g = WaveActiveSum(a1[j]);
        const float u = WaveActiveSum(a3[j]);
        const uint m = t0 + j;
        if (lane != 0 || m >= jb.n || row >= pc.rows) continue;
        const float rw = load_f32(Ptr[kRW] + (uint64_t)(jb.rows_off + m) * 4);
        const float h  = swiglu(maybe_round(g, pc.flags), maybe_round(u, pc.flags),
                                pc.swiglu_limit) * rw;
        store_f32(Ptr[kY] + ((uint64_t)(jb.h_off + m) * pc.rows + row) * 4,
                  maybe_round(h, pc.flags));
    }
}

// --- stage 2 ------------------------------------------------------------------
void stage_down(uint3 gid, uint tid) {
    gTbl[tid] = fp8_decode(tid);
    GroupMemoryBarrierWithGroupSync();
    const Job jb = load_job(pc.job);
    const uint lane = tid % LanesPerRow;
    const uint row  = gid.x * (kWg / LanesPerRow) + tid / LanesPerRow;
    const uint t0   = gid.y * TileM;
    const uint blocks = pc.k / kBlockK;

    float acc[TileM];
    for (uint j = 0; j < TileM; ++j) acc[j] = 0.0f;
    if (row < pc.rows) {
        for (uint blk = lane; blk < blocks; blk += LanesPerRow) {
            float w[32];
            int e;
            load_wblock(jb.fmt, jb.w2, jb.s2, row, pc.k, pc.scale_cols, blk, w, e);
            for (uint j = 0; j < TileM; ++j) {
                const uint m = t0 + j;
                if (m >= jb.n) break;
                const uint hrow = jb.h_off + m;
                const uint64_t o = Ptr[kX] + ((uint64_t)hrow * pc.k + (uint64_t)blk * 32) * 2;
                float p = 0.0f;
                [unroll] for (uint i = 0; i < 4; ++i) {
                    const uint4 av = load16(o + (uint64_t)i * 16);
                    [unroll] for (uint q = 0; q < 4; ++q)
                        p += w[i * 8 + q * 2] * f16tof32(av[q] & 0xFFFFu)
                           + w[i * 8 + q * 2 + 1] * f16tof32(av[q] >> 16);
                }
                acc[j] += ldexp(p, e) * load_f32(Ptr[kXS] + ((uint64_t)hrow * blocks + blk) * 4);
            }
        }
    }
    for (uint j = 0; j < TileM; ++j) {
        const float s = WaveActiveSum(acc[j]);
        const uint m = t0 + j;
        if (lane != 0 || m >= jb.n || row >= pc.rows) continue;
        const uint yrow = load_u32(Ptr[kIdx] + (uint64_t)(jb.rows_off + m) * 4);
        const uint64_t ya = Ptr[kY] + ((uint64_t)yrow * pc.y_stride + row) * 4;
        // `y[idx] += expert(x[idx])`: the expert's output is a bf16 tensor and
        // the accumulator is fp32, so the round is on the addend, not the sum.
        store_f32(ya, load_f32(ya) + maybe_round(s, pc.flags));
    }
}

// --- stage 3 ------------------------------------------------------------------
void stage_decode(uint3 gid, uint tid) {
    gTbl[tid] = fp8_decode(tid);
    GroupMemoryBarrierWithGroupSync();
    const uint blocks = pc.k / kBlockK;
    const uint idx = gid_linear(gid) * kWg + tid;
    const uint row = pc.row_base + idx / blocks;
    const uint blk = idx % blocks;
    if (row >= pc.row_base + pc.rows) return;
    // kFlagFromJob: the weight is job[pc.job]'s part pc.idx_off (0 w1, 1 w3,
    // 2 w2) in the job's own format, so one command buffer decodes a whole
    // batch of experts into the same transit buffer, one after another.
    uint64_t wa = Ptr[kW], sa = Ptr[kS];
    uint fmt = WFmt;
    if ((pc.flags & kFlagFromJob) != 0u) {
        const Job jb = load_job(pc.job);
        fmt = jb.fmt;
        wa = (pc.idx_off == 0u) ? jb.w1 : ((pc.idx_off == 1u) ? jb.w3 : jb.w2);
        sa = (pc.idx_off == 0u) ? jb.s1 : ((pc.idx_off == 1u) ? jb.s3 : jb.s2);
    }
    float w[32];
    int e;
    load_wblock(fmt, wa, sa, row, pc.k, pc.scale_cols, blk, w, e);
    const uint64_t o = Ptr[kY] + ((uint64_t)(row - pc.row_base) * pc.k + (uint64_t)blk * 32) * 2;
    [unroll] for (uint i = 0; i < 32; ++i)
        *(uint16_t*)(o + (uint64_t)i * 2) = (uint16_t)f32tof16(ldexp(w[i], e));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0)      stage_dense(gid, tid);
    else if (Stage == 1) stage_gateup(gid, tid);
    else if (Stage == 2) stage_down(gid, tid);
    else                 stage_decode(gid, tid);
}
