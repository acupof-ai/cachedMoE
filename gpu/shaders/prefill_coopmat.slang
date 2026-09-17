// prefill_coopmat -- option (a) of docs/p3_prefill.md §5: the GEMM design §7.13
// wrote down, on `VK_KHR_cooperative_matrix`.
//
// `linalg.CoopMat<half, Subgroup, 16, 16, use>` is a 16x16 matrix distributed
// across one wave's lanes, and `coopMatMulAdd(A, B, C)` is C += A x B in one
// operation. The prefill mapping is:
//
//   A = a 16x16 tile of the weight, rows r..r+15, cols k..k+15, from an fp16
//       transit buffer [R][K] (prefill_gemm.slang stage 3 decodes it);
//   B = a 16x16 tile of the ACTIVATIONS whose 16 columns are 16 tokens, rows
//       k..k+15 of each -- loaded straight from a contiguous fp16 [n][K] plane
//       with `ColumnMajor` and stride K, so no shared memory and no barriers;
//   C = the float accumulator for output rows r..r+15 of the same 16 tokens.
//
// One workgroup is one wave of 32 threads = 32 tokens, two B/C columns of 16.
// The loop is output-tile-major (r outer, k inner) so the only live tiles are
// the current A and two B and two C -- a C array over all R/16 output tiles
// would be sized by a specialisation constant but still spill (p2_attention.md
// §4.4), and would have to be measured separately. The cost of the order is
// that the activation tiles are re-loaded once per output tile row.
//
// Stage 0  the GEMM. Rows and cols are specialisation constants (multiples of
//          16) so every loop bound is folded at pipeline creation.
// Stage 1  x -> the contiguous fp16 plane stage 0 reads: gathers row
//          Idx[idx_off + m] of a pre-quantised activation (E4M3 value as fp16,
//          fp32 block scale) and writes value x scale as fp16. The scale has to
//          be folded in -- a tile multiply cannot apply a per-(token, block)
//          factor -- which is an fp16 range question the bench measures.
// Stage 2  x -> the fp16 plane from an fp32 activation [n][K] (the bf16 / fp32
//          linears: the gate, the compressor, wo_a's input), rows m + idx_off.
// Stage 3  band attention scores (docs/p4_prefill_speed.md §3): for query b,
//          S[b][h][t] = sum_d Q[b][h][d] G[b][t][d] with Q the fp16 queries
//          [B][H][D] and G the fp16 KV rows each query's index list gathers
//          [B][E][D] (prefill_attn.slang stage 3). A = a 16x16 tile of heads x
//          dims, B = dims x entries, C = heads x entries. CmRows = E (a
//          multiple of 16), CmCols = D.
// Stage 4  band attention P.V: O[b][h][d] = sum_t P[b][h][t] G[b][t][d] with
//          P the bf16 probabilities as fp16 [B][H][E] (prefill_attn.slang
//          stage 4). A = heads x entries, B = entries x dims, C = heads x dims.
//
// Stage 0's x plane and y plane strides are push constants (0 = K / R), so a
// grouped linear (wo_a: 8 groups of [1024][4096] over one [32768] row) is
// eight dispatches over the same planes with a column / row offset each.

#include "prefill_common.slang"

[vk::constant_id(8)] const uint CmRows = 2304;
[vk::constant_id(9)] const uint CmCols = 5120;
// Stage 0 geometry (docs/p4_prefill_speed.md): 16-token tiles per workgroup
// (0 = 2, one 32-thread wave's worth) and 16-row output tiles per workgroup in
// the row-tile mode (0 = 1). Stages 3 / 4: CmTokTiles is the 16-head tiles
// per workgroup (0 = 1).
[vk::constant_id(10)] const uint CmTokTiles = 2;
[vk::constant_id(11)] const uint CmRowTilesPerWg = 1;

typealias TileA = linalg.CoopMat<half,  MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA>;
typealias TileB = linalg.CoopMat<half,  MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB>;
typealias TileC = linalg.CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixAccumulator>;

static const uint kW = 0, kX = 1, kY = 2, kQ = 3, kQS = 4, kIdx = 5;
static const uint kFlagRowTiles = 64u;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct CoopPush {
    uint n;          // tokens, padded by the host to a multiple of 32
    uint k;          // stage 1: activation width
    uint idx_off;    // stage 1; stage 0: output row offset
    uint flags;      // stage 1: kFlagGather; stage 0: kFlagRowTiles
    uint row0;       // stage 0, row-tile mode: first output row of gid.y == 0
    uint x_off;      // stage 0: first token of gid.x == 0 (a token-block dispatch);
                     // stages 3 / 4: first query of gid.x == 0
    uint x_stride;   // stage 0: elements between x rows (0 = K)
    uint x_col0;     // stage 0: first x column of the multiply (a group's offset)
    uint y_stride;   // stage 0: elements between y rows (0 = R)
    uint y_row0;     // stage 0: first y element of output row 0 (a group's offset)
};
[[vk::push_constant]] CoopPush pc;

void stage_gemm(uint3 gid, uint tid) {
    const uint tt = (CmTokTiles == 0u) ? 2u : CmTokTiles;
    const uint rt = (CmRowTilesPerWg == 0u) ? 1u : CmRowTilesPerWg;
    const uint m0 = pc.x_off + gid.x * 16 * tt;
    half*  W = (half*)Ptr[kW];
    half*  X = (half*)Ptr[kX];
    float* Y = (float*)Ptr[kY];
    const uint R = CmRows, K = CmCols;
    const uint xs = (pc.x_stride == 0u) ? K : pc.x_stride;
    const uint ys = (pc.y_stride == 0u) ? R : pc.y_stride;
    // Tile-row-major workgroups (gid.y selects `rt` output tile rows from
    // row0) when the host dispatches that many; one workgroup walking every
    // row otherwise. pc.idx_off shifts the output rows, so gate and up can
    // share a plane.
    const bool rowt = (pc.flags & kFlagRowTiles) != 0u;
    const uint r0 = rowt ? pc.row0 + gid.y * 16 * rt : 0u;
    const uint r1 = rowt ? min(r0 + 16 * rt, R) : R;
    const uint yo = pc.idx_off + m0;
    // The token tiles are unrolled by hand (no arrays of cooperative
    // matrices); every `tt > i` folds at pipeline creation.
    for (uint r = r0; r < r1; r += 16) {
        TileC c0 = TileC(0.0f);
        TileC c1 = TileC(0.0f);
        TileC c2 = TileC(0.0f);
        TileC c3 = TileC(0.0f);
        TileC c4 = TileC(0.0f);
        TileC c5 = TileC(0.0f);
        TileC c6 = TileC(0.0f);
        TileC c7 = TileC(0.0f);
        for (uint k = 0; k < K; k += 16) {
            TileA a = TileA.Load<linalg.CoopMatMatrixLayout.RowMajor>(W, r * K + k, K);
            { TileB b = TileB.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(X, (m0 + 0) * xs + pc.x_col0 + k, xs); c0 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, b, c0); }
            if (tt > 1) { TileB b = TileB.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(X, (m0 + 16) * xs + pc.x_col0 + k, xs); c1 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, b, c1); }
            if (tt > 2) { TileB b = TileB.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(X, (m0 + 32) * xs + pc.x_col0 + k, xs); c2 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, b, c2); }
            if (tt > 3) { TileB b = TileB.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(X, (m0 + 48) * xs + pc.x_col0 + k, xs); c3 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, b, c3); }
            if (tt > 4) { TileB b = TileB.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(X, (m0 + 64) * xs + pc.x_col0 + k, xs); c4 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, b, c4); }
            if (tt > 5) { TileB b = TileB.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(X, (m0 + 80) * xs + pc.x_col0 + k, xs); c5 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, b, c5); }
            if (tt > 6) { TileB b = TileB.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(X, (m0 + 96) * xs + pc.x_col0 + k, xs); c6 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, b, c6); }
            if (tt > 7) { TileB b = TileB.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(X, (m0 + 112) * xs + pc.x_col0 + k, xs); c7 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, b, c7); }
        }
        c0.Store<linalg.CoopMatMatrixLayout.ColumnMajor>(Y, (yo + 0) * ys + pc.y_row0 + r, ys);
        if (tt > 1) c1.Store<linalg.CoopMatMatrixLayout.ColumnMajor>(Y, (yo + 16) * ys + pc.y_row0 + r, ys);
        if (tt > 2) c2.Store<linalg.CoopMatMatrixLayout.ColumnMajor>(Y, (yo + 32) * ys + pc.y_row0 + r, ys);
        if (tt > 3) c3.Store<linalg.CoopMatMatrixLayout.ColumnMajor>(Y, (yo + 48) * ys + pc.y_row0 + r, ys);
        if (tt > 4) c4.Store<linalg.CoopMatMatrixLayout.ColumnMajor>(Y, (yo + 64) * ys + pc.y_row0 + r, ys);
        if (tt > 5) c5.Store<linalg.CoopMatMatrixLayout.ColumnMajor>(Y, (yo + 80) * ys + pc.y_row0 + r, ys);
        if (tt > 6) c6.Store<linalg.CoopMatMatrixLayout.ColumnMajor>(Y, (yo + 96) * ys + pc.y_row0 + r, ys);
        if (tt > 7) c7.Store<linalg.CoopMatMatrixLayout.ColumnMajor>(Y, (yo + 112) * ys + pc.y_row0 + r, ys);
    }
}

// --- stage 3: attention scores ----------------------------------------------
// kW = Q16 [B][H][D], kX = G16 [B][E][D], kY = S f32 [B][H][E]. pc.k = H.
// Workgroup (query, head-tile group): the query's gathered KV is walked once
// per workgroup, entry tile by entry tile, against every head tile it holds.
void stage_attn_score(uint3 gid, uint tid) {
    const uint E = CmRows, D = CmCols, H = pc.k;
    const uint ht = (CmTokTiles == 0u) ? 1u : CmTokTiles;
    const uint b = pc.x_off + gid.x;
    const uint h0 = gid.y * 16 * ht;
    half*  Q = (half*)Ptr[kW];
    half*  G = (half*)Ptr[kX];
    float* S = (float*)Ptr[kY];
    for (uint t = 0; t < E; t += 16) {
        TileC c0 = TileC(0.0f);
        TileC c1 = TileC(0.0f);
        TileC c2 = TileC(0.0f);
        TileC c3 = TileC(0.0f);
        for (uint d = 0; d < D; d += 16) {
            TileB g = TileB.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(G, (b * E + t) * D + d, D);
            { TileA a = TileA.Load<linalg.CoopMatMatrixLayout.RowMajor>(Q, (b * H + h0 + 0) * D + d, D); c0 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, g, c0); }
            if (ht > 1) { TileA a = TileA.Load<linalg.CoopMatMatrixLayout.RowMajor>(Q, (b * H + h0 + 16) * D + d, D); c1 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, g, c1); }
            if (ht > 2) { TileA a = TileA.Load<linalg.CoopMatMatrixLayout.RowMajor>(Q, (b * H + h0 + 32) * D + d, D); c2 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, g, c2); }
            if (ht > 3) { TileA a = TileA.Load<linalg.CoopMatMatrixLayout.RowMajor>(Q, (b * H + h0 + 48) * D + d, D); c3 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, g, c3); }
        }
        c0.Store<linalg.CoopMatMatrixLayout.RowMajor>(S, (b * H + h0 + 0) * E + t, E);
        if (ht > 1) c1.Store<linalg.CoopMatMatrixLayout.RowMajor>(S, (b * H + h0 + 16) * E + t, E);
        if (ht > 2) c2.Store<linalg.CoopMatMatrixLayout.RowMajor>(S, (b * H + h0 + 32) * E + t, E);
        if (ht > 3) c3.Store<linalg.CoopMatMatrixLayout.RowMajor>(S, (b * H + h0 + 48) * E + t, E);
    }
}

// --- stage 4: attention P.V ---------------------------------------------------
// kW = P16 [B][H][E], kX = G16 [B][E][D], kY = O f32 [B][H][D] (written, not
// accumulated). pc.k = H.
// CmRowTilesPerWg is the number of 16-wide OUTPUT DIM tiles one accumulator
// group covers (0 = 1). It is what fixes the kernel's memory order: the
// contraction here is over `t` (entries), the output index is `d` (dims), and G
// is stored [E][D] -- so a `d`-outer / `t`-inner loop walks G with a 16*D*2 =
// 16 KiB stride, touching 32 B of every 1 KiB row, and does it D/16 = 32 times
// over. Holding `dv` dim tiles at once makes each visited G row 32*dv bytes of
// CONTIGUOUS memory and cuts the number of passes over G to D/(16*dv). At
// D = 512, dv = 8 that is 4 passes of 256 B rows instead of 32 passes of 32 B.
// ht (head tiles) and dv multiply the live accumulators, so the host keeps
// ht * dv <= 8.
void stage_attn_pv(uint3 gid, uint tid) {
    const uint E = CmRows, D = CmCols, H = pc.k;
    const uint ht = (CmTokTiles == 0u) ? 1u : CmTokTiles;
    const uint dv = (CmRowTilesPerWg == 0u) ? 1u : CmRowTilesPerWg;
    const uint b = pc.x_off + gid.x;
    const uint h0 = gid.y * 16 * ht;
    half*  P = (half*)Ptr[kW];
    half*  G = (half*)Ptr[kX];
    float* O = (float*)Ptr[kY];
    if (dv == 1u) {
        // the head-tile form: one dim tile, up to four head tiles
        for (uint d = 0; d < D; d += 16) {
            TileC c0 = TileC(0.0f);
            TileC c1 = TileC(0.0f);
            TileC c2 = TileC(0.0f);
            TileC c3 = TileC(0.0f);
            for (uint t = 0; t < E; t += 16) {
                TileB g = TileB.Load<linalg.CoopMatMatrixLayout.RowMajor>(G, (b * E + t) * D + d, D);
                { TileA a = TileA.Load<linalg.CoopMatMatrixLayout.RowMajor>(P, (b * H + h0 + 0) * E + t, E); c0 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, g, c0); }
                if (ht > 1) { TileA a = TileA.Load<linalg.CoopMatMatrixLayout.RowMajor>(P, (b * H + h0 + 16) * E + t, E); c1 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, g, c1); }
                if (ht > 2) { TileA a = TileA.Load<linalg.CoopMatMatrixLayout.RowMajor>(P, (b * H + h0 + 32) * E + t, E); c2 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, g, c2); }
                if (ht > 3) { TileA a = TileA.Load<linalg.CoopMatMatrixLayout.RowMajor>(P, (b * H + h0 + 48) * E + t, E); c3 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, g, c3); }
            }
            c0.Store<linalg.CoopMatMatrixLayout.RowMajor>(O, (b * H + h0 + 0) * D + d, D);
            if (ht > 1) c1.Store<linalg.CoopMatMatrixLayout.RowMajor>(O, (b * H + h0 + 16) * D + d, D);
            if (ht > 2) c2.Store<linalg.CoopMatMatrixLayout.RowMajor>(O, (b * H + h0 + 32) * D + d, D);
            if (ht > 3) c3.Store<linalg.CoopMatMatrixLayout.RowMajor>(O, (b * H + h0 + 48) * D + d, D);
        }
        return;
    }
    // the dim-tile form: one head tile (the host forces ht = 1), dv dim tiles,
    // so the A tile is loaded once for dv multiplies and G is read in runs.
    const uint prow = (b * H + h0) * E;
    for (uint d = 0; d < D; d += 16 * dv) {
        TileC c0 = TileC(0.0f);
        TileC c1 = TileC(0.0f);
        TileC c2 = TileC(0.0f);
        TileC c3 = TileC(0.0f);
        TileC c4 = TileC(0.0f);
        TileC c5 = TileC(0.0f);
        TileC c6 = TileC(0.0f);
        TileC c7 = TileC(0.0f);
        for (uint t = 0; t < E; t += 16) {
            const uint go = (b * E + t) * D + d;
            TileA a = TileA.Load<linalg.CoopMatMatrixLayout.RowMajor>(P, prow + t, E);
            { TileB g = TileB.Load<linalg.CoopMatMatrixLayout.RowMajor>(G, go + 16 * 0, D); c0 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, g, c0); }
            { TileB g = TileB.Load<linalg.CoopMatMatrixLayout.RowMajor>(G, go + 16 * 1, D); c1 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, g, c1); }
            if (dv > 2) { TileB g = TileB.Load<linalg.CoopMatMatrixLayout.RowMajor>(G, go + 16 * 2, D); c2 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, g, c2); }
            if (dv > 3) { TileB g = TileB.Load<linalg.CoopMatMatrixLayout.RowMajor>(G, go + 16 * 3, D); c3 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, g, c3); }
            if (dv > 4) { TileB g = TileB.Load<linalg.CoopMatMatrixLayout.RowMajor>(G, go + 16 * 4, D); c4 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, g, c4); }
            if (dv > 5) { TileB g = TileB.Load<linalg.CoopMatMatrixLayout.RowMajor>(G, go + 16 * 5, D); c5 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, g, c5); }
            if (dv > 6) { TileB g = TileB.Load<linalg.CoopMatMatrixLayout.RowMajor>(G, go + 16 * 6, D); c6 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, g, c6); }
            if (dv > 7) { TileB g = TileB.Load<linalg.CoopMatMatrixLayout.RowMajor>(G, go + 16 * 7, D); c7 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, g, c7); }
        }
        const uint oo = (b * H + h0) * D + d;
        c0.Store<linalg.CoopMatMatrixLayout.RowMajor>(O, oo + 16 * 0, D);
        c1.Store<linalg.CoopMatMatrixLayout.RowMajor>(O, oo + 16 * 1, D);
        if (dv > 2) c2.Store<linalg.CoopMatMatrixLayout.RowMajor>(O, oo + 16 * 2, D);
        if (dv > 3) c3.Store<linalg.CoopMatMatrixLayout.RowMajor>(O, oo + 16 * 3, D);
        if (dv > 4) c4.Store<linalg.CoopMatMatrixLayout.RowMajor>(O, oo + 16 * 4, D);
        if (dv > 5) c5.Store<linalg.CoopMatMatrixLayout.RowMajor>(O, oo + 16 * 5, D);
        if (dv > 6) c6.Store<linalg.CoopMatMatrixLayout.RowMajor>(O, oo + 16 * 6, D);
        if (dv > 7) c7.Store<linalg.CoopMatMatrixLayout.RowMajor>(O, oo + 16 * 7, D);
    }
}

void stage_stage_x(uint3 gid, uint tid) {
    const uint blocks = pc.k / kBlockK;
    const uint idx = gid_linear(gid) * 32 + tid;
    const uint m = idx / blocks, blk = idx % blocks;
    if (m >= pc.n) return;
    const uint src = ((pc.flags & kFlagGather) != 0u)
                         ? load_u32(Ptr[kIdx] + (uint64_t)(pc.idx_off + m) * 4) : (pc.idx_off + m);
    const float s = load_f32(Ptr[kQS] + ((uint64_t)src * blocks + blk) * 4);
    const uint64_t qa = Ptr[kQ] + ((uint64_t)src * pc.k + (uint64_t)blk * 32) * 2;
    const uint64_t xa = Ptr[kX] + ((uint64_t)m * pc.k + (uint64_t)blk * 32) * 2;
    [unroll] for (uint i = 0; i < 32; ++i) {
        const float v = f16tof32(load_u16(qa + (uint64_t)i * 2)) * s;
        *(uint16_t*)(xa + (uint64_t)i * 2) = (uint16_t)f32tof16(v);
    }
}

void stage_stage_f32(uint3 gid, uint tid) {
    const uint blocks = pc.k / kBlockK;
    const uint idx = gid_linear(gid) * 32 + tid;
    const uint m = idx / blocks, blk = idx % blocks;
    if (m >= pc.n) return;
    const uint64_t xa = Ptr[kQ] + ((uint64_t)(pc.idx_off + m) * pc.k + (uint64_t)blk * 32) * 4;
    const uint64_t ya = Ptr[kX] + ((uint64_t)m * pc.k + (uint64_t)blk * 32) * 2;
    [unroll] for (uint i = 0; i < 32; ++i)
        *(uint16_t*)(ya + (uint64_t)i * 2) = (uint16_t)f32tof16(load_f32(xa + (uint64_t)i * 4));
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0)      stage_gemm(gid, tid);
    else if (Stage == 1) stage_stage_x(gid, tid);
    else if (Stage == 2) stage_stage_f32(gid, tid);
    else if (Stage == 3) stage_attn_score(gid, tid);
    else                 stage_attn_pv(gid, tid);
}
