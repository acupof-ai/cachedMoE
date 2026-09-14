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

#include "prefill_common.slang"

[vk::constant_id(8)] const uint CmRows = 2304;
[vk::constant_id(9)] const uint CmCols = 5120;

typealias TileA = linalg.CoopMat<half,  MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA>;
typealias TileB = linalg.CoopMat<half,  MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB>;
typealias TileC = linalg.CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixAccumulator>;

static const uint kW = 0, kX = 1, kY = 2, kQ = 3, kQS = 4, kIdx = 5;
static const uint kFlagRowTiles = 64u;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct CoopPush {
    uint n;          // tokens, padded by the host to a multiple of 32
    uint k;          // stage 1: activation width
    uint idx_off;    // stage 1
    uint flags;      // stage 1: kFlagGather
};
[[vk::push_constant]] CoopPush pc;

void stage_gemm(uint3 gid, uint tid) {
    const uint m0 = gid.x * 32;
    half*  W = (half*)Ptr[kW];
    half*  X = (half*)Ptr[kX];
    float* Y = (float*)Ptr[kY];
    const uint R = CmRows, K = CmCols;
    // Tile-row-major workgroups (gid.y = output tile row) when the host
    // dispatches R/16 rows; one workgroup walking every row otherwise.
    const uint r0 = (pc.flags & kFlagRowTiles) != 0u ? gid.y * 16 : 0u;
    const uint r1 = (pc.flags & kFlagRowTiles) != 0u ? r0 + 16 : R;
    for (uint r = r0; r < r1; r += 16) {
        TileC c0 = TileC(0.0f);
        TileC c1 = TileC(0.0f);
        for (uint k = 0; k < K; k += 16) {
            TileA a  = TileA.Load<linalg.CoopMatMatrixLayout.RowMajor>(W, r * K + k, K);
            TileB b0 = TileB.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(X, m0 * K + k, K);
            TileB b1 = TileB.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(X, (m0 + 16) * K + k, K);
            c0 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, b0, c0);
            c1 = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, b1, c1);
        }
        c0.Store<linalg.CoopMatMatrixLayout.ColumnMajor>(Y, m0 * R + r, R);
        c1.Store<linalg.CoopMatMatrixLayout.ColumnMajor>(Y, (m0 + 16) * R + r, R);
    }
}

void stage_stage_x(uint3 gid, uint tid) {
    const uint blocks = pc.k / kBlockK;
    const uint idx = gid.x * 32 + tid;
    const uint m = idx / blocks, blk = idx % blocks;
    if (m >= pc.n) return;
    const uint src = ((pc.flags & kFlagGather) != 0u)
                         ? load_u32(Ptr[kIdx] + (uint64_t)(pc.idx_off + m) * 4) : m;
    const float s = load_f32(Ptr[kQS] + ((uint64_t)src * blocks + blk) * 4);
    const uint64_t qa = Ptr[kQ] + ((uint64_t)src * pc.k + (uint64_t)blk * 32) * 2;
    const uint64_t xa = Ptr[kX] + ((uint64_t)m * pc.k + (uint64_t)blk * 32) * 2;
    [unroll] for (uint i = 0; i < 32; ++i) {
        const float v = f16tof32(load_u16(qa + (uint64_t)i * 2)) * s;
        *(uint16_t*)(xa + (uint64_t)i * 2) = (uint16_t)f32tof16(v);
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0) stage_gemm(gid, tid);
    else            stage_stage_x(gid, tid);
}
