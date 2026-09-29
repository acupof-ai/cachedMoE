// prefill_gemm_lds -- prefill_coopmat stage 0's GEMM with four waves sharing
// LDS tiles: the rewrite docs/STATUS.md §2.5 names (F3) as the only change that
// moves the prefill's compute rate.
//
//   Y[m][r] = sum_k W[r][k] X[m][k]     W fp16 [R][K], X fp16 [n][xs], Y fp32 [n][ys]
//
// Stage 0 gives each wave its own 16 weight rows and walks K reading every
// 16x16 tile straight from global memory: 32 bytes of each of 16 rows per
// tile, one A tile per step shared by at most eight B tiles. Here a workgroup
// owns a BM x BN block of (weight rows x tokens) as 2 x 2 waves of WM x WN
// tiles (BM = 32 WM, BN = 32 WN). For each 64-wide slice of K the workgroup
// copies the block's W rows and X rows into LDS once, 16 bytes a load and 128
// contiguous bytes a row, and every wave then runs WM x WN tile multiplies per
// 16-wide step out of LDS. The next slice is loaded into registers while this
// one is multiplied. Each output tile still accumulates k = 0, 16, 32, ... in
// order, so Y is bit for bit stage 0's.
//
// Push constants are prefill_coopmat's (CoopPush); gid.x is the token block
// from x_off, gid.y the row block from row0. R must be a multiple of BM, the
// token count (padded by the host) of BN, and K of 64.

#include "prefill_common.slang"

[vk::constant_id(8)]  const uint LdsRows = 2304;   // R
[vk::constant_id(9)]  const uint LdsCols = 5120;   // K
[vk::constant_id(10)] const uint LdsWm = 2;        // 16-row tiles per wave
[vk::constant_id(11)] const uint LdsWn = 2;        // 16-token tiles per wave

typealias TileA = linalg.CoopMat<half,  MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA>;
typealias TileB = linalg.CoopMat<half,  MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB>;
typealias TileC = linalg.CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixAccumulator>;

static const uint kW = 0, kX = 1, kY = 2;
static const uint kBk = 64;               // K slice, in halfs
static const uint kRowQ = kBk / 8 + 1;    // uint4 per LDS row: 8 of data + 1 of padding
static const uint kMaxW = 4;              // LdsWm, LdsWn <= 4
static const uint kThreads = 128;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct CoopPush {
    uint n, k, idx_off, flags, row0, x_off, x_stride, x_col0, y_stride, y_row0;
};
[[vk::push_constant]] CoopPush pc;

// Both blocks at their largest (4 x 4 tiles a wave: 128 rows, 128 tokens).
groupshared uint4 As[128 * kRowQ];
groupshared uint4 Bs[128 * kRowQ];

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint wm = LdsWm, wn = LdsWn;
    const uint bm = 32 * wm, bn = 32 * wn;
    const uint R = LdsRows, K = LdsCols;
    const uint xs = (pc.x_stride == 0u) ? K : pc.x_stride;
    const uint ys = (pc.y_stride == 0u) ? R : pc.y_stride;
    const uint r_base = pc.row0 + gid.y * bm;
    const uint m_base = pc.x_off + gid.x * bn;
    uint4* W = (uint4*)Ptr[kW];
    uint4* X = (uint4*)Ptr[kX];
    float* Y = (float*)Ptr[kY];

    // This thread's share of a slice: A is bm rows x 8 uint4, B bn rows x 8.
    const uint a_per = bm * 8 / kThreads, b_per = bn * 8 / kThreads;
    uint4 ra[8], rb[8];
    [ForceUnroll] for (uint i = 0; i < 8; ++i) {
        if (i < a_per) { const uint q = tid + i * kThreads; ra[i] = W[((r_base + q / 8) * K) / 8 + q % 8]; }
        if (i < b_per) { const uint q = tid + i * kThreads; rb[i] = X[((m_base + q / 8) * xs + pc.x_col0) / 8 + q % 8]; }
    }

    const uint wave = tid / 32;
    const uint w_r = (wave / 2) * 16 * wm;     // this wave's first row in the block
    const uint w_m = (wave % 2) * 16 * wn;     // and first token
    TileC c[kMaxW * kMaxW];
    [ForceUnroll] for (uint t = 0; t < kMaxW * kMaxW; ++t) c[t] = TileC(0.0f);

    for (uint k0 = 0; k0 < K; k0 += kBk) {
        GroupMemoryBarrierWithGroupSync();     // the last slice's reads are done
        [ForceUnroll] for (uint i = 0; i < 8; ++i) {
            const uint q = tid + i * kThreads;
            if (i < a_per) As[(q / 8) * kRowQ + q % 8] = ra[i];
            if (i < b_per) Bs[(q / 8) * kRowQ + q % 8] = rb[i];
        }
        GroupMemoryBarrierWithGroupSync();
        if (k0 + kBk < K) {
            const uint kq = (k0 + kBk) / 8;
            [ForceUnroll] for (uint i = 0; i < 8; ++i) {
                const uint q = tid + i * kThreads;
                if (i < a_per) ra[i] = W[((r_base + q / 8) * K) / 8 + kq + q % 8];
                if (i < b_per) rb[i] = X[((m_base + q / 8) * xs + pc.x_col0) / 8 + kq + q % 8];
            }
        }
        [ForceUnroll] for (uint kk = 0; kk < kBk / 16; ++kk) {
            TileB b[kMaxW];
            [ForceUnroll] for (uint j = 0; j < kMaxW; ++j)
                if (j < wn) b[j] = TileB.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(Bs, (w_m + 16 * j) * kRowQ + 2 * kk, kRowQ);
            [ForceUnroll] for (uint i = 0; i < kMaxW; ++i) {
                if (i >= wm) continue;
                TileA a = TileA.Load<linalg.CoopMatMatrixLayout.RowMajor>(As, (w_r + 16 * i) * kRowQ + 2 * kk, kRowQ);
                [ForceUnroll] for (uint j = 0; j < kMaxW; ++j)
                    if (j < wn)
                        c[i * kMaxW + j] = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, b[j], c[i * kMaxW + j]);
            }
        }
    }
    [ForceUnroll] for (uint i = 0; i < kMaxW; ++i)
        [ForceUnroll] for (uint j = 0; j < kMaxW; ++j)
            if (i < wm && j < wn)
                c[i * kMaxW + j].Store<linalg.CoopMatMatrixLayout.ColumnMajor>(
                    Y, (pc.idx_off + m_base + w_m + 16 * j) * ys + pc.y_row0 + r_base + w_r + 16 * i, ys);
}
