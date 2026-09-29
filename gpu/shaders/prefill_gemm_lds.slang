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
//
// LdsWt: W is stored [K][R] (the band attention's P.V, whose "weight" is the
// gathered KV G [E][D] contracted over E). A slice of it is 64 k-rows of BM
// contiguous elements; each thread loads two k-rows of the same 8 columns
// and writes them into the row-major [r][k] layout as 8 dwords of
// {W[k][r], W[k+1][r]}, so the A tiles are read exactly as an [R][K] weight's
// (column-major tile reads from LDS were 8 two-byte reads a lane with bank
// conflicts, STATUS §7 0af; the transposing writes are 4 dwords a uint4).
// kFlagBatched: gid.y is the batch index and gid.x the row block, with one
// problem's W, X and Y per batch entry back to back (the attention: one
// query's G, Q or P, and S or O). Row block fastest, so the workgroups in
// flight read different 256-byte columns of a [K][R] weight's 1 KB rows:
// batch fastest had every resident workgroup on the same column, the same
// DRAM channels, and the P.V at half the score's rate (STATUS §7 0ag).

#include "prefill_common.slang"

// LDS_BK: the K slice in halfs. 128 was measured (STATUS §3 68): +12% on
// w1, nothing on wq_b, slower on wo_b, at twice the LDS and staging registers.
#ifndef LDS_BK
#define LDS_BK 64
#endif

[vk::constant_id(8)]  const uint LdsRows = 2304;   // R
[vk::constant_id(9)]  const uint LdsCols = 5120;   // K
[vk::constant_id(10)] const uint LdsWm = 2;        // 16-row tiles per wave
[vk::constant_id(11)] const uint LdsWn = 2;        // 16-token tiles per wave
[vk::constant_id(12)] const uint LdsWt = 0;        // W stored [K][R]

typealias TileA = linalg.CoopMat<half,  MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA>;
typealias TileB = linalg.CoopMat<half,  MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB>;
typealias TileC = linalg.CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixAccumulator>;

static const uint kW = 0, kX = 1, kY = 2;
static const uint kBk = LDS_BK;           // K slice, in halfs
static const uint kQ = kBk / 8;           // uint4 of data per LDS row
static const uint kRowQ = kQ + 1;         // uint4 per LDS row: kQ of data + 1 of padding
static const uint kMaxW = 4;              // LdsWm, LdsWn <= 4
static const uint kThreads = 128;
static const uint kRegs = kMaxW * 32 * kQ / kThreads;   // a slice's uint4 per thread at the widest block

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct CoopPush {
    uint n, k, idx_off, flags, row0, x_off, x_stride, x_col0, y_stride, y_row0;
};
[[vk::push_constant]] CoopPush pc;

// The blocks at this geometry's size, not the largest one's: a 64-row block
// takes 18 KB for both, a 128-row one 28 KB, and the CU's 64 KB then holds
// three or two workgroups instead of one (STATUS §7 0af: with one, the next
// slice's global loads are hidden by nothing but this workgroup's own MMAs).
groupshared uint4 As[32 * LdsWm * kRowQ];
groupshared uint4 Bs[32 * LdsWn * kRowQ];

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint wm = LdsWm, wn = LdsWn;
    const uint bm = 32 * wm, bn = 32 * wn;
    const uint R = LdsRows, K = LdsCols;
    const uint xs = (pc.x_stride == 0u) ? K : pc.x_stride;
    const uint ys = (pc.y_stride == 0u) ? R : pc.y_stride;
    const bool batched = (pc.flags & kFlagBatched) != 0u;
    const uint r_base = pc.row0 + (batched ? gid.x : gid.y) * bm;
    const uint m_base = pc.x_off + (batched ? 0u : gid.x * bn);
    const uint64_t batch = batched ? gid.y : 0u;
    uint4* W = (uint4*)(Ptr[kW] + batch * R * K * 2);
    uint4* X = (uint4*)(Ptr[kX] + batch * bn * xs * 2);
    float* Y = (float*)(Ptr[kY] + batch * bn * ys * 4);

    // This thread's share of a slice: A is bm rows x 8 uint4 (LdsWt: pairs
    // of k-rows, 2i and 2i + 1, of the same 8 columns), B bn rows x 8.
    const uint a_per = bm * kQ / kThreads, b_per = bn * kQ / kThreads;
    const uint aq = bm / 8;                    // LdsWt: uint4 per k-row of the slice
    uint4 ra[kRegs], rb[kRegs];
    [ForceUnroll] for (uint i = 0; i < kRegs; ++i) {
        if (i < a_per) {
            const uint q = tid + i * kThreads, qp = tid + (i / 2) * kThreads;
            ra[i] = LdsWt ? W[((2 * (qp / aq) + i % 2) * R + r_base) / 8 + qp % aq]
                          : W[((r_base + q / kQ) * K) / 8 + q % kQ];
        }
        if (i < b_per) { const uint q = tid + i * kThreads; rb[i] = X[((m_base + q / kQ) * xs + pc.x_col0) / 8 + q % kQ]; }
    }

    const uint wave = tid / 32;
    const uint w_r = (wave / 2) * 16 * wm;     // this wave's first row in the block
    const uint w_m = (wave % 2) * 16 * wn;     // and first token
    TileC c[kMaxW * kMaxW];
    [ForceUnroll] for (uint t = 0; t < kMaxW * kMaxW; ++t) c[t] = TileC(0.0f);

    for (uint k0 = 0; k0 < K; k0 += kBk) {
        GroupMemoryBarrierWithGroupSync();     // the last slice's reads are done
        [ForceUnroll] for (uint i = 0; i < kRegs; ++i) {
            const uint q = tid + i * kThreads;
            if (i < a_per && !LdsWt) As[(q / kQ) * kRowQ + q % kQ] = ra[i];
            if (i < a_per && LdsWt && i % 2 == 0) {
                // The 16 lanes of one k-row pair hold rows 8 apart, which share
                // an LDS bank (8 x 36 dwords); each lane walks its 8 rows in
                // its own order so a store instruction spreads over 8 banks.
                const uint qp = tid + (i / 2) * kThreads, kd = qp / aq, r0 = (qp % aq) * 8, rot = tid & 7u;
                [ForceUnroll] for (uint j = 0; j < 8; ++j) {
                    const uint jj = j ^ rot, sh = 16 * (jj & 1u);
                    const uint lo = (ra[i][jj >> 1] >> sh) & 0xFFFFu;
                    const uint hi = (ra[i + 1][jj >> 1] >> sh) & 0xFFFFu;
                    As[(r0 + jj) * kRowQ + kd / 4][kd % 4] = lo | (hi << 16);
                }
            }
            if (i < b_per) Bs[(q / kQ) * kRowQ + q % kQ] = rb[i];
        }
        GroupMemoryBarrierWithGroupSync();
        if (k0 + kBk < K) {
            const uint kq = (k0 + kBk) / 8, k1 = k0 + kBk;
            [ForceUnroll] for (uint i = 0; i < kRegs; ++i) {
                const uint q = tid + i * kThreads, qp = tid + (i / 2) * kThreads;
                if (i < a_per)
                    ra[i] = LdsWt ? W[((k1 + 2 * (qp / aq) + i % 2) * R + r_base) / 8 + qp % aq]
                                  : W[((r_base + q / kQ) * K) / 8 + kq + q % kQ];
                if (i < b_per) rb[i] = X[((m_base + q / kQ) * xs + pc.x_col0) / 8 + kq + q % kQ];
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
    // kFlagRound: the caller wants y rounded onto bf16 -- on the accumulator
    // here, so y is written once instead of copied through a rounding pass
    // (exactly the same values: round then store is store then round).
    const bool rnd = (pc.flags & kFlagRound) != 0u;
    [ForceUnroll] for (uint i = 0; i < kMaxW; ++i)
        [ForceUnroll] for (uint j = 0; j < kMaxW; ++j)
            if (i < wm && j < wn) {
                if (rnd)
                    for (uint e = 0; e < c[i * kMaxW + j].GetLength(); ++e)
                        c[i * kMaxW + j][e] = bf16_round(c[i * kMaxW + j][e]);
                c[i * kMaxW + j].Store<linalg.CoopMatMatrixLayout.ColumnMajor>(
                    Y, (pc.idx_off + m_base + w_m + 16 * j) * ys + pc.y_row0 + r_base + w_r + 16 * i, ys);
            }
}
