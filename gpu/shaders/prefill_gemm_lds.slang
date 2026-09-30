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

// Every tile read out of LDS is four ds_load_b64 (STATUS §3 71): Mesa lowers
// shared loads to 64 bits at most ("for performance reasons"), so 128-bit
// reads cannot be had from a Vulkan shader, by hand or by the compiler --
// true of STOCK RADV only. STATUS §7 0bl patches our own Mesa build to lift
// that cap and every one of these becomes a ds_read_b128 (gate/up -5.6%); the
// shader has no toggle for it, the driver does (VK_DRIVER_FILES).
//
// LDS_BK: 64 halfs = 128 B a row a slice = exactly one cache line, and it is a
// local optimum pinned on BOTH sides for two different reasons (STATUS §3 86):
// 128 costs register pressure (§3 68, 81: +12%/+19% on the FP4 pair) and 32
// halves the read granularity below the line, so VRAM reads double (102.7 ->
// 200.8 MB at n=512) and the kernel is 40% slower even though occupancy goes
// from 2 to 4 waves a SIMD. Do not turn this knob again without a new reason.
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
[vk::constant_id(13)] const uint LdsGroupRows = 0; // grouped weight: rows a group, X [groups][n][K]
// LdsWFp4: W is a routed expert's FP4 E2M1 [R][K/2] with UE8M0 scales
// [R][K/32], part pc.part (0 w1, 1 w3, 2 w2) of job pc.job. The A tiles are
// decoded as they are staged -- the same f32tof16(ldexp(v, e)) that
// prefill_gemm stage 3 wrote into the fp16 transit, so the tiles and Y are bit
// for bit the decoded path's -- and the decode pass and its 2-byte plane go
// away (STATUS §7 0ba: 17K 2.9 s of DRAM-bound decodes).
[vk::constant_id(14)] const uint LdsWFp4 = 0;

typealias TileA = linalg.CoopMat<half,  MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA>;
typealias TileB = linalg.CoopMat<half,  MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB>;
typealias TileC = linalg.CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixAccumulator>;

static const uint kW = 0, kX = 1, kY = 2, kJob = 7;
static const uint kBk = LDS_BK;           // K slice, in halfs
static const uint kQ = kBk / 8;           // uint4 of data per LDS row
// LDS_SPLIT: stage the slice into LDS in two halves and run two sub-slices of
// MMA per global read (STATUS §7 0br). The global read stays a whole kBk row --
// 128 B at kBk 64, exactly one cache line, which is what §3 86 showed must not
// shrink -- while the LDS footprint halves, so the CU holds twice the
// workgroups. The cost is twice the barriers and a prefetch hidden by half as
// many MMA steps, because ra/rb must survive until the second half is staged.
#ifndef LDS_SPLIT
#define LDS_SPLIT 0
#endif
static const uint kHalves = LDS_SPLIT ? 2u : 1u;
static const uint kQh = kQ / kHalves;     // uint4 of data per LDS row, per half
static const uint kRowQ = kQh + 1;        // uint4 per LDS row: kQh of data + 1 of padding
static const uint kMaxW = 4;              // LdsWm, LdsWn <= 4
static const uint kThreads = 128;
static const uint kRegs = kMaxW * 32 * kQ / kThreads;   // a slice's uint4 per thread at the widest block

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct CoopPush {
    uint n, k, idx_off, flags, row0, x_off, x_stride, x_col0, y_stride, y_row0;
    uint rope_pos0, rope_hd, rope_dim;   // prefill_coopmat stage 2's
    uint job, part;                      // LdsWFp4: the expert and its matrix (3 = w1 and w3 in one dispatch,
    uint up_off;                         //   row blocks past R are w3's, their y tokens from up_off)
};
[[vk::push_constant]] CoopPush pc;

// Eight halfs of FP4 row `row` from column k (a multiple of 8), decoded and
// packed as the fp16 transit holds them.
uint4 load_a_fp4(uint64_t W4, uint64_t S4, uint row, uint k) {
    const uint nib = load_u32(W4 + (uint64_t)row * (LdsCols / 2) + k / 2);
    const int  e   = int(load_u8(S4 + (uint64_t)row * (LdsCols / 32) + k / 32)) - 127;
    uint h[8];
    // (integer construction of the half's bits: exact for |e| <= 13 but 9 ops
    // an element against 4, gate/up 6.7 -> 9.5 s at 17K; the loads alone
    // cost nothing against the fp16 plane's, the decode 0.7 s a prefill)
    [ForceUnroll] for (uint j = 0; j < 8; ++j)
        h[j] = f32tof16(ldexp(kE2M1[(nib >> (4 * j)) & 0xFu], e)) & 0xFFFFu;
    return uint4(h[0] | (h[1] << 16), h[2] | (h[3] << 16), h[4] | (h[5] << 16), h[6] | (h[7] << 16));
}

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
    const uint rblk = batched ? gid.x : gid.y;
    const bool up = LdsWFp4 && pc.part == 3u && rblk * bm >= R;   // the w3 half of a gate/up dispatch
    const uint r_base = pc.row0 + rblk * bm - (up ? R : 0u);
    const uint m_base = pc.x_off + (batched ? 0u : gid.x * bn);
    const uint64_t batch = batched ? gid.y : 0u;
    uint4* W = (uint4*)(Ptr[kW] + batch * R * K * 2);
    uint64_t W4 = 0, S4 = 0;
    if (LdsWFp4) {
        const Job jb = load_job(Ptr[kJob], pc.job);
        const uint part = (pc.part == 3u) ? (up ? 1u : 0u) : pc.part;
        W4 = (part == 0u) ? jb.w1 : ((part == 1u) ? jb.w3 : jb.w2);
        S4 = (part == 0u) ? jb.s1 : ((part == 1u) ? jb.s3 : jb.s2);
    }
    const uint y_tok = pc.idx_off + (up ? pc.up_off : 0u);
    uint4* X = (uint4*)(Ptr[kX] + batch * bn * xs * 2);
    float* Y = (float*)(Ptr[kY] + batch * bn * ys * 4);
    // A grouped weight's row block multiplies its own group's slice of X,
    // staged [groups][n][K] (one dispatch over every group, STATUS §7 0aq).
    const uint xc = pc.x_col0 + (LdsGroupRows ? (r_base / LdsGroupRows) * pc.n * K : 0u);

    // This thread's share of a slice: A is bm rows x 8 uint4 (LdsWt: pairs
    // of k-rows, 2i and 2i + 1, of the same 8 columns), B bn rows x 8.
    const uint a_per = bm * kQ / kThreads, b_per = bn * kQ / kThreads;
    const uint aq = bm / 8;                    // LdsWt: uint4 per k-row of the slice
    uint4 ra[kRegs], rb[kRegs];
    [ForceUnroll] for (uint i = 0; i < kRegs; ++i) {
        if (i < a_per) {
            const uint q = tid + i * kThreads, qp = tid + (i / 2) * kThreads;
            ra[i] = LdsWt   ? W[((2 * (qp / aq) + i % 2) * R + r_base) / 8 + qp % aq]
                  : LdsWFp4 ? load_a_fp4(W4, S4, r_base + q / kQ, (q % kQ) * 8)
                            : W[((r_base + q / kQ) * K) / 8 + q % kQ];
        }
        if (i < b_per) { const uint q = tid + i * kThreads; rb[i] = X[((m_base + q / kQ) * xs + xc) / 8 + q % kQ]; }
    }

    const uint wave = tid / 32;
    const uint w_r = (wave / 2) * 16 * wm;     // this wave's first row in the block
    const uint w_m = (wave % 2) * 16 * wn;     // and first token
    TileC c[kMaxW * kMaxW];
    [ForceUnroll] for (uint t = 0; t < kMaxW * kMaxW; ++t) c[t] = TileC(0.0f);

    for (uint k0 = 0; k0 < K; k0 += kBk) {
      [ForceUnroll] for (uint h = 0; h < kHalves; ++h) {
        GroupMemoryBarrierWithGroupSync();     // the last (sub)slice's reads are done
        [ForceUnroll] for (uint i = 0; i < kRegs; ++i) {
            const uint q = tid + i * kThreads;
            // Each register holds one (row, column) of the slice, so it belongs
            // to exactly one half: column q % kQ, half (q % kQ) / kQh.
            const bool mine = kHalves == 1u || (q % kQ) / kQh == h;
            if (i < a_per && !LdsWt && mine) As[(q / kQ) * kRowQ + (q % kQ) % kQh] = ra[i];
            if (i < a_per && LdsWt && i % 2 == 0 && kHalves == 1u) {
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
            if (i < b_per && mine) Bs[(q / kQ) * kRowQ + (q % kQ) % kQh] = rb[i];
        }
        GroupMemoryBarrierWithGroupSync();
        // The prefetch overwrites ra/rb, so with two halves it cannot be issued
        // until the second half has been staged.
        if (k0 + kBk < K && h + 1 == kHalves) {
            const uint kq = (k0 + kBk) / 8, k1 = k0 + kBk;
            [ForceUnroll] for (uint i = 0; i < kRegs; ++i) {
                const uint q = tid + i * kThreads, qp = tid + (i / 2) * kThreads;
                if (i < a_per)
                    ra[i] = LdsWt   ? W[((k1 + 2 * (qp / aq) + i % 2) * R + r_base) / 8 + qp % aq]
                          : LdsWFp4 ? load_a_fp4(W4, S4, r_base + q / kQ, k1 + (q % kQ) * 8)
                                    : W[((r_base + q / kQ) * K) / 8 + kq + q % kQ];
                if (i < b_per) rb[i] = X[((m_base + q / kQ) * xs + xc) / 8 + kq + q % kQ];
            }
        }
        [ForceUnroll] for (uint kk = 0; kk < kBk / (16 * kHalves); ++kk) {
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
                    Y, (y_tok + m_base + w_m + 16 * j) * ys + pc.y_row0 + r_base + w_r + 16 * i, ys);
            }
}
