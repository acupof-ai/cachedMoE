// probe_occ -- why the real FP4 MoE GEMM sits at 14.2 TFLOP/s while probe_mma
// reaches 48.6 on the same machine (STATUS §7 0bt, 0bn). probe_mma holds its A
// and B tiles in registers and allocates no LDS, so it measures the matrix unit
// alone at full occupancy. The real kernel differs in exactly two ways that
// could cost 3.4x, and this probe separates them by adding them one at a time
// to the same WMMA loop:
//
//   mode 0  the probe_mma loop again, no LDS                    -- the control
//   mode 1  + 27,648 B of LDS allocated and live, never read in the loop.
//           That is `prefill_gemm_lds`'s wm4wn2 footprint, and a CU's 64 KB
//           then holds only 2 workgroups = 4 waves/SIMD. Pure occupancy.
//   mode 2  + A and B re-read from that LDS every iteration, which is what the
//           real kernel's K loop does. Occupancy AND the LDS read path.
//   mode 3  + the real tile ratio: wm4wn2 loads 4 A tiles and 2 B tiles to feed
//           8 MACs, where mode 2 loads 2 to feed 8, so mode 2 understates the
//           load side threefold.
//   mode 4  + the FP4 decode's VALU work. `tools/isa_blocks.py` counts about
//           791 non-WMMA instructions per K iteration per wave on the
//           specialised path against 8 WMMAs; this mode issues a dependent
//           v_fma chain of that length so the VALU and the matrix unit compete
//           the way they do in the real kernel. If this is where the rate
//           falls, the kernel is VALU-bound on unpacking FP4, not limited by
//           the matrix unit, LDS or occupancy -- and no API changes that.
//   mode 5  the same 791 ops as eight independent chains instead of one. Mode 4
//           is a serial dependency and therefore a worst case; mode 5 is the
//           best a perfect scheduler could do with the same instruction count.
//           The two together bracket what better VALU scheduling -- the one
//           thing a different compiler could plausibly offer -- is worth.
//
// Mode 1 against mode 0 prices occupancy; mode 2 against mode 1 prices the LDS
// reads. Both would still apply under any other API: the 64 KB LDS per CU and
// RDNA 3.5's 256-VGPR ceiling are architectural, and gfx1151 has no
// direct-to-LDS load instruction to stage the tiles with (verified with
// llvm-mc: `global_load_lds_*` and `buffer_load ... lds` assemble for gfx942 /
// gfx950 and for no gfx11 target).
//
// Wave32 required (the pipeline sets it): 256 threads are 8 waves.

[[vk::binding(0, 0)]] StructuredBuffer<uint4>   Src;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> Sink;
[[vk::binding(2, 0)]] StructuredBuffer<uint>    Chase;

typealias TileA = linalg.CoopMat<half,  MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA>;
typealias TileB = linalg.CoopMat<half,  MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB>;
typealias TileC = linalg.CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixAccumulator>;

struct OccPush {
    uint mode, n, per_group, base;
};
[[vk::push_constant]] OccPush pc;

// 27,648 B = prefill_gemm_lds's wm4wn2 footprint, as halves.
static const uint kLdsHalves = 27648 / 2;
groupshared half Stage[kLdsHalves];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    const half av = half(float(pc.per_group) * 1e-3f);
    const half bv = half(float(pc.base) * 1e-3f);

    // Fill the stage so the allocation cannot be dead-code eliminated. Values
    // come from the push constant, which the compiler cannot fold.
    if (pc.mode != 0) {
        for (uint i = lane; i < kLdsHalves; i += 256) Stage[i] = (i & 1) == 0 ? av : bv;
        GroupMemoryBarrierWithGroupSync();
    }

    TileC c[8];
    [ForceUnroll] for (uint t = 0; t < 8; ++t) c[t] = TileC(float(t));

    if (pc.mode >= 3) {
        const uint stride = 16;
        // 4 A + 2 B loads feeding 8 MACs, as prefill_gemm_lds's wm4wn2 does.
        float valu = float(pc.base) * 1e-3f;
        float vpar[8];
        [ForceUnroll] for (uint q = 0; q < 8; ++q) vpar[q] = float(pc.base + q) * 1e-3f;
        for (uint i = 0; i < pc.n; ++i) {
            const uint off = ((i * 256) % (kLdsHalves - 1024));
            TileA a[4];
            [ForceUnroll] for (uint t = 0; t < 4; ++t)
                a[t] = TileA.Load<linalg.CoopMatMatrixLayout.RowMajor>(Stage, off + t * 256, stride);
            TileB b[2];
            [ForceUnroll] for (uint t = 0; t < 2; ++t)
                b[t] = TileB.Load<linalg.CoopMatMatrixLayout.RowMajor>(Stage, off + t * 256, stride);
            [ForceUnroll] for (uint m = 0; m < 4; ++m)
                [ForceUnroll] for (uint nn = 0; nn < 2; ++nn)
                    c[m * 2 + nn] = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a[m], b[nn], c[m * 2 + nn]);
            if (pc.mode == 4) {
                // 791 dependent VALU ops, the specialised FP4 path's count.
                [ForceUnroll] for (uint k = 0; k < 791; ++k) valu = valu * 1.0000001f + 1e-7f;
            } else if (pc.mode == 5) {
                // The same count with eight independent chains.
                [ForceUnroll] for (uint k = 0; k < 99; ++k)
                    [ForceUnroll] for (uint q = 0; q < 8; ++q)
                        vpar[q] = vpar[q] * 1.0000001f + 1e-7f;
            }
        }
        if (lane == 0) {
            float acc = valu;
            [ForceUnroll] for (uint q = 0; q < 8; ++q) acc += vpar[q];
            Sink[2049] = acc;
        }
    } else if (pc.mode == 2) {
        // A and B come out of LDS every iteration, 16x16 row-major tiles, the
        // same shape the real K loop loads. Two tile loads feed eight MACs,
        // which is the real kernel's ratio at wm4wn2 as well.
        const uint stride = 16;
        for (uint i = 0; i < pc.n; ++i) {
            const uint off = ((i * 256) % (kLdsHalves - 256));
            const TileA a = TileA.Load<linalg.CoopMatMatrixLayout.RowMajor>(Stage, off, stride);
            const TileB b = TileB.Load<linalg.CoopMatMatrixLayout.RowMajor>(Stage, off, stride);
            [ForceUnroll] for (uint t = 0; t < 8; ++t)
                c[t] = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, b, c[t]);
        }
    } else {
        const TileA a = TileA(av);
        const TileB b = TileB(bv);
        for (uint i = 0; i < pc.n; ++i)
            [ForceUnroll] for (uint t = 0; t < 8; ++t)
                c[t] = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, b, c[t]);
    }

    [ForceUnroll] for (uint t = 0; t < 8; ++t)
        c[t].Store<linalg.CoopMatMatrixLayout.RowMajor>(Sink, ((lane / 32) * 8 + t) * 256, 16);
    // Keeps Stage live in mode 1, where the loop never reads it.
    if (pc.mode == 1 && lane == 0) Sink[2048] = float(Stage[pc.base % kLdsHalves]);
}
