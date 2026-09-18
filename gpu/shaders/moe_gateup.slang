// moe_gateup -- fused gate/up + SwiGLU for the experts of one MoE layer
// (design §7.9, dispatch A; #10 of the §7.14 list).
//
// One workgroup owns (expert slot, row block). A row block is
// kWgSize / LanesPerRow row groups of RowsPerLane rows each; the LanesPerRow
// lanes of a row group walk those rows' 5120 K-elements in 32-element UE8M0
// blocks, 16 B of packed FP4 (or 32 B of FP8) per lane per row per block.
//
// Two independent reuse knobs, both "read the expensive operand once":
//   M           activation columns per lane -- weights are read once and used
//               M times (design §7.1 rule 7). M=1 is decode, M=6 a full
//               speculative verify batch.
//   RowsPerLane weight rows per lane -- x is read once and used RowsPerLane
//               times. Without it a workgroup re-reads all of x once per row
//               group, which is 1.9x the weight traffic in L0 requests even
//               though it never leaves cache.
//
// The set of experts to compute is an indirection list, not 0..6: design §7.9's
// "compute the three that arrived first" only needs a shorter SlotList and a
// smaller gid.y, with no change to the kernel.
//
// Three things P1 left open and P2 adds (docs/kernel_p2_moe.md):
//
//   XMode      v0.5 read x straight from global for every (m, 8 elements),
//              which at M=6 is 24 activation loads against 2 weight loads per
//              block -- the measured M=6 bottleneck (kernel_p1.md §3.3).
//              XMode >= 1 stages a K-tile of x (LanesPerRow blocks, all M
//              columns) in LDS once per workgroup and every row group reads it
//              from there; XMode 2 and 3 additionally change the arithmetic to
//              packed fp16 or to int8 dot4.
//   HQuant     design §7.9 v0.6: `silu(gate)*up` is fp8-quantised before w2.
//              HQuant == 2 does it here, writing fp8 h plus a UE8M0 scale
//              plane; HQuant == 1 leaves it to dispatch B; HQuant == 3 leaves
//              it to gpu/shaders/moe_hquant.slang and changes nothing here,
//              which is how M=1 keeps L32 R1 (docs/kernel_p2_moe.md §8 item 1).
//   Fp8Slots   the shared expert is fp8 E4M3 with a [rows/32][K/32] scale
//              plane. Its slot carries kSlotFp8Bit in Ids, so the same
//              dispatch covers both weight formats.

#include "moe_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> ExpertAddr;  // pointer table (design §5.3)
[[vk::binding(1, 0)]] StructuredBuffer<uint>     Ids;         // Ids[slot] = expert id | fp8 bit
[[vk::binding(2, 0)]] StructuredBuffer<uint>     SlotList;    // compute index -> slot
[[vk::binding(3, 0)]] StructuredBuffer<float>    RouteW;      // [M][slots] routing weights
[[vk::binding(4, 0)]] StructuredBuffer<uint4>    X;           // x[M][K] fp16, 8 per uint4
[[vk::binding(5, 0)]] RWStructuredBuffer<half>   H16;         // h[M][slots][N] fp16
[[vk::binding(6, 0)]] RWStructuredBuffer<float>  H32;         // h[M][slots][N] fp32
[[vk::binding(7, 0)]] RWStructuredBuffer<uint>   HU;          // the same allocation, raw words
[[vk::binding(8, 0)]] StructuredBuffer<uint>     XU;          // x, raw words (XMode 6's int8 tail)

struct GateUpPush {
    uint layer;
    uint experts_per_layer;
    uint num_slots;      // stride of the h/RouteW slot dimension
    uint n_rows;         // 2304
    uint k;              // 5120
    float swiglu_limit;  // 10.0 (design §2.4)
    // Track T / DSpark: live activation columns. M is the specialisation
    // constant (the array bound), `m` is how many columns this dispatch actually
    // computes -- 1 for a decode token, the batch size for a verify batch. It is
    // the tail of the struct so the first six words keep their offsets.
    uint m;
};
[[vk::push_constant]] GateUpPush pc;

// Track K1a: the trip count of every column loop below. At M == 1 the host
// specialises StaticM to 1 and this folds to a literal, which is what the
// pre-fb53514 `m < M` loops were; StaticM == 0 keeps Track T's push-constant
// mask for any M > 1 caller. See moe_common.slang's StaticM.
#define kLiveM ((StaticM != 0u) ? StaticM : pc.m)

static const uint kRowsPerWg = kWgSize / LanesPerRow;
static const uint kWgRows    = kRowsPerWg * RowsPerLane;

// LDS. Every array is sized from specialisation constants, so a variant that
// does not use one pays a single word for it and the occupancy of the P1
// kernel is unchanged when XMode == 0, HQuant != 2 and Fp8Slots == 0.
groupshared float sRed[kWgSize];
static const uint kXTileU  = (XMode == 0) ? 1u
                           : ((XMode == 3) ? (M * LanesPerRow * 8u) : (M * LanesPerRow * 16u));
groupshared uint  sX[kXTileU];
static const uint kXScaleN = (XMode == 3) ? (M * LanesPerRow) : 1u;
groupshared float sXS[kXScaleN];
static const uint kFp8TblN = (Fp8Slots == 0) ? 1u : kFp8TableN;
groupshared float sFp8[kFp8TblN];
static const uint kHqN     = (HQuant == 2) ? (M * kWgRows) : 1u;
groupshared float sHq[kHqN];
// XMode 6's weight decode. Measuring the first version of the pre-quantised
// int8 path showed dispatch A was not limited by the activation at all any
// more: with 8 dot4add_i8packed per (column, block, row) against
// fp4_nibbles_to_i8x4 called 8 times per (block, row, matrix), the FP4 -> int8
// weight decode is ~3/4 of the VALU work, which is why docs/kernel_p2_moe.md
// §3.6's model (which counted only the activation side) came out too
// optimistic. One byte of packed FP4 is two elements, so a 256-entry table
// turns the whole decode into two LDS reads and a shift-or per output word.
static const uint kI8TblN = (XMode == 6) ? kFp8TableN : 1u;
groupshared uint sI8[kI8TblN];

// Four consecutive elements of `pw` as packed signed int8, through that table.
uint fp4_i8x4_tbl(uint pw, uint g) {
    const uint n = pw >> (16u * g);
    return sI8[n & 0xFFu] | (sI8[(n >> 8) & 0xFFu] << 16);
}

// Sum `v` across the LanesPerRow lanes of one row group and broadcast it.
// An LDS tree rather than WaveActiveSum: LanesPerRow is independent of the
// subgroup size (a 64-lane row spans two wave32 subgroups), and at
// RowsPerLane x M calls per workgroup the barriers do not show up in the
// bandwidth measurement.
float row_reduce(float v, uint tid, uint sublane) {
    sRed[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = LanesPerRow >> 1; s > 0; s >>= 1) {
        if (sublane < s) sRed[tid] = sRed[tid] + sRed[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float r = sRed[tid - sublane];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Stage x[0..M) over blocks [base, base + LanesPerRow) into LDS. Called by the
// whole workgroup between two barriers, so it must sit in uniform control flow.
void stage_x(uint base, uint tid) {
    const uint xu4 = pc.k / 8;                  // uint4 entries per column of x
    if (XMode == 3) {
        // One (m, block) per thread: read the 32 halves, take the block amax,
        // and write 32 int8 plus the block scale. M * LanesPerRow <= 256.
        const uint pairs = M * LanesPerRow;
        if (tid < pairs) {
            const uint m   = tid / LanesPerRow;
            const uint b   = tid % LanesPerRow;
            const uint blk = base + b;
            float v[32];
            [unroll] for (uint c = 0; c < 4; ++c) {
                const uint gi = m * xu4 + blk * 4 + c;
                const uint4 w = (blk * 4 + c < xu4) ? X[gi] : uint4(0, 0, 0, 0);
                [unroll] for (uint j = 0; j < 8; ++j) v[c * 8 + j] = unpack_half(w, j);
            }
            float amax = 0.0f;
            [unroll] for (uint j = 0; j < 32; ++j) amax = max(amax, abs(v[j]));
            const float s    = max(amax, 1e-30f) * (1.0f / 127.0f);
            const float inv  = 1.0f / s;
            sXS[m * LanesPerRow + b] = s;
            [unroll] for (uint u = 0; u < 8; ++u) {
                uint packed = 0;
                [unroll] for (uint j = 0; j < 4; ++j) {
                    const int q = clamp(int(floor(v[u * 4 + j] * inv + 0.5f)), -127, 127);
                    packed |= (uint(q) & 0xFFu) << (8 * j);
                }
                sX[m * LanesPerRow * 8 + b * 8 + u] = packed;
            }
        }
        return;
    }
    // XMode 1 and 2 keep the fp16 bits and stage them as raw words.
    const uint total = M * LanesPerRow * 4u;    // uint4 entries in the tile
    for (uint i = tid; i < total; i += kWgSize) {
        const uint m = i / (LanesPerRow * 4u);
        const uint t = i % (LanesPerRow * 4u);
        const uint4 w = (base * 4 + t < xu4) ? X[m * xu4 + base * 4 + t] : uint4(0, 0, 0, 0);
        const uint o = m * LanesPerRow * 16u + t * 4u;
        sX[o + 0] = w.x; sX[o + 1] = w.y; sX[o + 2] = w.z; sX[o + 3] = w.w;
    }
}

// The 8 activation values of column `m`, element group `g` of block `b`, from
// wherever XMode put them.
void read_x8(uint m, uint b, uint blk, uint g, out float v[8]) {
    if (XMode == 0) {
        const uint4 xr = X[m * (pc.k / 8) + blk * 4 + g];
        [unroll] for (uint j = 0; j < 8; ++j) v[j] = unpack_half(xr, j);
    } else {
        const uint o = m * LanesPerRow * 16u + b * 16u + g * 4u;
        [unroll] for (uint j = 0; j < 8; ++j)
            v[j] = f16tof32(((j & 1) != 0) ? (sX[o + (j >> 1)] >> 16) : (sX[o + (j >> 1)] & 0xFFFFu));
    }
}

// The same eight values as four half2, for the packed-fp16 path.
void read_x8_h2(uint m, uint b, uint g, out float16_t2 v[4]) {
    const uint o = m * LanesPerRow * 16u + b * 16u + g * 4u;
    [unroll] for (uint j = 0; j < 4; ++j) v[j] = bit_cast<float16_t2>(sX[o + j]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint sublane = tid % LanesPerRow;
    const uint rowInWg = tid / LanesPerRow;
    const uint rowBase = (gid.x * kRowsPerWg + rowInWg) * RowsPerLane;

    // One expert set for the whole dispatch. A verify batch whose columns route
    // to DIFFERENT experts needs the two-index form written up in
    // docs/p4_dspark_runtime.md §2.2 -- TokenIds[m][slot] resolving to an expert
    // id, then that id to a table row, the shared expert's row being the case
    // that forces the second index -- and it is not implemented.
    const uint slot   = SlotList[gid.y];
    const uint rawId  = Ids[slot];
    const bool isFp8  = (Fp8Slots != 0) && ((rawId & kSlotFp8Bit) != 0);
    const uint expert = rawId & ~kSlotFp8Bit;
    const uint tbase  = (pc.layer * pc.experts_per_layer + expert) * 6;

    const uint64_t w1Base = ExpertAddr[tbase + 0];
    const uint64_t s1Base = ExpertAddr[tbase + 1];
    const uint64_t w3Base = ExpertAddr[tbase + 4];
    const uint64_t s3Base = ExpertAddr[tbase + 5];

    if (Fp8Slots != 0) {
        // design §7.1 rule 5: a 256-entry table in LDS, built once per
        // workgroup by 256 threads doing one arithmetic decode each.
        if (tid < kFp8TableN) sFp8[tid] = fp8_e4m3_decode(tid);
        GroupMemoryBarrierWithGroupSync();
    }
    if (XMode == 6) {
        // byte of packed FP4 -> the two 2*E2M1 int8 it holds, low nibble first
        // (design appendix A). Same shape as the fp8 table above.
        if (tid < kFp8TableN)
            sI8[tid] = (uint(kE2M1x2[tid & 15u]) & 0xFFu)
                     | ((uint(kE2M1x2[tid >> 4]) & 0xFFu) << 8);
        GroupMemoryBarrierWithGroupSync();
    }

    const uint blocks   = pc.k / kBlockK;                    // 160
    const uint rowBytes = isFp8 ? pc.k : (pc.k / 2);         // 5120 or 2560 per row
    // fp4: one scale row per weight row. fp8: one per 32 weight rows.
    const uint scaleRow = isFp8 ? (rowBase >> 5) : rowBase;
    const bool live = (rowBase + RowsPerLane <= pc.n_rows) && (w1Base != 0) && (w3Base != 0);
    // The LDS tile is only staged for FP4 slots: the fp8 shared expert is one
    // slot of seven and keeps the simple global-x path. XMode 4 and 5 change
    // only the arithmetic, so they take the un-tiled path too.
    const bool tiled = x_mode_tiles() && !isFp8;

    float accGate[RowsPerLane][M];
    float accUp[RowsPerLane][M];
    for (uint i = 0; i < RowsPerLane; ++i)
        for (uint j = 0; j < M; ++j) { accGate[i][j] = 0.0f; accUp[i][j] = 0.0f; }

    // One iteration of the block loop, for whichever of the two addressings.
    // `b` is the block's index inside the staged tile (meaningless when !tiled).
    if (tiled) {
        for (uint base = 0; base < blocks; base += LanesPerRow) {
            GroupMemoryBarrierWithGroupSync();
            stage_x(base, tid);
            GroupMemoryBarrierWithGroupSync();
            const uint blk = base + sublane;
            if (live && blk < blocks) {
            uint4 w1q[RowsPerLane];
            uint4 w3q[RowsPerLane];
            int   e1[RowsPerLane];
            int   e3[RowsPerLane];
            for (uint r = 0; r < RowsPerLane; ++r) {
                const uint64_t off = (uint64_t)(rowBase + r) * rowBytes + (uint64_t)blk * 16;
                const uint64_t sof = (uint64_t)(scaleRow + r) * blocks + blk;
                w1q[r] = load_fp4_block(w1Base + off);
                w3q[r] = load_fp4_block(w3Base + off);
                e1[r]  = load_scale_exp(s1Base + sof);
                e3[r]  = load_scale_exp(s3Base + sof);
            }

            float p1[RowsPerLane][M];
            float p3[RowsPerLane][M];
            for (uint i = 0; i < RowsPerLane; ++i)
                for (uint j = 0; j < M; ++j) { p1[i][j] = 0.0f; p3[i][j] = 0.0f; }

            if (XMode == 3) {
                // int8 dot4: 8 dot4add per 32 elements per column instead of
                // 32 fp32 FMAs. 2*E2M1 is an exact int8, so the factor 2 rides
                // in the block exponent (e - 128 instead of e - 127).
                for (uint r = 0; r < RowsPerLane; ++r) {
                    int a1[M];
                    int a3[M];
                    for (uint j = 0; j < M; ++j) { a1[j] = 0; a3[j] = 0; }
                    [unroll] for (uint w = 0; w < 4; ++w) {
                        const uint pw1 = w1q[r][w];
                        const uint pw3 = w3q[r][w];
                        [unroll] for (uint g = 0; g < 2; ++g) {
                            const uint d1 = fp4_nibbles_to_i8x4(pw1, g);
                            const uint d3 = fp4_nibbles_to_i8x4(pw3, g);
                            for (uint m = 0; m < kLiveM; ++m) {
                                const uint xw = sX[m * LanesPerRow * 8 + sublane * 8 + w * 2 + g];
                                a1[m] = dot4add_i8packed(d1, xw, a1[m]);
                                a3[m] = dot4add_i8packed(d3, xw, a3[m]);
                            }
                        }
                    }
                    for (uint m = 0; m < kLiveM; ++m) {
                        const float s = sXS[m * LanesPerRow + sublane];
                        p1[r][m] = float(a1[m]) * s;
                        p3[r][m] = float(a3[m]) * s;
                    }
                }
                for (uint r = 0; r < RowsPerLane; ++r)
                    for (uint m = 0; m < kLiveM; ++m) {
                        accGate[r][m] += ldexp(p1[r][m], e1[r] - 1);
                        accUp[r][m]   += ldexp(p3[r][m], e3[r] - 1);
                    }
            } else {
            [unroll] for (uint w = 0; w < 4; ++w) {
                if (XMode == 2) {
                    float16_t2 g1[RowsPerLane][4];
                    float16_t2 g3[RowsPerLane][4];
                    for (uint r = 0; r < RowsPerLane; ++r) {
                        const uint pw1 = w1q[r][w];
                        const uint pw3 = w3q[r][w];
                        [unroll] for (uint j = 0; j < 4; ++j) {
                            g1[r][j] = float16_t2(float16_t(fp4_decode((pw1 >> (8 * j)) & 0xFu)),
                                                  float16_t(fp4_decode((pw1 >> (8 * j + 4)) & 0xFu)));
                            g3[r][j] = float16_t2(float16_t(fp4_decode((pw3 >> (8 * j)) & 0xFu)),
                                                  float16_t(fp4_decode((pw3 >> (8 * j + 4)) & 0xFu)));
                        }
                    }
                    for (uint m = 0; m < kLiveM; ++m) {
                        float16_t2 xv[4];
                        read_x8_h2(m, sublane, w, xv);
                        for (uint r = 0; r < RowsPerLane; ++r) {
                            float16_t2 t1 = float16_t2(float16_t(0.0f), float16_t(0.0f));
                            float16_t2 t3 = float16_t2(float16_t(0.0f), float16_t(0.0f));
                            [unroll] for (uint j = 0; j < 4; ++j) {
                                t1 += g1[r][j] * xv[j];
                                t3 += g3[r][j] * xv[j];
                            }
                            p1[r][m] += float(t1.x) + float(t1.y);
                            p3[r][m] += float(t3.x) + float(t3.y);
                        }
                    }
                } else {
                    float d1[RowsPerLane][8];
                    float d3[RowsPerLane][8];
                    for (uint r = 0; r < RowsPerLane; ++r) {
                        const uint pw1 = w1q[r][w];
                        const uint pw3 = w3q[r][w];
                        [unroll] for (uint j = 0; j < 8; ++j) {
                            d1[r][j] = fp4_decode((pw1 >> (4 * j)) & 0xFu);
                            d3[r][j] = fp4_decode((pw3 >> (4 * j)) & 0xFu);
                        }
                    }
                    for (uint m = 0; m < kLiveM; ++m) {
                        float xv[8];
                        read_x8(m, sublane, blk, w, xv);
                        [unroll] for (uint j = 0; j < 8; ++j)
                            for (uint r = 0; r < RowsPerLane; ++r) {
                                p1[r][m] += d1[r][j] * xv[j];
                                p3[r][m] += d3[r][j] * xv[j];
                            }
                    }
                }
            }
            for (uint r = 0; r < RowsPerLane; ++r)
                for (uint m = 0; m < kLiveM; ++m) {
                    accGate[r][m] += ldexp(p1[r][m], e1[r]);
                    accUp[r][m]   += ldexp(p3[r][m], e3[r]);
                }
            }  // XMode != 3
            }  // live && blk < blocks
        }
    } else if (live) {
        for (uint blk = sublane; blk < blocks; blk += LanesPerRow) {
            // fp8 reads 32 B per block per row, fp4 16 B.
            uint4 w1q[RowsPerLane], w1r[RowsPerLane];
            uint4 w3q[RowsPerLane], w3r[RowsPerLane];
            int   e1[RowsPerLane];
            int   e3[RowsPerLane];
            for (uint r = 0; r < RowsPerLane; ++r) {
                const uint64_t off = (uint64_t)(rowBase + r) * rowBytes
                                   + (uint64_t)blk * (isFp8 ? 32 : 16);
                const uint64_t sof = (uint64_t)(scaleRow + (isFp8 ? 0 : r)) * blocks + blk;
                w1q[r] = load_raw16(w1Base + off);
                w3q[r] = load_raw16(w3Base + off);
                if (isFp8) {
                    w1r[r] = load_raw16(w1Base + off + 16);
                    w3r[r] = load_raw16(w3Base + off + 16);
                }
                e1[r] = load_scale_exp(s1Base + sof);
                e3[r] = load_scale_exp(s3Base + sof);
            }

            float p1[RowsPerLane][M];
            float p3[RowsPerLane][M];
            for (uint i = 0; i < RowsPerLane; ++i)
                for (uint j = 0; j < M; ++j) { p1[i][j] = 0.0f; p3[i][j] = 0.0f; }

            if (XMode == 4 && !isFp8) {
                // Packed fp16 straight from global: the activation keeps its
                // fp16 bit pattern all the way to v_pk_fma_f16, so the 32
                // f16tof32 and 64 FMA per (column, block) of the fp32 path
                // become 16 packed FMA and nothing else.
                // The half2 accumulator lives for the whole 32-element block,
                // not for each group of 8: reducing per group would cost four
                // fp16->fp32 conversions and two adds per (row, column, group),
                // which on its own is as large as the FMA stream it saves.
                float16_t2 t1[RowsPerLane][M];
                float16_t2 t3[RowsPerLane][M];
                for (uint i = 0; i < RowsPerLane; ++i)
                    for (uint j = 0; j < M; ++j) {
                        t1[i][j] = float16_t2(float16_t(0.0f), float16_t(0.0f));
                        t3[i][j] = float16_t2(float16_t(0.0f), float16_t(0.0f));
                    }
                [unroll] for (uint w = 0; w < 4; ++w) {
                    float16_t2 g1[RowsPerLane][4];
                    float16_t2 g3[RowsPerLane][4];
                    for (uint r = 0; r < RowsPerLane; ++r) {
                        const uint pw1 = w1q[r][w];
                        const uint pw3 = w3q[r][w];
                        [unroll] for (uint j = 0; j < 4; ++j) {
                            g1[r][j] = fp4_pair_h2(pw1, j);
                            g3[r][j] = fp4_pair_h2(pw3, j);
                        }
                    }
                    for (uint m = 0; m < kLiveM; ++m) {
                        const uint4 xr = X[m * (pc.k / 8) + blk * 4 + w];
                        float16_t2 xv[4];
                        [unroll] for (uint j = 0; j < 4; ++j) xv[j] = bit_cast<float16_t2>(xr[j]);
                        for (uint r = 0; r < RowsPerLane; ++r)
                            [unroll] for (uint j = 0; j < 4; ++j) {
                                t1[r][m] += g1[r][j] * xv[j];
                                t3[r][m] += g3[r][j] * xv[j];
                            }
                    }
                }
                for (uint r = 0; r < RowsPerLane; ++r)
                    for (uint m = 0; m < kLiveM; ++m) {
                        p1[r][m] = float(t1[r][m].x) + float(t1[r][m].y);
                        p3[r][m] = float(t3[r][m].x) + float(t3[r][m].y);
                    }
            } else if (XMode == 6 && !isFp8) {
                // The activation is already int8 with a per-block-32 scale
                // (gpu/shaders/moe_xquant.slang), so the lane's whole cost per
                // (column, block) is 8 loads and 16 dot4add_i8packed for every
                // RowsPerLane weight rows -- no f16tof32, no amax, no rounding.
                // 2*E2M1 is the exact int8, hence the `esh` shift below.
                uint q1[RowsPerLane][8];
                uint q3[RowsPerLane][8];
                for (uint r = 0; r < RowsPerLane; ++r)
                    [unroll] for (uint w = 0; w < 4; ++w)
                        [unroll] for (uint g = 0; g < 2; ++g) {
                            q1[r][w * 2 + g] = fp4_i8x4_tbl(w1q[r][w], g);
                            q3[r][w * 2 + g] = fp4_i8x4_tbl(w3q[r][w], g);
                        }
                const uint xElems = M * pc.k;
                const uint xBase4 = xq_value_words(xElems) / 4;   // in uint4
                const uint xScale = xq_scale_words(xElems);
                for (uint m = 0; m < kLiveM; ++m) {
                    const float sx = asfloat(XU[xScale + m * blocks + blk]);
                    // 32 int8 are two uint4, i.e. two global_load_dwordx4 --
                    // the same instruction count as one fp16 element group of
                    // the fp32 path, for the whole block.
                    const uint4 xa = X[xBase4 + m * (pc.k / 16) + blk * 2];
                    const uint4 xb = X[xBase4 + m * (pc.k / 16) + blk * 2 + 1];
                    uint xq[8];
                    [unroll] for (uint j = 0; j < 4; ++j) { xq[j] = xa[j]; xq[4 + j] = xb[j]; }
                    for (uint r = 0; r < RowsPerLane; ++r) {
                        int a1 = 0, a3 = 0;
                        [unroll] for (uint u = 0; u < 8; ++u) {
                            a1 = dot4add_i8packed(q1[r][u], xq[u], a1);
                            a3 = dot4add_i8packed(q3[r][u], xq[u], a3);
                        }
                        p1[r][m] += float(a1) * sx;
                        p3[r][m] += float(a3) * sx;
                    }
                }
            } else if (XMode == 5 && !isFp8) {
                // int8 dot4 straight from global (design §7.9's "int8 dot4 as
                // the M>=4 fallback"): the weights are decoded to packed int8
                // once per block and every RowsPerLane row then costs 8
                // dot4add_i8packed per column instead of 64 FMA.
                uint q1[RowsPerLane][8];
                uint q3[RowsPerLane][8];
                for (uint r = 0; r < RowsPerLane; ++r)
                    [unroll] for (uint w = 0; w < 4; ++w)
                        [unroll] for (uint g = 0; g < 2; ++g) {
                            q1[r][w * 2 + g] = fp4_nibbles_to_i8x4(w1q[r][w], g);
                            q3[r][w * 2 + g] = fp4_nibbles_to_i8x4(w3q[r][w], g);
                        }
                for (uint m = 0; m < kLiveM; ++m) {
                    float xv[32];
                    [unroll] for (uint w = 0; w < 4; ++w) {
                        const uint4 xr = X[m * (pc.k / 8) + blk * 4 + w];
                        [unroll] for (uint j = 0; j < 8; ++j) xv[w * 8 + j] = unpack_half(xr, j);
                    }
                    float amax = 0.0f;
                    [unroll] for (uint j = 0; j < 32; ++j) amax = max(amax, abs(xv[j]));
                    const float sx  = max(amax, 1e-30f) * (1.0f / 127.0f);
                    const float inv = 1.0f / sx;
                    uint xq[8];
                    [unroll] for (uint u = 0; u < 8; ++u) {
                        uint packed = 0;
                        [unroll] for (uint j = 0; j < 4; ++j) {
                            const int q = clamp(int(floor(xv[u * 4 + j] * inv + 0.5f)), -127, 127);
                            packed |= (uint(q) & 0xFFu) << (8 * j);
                        }
                        xq[u] = packed;
                    }
                    for (uint r = 0; r < RowsPerLane; ++r) {
                        int a1 = 0, a3 = 0;
                        [unroll] for (uint u = 0; u < 8; ++u) {
                            a1 = dot4add_i8packed(q1[r][u], xq[u], a1);
                            a3 = dot4add_i8packed(q3[r][u], xq[u], a3);
                        }
                        p1[r][m] += float(a1) * sx;
                        p3[r][m] += float(a3) * sx;
                    }
                }
            } else
            [unroll] for (uint w = 0; w < 4; ++w) {
                float d1[RowsPerLane][8];
                float d3[RowsPerLane][8];
                for (uint r = 0; r < RowsPerLane; ++r) {
                    if (isFp8) {
                        const uint a = (w < 2) ? w1q[r][2 * w] : w1r[r][2 * w - 4];
                        const uint b = (w < 2) ? w1q[r][2 * w + 1] : w1r[r][2 * w - 3];
                        const uint c = (w < 2) ? w3q[r][2 * w] : w3r[r][2 * w - 4];
                        const uint e = (w < 2) ? w3q[r][2 * w + 1] : w3r[r][2 * w - 3];
                        [unroll] for (uint j = 0; j < 4; ++j) {
                            d1[r][j]     = sFp8[(a >> (8 * j)) & 0xFFu];
                            d1[r][4 + j] = sFp8[(b >> (8 * j)) & 0xFFu];
                            d3[r][j]     = sFp8[(c >> (8 * j)) & 0xFFu];
                            d3[r][4 + j] = sFp8[(e >> (8 * j)) & 0xFFu];
                        }
                    } else {
                        const uint pw1 = w1q[r][w];
                        const uint pw3 = w3q[r][w];
                        [unroll] for (uint j = 0; j < 8; ++j) {
                            d1[r][j] = fp4_decode((pw1 >> (4 * j)) & 0xFu);
                            d3[r][j] = fp4_decode((pw3 >> (4 * j)) & 0xFu);
                        }
                    }
                }
                for (uint m = 0; m < kLiveM; ++m) {
                    const uint4 xr = X[m * (pc.k / 8) + blk * 4 + w];
                    [unroll] for (uint j = 0; j < 8; ++j) {
                        const float xv = unpack_half(xr, j);
                        for (uint r = 0; r < RowsPerLane; ++r) {
                            p1[r][m] += d1[r][j] * xv;
                            p3[r][m] += d3[r][j] * xv;
                        }
                    }
                }
            }
            // 2*E2M1 is the exact int8, so the int8 path's block exponent is
            // one lower than the fp4 path's.
            const int esh = ((XMode == 5 || XMode == 6) && !isFp8) ? 1 : 0;
            for (uint r = 0; r < RowsPerLane; ++r)
                for (uint m = 0; m < kLiveM; ++m) {
                    accGate[r][m] += ldexp(p1[r][m], e1[r] - esh);
                    accUp[r][m]   += ldexp(p3[r][m], e3[r] - esh);
                }
        }
    }

    // --- SwiGLU, the route weight, and the write-out -------------------------
    for (uint r = 0; r < RowsPerLane; ++r) {
        for (uint m = 0; m < kLiveM; ++m) {
            const float g = row_reduce(accGate[r][m], tid, sublane);
            const float u = row_reduce(accUp[r][m], tid, sublane);
            if (HQuant == 2) {
                // Park the value; the fp8 block scale needs 32 consecutive rows
                // and those are spread over the workgroup's row groups.
                if (sublane == 0) {
                    const float rw = RouteW[m * pc.num_slots + slot];
                    const float h  = swiglu(g, u, pc.swiglu_limit) * rw;
                    // Through fp16 first so HQuant 1 and 2 quantise the same
                    // number (design §6 keeps h in fp16 either way).
                    sHq[m * kWgRows + rowInWg * RowsPerLane + r] = live ? float(half(h)) : 0.0f;
                }
            } else if (live && sublane == 0) {
                const float rw = RouteW[m * pc.num_slots + slot];
                const float h  = swiglu(g, u, pc.swiglu_limit) * rw;
                const uint  oi = (m * pc.num_slots + slot) * pc.n_rows + rowBase + r;
                if (HPrecision == 0) H16[oi] = (half)h;
                else                 H32[oi] = h;
            }
        }
    }

    if (HQuant == 2) {
        GroupMemoryBarrierWithGroupSync();
        // design §7.9 v0.6 / inference/kernel.py act_quant: per row of h, one
        // UE8M0 scale per 32 elements along the intermediate dimension. The
        // workgroup owns kWgRows >= 32 consecutive rows, so it owns whole
        // blocks and nothing crosses a workgroup boundary.
        const uint blocksH = kWgRows / kBlockK;              // >= 1, enforced host-side
        const uint nScale  = M * blocksH;
        const uint sBase   = (M * pc.num_slots * pc.n_rows) / 4;   // scale plane, in words
        if (tid < nScale) {
            const uint m  = tid / blocksH;
            const uint hb = tid % blocksH;
            float amax = 0.0f;
            [unroll] for (uint j = 0; j < kBlockK; ++j)
                amax = max(amax, abs(sHq[m * kWgRows + hb * kBlockK + j]));
            const float s = fast_round_scale(max(amax, kFp8ActMin), 1.0f / kFp8ActMax);
            const uint  gb = (gid.x * kWgRows) / kBlockK + hb;
            HU[sBase + (m * pc.num_slots + slot) * (pc.n_rows / kBlockK) + gb] = asuint(s);
            sRed[tid] = s;                                   // reuse: row_reduce is done
        }
        GroupMemoryBarrierWithGroupSync();
        // 4 fp8 bytes per word; the workgroup's rows are contiguous so every
        // word belongs to exactly one workgroup and no read-modify-write is
        // needed.
        const uint words = M * kWgRows / 4;
        if (tid < words) {
            const uint m = tid / (kWgRows / 4);
            const uint q = tid % (kWgRows / 4);
            const float s   = sRed[m * blocksH + (q * 4) / kBlockK];
            const float inv = 1.0f / s;
            uint packed = 0;
            [unroll] for (uint j = 0; j < 4; ++j) {
                const float v = clamp(sHq[m * kWgRows + q * 4 + j] * inv, -kFp8ActMax, kFp8ActMax);
                packed |= fp8_e4m3_pack(fp8_e4m3_round(v)) << (8 * j);
            }
            const uint row0 = gid.x * kWgRows + q * 4;
            HU[((m * pc.num_slots + slot) * pc.n_rows + row0) / 4] = packed;
        }
    }
}
