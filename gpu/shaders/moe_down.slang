// moe_down -- w2 projection with the expert reduction folded in
// (design §7.9, dispatch B; #11 of the §7.14 list).
//
// One workgroup owns a row block of y and loops over every entry of SlotList
// internally, so the sum over experts happens in the lane's own fp32
// accumulator: no atomics, no third kernel. Because the reduction across a row
// is linear, the per-lane partials of all slots are accumulated first and
// reduced across the row exactly once at the end.
//
// RowsPerLane matters more here than in dispatch A: w2 is half the bytes of
// w1+w3 but h is read for every row of every slot, so at RowsPerLane = 1 the
// h traffic is 3.7x the weight traffic in L0 requests.
//
// h arrives in one of three forms and `load_h_block` is the only place that
// knows which (design §7.9 v0.6, docs/kernel_p2_moe.md):
//   HQuant 0  fp16 (or fp32) straight from dispatch A -- the P1 kernel
//   HQuant 1  the same fp16, with the reference's fp8 E4M3 + UE8M0 block-32
//             round trip applied here. A lane already holds all 32 elements of
//             a block, so the block amax costs no extra traffic at all.
//   HQuant 2  fp8 bytes plus a UE8M0 scale plane, both written by dispatch A
//   HQuant 3  the same two planes, written by the tiny moe_hquant dispatch that
//             runs between A and B, so A keeps the shape it wants. Only the
//             base offsets differ from mode 2 (moe_common.slang hq_value_words)
//
// Partial dispatch (design §7.9 "compute the experts that arrived first"):
// `list_count` entries of SlotList are summed and `flags & 1` says whether to
// add into y or overwrite it, so a layer can be run as several dispatches over
// disjoint slot subsets.

#include "moe_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> ExpertAddr;  // pointer table (design §5.3)
[[vk::binding(1, 0)]] StructuredBuffer<uint>     Ids;
[[vk::binding(2, 0)]] StructuredBuffer<uint>     SlotList;
[[vk::binding(3, 0)]] StructuredBuffer<uint>     H;   // h[M][slots][K] as raw words
[[vk::binding(4, 0)]] RWStructuredBuffer<float>  Y;   // y[M][N] fp32 (design §6)

struct DownPush {
    uint layer;
    uint experts_per_layer;
    uint num_slots;    // stride of the h slot dimension
    uint list_count;   // how many entries of SlotList to sum (design §7.9)
    uint n_rows;       // 5120
    uint k;            // 2304
    uint flags;        // bit 0: accumulate into y instead of overwriting
    // Track T / DSpark: live activation columns -- see GateUpPush::m. `M` is the
    // specialisation constant (the array bound), `pc.m` is what this dispatch
    // computes. Tail of the struct so the first seven words keep their offsets.
    uint m;
};
[[vk::push_constant]] DownPush pc;

// Track K1a: the trip count of every column loop below. At M == 1 the host
// specialises StaticM to 1 and this folds to a literal, which is what the
// pre-fb53514 `m < M` loops were; StaticM == 0 keeps Track T's push-constant
// mask for any M > 1 caller. See moe_common.slang's StaticM.
#define kLiveM ((StaticM != 0u) ? StaticM : pc.m)

static const uint kRowsPerWg = kWgSize / LanesPerRow;

groupshared float sRed[kWgSize];
static const uint kXTileU  = (XMode == 0) ? 1u
                           : ((XMode == 3) ? (M * LanesPerRow * 8u) : (M * LanesPerRow * 16u));
groupshared uint  sX[kXTileU];
static const uint kXScaleN = (XMode == 3) ? (M * LanesPerRow) : 1u;
groupshared float sXS[kXScaleN];
static const uint kFp8TblN = (Fp8Slots == 0 && HQuant < 2) ? 1u : kFp8TableN;
groupshared float sFp8[kFp8TblN];

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

// Eight consecutive h values starting at element `e` of the flat h array.
// Only the HQuant == 0, XMode == 0 path uses it; it is kept because it is the
// minimal-register form the P1 measurement was taken with.
void load_h8(uint e, out float v[8]) {
    if (HPrecision == 0) {
        const uint w0 = e >> 1;                       // two halves per word
        [unroll] for (uint i = 0; i < 4; ++i) {
            const uint w = H[w0 + i];
            v[2 * i + 0] = f16tof32(w & 0xFFFFu);
            v[2 * i + 1] = f16tof32(w >> 16);
        }
    } else {
        [unroll] for (uint i = 0; i < 8; ++i) v[i] = asfloat(H[e + i]);
    }
}

// One 32-element block of h as dispatch A left it: fp16 or fp32, no round trip.
void load_h_block_raw(uint row, uint blk, out float v[32]) {
    const uint e = row * pc.k + blk * kBlockK;
    if (HPrecision == 0) {
        const uint w0 = e >> 1;
        [unroll] for (uint i = 0; i < 16; ++i) {
            const uint w = H[w0 + i];
            v[2 * i + 0] = f16tof32(w & 0xFFFFu);
            v[2 * i + 1] = f16tof32(w >> 16);
        }
    } else {
        [unroll] for (uint i = 0; i < 32; ++i) v[i] = asfloat(H[e + i]);
    }
}

// One 32-element block of h for column `m` of slot `slot`, dequantised to fp32
// and with design §7.9 v0.6's fp8 round trip already applied.
void load_h_block(uint m, uint slot, uint blk, out float v[32]) {
    const uint row = m * pc.num_slots + slot;
    if (HQuant >= 2) {
        const uint elems = M * pc.num_slots * pc.k;
        const float s = asfloat(H[hq_scale_words(elems) + row * (pc.k / kBlockK) + blk]);
        const uint  w0 = hq_value_words(elems) + (row * pc.k + blk * kBlockK) / 4;
        [unroll] for (uint u = 0; u < 8; ++u) {
            const uint w = H[w0 + u];
            [unroll] for (uint j = 0; j < 4; ++j)
                v[u * 4 + j] = sFp8[(w >> (8 * j)) & 0xFFu] * s;
        }
        return;
    }
    load_h_block_raw(row, blk, v);
    if (HQuant == 1) {
        // inference/kernel.py act_quant(x, 32, scale_fmt="ue8m0"): amax floored
        // at 1e-4, scale rounded up to a power of two, fp8 E4M3 round trip.
        float amax = 0.0f;
        [unroll] for (uint j = 0; j < 32; ++j) amax = max(amax, abs(v[j]));
        const float s   = fast_round_scale(max(amax, kFp8ActMin), 1.0f / kFp8ActMax);
        const float inv = 1.0f / s;
        [unroll] for (uint j = 0; j < 32; ++j)
            v[j] = fp8_e4m3_round(clamp(v[j] * inv, -kFp8ActMax, kFp8ActMax)) * s;
    }
}

// The UE8M0 block scale of one h block, without keeping the values. The
// XMode == 0 path reads h twice -- once for the amax, once per element group --
// because holding M x 32 values would cost 192 VGPRs at M=6; both reads are L2
// hits on a 4.5 KiB working set.
float h_block_scale(uint m, uint slot, uint blk) {
    const uint row = m * pc.num_slots + slot;
    if (HQuant >= 2) {
        const uint elems = M * pc.num_slots * pc.k;
        return asfloat(H[hq_scale_words(elems) + row * (pc.k / kBlockK) + blk]);
    }
    float v[32];
    load_h_block_raw(row, blk, v);
    float amax = 0.0f;
    [unroll] for (uint j = 0; j < 32; ++j) amax = max(amax, abs(v[j]));
    return fast_round_scale(max(amax, kFp8ActMin), 1.0f / kFp8ActMax);
}

// Eight h values of element group `w` of block `blk` with the fp8 round trip
// applied, but left ON the quantised grid: the caller multiplies the finished
// per-(row, column, block) partial by the block scale exactly once instead of
// scaling all 32 elements. The block scale is a power of two, so pulling it out
// of the sum is bit-exact, and at RowsPerLane = 1 those 32 multiplies were
// pure overhead against 32 FMAs -- the +18% dispatch B paid for fp8 h on the
// M=1 decode shape (docs/kernel_p2_moe.md §9).
void load_h8q(uint m, uint slot, uint blk, uint w, float s, out float v[8]) {
    const uint row = m * pc.num_slots + slot;
    if (HQuant >= 2) {
        const uint w0 = hq_value_words(M * pc.num_slots * pc.k)
                      + (row * pc.k + blk * kBlockK) / 4 + w * 2;
        [unroll] for (uint u = 0; u < 2; ++u) {
            const uint word = H[w0 + u];
            [unroll] for (uint j = 0; j < 4; ++j)
                v[u * 4 + j] = sFp8[(word >> (8 * j)) & 0xFFu];
        }
        return;
    }
    load_h8(row * pc.k + blk * kBlockK + w * 8, v);
    const float inv = 1.0f / s;
    [unroll] for (uint j = 0; j < 8; ++j)
        v[j] = fp8_e4m3_round(clamp(v[j] * inv, -kFp8ActMax, kFp8ActMax));
}

// Stage blocks [base, base + LanesPerRow) of h for every column into LDS. One
// thread per (m, block): the fp8 round trip is block-granular, so splitting a
// block across threads would need a cross-lane amax.
void stage_h(uint slot, uint base, uint blocks, uint tid) {
    if (HQuant == 0 && HPrecision == 0 && XMode != 3) {
        // Nothing to compute: h is already fp16 pairs, so all 256 threads copy
        // words. Block-granular staging would leave 256 - M*LanesPerRow threads
        // idle behind two barriers, which at M=1 is 7 threads in 8.
        const uint total = M * LanesPerRow * 16u;
        const uint words = pc.k / 2;                   // h words per (m, slot)
        for (uint i = tid; i < total; i += kWgSize) {
            const uint m = i / (LanesPerRow * 16u);
            const uint t = i % (LanesPerRow * 16u);
            const uint g = base * 16u + t;
            sX[m * LanesPerRow * 16u + t] =
                (g < words) ? H[(m * pc.num_slots + slot) * words + g] : 0u;
        }
        return;
    }
    const uint pairs = M * LanesPerRow;
    if (tid >= pairs) return;
    const uint m   = tid / LanesPerRow;
    const uint b   = tid % LanesPerRow;
    const uint blk = base + b;
    float v[32];
    if (blk < blocks) load_h_block(m, slot, blk, v);
    else [unroll] for (uint j = 0; j < 32; ++j) v[j] = 0.0f;

    if (XMode == 3) {
        float amax = 0.0f;
        [unroll] for (uint j = 0; j < 32; ++j) amax = max(amax, abs(v[j]));
        const float s   = max(amax, 1e-30f) * (1.0f / 127.0f);
        const float inv = 1.0f / s;
        sXS[m * LanesPerRow + b] = s;
        [unroll] for (uint u = 0; u < 8; ++u) {
            uint packed = 0;
            [unroll] for (uint j = 0; j < 4; ++j) {
                const int q = clamp(int(floor(v[u * 4 + j] * inv + 0.5f)), -127, 127);
                packed |= (uint(q) & 0xFFu) << (8 * j);
            }
            sX[m * LanesPerRow * 8 + b * 8 + u] = packed;
        }
    } else {
        [unroll] for (uint u = 0; u < 16; ++u)
            sX[m * LanesPerRow * 16 + b * 16 + u] =
                (f32tof16(v[2 * u]) & 0xFFFFu) | (f32tof16(v[2 * u + 1]) << 16);
    }
}

void read_h8_lds(uint m, uint b, uint g, out float v[8]) {
    const uint o = m * LanesPerRow * 16u + b * 16u + g * 4u;
    [unroll] for (uint j = 0; j < 8; ++j)
        v[j] = f16tof32(((j & 1) != 0) ? (sX[o + (j >> 1)] >> 16) : (sX[o + (j >> 1)] & 0xFFFFu));
}

void read_h8_h2_lds(uint m, uint b, uint g, out float16_t2 v[4]) {
    const uint o = m * LanesPerRow * 16u + b * 16u + g * 4u;
    [unroll] for (uint j = 0; j < 4; ++j) v[j] = bit_cast<float16_t2>(sX[o + j]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint sublane = tid % LanesPerRow;
    const uint rowInWg = tid / LanesPerRow;
    const uint rowBase = (gid.x * kRowsPerWg + rowInWg) * RowsPerLane;

    const uint blocks = pc.k / kBlockK;   // 72
    const bool live   = rowBase + RowsPerLane <= pc.n_rows;

    if (Fp8Slots != 0 || HQuant >= 2) {
        if (tid < kFp8TableN) sFp8[tid] = fp8_e4m3_decode(tid);
        GroupMemoryBarrierWithGroupSync();
    }

    float acc[RowsPerLane][M];
    for (uint i = 0; i < RowsPerLane; ++i)
        for (uint j = 0; j < M; ++j) acc[i][j] = 0.0f;

    for (uint ci = 0; ci < pc.list_count; ++ci) {
        const uint slot   = SlotList[ci];
        const uint rawId  = Ids[slot];
        const bool isFp8  = (Fp8Slots != 0) && ((rawId & kSlotFp8Bit) != 0);
        const uint expert = rawId & ~kSlotFp8Bit;
        const uint tbase  = (pc.layer * pc.experts_per_layer + expert) * 6;
        const uint64_t w2Base = ExpertAddr[tbase + 2];
        const uint64_t s2Base = ExpertAddr[tbase + 3];
        if (w2Base == 0) continue;                       // uniform across the workgroup

        const uint rowBytes = isFp8 ? pc.k : (pc.k / 2);
        const uint scaleRow = isFp8 ? (rowBase >> 5) : rowBase;
        const bool tiled    = x_mode_tiles() && !isFp8;
        const uint stride   = isFp8 ? 32u : 16u;         // weight bytes per block per row

        // How many block iterations: the tiled form walks LanesPerRow blocks at
        // a time with the whole workgroup, the plain form strides by lane.
        const uint iters = tiled ? ((blocks + LanesPerRow - 1) / LanesPerRow) : 0u;
        for (uint it = 0; it < iters; ++it) {
            const uint base = it * LanesPerRow;
            GroupMemoryBarrierWithGroupSync();
            stage_h(slot, base, blocks, tid);
            GroupMemoryBarrierWithGroupSync();
            const uint blk = base + sublane;
            if (live && blk < blocks) {
                uint4 w2q[RowsPerLane];
                int   e2[RowsPerLane];
                for (uint r = 0; r < RowsPerLane; ++r) {
                    w2q[r] = load_fp4_block(w2Base + (uint64_t)(rowBase + r) * rowBytes
                                                   + (uint64_t)blk * stride);
                    e2[r]  = load_scale_exp(s2Base + (uint64_t)(scaleRow + r) * blocks + blk);
                }
                float p[RowsPerLane][M];
                for (uint i = 0; i < RowsPerLane; ++i)
                    for (uint j = 0; j < M; ++j) p[i][j] = 0.0f;

                if (XMode == 3) {
                    for (uint r = 0; r < RowsPerLane; ++r) {
                        int a[M];
                        for (uint j = 0; j < M; ++j) a[j] = 0;
                        [unroll] for (uint w = 0; w < 4; ++w) {
                            const uint pw = w2q[r][w];
                            [unroll] for (uint g = 0; g < 2; ++g) {
                                const uint d = fp4_nibbles_to_i8x4(pw, g);
                                for (uint m = 0; m < kLiveM; ++m)
                                    a[m] = dot4add_i8packed(
                                        d, sX[m * LanesPerRow * 8 + sublane * 8 + w * 2 + g], a[m]);
                            }
                        }
                        for (uint m = 0; m < kLiveM; ++m)
                            p[r][m] = float(a[m]) * sXS[m * LanesPerRow + sublane];
                    }
                    for (uint r = 0; r < RowsPerLane; ++r)
                        for (uint m = 0; m < kLiveM; ++m) acc[r][m] += ldexp(p[r][m], e2[r] - 1);
                } else {
                    [unroll] for (uint w = 0; w < 4; ++w) {
                        if (XMode == 2) {
                            float16_t2 gw[RowsPerLane][4];
                            for (uint r = 0; r < RowsPerLane; ++r) {
                                const uint pw = w2q[r][w];
                                [unroll] for (uint j = 0; j < 4; ++j)
                                    gw[r][j] = float16_t2(
                                        float16_t(fp4_decode((pw >> (8 * j)) & 0xFu)),
                                        float16_t(fp4_decode((pw >> (8 * j + 4)) & 0xFu)));
                            }
                            for (uint m = 0; m < kLiveM; ++m) {
                                float16_t2 hv[4];
                                read_h8_h2_lds(m, sublane, w, hv);
                                for (uint r = 0; r < RowsPerLane; ++r) {
                                    float16_t2 t = float16_t2(float16_t(0.0f), float16_t(0.0f));
                                    [unroll] for (uint j = 0; j < 4; ++j) t += gw[r][j] * hv[j];
                                    p[r][m] += float(t.x) + float(t.y);
                                }
                            }
                        } else {
                            float d[RowsPerLane][8];
                            for (uint r = 0; r < RowsPerLane; ++r) {
                                const uint pw = w2q[r][w];
                                [unroll] for (uint j = 0; j < 8; ++j)
                                    d[r][j] = fp4_decode((pw >> (4 * j)) & 0xFu);
                            }
                            for (uint m = 0; m < kLiveM; ++m) {
                                float hv[8];
                                read_h8_lds(m, sublane, w, hv);
                                [unroll] for (uint j = 0; j < 8; ++j)
                                    for (uint r = 0; r < RowsPerLane; ++r) p[r][m] += d[r][j] * hv[j];
                            }
                        }
                    }
                    for (uint r = 0; r < RowsPerLane; ++r)
                        for (uint m = 0; m < kLiveM; ++m) acc[r][m] += ldexp(p[r][m], e2[r]);
                }
            }
        }

        if (!tiled && live) {
            for (uint blk = sublane; blk < blocks; blk += LanesPerRow) {
                uint4 w2q[RowsPerLane], w2r[RowsPerLane];
                int   e2[RowsPerLane];
                for (uint r = 0; r < RowsPerLane; ++r) {
                    const uint64_t off = (uint64_t)(rowBase + r) * rowBytes
                                       + (uint64_t)blk * stride;
                    w2q[r] = load_raw16(w2Base + off);
                    if (isFp8) w2r[r] = load_raw16(w2Base + off + 16);
                    e2[r] = load_scale_exp(s2Base
                                + (uint64_t)(scaleRow + (isFp8 ? 0 : r)) * blocks + blk);
                }

                float p[RowsPerLane][M];
                for (uint i = 0; i < RowsPerLane; ++i)
                    for (uint j = 0; j < M; ++j) p[i][j] = 0.0f;

                // Only the block scale is kept between the two passes over h:
                // holding all M x 32 values would cost 192 VGPRs at M=6.
                float hs[M];
                for (uint j = 0; j < M; ++j) hs[j] = 1.0f;
                if (HQuant != 0)
                    for (uint m = 0; m < kLiveM; ++m) hs[m] = h_block_scale(m, slot, blk);

                const bool packed = (XMode == 4) && !isFp8 && HQuant == 0 && HPrecision == 0;
                const bool asI8   = (XMode == 5) && !isFp8;
                if (packed) {
                    // h keeps its fp16 bit pattern into v_pk_fma_f16: no
                    // f16tof32, half the FMA stream (docs/kernel_p2_moe.md §3).
                    float16_t2 t[RowsPerLane][M];
                    for (uint i = 0; i < RowsPerLane; ++i)
                        for (uint j = 0; j < M; ++j)
                            t[i][j] = float16_t2(float16_t(0.0f), float16_t(0.0f));
                    [unroll] for (uint w = 0; w < 4; ++w) {
                        float16_t2 gw[RowsPerLane][4];
                        for (uint r = 0; r < RowsPerLane; ++r) {
                            const uint pw = w2q[r][w];
                            [unroll] for (uint j = 0; j < 4; ++j) gw[r][j] = fp4_pair_h2(pw, j);
                        }
                        for (uint m = 0; m < kLiveM; ++m) {
                            const uint e0 = ((m * pc.num_slots + slot) * pc.k
                                             + blk * kBlockK + w * 8) >> 1;
                            float16_t2 hv[4];
                            [unroll] for (uint j = 0; j < 4; ++j) hv[j] = bit_cast<float16_t2>(H[e0 + j]);
                            for (uint r = 0; r < RowsPerLane; ++r)
                                [unroll] for (uint j = 0; j < 4; ++j) t[r][m] += gw[r][j] * hv[j];
                        }
                    }
                    for (uint r = 0; r < RowsPerLane; ++r)
                        for (uint m = 0; m < kLiveM; ++m)
                            p[r][m] = float(t[r][m].x) + float(t[r][m].y);
                } else if (asI8) {
                    uint q[RowsPerLane][8];
                    for (uint r = 0; r < RowsPerLane; ++r)
                        [unroll] for (uint w = 0; w < 4; ++w)
                            [unroll] for (uint g = 0; g < 2; ++g)
                                q[r][w * 2 + g] = fp4_nibbles_to_i8x4(w2q[r][w], g);
                    for (uint m = 0; m < kLiveM; ++m) {
                        float hv[32];
                        if (HQuant != 0) load_h_block(m, slot, blk, hv);
                        else load_h_block_raw(m * pc.num_slots + slot, blk, hv);
                        float amax = 0.0f;
                        [unroll] for (uint j = 0; j < 32; ++j) amax = max(amax, abs(hv[j]));
                        const float sh  = max(amax, 1e-30f) * (1.0f / 127.0f);
                        const float inv = 1.0f / sh;
                        uint hq[8];
                        [unroll] for (uint u = 0; u < 8; ++u) {
                            uint word = 0;
                            [unroll] for (uint j = 0; j < 4; ++j) {
                                const int qv = clamp(int(floor(hv[u * 4 + j] * inv + 0.5f)), -127, 127);
                                word |= (uint(qv) & 0xFFu) << (8 * j);
                            }
                            hq[u] = word;
                        }
                        for (uint r = 0; r < RowsPerLane; ++r) {
                            int a = 0;
                            [unroll] for (uint u = 0; u < 8; ++u)
                                a = dot4add_i8packed(q[r][u], hq[u], a);
                            p[r][m] += float(a) * sh;
                        }
                    }
                } else
                [unroll] for (uint w = 0; w < 4; ++w) {
                    float d[RowsPerLane][8];
                    for (uint r = 0; r < RowsPerLane; ++r) {
                        if (isFp8) {
                            const uint a = (w < 2) ? w2q[r][2 * w]     : w2r[r][2 * w - 4];
                            const uint b = (w < 2) ? w2q[r][2 * w + 1] : w2r[r][2 * w - 3];
                            [unroll] for (uint j = 0; j < 4; ++j) {
                                d[r][j]     = sFp8[(a >> (8 * j)) & 0xFFu];
                                d[r][4 + j] = sFp8[(b >> (8 * j)) & 0xFFu];
                            }
                        } else {
                            const uint pw = w2q[r][w];
                            [unroll] for (uint j = 0; j < 8; ++j)
                                d[r][j] = fp4_decode((pw >> (4 * j)) & 0xFu);
                        }
                    }
                    for (uint m = 0; m < kLiveM; ++m) {
                        float hv[8];
                        if (HQuant != 0) load_h8q(m, slot, blk, w, hs[m], hv);
                        else load_h8((m * pc.num_slots + slot) * pc.k + blk * kBlockK + w * 8, hv);
                        [unroll] for (uint j = 0; j < 8; ++j)
                            for (uint r = 0; r < RowsPerLane; ++r) p[r][m] += d[r][j] * hv[j];
                    }
                }
                const int esh = asI8 ? 1 : 0;
                // `hs[m]` is 1 unless the generic path above left h on its fp8
                // grid, in which case this is where the block scale lands.
                const bool folded = (HQuant != 0) && !packed && !asI8;
                for (uint r = 0; r < RowsPerLane; ++r)
                    for (uint m = 0; m < kLiveM; ++m)
                        acc[r][m] += ldexp(folded ? (p[r][m] * hs[m]) : p[r][m], e2[r] - esh);
            }
        }
    }

    for (uint r = 0; r < RowsPerLane; ++r)
        for (uint m = 0; m < kLiveM; ++m) {
            const float s = row_reduce(acc[r][m], tid, sublane);
            if (live && sublane == 0) {
                const uint oi = m * pc.n_rows + rowBase + r;
                Y[oi] = ((pc.flags & 1u) != 0) ? (Y[oi] + s) : s;
            }
        }
}
