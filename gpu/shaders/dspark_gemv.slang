// The DSpark linear-algebra kernel: four stages of one .spv covering every
// projection of a draft cycle (design §7.12).
//
//   Stage 0  Gemv      y[m][row] = sum_k W[row][k] * x[m][k], fp8 E4M3 weight
//   Stage 1  RmsNorm   y[m][i]   = w[i] * x[m][i] * rsqrt(mean(x[m]^2) + eps)
//   Stage 2  RopeQuant RoPE over the last `rope_dim` dims of each row, and the
//                      `act_quant` round trip the KV ring stores
//   Stage 3  WoA       o[m][g][r] = sum_d o_in[m][g][d] * wo_a[g][r][d], fp8
//                      weight, NO act_quant on the activation
//
// Every fp8 projection of the draft path is "stage 0, then stage 1 or 2":
//
//   main_proj + main_norm   0 (K=15360, rows=5120, M=1) then 1 (n=5120)
//   wq_a      + q_norm      0 (K=5120,  rows=1280, M=5) then 1 (n=1280)
//   wq_b      + RoPE        0 (K=1280,  rows=32768,M=5) then 2 (rows_per_pos=64)
//   wkv (main)+ kv_norm     0 (K=5120,  rows=512,  M=1) then 1 then 2 (quant=1)
//   wkv (draft)+ kv_norm    0 (K=5120,  rows=512,  M=5) then 1 then 2 (quant=1)
//   attention output        2 with inverse=1, quant=0 (rows_per_pos=64)
//   wo_a                    3 (rows=8192, K=4096, groups of 1024)
//   wo_b                    0 (K=8192,   rows=5120,M=5)
//
// **Why the RMSNorm is its own stage rather than fused into the GEMV.** An
// RMSNorm needs a statistic over the WHOLE output row, and the outputs here are
// 5120 / 1280 / 512 wide while the GEMV that produces them is spread over
// hundreds of workgroups (wq_b is 32768 rows). Fusing would mean either one
// workgroup owning every row -- 79 MB of `main_proj` read by a single workgroup
// -- or wkv.slang's two-dispatch partial-sum dance. The draft path runs the norm
// five times a cycle over at most 5120 elements; a standalone one-workgroup-per-
// column kernel costs microseconds and is obviously correct, and obviously
// correct is what §7.12 needs first. wkv.slang's fused form is the optimisation
// to reach for later, and it is already written there.
//
// The M dimension: read the weight once, use it M times
// ----------------------------------------------------
// This is the whole reason DSpark is worth anything. A draft cycle re-reads
// `wq_b` (41.9 MB) and `wo_b` (41.9 MB) once per stage whatever M is, so at
// M = 5 the same bytes retire five positions. The loop order that gets that is
// weight-outer / batch-inner, exactly what moe_gateup.slang does for the FP4
// experts (its `for (uint m = 0; m < M; ++m)` inside the weight-block decode):
// one 32-byte weight block is loaded and decoded into registers ONCE, then
// dotted against `mlive` staged activation blocks.
//
// The activation staging, and why it is K-TILED
// ---------------------------------------------
// fp8_gemv.slang stages the whole activation into LDS once per workgroup. At
// M = 1 that is 8192 elements at worst; at M = 5 it would be 40960, which is
// 40 KB of LDS as raw fp8 bytes and 80 KB as the bf16 form fp8_gemv measured as
// faster -- past the 64 KB a workgroup may have, never mind occupancy.
//
// So the staging is tiled along K instead: `kTileK` = 1024 elements, i.e.
// `kTileBlocks` = 32 UE8M0 blocks, for every live draft column. That is
// `kMaxDraftM * 32 * 17` words = 13 KB whatever K is, which puts this kernel in
// the same LDS bracket as the rest of gpu/shaders (sparse_attn 23 KB, head
// 25 KB, fp8_gemv 20 KB) and makes K unbounded. The cost is one extra pair of
// workgroup barriers per 1024 K-elements -- eight of them for wo_b, fifteen for
// main_proj.
//
// Three details that are not free to get wrong:
//
//   * **A block is 17 LDS words, not 16.** Sixteen puts a 16-way bank conflict
//     on every read of the staged activation: lane `l` reads word `i` of block
//     `l` at index `16 l + i`, whose bank `(16 l + i) mod 32` takes two values
//     across 32 lanes. 17 is odd, so `17 l mod 32` is a bijection. The full
//     argument and the measurement are in fp8_gemv.slang; `kTileBlocks` is 32,
//     so `(m * 32 + l) * 17 + i` keeps the same property for every m (544 is a
//     multiple of 32, so the m term drops out of the bank index).
//   * **One thread owns one whole `act_quant` block.** `kTileBlocks` is 32 and
//     `kMaxDraftM` is 6, so `mlive * 32 <= 192 <= 256` slots are staged by one
//     thread each and the block amax is a pure register reduction -- no
//     cross-lane work, no `stage_amax` tree. This is the property
//     attn_common.slang's point 1 notes: with a 32-element UE8M0 block the
//     quantisation is free if the staging is arranged this way.
//   * **The scale is read off the fp32 bit pattern** (`act_scale`), never via
//     `ceil(log2(x))`, which gets exact powers of two wrong (design §2.4). The
//     stored value is the E4M3 quotient q, not q * s: bf16 holds every E4M3
//     number exactly and the power-of-two scale rejoins the accumulator through
//     `ldexp`, so folding it out of the inner loop is exact.
//
// Byte budget of a draft cycle, stage 0 only (design §7.12's 2.2 GB):
// main_proj 79 MB once, then per mtp stage wq_a 6.6 + wq_b 41.9 + wkv 2.6 x2 +
// wo_b 41.9 = 95.6 MB, plus stage 3's wo_a at 33.6 MB -- 129 MB a stage, about
// 470 MB of the cycle, against 1.32 GB for the head.
//
// **wo_a: fp8 weight, unquantised activation.** The checkpoint stores wo_a as
// fp8 E4M3 with a `[256][128]` UE8M0 scale plane (33.6 MB), and that is what the
// pinned store (store/pinned.h PinnedTensor data/scale) holds and what stage 3
// reads -- through the same table decode and `ldexp` scale fold as stage 0, the
// way gpu/shaders/wo_a.slang does for the M = 1 path. `convert.py` dequantises
// it to bf16 and the module is declared bf16, but an E4M3 value times a power of
// two is exactly representable, so decoding the fp8 bytes here IS that bf16
// weight; reading a dequantised copy instead would be a separate 67 MB tensor
// and double the bytes read per stage, for the same number.
//
// What stage 3 must NOT do is quantise the ACTIVATION. `Attention.forward`
// reaches wo_a through `torch.einsum` rather than `linear()`, so there is no
// `act_quant` round trip on `o`; doing one anyway would not cost a little
// precision, it would compute a different function -- E4M3 has three mantissa
// bits, so an element moves by up to 6% (attn_common.slang, point 1). Stage 3's
// pipeline is therefore built with `ActQuant = 0`, and the staging path it
// shares with stage 0 keys off exactly that constant.

#include "dspark_common.slang"

// Stage 0/3 slots. Every stage owns its own 32-slot slice of the address table
// (one descriptor per pipeline), so the four stages are free to reuse indices.
static const uint kW = 0, kS = 1, kX = 2, kY = 3;
// Stage 1: RMSNorm.
static const uint kNormX = 0, kNormW = 1, kNormY = 2;
// Stage 2: RoPE (+ act_quant).
static const uint kRopeX = 0, kRopeTab = 1, kRopeY = 2;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

// One struct for all four stages, the way wkv.slang and head.slang carry one
// for theirs: a stage is one pipeline and a pipeline has one push range, and
// four structs at one binding would only let a caller push the wrong one.
struct DsparkGemvPush {
    uint  m;              // live draft columns; the loops run min(M, m)
    uint  rows;           // 0: weight rows. 2: rows per draft position. 3: 8192
    uint  k;              // 0: contraction. 1: vector length. 2: row width. 3: 4096
    uint  scale_cols;     // 0, 3: k / 32, the UE8M0 plane's row stride (wo_a: 128)
    uint  row_base;       // 0, 3: first row of the dispatch
    uint  x_stride;       // elements between draft columns of the input
    uint  y_stride;       // elements between draft columns of the output
    uint  rows_per_group; // 3: o_lora_rank = 1024
    uint  rope_dim;       // 2: rope_head_dim = 64
    uint  inverse;        // 2: 1 = conjugate the rotation (the output's un-rope)
    uint  quant;          // 2: 1 = the act_quant round trip the KV ring stores
    uint  flags;          // kFlagInBf16 | kFlagOutBf16 | kFlagRoundIn
    float eps;            // 1: rms_norm_eps = 1e-20
};
[[vk::push_constant]] DsparkGemvPush pc;

// --- the K-tile staging -----------------------------------------------------

static const uint kTileBlocks = 32;                      // UE8M0 blocks per tile
static const uint kTileK      = kTileBlocks * kBlockK;   // 1024 elements
static const uint kXStride    = kBlockK / 2 + 1;         // 17 words, odd on purpose

groupshared uint  gXQ[kMaxDraftM * kTileBlocks * kXStride];  // 3264 words, 13.1 KiB
groupshared float gXS[kMaxDraftM * kTileBlocks];             // 192 floats, 768 B

uint gxq_word(uint m, uint blk, uint w) { return (m * kTileBlocks + blk) * kXStride + w; }

// One K-tile of the activation for every live draft column, into gXQ/gXS.
// Whole workgroup, uniform control flow, barriers on both sides (the caller's).
//
// `xoff` is an element offset added inside a column, so wo_a's group `g` can
// stage the 4096-wide slice at `g * k` of a 32768-wide `o` without a copy.
// `klen` bounds the tile: K is always a multiple of 32 (it is a multiple of the
// UE8M0 block) but not of 1024 -- wq_b's 1280 is one full tile plus eight blocks
// -- so the blocks past the end stage zeros and the accumulator skips them.
void stage_tile(uint64_t xaddr, uint xstride, uint xoff, uint klen, uint k0,
                uint ml, bool xb, uint tid) {
    if (tid >= ml * kTileBlocks) return;
    const uint m   = tid / kTileBlocks;
    const uint blk = tid % kTileBlocks;
    const uint d0  = k0 + blk * kBlockK;

    float v[32];
    float s = 1.0f;
    if (d0 < klen) {
        act_load32(xaddr, act_index(m, xstride, xoff + d0), xb, v);
        // ActQuant is the specialisation constant of attn_common.slang: 1 for
        // every fp8 `Linear` (stage 0), 0 for wo_a's einsum (stage 3).
        if (ActQuant != 0) {
            float amax = 0.0f;
            [unroll] for (uint i = 0; i < 32; ++i) amax = max(amax, abs(v[i]));
            s = act_scale(amax);
            const float inv = 1.0f / s;
            [unroll] for (uint i = 0; i < 32; ++i)
                v[i] = fp8_round(clamp(v[i] * inv, -kFp8MaxF, kFp8MaxF));
        }
    } else {
        [unroll] for (uint i = 0; i < 32; ++i) v[i] = 0.0f;
    }
    [unroll] for (uint w = 0; w < 16; ++w)
        gXQ[gxq_word(m, blk, w)] = bf16_encode(v[w * 2]) | (bf16_encode(v[w * 2 + 1]) << 16);
    gXS[m * kTileBlocks + blk] = s;
}

// The 32 staged activations of one (column, block): eight LDS words rather than
// 32 indexed reads.
void read_x_block(uint m, uint blk, out float v[32]) {
    [unroll] for (uint i = 0; i < 16; ++i) {
        const uint w = gXQ[gxq_word(m, blk, i)];
        v[i * 2]     = bf16_decode(w & 0xFFFFu);
        v[i * 2 + 1] = bf16_decode(w >> 16);
    }
}

uint rows_per_wg() { return (kWg / LanesPerRow) * RowsPerLane; }

// One 32-element fp8 E4M3 weight block into registers, and its UE8M0 exponent.
// Shared by stage 0 and stage 3. Weight layout, from model/manifest.h:
// `[rows, K]` E4M3 bytes plus a `[rows/32, K/32]` UE8M0 scale plane -- a 32x32
// TILE scale, so row `row`'s block at K-offset `kd` takes its scale byte from row
// `row >> 5` of the plane, at `sbase + (row >> 5) * scale_cols + kd / 32`,
// unbiased by -127 and folded in by the caller with `ldexp`. gTbl must already
// be filled and barriered.
void fp8_weight_block(uint64_t wbase, uint64_t sbase, uint row, uint k, uint kd,
                      uint scale_cols, out float wv[32], out int e) {
    const uint64_t off = (uint64_t)row * (uint64_t)k + kd;
    const uint4 b0 = load16(wbase + off);
    const uint4 b1 = load16(wbase + off + 16);
    e = (int)load_u8(sbase + (uint64_t)(row >> 5) * scale_cols + (kd / kBlockK)) - 127;
    [unroll] for (uint i = 0; i < 4; ++i) {
        const uint a = b0[i], b = b1[i];
        [unroll] for (uint j = 0; j < 4; ++j) {
            wv[i * 4 + j]      = gTbl[(a >> (8 * j)) & 0xFFu];
            wv[16 + i * 4 + j] = gTbl[(b >> (8 * j)) & 0xFFu];
        }
    }
}

// --- stage 0: the fp8 E4M3 GEMV, M columns ----------------------------------
// The weight decode is `fp8_weight_block` above; the activation is act-quantised.
void stage_gemv(uint gid, uint tid) {
    fp8_table_fill(tid);
    // fp8_table_fill must complete before ANY read of gTbl, in uniform control
    // flow, and the barrier below is the one that guarantees it (the first tile
    // barrier). docs/p2_decode.md §3.2 is the record of what forgetting it
    // costs.
    const uint ml      = mlive(pc.m);
    const bool xb      = in_bf16(pc.flags);
    const bool yb      = out_bf16(pc.flags);
    const uint sublane = tid % LanesPerRow;
    const uint rowBase = pc.row_base + gid * rows_per_wg() + (tid / LanesPerRow) * RowsPerLane;

    float acc[kMaxRowsPerLane][kMaxDraftM];
    [unroll] for (uint r0 = 0; r0 < kMaxRowsPerLane; ++r0)
        [unroll] for (uint m0 = 0; m0 < kMaxDraftM; ++m0) acc[r0][m0] = 0.0f;

    // `pc.k`, `ml` and `kTileK` are all uniform, so every barrier below is
    // reached by all 256 threads the same number of times.
    for (uint k0 = 0; k0 < pc.k; k0 += kTileK) {
        GroupMemoryBarrierWithGroupSync();      // last tile's readers are done
        stage_tile(Ptr[kX], pc.x_stride, 0, pc.k, k0, ml, xb, tid);
        GroupMemoryBarrierWithGroupSync();

        for (uint blk = sublane; blk < kTileBlocks; blk += LanesPerRow) {
            const uint kd = k0 + blk * kBlockK;
            if (kd >= pc.k) continue;
            // Unrolled against the compile-time bound with a predicate rather
            // than bounded by RowsPerLane: a loop bounded by a specialisation
            // constant leaves acc[][] dynamically indexed and the compiler then
            // spills it to scratch (docs/p2_attention.md §4.4).
            [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) {
                if (r >= RowsPerLane) continue;
                const uint row = rowBase + r;
                if (row >= pc.rows) continue;
                // Decoded ONCE and reused for every draft column: this is the
                // M-awareness the whole draft path is built on.
                float wv[32];
                int   e;
                fp8_weight_block(Ptr[kW], Ptr[kS], row, pc.k, kd, pc.scale_cols, wv, e);
                [unroll] for (uint m = 0; m < kMaxDraftM; ++m) {
                    if (m >= ml) continue;
                    float x[32];
                    read_x_block(m, blk, x);
                    float p = 0.0f;
                    [unroll] for (uint i = 0; i < 32; ++i) p += wv[i] * x[i];
                    acc[r][m] += ldexp(p, e) * gXS[m * kTileBlocks + blk];
                }
            }
        }
    }
    // row_reduce may be the LDS tree, which writes gRed; nothing above touches
    // gRed, but the tile loop's last readers must be done before the reduction's
    // own barriers start pacing the workgroup differently.
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) {
        if (r >= RowsPerLane) continue;
        [unroll] for (uint m = 0; m < kMaxDraftM; ++m) {
            if (m >= ml) continue;
            // Uniform: RowsPerLane is a specialisation constant and `ml` a push
            // constant, so every thread runs the same number of row_reduces.
            const float y = row_reduce(acc[r][m], tid, sublane);
            if (sublane == 0 && rowBase + r < pc.rows)
                act_store(Ptr[kY], act_index(m, pc.y_stride, rowBase + r), yb, y);
        }
    }
}

// --- stage 1: RMSNorm -------------------------------------------------------
// `y[m][i] = w[i] * x[m][i] * rsqrt(mean(x[m]^2) + eps)`, fp32 accumulate, one
// workgroup per draft column, `w` a bf16 vector.
//
// The output is rounded onto the bf16 grid whatever `kFlagOutBf16` says.
// `RMSNorm.forward` ends in `.to(dtype)` and `generate.py` runs under
// `torch.set_default_dtype(bfloat16)`, so the reference's norm output IS a bf16
// number; keeping our extra mantissa bits is not a free precision win, because
// the value goes straight into `act_quant` and a wider input moves the block
// amax into a different power-of-two bracket, re-rounding all 32 elements one
// grid step coarser. attn_common.slang has the measurement (wq_b: relative L2
// 1.5e-2 -> 2e-3 when this was added).
//
// The INPUT is a different question. The reference's `linear()` returns bf16, so
// its RMS statistic is taken over bf16 values; stage 0 leaves fp32, the way
// every GEMV in gpu/shaders does, and measures cos 0.999998 either way
// (docs/p2_attention.md §7.15.1). `kFlagRoundIn` lets a caller ask for the
// reference's exact order instead of arguing about which is right.
void stage_rmsnorm(uint gid, uint tid) {
    const uint m   = gid;                       // one workgroup per draft column
    const bool xb  = in_bf16(pc.flags);
    const bool rnd = xb || (pc.flags & kFlagRoundIn) != 0u;
    const bool yb  = out_bf16(pc.flags);

    float sq = 0.0f;
    for (uint i = tid; i < pc.k; i += kWg) {
        float v = act_load(Ptr[kNormX], act_index(m, pc.x_stride, i), xb);
        if (rnd) v = bf16_round(v);
        sq += v * v;
    }
    const float rs = rsqrt(wg_reduce(sq, tid) / float(pc.k) + pc.eps);

    for (uint i = tid; i < pc.k; i += kWg) {
        float v = act_load(Ptr[kNormX], act_index(m, pc.x_stride, i), xb);
        if (rnd) v = bf16_round(v);
        const float w = bf16_decode(load_u16(Ptr[kNormW] + (uint64_t)i * 2));
        act_store(Ptr[kNormY], act_index(m, pc.y_stride, i), yb, bf16_round(v * rs * w));
    }
}

// --- stage 2: RoPE, and the KV ring's act_quant ------------------------------
// The tensor is `[M][rows_per_pos][row_dim]`: `rows_per_pos` is 1 for a kv row
// (`[M][512]`) and `n_heads` for a q or an attention output (`[M][64][512]`).
// The rotation covers the last `rope_dim` dims of every row, pairing ADJACENT
// elements -- (row_dim - rope_dim + 2j, ... + 2j + 1), frequency j driving pair
// j -- and a pair never straddles a 32-element block because both `rope_lo` and
// `kBlockK` are even.
//
// `inverse != 0` conjugates, which is the un-rotation
// `apply_rotary_emb(o[..., -64:], freqs_cis, True)` applies to the attention
// output. dspark_attn.slang does NOT fuse it: one RoPE implementation for the
// whole family is worth more than one saved dispatch on a kernel that moves
// 320 KB.
//
// `quant != 0` adds the `act_quant(kv, 32, "ue8m0", e8m0, inplace=True)` round
// trip. There is no fp8 byte plane: the reference's window cache is a plain bf16
// tensor and `inplace=True` writes the dequantised value back into it, which is
// exact (three E4M3 mantissa bits and a power-of-two scale fit bf16's eight). So
// the ring stores bf16 and dspark_attn.slang reads bf16, and wkv.slang's
// byte-plus-scale-plane ring -- which the M = 1 decode path needs because its
// attention decodes fp8 in the inner loop -- has no counterpart here.
//
// One thread owns one 32-element block: `mlive * rows_per_pos * (row_dim / 32)`
// threads in all, 160 for a draft kv and 10240 for a 5 x 64 x 512 q.
void stage_ropequant(uint gid, uint tid) {
    const uint ml    = mlive(pc.m);
    const uint nblk  = pc.k / kBlockK;                  // blocks in one row
    const uint total = ml * pc.rows * nblk;
    const uint g     = gid * kWg + tid;
    if (g >= total) return;

    const uint blk  = g % nblk;
    const uint row  = g / nblk;                          // flat [m][rows_per_pos]
    const uint m    = row / pc.rows;                     // draft position
    const uint rr   = row % pc.rows;                     // row within the position
    const bool xb   = in_bf16(pc.flags);
    const bool yb   = out_bf16(pc.flags);
    const uint d0   = blk * kBlockK;
    const uint64_t idx = act_index(m, pc.x_stride, rr * pc.k + d0);

    float v[32];
    act_load32(Ptr[kRopeX], idx, xb, v);

    const uint rope_lo  = pc.k - pc.rope_dim;
    const uint half_dim = pc.rope_dim / 2;
    const bool inv      = pc.inverse != 0u;
    [unroll] for (uint i = 0; i < 32; i += 2) {
        const uint d = d0 + i;
        if (d < rope_lo) continue;
        const float2 cs = rope_cs(Ptr[kRopeTab], m, half_dim, (d - rope_lo) >> 1);
        rope_pair(cs.x, cs.y, v[i], v[i + 1], inv);
        // apply_rotary_emb writes back into the bf16 tensor it was handed.
        v[i]     = bf16_round(v[i]);
        v[i + 1] = bf16_round(v[i + 1]);
    }
    if (pc.quant != 0u) dspark_act_quant32(v);

    const uint64_t oidx = act_index(m, pc.y_stride, rr * pc.k + d0);
    [unroll] for (uint i = 0; i < 32; ++i)
        act_store(Ptr[kRopeY], oidx + i, yb, v[i]);
}

// --- stage 3: wo_a, the grouped matmul ---------------------------------------
// `o[m][g][r] = sum_d o_in[m][g][d] * wo_a[g][r][d]`, g = 0..7, r = 0..1023,
// d = 0..4095 -- `torch.einsum("bsgd,grd->bsgr", o, wo_a)` with the weight seen
// as the flat `[8192][4096]` fp8 E4M3 matrix it is stored as, row `g * 1024 + r`,
// with its `[256][128]` UE8M0 scale plane (scale_cols = 128).
//
// fp8 weight decoded exactly, NO activation quantisation: see the file header.
// A workgroup covers `(256 / LanesPerRow) * RowsPerLane` consecutive rows and a
// group is 1024 rows, so a workgroup never straddles two groups and can stage
// the one 4096-wide slice of `o` its group needs -- which is why `group` is
// derived from the workgroup's base row and not per lane.
void stage_woa(uint gid, uint tid) {
    // Filled in uniform control flow; the first tile barrier below is the one
    // that makes it visible before any read (docs/p2_decode.md §3.2).
    fp8_table_fill(tid);
    const uint ml      = mlive(pc.m);
    const bool xb      = in_bf16(pc.flags);
    const bool yb      = out_bf16(pc.flags);
    const uint base    = pc.row_base + gid * rows_per_wg();
    const uint group   = base / pc.rows_per_group;
    const uint sublane = tid % LanesPerRow;
    const uint rowBase = base + (tid / LanesPerRow) * RowsPerLane;

    float acc[kMaxRowsPerLane][kMaxDraftM];
    [unroll] for (uint r0 = 0; r0 < kMaxRowsPerLane; ++r0)
        [unroll] for (uint m0 = 0; m0 < kMaxDraftM; ++m0) acc[r0][m0] = 0.0f;

    for (uint k0 = 0; k0 < pc.k; k0 += kTileK) {
        GroupMemoryBarrierWithGroupSync();
        stage_tile(Ptr[kX], pc.x_stride, group * pc.k, pc.k, k0, ml, xb, tid);
        GroupMemoryBarrierWithGroupSync();

        for (uint blk = sublane; blk < kTileBlocks; blk += LanesPerRow) {
            const uint kd = k0 + blk * kBlockK;
            if (kd >= pc.k) continue;
            [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) {
                if (r >= RowsPerLane) continue;
                const uint row = rowBase + r;
                if (row >= pc.rows) continue;
                // Decoded once per (row, block), reused for every draft column.
                float wv[32];
                int   e;
                fp8_weight_block(Ptr[kW], Ptr[kS], row, pc.k, kd, pc.scale_cols, wv, e);
                [unroll] for (uint m = 0; m < kMaxDraftM; ++m) {
                    if (m >= ml) continue;
                    float x[32];
                    read_x_block(m, blk, x);
                    float p = 0.0f;
                    [unroll] for (uint i = 0; i < 32; ++i) p += wv[i] * x[i];
                    // No activation scale: gXS is 1.0 with ActQuant = 0, and the
                    // staged value is `o` rounded to bf16, which is exact -- the
                    // reference's `o` is a bf16 tensor. The WEIGHT scale is a
                    // power of two and rejoins exactly through ldexp.
                    acc[r][m] += ldexp(p, e);
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) {
        if (r >= RowsPerLane) continue;
        [unroll] for (uint m = 0; m < kMaxDraftM; ++m) {
            if (m >= ml) continue;
            const float y = row_reduce(acc[r][m], tid, sublane);
            if (sublane == 0 && rowBase + r < pc.rows)
                act_store(Ptr[kY], act_index(m, pc.y_stride, rowBase + r), yb, y);
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0)      stage_gemv(gid.x, tid);
    else if (Stage == 1) stage_rmsnorm(gid.x, tid);
    else if (Stage == 2) stage_ropequant(gid.x, tid);
    else                 stage_woa(gid.x, tid);
}
