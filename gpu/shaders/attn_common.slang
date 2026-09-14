// Shared pieces of the non-MoE decode path (design §7.2-§7.8, §7.11).
//
// Deliberately separate from moe_common.slang: that file is the FP4 expert
// path and is owned by the MoE track. Everything here is FP8 E4M3 weights,
// BF16 weights, or fp32 mHC arithmetic.
//
// The three things this file exists to get exactly right
// -----------------------------------------------------
//
// 1. **The activation quantisation nobody wrote down.** `inference/model.py`'s
//    `linear()` dispatches on the weight dtype, and for BOTH fp8 and fp4
//    weights it first does `act_quant(x, 32, "ue8m0", e8m0)` -- the activation
//    is rounded onto the FP8 E4M3 grid with a per-32-element power-of-two
//    scale before the GEMM. design §6 records that round trip only for the MoE
//    intermediate `h`; in fact **every fp8 Linear in the attention path does
//    it too**: wq_a, wq_b, wkv, wo_b and the indexer's wq_b. Skipping it does
//    not cost a little precision, it computes a different function -- E4M3 has
//    three mantissa bits, so an element moves by up to 6%.
//    `wo_a` is the exception: its module is declared bf16 (convert.py would
//    have dequantised it) and `Attention.forward` reaches its weight through
//    an einsum rather than `linear`, so its activation is NOT quantised even
//    though the checkpoint stores the weight as fp8.
//    Cost here: zero. With LanesPerRow = 32 and a 32-element UE8M0 block, one
//    lane already owns exactly one whole block of x, so the block amax is a
//    register reduction over values it has loaded anyway.
//
// 2. **RoPE pairs adjacent elements, not (d, d+32).** `apply_rotary_emb` does
//    `view_as_complex(x.unflatten(-1, (-1, 2)))`, so within the last
//    `rope_head_dim` = 64 dims the pairs are (448+2j, 448+2j+1) and frequency
//    j drives pair j. design §7.3's "rows (d, d+32) rotate in pairs" is wrong;
//    following it would rotate the wrong pairs.
//
// 3. **FP8 E4M3 decode from a 256-entry LDS table** (design §7.1 rule 5). The
//    FP4 sweep of kernel_p1.md §3.2 found the table beats arithmetic by 1.4x
//    on identical byte counts, and an fp8 table is 512 B of LDS against a
//    5-instruction decode, so the same call goes the same way. `fp8_decode` is
//    the arithmetic form, used once per thread to build that table.

// Specialisation constant ids follow gpu/vulkan/pipeline.h PipelineSpec, which
// fixes 0..3 and starts `extra` at 4 -- the same contract moe_common.slang
// uses, so one Pipeline::create serves both kernel families.
[vk::constant_id(0)] const uint M               = 1;   // batch; 1 until DSpark (§7.12)
[vk::constant_id(1)] const uint LanesPerRow     = 32;  // design §7.1 rule 3
[vk::constant_id(2)] const uint RowsPerWgUnused = 8;   // reserved: rows/wg is WG/LanesPerRow
[vk::constant_id(3)] const uint SubgroupSize    = 32;  // VK_EXT_subgroup_size_control
[vk::constant_id(4)] const uint Stage           = 0;   // multi-stage kernels
[vk::constant_id(5)] const uint ActQuant        = 1;   // 1 = quantise x to fp8 E4M3 (see above)
// Weight rows per lane, the dual of M: x is staged once per workgroup and used
// RowsPerLane times more. The attention GEMVs are tall and thin (wq_b is
// 32768 x 1280), so a workgroup that owns only 256/LanesPerRow rows spends a
// third of its L2 traffic re-staging the same 1280-wide activation. Same knob
// moe_common.slang carries for the FP4 path.
[vk::constant_id(6)] const uint RowsPerLane     = 1;
// Attention heads one sparse_attn workgroup covers. Its only effect is how
// many times the 320 KiB of KV is pulled through L2 -- once per workgroup --
// and it is opt-in: sparse_attn.slang ignores it unless the caller pushes a
// head count. Nothing else in this family reads it.
[vk::constant_id(7)] const uint HeadsPerWg      = 1;
// Whether `row_reduce` collapses to one WaveActiveSum. Per stage, and measured
// rather than reasoned about: see the function itself.
[vk::constant_id(8)] const uint WaveReduce      = 1;

static const uint  kWg          = 256;
static const uint  kMaxRowsPerLane = 4;
static const uint  kBlockK    = 32;      // elements per UE8M0 scale
static const uint  kFp8Max    = 448;
static const float kFp8MaxF   = 448.0f;
static const float kActAmaxMin = 1e-4f;  // act_quant_kernel's floor
static const float kE4M3Sub   = 0.001953125f;   // 2^-9, the subnormal step
static const float kE4M3MinN  = 0.015625f;      // 2^-6, the smallest normal

// --- addressing -------------------------------------------------------------
// Weights are reached through buffer_device_address, not descriptors: the
// pinned set is ~17.7 GB across many 2 GiB slabs (design §1.1, §5.2) and one
// descriptor per tensor per layer would be thousands of them. Slang emits
// `OpLoad ... Aligned 4` for a raw pointer dereference, which is all the 8-byte
// skew of design §5.1 can promise and all RDNA's global_load_dwordx4 needs.
uint4 load16(uint64_t addr)   { return *(uint4*)addr; }
uint  load_u8(uint64_t addr)  { return (uint)(*(uint8_t*)addr); }
uint  load_u16(uint64_t addr) { return (uint)(*(uint16_t*)addr); }
uint  load_u32(uint64_t addr) { return *(uint*)addr; }
float load_f32(uint64_t addr) { return *(float*)addr; }

// --- FP8 E4M3 ---------------------------------------------------------------

// e4m3fn: bias 7, no infinities, 0x7F/0xFF are NaN, max finite 448. Exponent 0
// is subnormal (m * 2^-9). The NaN codes come out as +-480; a weight never
// holds one and testing per element would cost more than it saves.
float fp8_decode(uint b) {
    const uint e = (b >> 3) & 0xFu;
    const uint m = b & 7u;
    const float mag = (e == 0) ? (float(m) * kE4M3Sub)
                               : ((1.0f + float(m) * 0.125f) * ldexp(1.0f, int(e) - 7));
    return ((b & 0x80u) != 0) ? -mag : mag;
}

// The 256-entry decode table plus the reduction scratch. Declared here rather
// than passed in: Slang would be free to copy an `inout float[256]` parameter,
// and 2 KiB of LDS in a kernel that does not use the table is cheaper than
// finding out that it did.
groupshared float gTbl[256];
groupshared float gRed[256];

// Must be called by a full 256-thread workgroup in uniform control flow,
// followed by a barrier.
void fp8_table_fill(uint tid) { gTbl[tid] = fp8_decode(tid); }

// `fast_round_scale(amax, 1/448)` from inference/kernel.py: the smallest power
// of two >= amax/448, read straight off the fp32 bit pattern. Doing it through
// log2 gets the exact powers of two wrong (design §2.4).
float act_scale(float amax) {
    const float v = max(amax, kActAmaxMin) * (1.0f / kFp8MaxF);
    const uint  b = asuint(v);
    const int   e = int((b >> 23) & 0xFFu) - 127 + (((b & 0x7FFFFFu) != 0u) ? 1 : 0);
    return ldexp(1.0f, e);
}

// Round-to-nearest-even onto the E4M3 grid and return the BYTE, for |v|
// already clamped to 448. This is what `torch.Tensor.to(torch.float8_e4m3fn)`
// does. Everything that needs the rounded *value* goes through
// `fp8_decode(fp8_encode_rn(v))`, so there is exactly one rounding rule here
// and the value and the byte can never disagree -- which matters because the
// KV cache write of design §7.4 stores the byte while the attention that reads
// it back consumes the value.
uint fp8_encode_rn(float v) {
    const uint sign = (asuint(v) >> 24) & 0x80u;
    const float a = abs(v);
    uint mag;
    if (a < kE4M3MinN) {                    // subnormal: multiples of 2^-9
        const float t = a * 512.0f;
        const float f = floor(t);
        const float d = t - f;
        const float n = (d > 0.5f) ? (f + 1.0f)
                      : ((d < 0.5f) ? f : ((fmod(f, 2.0f) == 0.0f) ? f : (f + 1.0f)));
        mag = (uint)n;                      // 8 carries into exponent 1, which is 2^-6
    } else {
        const uint b = asuint(a);
        int  e    = int((b >> 23) & 0xFFu);
        uint keep = (b & 0x7FFFFFu) >> 20;  // the three mantissa bits we keep
        const uint rem = b & 0xFFFFFu;      // round + sticky
        if (rem > 0x80000u || (rem == 0x80000u && (keep & 1u) != 0u)) {
            keep += 1u;
            if (keep == 8u) { keep = 0u; e += 1; }
        }
        int ee = e - 127 + 7;               // the E4M3 exponent field
        if (ee > 15 || (ee == 15 && keep == 7)) { ee = 15; keep = 6; }   // 448
        if (ee < 0) { ee = 0; keep = 0; }
        mag = (uint(ee) << 3) | keep;
    }
    return sign | mag;
}

// The same rounding, branchless, returning the VALUE and never the byte.
//
// `fp8_encode_rn` above is the definition; this is what the GEMV activation
// staging actually runs, and it is here because the staging turned out to be
// the thing wq_b and wo_b were spending their time on rather than reading
// weights. wo_b stages 8192 activations across 256 threads, so every thread
// runs this 32 times before the first weight byte is touched; at ~25
// instructions and three branches the encode alone was more work than the 256
// multiply-adds that lane then performs. This form is nine instructions and
// branch-free.
//
// Two ranges, selected rather than branched:
//
//   * |v| >= 2^-6 is a normal. Keeping three mantissa bits with
//     round-to-nearest-even is an integer add of (0x7FFFF + lsb) followed by a
//     mask -- the carry out of the mantissa walks into the exponent by itself,
//     which is why no exponent arithmetic appears. The caller has already
//     clamped to +-448, which is exactly representable, so the add can never
//     reach the 0x7F/0xFF NaN codes.
//   * |v| < 2^-6 is a subnormal, i.e. a multiple of 2^-9. `(a + M) - M` with
//     M = 1.5 * 2^14 rounds to that grid with round-to-nearest-even in two
//     instructions: every number in [2^14, 2^15) has ulp 2^-9, and a is small
//     enough that a + M stays in that binade. 8 * 2^-9 comes out as 2^-6,
//     which is the smallest normal and the same answer `fp8_encode_rn` gives
//     by carrying its mantissa into the exponent field.
static const float kE4M3SubMagic = 24576.0f;   // 1.5 * 2^14

float fp8_round(float v) {
    const float a = abs(v);
    const uint  b = asuint(a);
    const uint  r = (b + 0x7FFFFu + ((b >> 20) & 1u)) & 0xFFF00000u;
    const float mag = (a < kE4M3MinN) ? ((a + kE4M3SubMagic) - kE4M3SubMagic)
                                      : asfloat(r);
    return asfloat(asuint(mag) | (asuint(v) & 0x80000000u));
}

// One 32-element activation block through `act_quant`, in place.
// `ActQuant == 0` leaves it alone: that is the wo_a path, and the A/B that
// shows what the round trip is worth.
void act_quant_block(inout float v[32]) {
    if (ActQuant == 0) return;
    float amax = 0.0f;
    [unroll] for (uint i = 0; i < 32; ++i) amax = max(amax, abs(v[i]));
    const float s = act_scale(amax);
    const float inv = 1.0f / s;
    [unroll] for (uint i = 0; i < 32; ++i)
        v[i] = fp8_round(clamp(v[i] * inv, -kFp8MaxF, kFp8MaxF)) * s;
}

// --- FP4 E2M1 ---------------------------------------------------------------
// The compressor's compressed KV (block 16, E4M3 scale) and the indexer's q/k
// (block 32, UE8M0 scale) both quantise to E2M1 -- design §7.4, §11.3. This is
// the encode side, which moe_common.slang does not have: the MoE path only ever
// DECODES fp4, because its weights arrive quantised from the checkpoint.
//
// Eight magnitudes and the seven midpoints between them. Ties round to even
// MANTISSA, which alternates along the list, so a tie on an odd boundary index
// takes the upper value: 0.25 -> 0, 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2.0,
// 2.5 -> 2.0, 3.5 -> 4.0, 5.0 -> 4.0. tools/dsref.py `_round_to_fp4` is the
// same rule and the oracle asserts against the reference's own bytes.
static const float kFp4Mag[8]   = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
static const float kFp4Bound[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};

// Round-to-nearest-even onto the 16 E2M1 values. `v` must already be clamped
// to +-6.
float fp4_round(float v) {
    const float mag = abs(v);
    uint i = 0;
    [unroll] for (uint j = 0; j < 7; ++j) if (kFp4Bound[j] < mag) i = j + 1;
    if (i < 7 && mag == kFp4Bound[i] && (i & 1u) != 0u) i += 1;
    return (v < 0.0f) ? -kFp4Mag[i] : kFp4Mag[i];
}

// The 4-bit code of an already-rounded E2M1 value: the magnitude's index in
// `kFp4Mag` (which is the E2M1 bit pattern) plus the sign bit. Negative zero
// codes as +0, the way `_pack_fp4`'s `v < 0` test does.
uint fp4_code(float v) {
    const float mag = abs(v);
    uint i = 0;
    [unroll] for (uint j = 1; j < 8; ++j) if (kFp4Mag[j] <= mag) i = j;
    return (v < 0.0f) ? (i | 8u) : i;
}

// --- BF16 -------------------------------------------------------------------

float bf16_decode(uint h)   { return asfloat(h << 16); }
uint  bf16_encode(float f) {
    // round to nearest even, the same rule torch uses for fp32 -> bf16
    const uint b = asuint(f);
    return (b + 0x7FFFu + ((b >> 16) & 1u)) >> 16;
}

// Round an fp32 value onto the bf16 grid without narrowing the storage.
//
// Used at exactly one kind of place: the output of an RMSNorm, because
// `RMSNorm.forward` ends in `.to(dtype)` and `generate.py` runs with the
// default dtype set to bf16. Keeping our extra mantissa bits there is not a
// free precision win -- the value goes straight into `act_quant`, whose block
// amax then lands in a different power-of-two bracket than the reference's and
// re-rounds the entire 32-element block one grid step coarser. Measured on
// wq_b: relative L2 against the oracle went from 1.5e-2 to the 2e-3 bf16 noise
// floor when this was added. The residual stream itself stays fp32 (design §6),
// where the extra bits do not feed a quantiser.
float bf16_round(float f) { return bf16_decode(bf16_encode(f)); }

// --- reductions -------------------------------------------------------------

// Sum `v` across the LanesPerRow lanes of one row group and broadcast it.
//
// When a row group IS a subgroup -- LanesPerRow == SubgroupSize, which is the
// measured configuration (32 lanes a row, Wave32) -- this can be one
// `WaveActiveSum` and no barriers at all, instead of an LDS tree costing
// log2(LanesPerRow) + 2 = seven workgroup barriers per row. (It MUST be the
// tree when a 64-lane row spans two Wave32 subgroups, which is the case
// moe_gateup.slang's comment is about.)
//
// Which is faster is not a question of instruction count, and the sweep said
// so: the wave form is worth +32% on wq_b and +24% on wkv and **-12% on wq_a
// and -14% on wo_a**. The two that lose are the two whose workgroups each
// re-stage a wide activation (K = 5120 and 4096) over many workgroups; the
// barriers evidently pace those workgroups against each other in a way the
// memory system likes, and removing them lets them all pile into L2 at once.
// So it is a per-stage specialisation constant, with the numbers in
// gpu/vulkan/attn_kernels.cpp's kStages table, not a global improvement.
float row_reduce(float v, uint tid, uint sublane) {
    if (WaveReduce != 0 && LanesPerRow == SubgroupSize) return WaveActiveSum(v);
    gRed[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = LanesPerRow >> 1; s > 0; s >>= 1) {
        if (sublane < s) gRed[tid] = gRed[tid] + gRed[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = gRed[tid - sublane];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Sum over the whole workgroup, broadcast to every thread.
float wg_reduce(float v, uint tid) {
    gRed[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = kWg >> 1; s > 0; s >>= 1) {
        if (tid < s) gRed[tid] = gRed[tid] + gRed[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = gRed[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

float wg_max(float v, uint tid) {
    gRed[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = kWg >> 1; s > 0; s >>= 1) {
        if (tid < s) gRed[tid] = max(gRed[tid], gRed[tid + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = gRed[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// --- RoPE -------------------------------------------------------------------
// `RopeTab` is [rows][rope_head_dim/2][2] of (cos, sin) for the positions the
// dispatch needs, so the kernel never evaluates a transcendental and YaRN lives
// entirely on the host (design §2.1: base and original_seq_len differ between a
// window-only layer and a compressed one).
void rope_pair(float c, float s, inout float re, inout float im, bool inverse) {
    const float sn = inverse ? -s : s;
    const float a = re * c - im * sn;
    const float b = re * sn + im * c;
    re = a; im = b;
}
