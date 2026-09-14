// Shared pieces of the two FP4 MoE dispatches of design §7.9.
//
// Weight addressing (design §5.1 v0.5, §5.3): an expert's six tensors live in
// the slab slot exactly as they lie in the safetensors shard, and the GPU-side
// pointer table holds one device address per part:
//
//     table[(layer * experts_per_layer + expert) * 6 + part]
//     part order = w1.weight, w1.scale, w2.weight, w2.scale, w3.weight, w3.scale
//
// Those addresses are `slot_base + run.slot_offset + part.skew`. skew is a
// multiple of 8 (safetensors offsets are 8-byte multiples, sector-aligned run
// starts are 4096-byte multiples), so every load below declares alignment 8 and
// never 16. This is the single hardest constraint the no-repack decision of
// design §5.1 puts on the kernels.
//
// FP4 E2M1: two elements per byte, LOW nibble = the EVEN element along K
// (design appendix A, pinned by cpu/dequant.cpp and the L0 oracle). One UE8M0
// exponent per 32 elements along K; the block sum is scaled with ldexp, which
// is exact, so the kernel and cpu/gemv_fp4_ref differ only in summation order.

// Specialisation constants; ids match gpu/vulkan/pipeline.h PipelineSpec.
[vk::constant_id(0)] const uint M            = 1;    // accumulators per lane (design §7.1 rule 7)
[vk::constant_id(1)] const uint LanesPerRow  = 32;   // A/B over {16, 32, 64} (rule 3)
[vk::constant_id(2)] const uint RowsPerWgUnused = 8; // reserved: rows_per_wg is WG/LanesPerRow here
[vk::constant_id(3)] const uint SubgroupSize = 32;   // applied via VK_EXT_subgroup_size_control
[vk::constant_id(4)] const uint DecodeMode   = 1;    // 0 = const table, 1 = arithmetic, 2 = select tree
[vk::constant_id(5)] const uint HPrecision   = 0;    // 0 = fp16 h, 1 = fp32 h
// Weight rows per lane: the dual of M. M reads the weights once and uses them
// for M activation columns; RowsPerLane reads the activation once and uses it
// for RowsPerLane weight rows. Both are pure register-blocking knobs.
[vk::constant_id(6)] const uint RowsPerLane  = 1;

static const uint kWgSize    = 256;
static const uint kMaxM      = 6;      // design §1.2: verify batch <= 6
static const uint kMaxRows   = 16;     // upper bound on RowsPerLane (host-checked)
static const uint kBlockK    = 32;     // elements per UE8M0 scale
static const uint kChunkBlks = 64;     // blocks staged per outer iteration

// design appendix A: {0, .5, 1, 1.5, 2, 3, 4, 6} with a sign bit.
static const float kE2M1[16] = { 0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 };

// The same sixteen values as fp16, for the packed path. Every E2M1 value is
// exact in fp16, so this is a pure format change; the constant-array indexing
// is the form P1 measured as 1.4x faster than any arithmetic decode
// (kernel_p1.md §3.2), kept here rather than converting kE2M1 per element.
static const float16_t kE2M1H[16] = {
    float16_t( 0.0),  float16_t( 0.5),  float16_t( 1.0),  float16_t( 1.5),
    float16_t( 2.0),  float16_t( 3.0),  float16_t( 4.0),  float16_t( 6.0),
    float16_t(-0.0),  float16_t(-0.5),  float16_t(-1.0),  float16_t(-1.5),
    float16_t(-2.0),  float16_t(-3.0),  float16_t(-4.0),  float16_t(-6.0) };

// Two consecutive E2M1 elements of `pw` (elements 2j and 2j+1 along K) as a
// half2, ready for v_pk_fma_f16 against the activation's own fp16 bits.
float16_t2 fp4_pair_h2(uint pw, uint j) {
    const uint n = pw >> (8 * j);
    return float16_t2(kE2M1H[n & 0xFu], kE2M1H[(n >> 4) & 0xFu]);
}

// The three decode variants design §7.1 rule 5 asks to A/B.
float fp4_decode(uint nib) {
    if (DecodeMode == 0) {
        return kE2M1[nib & 15];
    } else if (DecodeMode == 1) {
        // Build the fp32 bit pattern directly: e2m1 exponent 0 is the
        // subnormal pair {0, 0.5}, otherwise (1 + m/2) * 2^(e-1).
        uint e    = (nib >> 1) & 3;
        uint mbit = nib & 1;
        uint bits = (e == 0) ? (mbit != 0 ? 0x3F000000u : 0u)
                             : (((e + 126u) << 23) | (mbit << 22));
        return asfloat(bits | ((nib & 8) << 28));
    } else {
        float mag = (nib & 4) != 0
            ? ((nib & 2) != 0 ? ((nib & 1) != 0 ? 6.0f : 4.0f) : ((nib & 1) != 0 ? 3.0f : 2.0f))
            : ((nib & 2) != 0 ? ((nib & 1) != 0 ? 1.5f : 1.0f) : ((nib & 1) != 0 ? 0.5f : 0.0f));
        return (nib & 8) != 0 ? -mag : mag;
    }
}

// 16 B of packed FP4 (32 elements) from a device address that is only
// guaranteed 8-byte aligned. Slang emits `OpLoad ... Aligned 4` for a pointer
// dereference, which is exactly the promise the skew arithmetic can keep and
// is enough for RDNA's global_load_dwordx4.
uint4 load_fp4_block(uint64_t addr) {
    return *(uint4*)addr;
}

// One UE8M0 exponent byte.
uint load_scale_byte(uint64_t addr) {
    return (uint)(*(uint8_t*)addr);
}

// Half `j` of a packed pair buffer entry.
float unpack_half(uint4 v, uint j) {
    uint w = v[j >> 1];
    return f16tof32((j & 1) != 0 ? (w >> 16) : (w & 0xFFFFu));
}

// design §2.4: gate is clamped above only, up on both sides, then SwiGLU.
float swiglu(float gate, float up, float limit) {
    float g = min(gate, limit);
    float u = clamp(up, -limit, limit);
    return (g / (1.0f + exp(-g))) * u;
}

// ---------------------------------------------------------------------------
// P2 additions (design §7.9 speculative-decode work, docs/kernel_p2_moe.md)
// ---------------------------------------------------------------------------

// How the activation operand (x in dispatch A, h in dispatch B) reaches the
// FMA. design §7.1 rule 6 and kernel_p1.md §3.3: at M=6 a lane issues M x 4
// `uint4` loads of activation against 2 loads of weight, so the activation is
// the bottleneck and it is an *instruction* bottleneck, not a byte one.
//   0 Global   one global load per (m, 8 elements)  -- the P1 kernel
//   1 Lds      a K-tile of LanesPerRow blocks staged in LDS by the whole
//              workgroup and re-read from LDS by every row group
//   2 LdsF16x2 the same tile, consumed as packed half2 (v_pk_fma_f16) with an
//              fp16 block accumulator, fp32 across blocks
//   3 LdsI8    the tile is quantised to int8 per (m, block-32) during staging
//              and consumed with dot4add_i8packed (design §7.9 "int8 dot4 as
//              the M>=4 fallback")
//   4 GlobF16x2 no staging and no barriers: the activation is read from global
//              exactly as in mode 0, but kept in its fp16 bit pattern and
//              consumed as half2 (v_pk_fma_f16). This halves the FMA stream and
//              removes the f16tof32 per element, which together are ~70% of the
//              M-proportional VALU work at M=6 (docs/kernel_p2_moe.md §3).
//   5 GlobI8   no staging either: the lane quantises its own 32-element block
//              to int8 in registers and uses dot4add_i8packed, so one
//              instruction covers four MACs for every weight row the lane owns.
[vk::constant_id(7)] const uint XMode = 0;
// True when XMode stages the activation in LDS, which is what costs the two
// workgroup barriers per K-chunk.
static bool x_mode_tiles() { return XMode >= 1 && XMode <= 3; }

// design §7.9 v0.6: the reference quantises `silu(gate)*up` to fp8 E4M3 with a
// UE8M0 block-32 scale before w2. Where that happens:
//   0 Off    v0.5 behaviour, h goes to w2 unquantised (the P1 kernel)
//   1 InB    dispatch B reproduces the round trip from the fp16 h it already
//            reads; no layout change, no extra traffic, every variant supports it
//   2 Fp8H   dispatch A writes fp8 h plus a UE8M0 scale plane and dispatch B
//            consumes those; halves the h bytes, needs >= 32 rows per workgroup
[vk::constant_id(8)] const uint HQuant = 0;

// Compile the FP8 E4M3 weight path (the shared expert of design §7.9). A slot
// is fp8 when bit 31 of its `Ids` entry is set, so one dispatch pair covers the
// six routed FP4 experts and the fp8 shared expert.
[vk::constant_id(9)] const uint Fp8Slots = 0;

static const uint  kSlotFp8Bit  = 0x80000000u;
static const float kFp8ActMax   = 448.0f;      // inference/kernel.py fp8_max
static const float kFp8ActMin   = 1e-4f;       // act_quant_kernel's amax floor
static const uint  kFp8TableN   = 256;

// One UE8M0 exponent byte from a device address, as an int already unbiased.
int load_scale_exp(uint64_t addr) { return (int)load_scale_byte(addr) - 127; }

// 16 B from an 8-byte-aligned device address -- see load_fp4_block.
uint4 load_raw16(uint64_t addr) { return *(uint4*)addr; }

// --- FP8 E4M3 ---------------------------------------------------------------

// The arithmetic decode, used once per thread to fill the 256-entry LDS table
// design §7.1 rule 5 asks for. e4m3fn: no infinities, 0x7F/0xFF are NaN, max
// finite 448. The NaN codes decode to +-480 here; weights never contain them
// and checking would cost a compare per element.
float fp8_e4m3_decode(uint b) {
    const uint e = (b >> 3) & 0xFu;
    const uint m = b & 7u;
    const float mag = (e == 0) ? (float(m) * 0.001953125f)              // m * 2^-9
                               : ((1.0f + float(m) * 0.125f) * ldexp(1.0f, int(e) - 7));
    return ((b & 0x80u) != 0) ? -mag : mag;
}

// Round `v` (already clamped to +-448) onto the E4M3 grid, in fp32. This is the
// quant/dequant round trip `act_quant(..., inplace=True)` performs; packing to
// a byte is a separate step because dispatch B never needs the byte.
float fp8_e4m3_round(float v) {
    const float a = abs(v);
    const float sgn = (v < 0.0f) ? -1.0f : 1.0f;
    if (a < 0.015625f) {                        // < 2^-6: subnormal, quantum 2^-9
        // Adding 2^23 makes the fp32 ulp exactly 1, so the add itself is the
        // round-to-nearest-even we want.
        const float t = a * 512.0f;
        const float r = (t + 8388608.0f) - 8388608.0f;
        return sgn * r * 0.001953125f;
    }
    // Normal: keep 3 mantissa bits of the fp32 significand, round to nearest
    // even at the 20 bits being dropped.
    const uint ab = asuint(a);
    const uint r  = (ab + 0x0007FFFFu + ((ab >> 20) & 1u)) & 0xFFF00000u;
    return sgn * asfloat(r);
}

// The byte for a value already on the E4M3 grid and inside +-448.
uint fp8_e4m3_pack(float v) {
    const float a = abs(v);
    const uint  s = (v < 0.0f) ? 0x80u : 0u;
    if (a < 0.015625f) return s | uint(a * 512.0f + 0.5f);
    const uint ab = asuint(a);
    const uint e  = ((ab >> 23) & 0xFFu) - 127u + 7u;
    const uint m  = (ab >> 20) & 7u;
    return s | (e << 3) | m;
}

// `fast_round_scale` from inference/kernel.py: the smallest power of two >=
// amax/max, computed on the bit pattern because ceil(log2(x)) in floating point
// gets the exact powers of two wrong (design §2.4, tools/dsref.py).
float fast_round_scale(float amax, float max_inv) {
    const uint bits = asuint(amax * max_inv);
    const int  e    = int((bits >> 23) & 0xFFu) - 127 + (((bits & 0x7FFFFFu) != 0u) ? 1 : 0);
    return ldexp(1.0f, e);
}

// --- FP4 as int8 (design §7.9's int8 dot4 path) ------------------------------
// Every E2M1 magnitude is a multiple of 0.5, so 2*value is an exact int8 and
// the weight side of the int8 path costs nothing in accuracy; only the
// activation is quantised. The factor 2 is folded into the block exponent.
static const int kE2M1x2[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

// Four consecutive nibbles of `pw` (elements 4*g .. 4*g+3 along K) as packed
// signed int8, ready for dot4add_i8packed.
uint fp4_nibbles_to_i8x4(uint pw, uint g) {
    const uint n = pw >> (16u * g);
    return (uint(kE2M1x2[(n >>  0) & 15u]) & 0xFFu)
         | ((uint(kE2M1x2[(n >>  4) & 15u]) & 0xFFu) << 8)
         | ((uint(kE2M1x2[(n >>  8) & 15u]) & 0xFFu) << 16)
         | ((uint(kE2M1x2[(n >> 12) & 15u]) & 0xFFu) << 24);
}
