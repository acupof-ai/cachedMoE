// Shared pieces of the batched prefill kernels (docs/p3_prefill.md, design
// §7.13).
//
// Deliberately NOT an include of attn_common.slang or moe_common.slang. Those
// two files belong to the attention and MoE tracks and change under their own
// sweeps; the prefill kernels need a dozen of their functions with exactly the
// same rounding rules, so the rules are restated here -- each with the name of
// the function it must stay bit-identical to -- rather than inherited through a
// dependency another track can move.
//
// Precision contract (docs/p3_prefill.md §1, design §2.4 / §6)
// -----------------------------------------------------------
// `generate.py` runs the reference under bf16, so every Linear output, the
// residual stream, the attention input and output, and every expert's output
// are bf16 tensors in the reference, while the mHC coefficients, the gate, the
// ratio > 1 compressor, SwiGLU and the MoE accumulation are fp32. The prefill
// kernels store fp32 everywhere and ROUND TO BF16 exactly where the reference
// holds a bf16 tensor, selected per dispatch with `kFlagRound`. p2_attention.md
// §3 measured what being more precise than the reference costs in front of a
// quantiser (1.5e-2 against 1.8e-3 on wq_b), and p2_decode.md §4 found the
// first-token near-ties are decided at that level.

// Specialisation ids 0..3 are gpu/vulkan/pipeline.h's PipelineSpec; `extra`
// starts at 4. The same contract attn_common.slang and moe_common.slang keep,
// so one Pipeline::create serves all three families.
[vk::constant_id(0)] const uint MUnused       = 1;
[vk::constant_id(1)] const uint LanesPerRow   = 32;   // design §7.1 rule 3
[vk::constant_id(2)] const uint RowsPerWgUnused = 8;
[vk::constant_id(3)] const uint SubgroupSize  = 32;
[vk::constant_id(4)] const uint Stage         = 0;
// 0 fp8 E4M3 + [R/32][K/32] UE8M0, 1 bf16, 2 fp32, 3 FP4 E2M1 + [R][K/32] UE8M0
[vk::constant_id(5)] const uint WFmt          = 0;
// 0 fp32 activation, 1 pre-quantised: E4M3 value as fp16 [n][K] + fp32 block
// scale [n][K/32] (what `pf_act_quant` writes)
[vk::constant_id(6)] const uint XFmt          = 0;
// Activation columns (tokens) one lane accumulates at once: the M of design
// §7.1 rule 7, generalised. The weight is read once per tile of TileM tokens.
[vk::constant_id(7)] const uint TileM         = 8;

static const uint kWg       = 256;
// A 1-D kernel whose workgroup count passes 65535 is dispatched as
// kWgRowX x ceil(n / kWgRowX) (PrefillRunner::record): gid.y counts rows of
// kWgRowX workgroups. Only kernels indexed through gid_linear may be split.
static const uint kWgRowX   = 16384;
uint gid_linear(uint3 gid) { return gid.y * kWgRowX + gid.x; }
static const uint kBlockK   = 32;
static const float kFp8MaxF = 448.0f;
static const float kActAmaxMin = 1e-4f;
static const float kE4M3Sub = 0.001953125f;   // 2^-9
static const float kE4M3MinN = 0.015625f;     // 2^-6
static const uint kMaxTile  = 32;

// Per-dispatch flags shared by every push struct that has a `flags` field.
static const uint kFlagAccumulate = 1u;   // y += result instead of y = result
static const uint kFlagRound      = 2u;   // round the written value onto bf16
static const uint kFlagGather     = 4u;   // x row m is x[Idx[idx_off + m]]
static const uint kFlagScatter    = 8u;   // y row m is y[Idx[idx_off + m]]
static const uint kFlagRowScale   = 16u;  // multiply row m's result by XS[m]
static const uint kFlagInverse    = 32u;  // RoPE: conjugate
static const uint kFlagRoundPre   = 64u;  // GEMM: round before out_scale too
static const uint kFlagFromJob    = 128u; // weight decode: addresses from the job table

// --- addressing -------------------------------------------------------------
uint4 load16(uint64_t a)    { return *(uint4*)a; }
uint  load_u8(uint64_t a)   { return (uint)(*(uint8_t*)a); }
uint  load_u16(uint64_t a)  { return (uint)(*(uint16_t*)a); }
uint  load_u32(uint64_t a)  { return *(uint*)a; }
int   load_i32(uint64_t a)  { return *(int*)a; }
float load_f32(uint64_t a)  { return *(float*)a; }
void  store_f32(uint64_t a, float v) { *(float*)a = v; }

// --- BF16 -------------------------------------------------------------------
// attn_common.slang `bf16_decode` / `bf16_encode` / `bf16_round`.
float bf16_decode(uint h) { return asfloat(h << 16); }
uint  bf16_encode(float f) {
    const uint b = asuint(f);
    return (b + 0x7FFFu + ((b >> 16) & 1u)) >> 16;
}
float bf16_round(float f) { return bf16_decode(bf16_encode(f)); }
float maybe_round(float f, uint flags) { return ((flags & kFlagRound) != 0u) ? bf16_round(f) : f; }

// --- FP8 E4M3 ---------------------------------------------------------------
// attn_common.slang `fp8_decode`.
float fp8_decode(uint b) {
    const uint e = (b >> 3) & 0xFu;
    const uint m = b & 7u;
    const float mag = (e == 0) ? (float(m) * kE4M3Sub)
                               : ((1.0f + float(m) * 0.125f) * ldexp(1.0f, int(e) - 7));
    return ((b & 0x80u) != 0) ? -mag : mag;
}

// attn_common.slang `act_scale`: fast_round_scale(amax, 1/448).
float act_scale(float amax) {
    const float v = max(amax, kActAmaxMin) * (1.0f / kFp8MaxF);
    const uint  b = asuint(v);
    const int   e = int((b >> 23) & 0xFFu) - 127 + (((b & 0x7FFFFFu) != 0u) ? 1 : 0);
    return ldexp(1.0f, e);
}

// attn_common.slang `fp8_round`: round-to-nearest-even onto E4M3, branch free,
// for |v| already clamped to 448.
static const float kE4M3SubMagic = 24576.0f;
float fp8_round(float v) {
    const float a = abs(v);
    const uint  b = asuint(a);
    const uint  r = (b + 0x7FFFFu + ((b >> 20) & 1u)) & 0xFFF00000u;
    const float mag = (a < kE4M3MinN) ? ((a + kE4M3SubMagic) - kE4M3SubMagic) : asfloat(r);
    return asfloat(asuint(mag) | (asuint(v) & 0x80000000u));
}

// The byte for a value already on the E4M3 grid (moe_common.slang
// `fp8_e4m3_pack`).
uint fp8_pack(float v) {
    const float a = abs(v);
    const uint  s = (v < 0.0f) ? 0x80u : 0u;
    if (a < kE4M3MinN) return s | uint(a * 512.0f + 0.5f);
    const uint ab = asuint(a);
    const uint e  = ((ab >> 23) & 0xFFu) - 127u + 7u;
    const uint m  = (ab >> 20) & 7u;
    return s | (e << 3) | m;
}

// --- FP4 E2M1 ---------------------------------------------------------------
// moe_common.slang kE2M1 (low nibble = even element along K).
static const float kE2M1[16] = { 0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 };
// attn_common.slang `fp4_round` / `fp4_code`.
static const float kFp4Mag[8]   = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
static const float kFp4Bound[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
float fp4_round(float v) {
    const float mag = abs(v);
    uint i = 0;
    [unroll] for (uint j = 0; j < 7; ++j) if (kFp4Bound[j] < mag) i = j + 1;
    if (i < 7 && mag == kFp4Bound[i] && (i & 1u) != 0u) i += 1;
    return (v < 0.0f) ? -kFp4Mag[i] : kFp4Mag[i];
}
uint fp4_code(float v) {
    const float mag = abs(v);
    uint i = 0;
    [unroll] for (uint j = 1; j < 8; ++j) if (kFp4Mag[j] <= mag) i = j;
    return (v < 0.0f) ? (i | 8u) : i;
}

// --- RoPE -------------------------------------------------------------------
// attn_common.slang `rope_pair`: adjacent elements (lo + 2j, lo + 2j + 1).
void rope_pair(float c, float s, inout float re, inout float im, bool inverse) {
    const float sn = inverse ? -s : s;
    const float a = re * c - im * sn;
    const float b = re * sn + im * c;
    re = a; im = b;
}

// design §2.4: gate clamped above only, up both sides, then SwiGLU.
float swiglu(float gate, float up, float limit) {
    const float g = min(gate, limit);
    const float u = clamp(up, -limit, limit);
    return (g / (1.0f + exp(-g))) * u;
}

// --- reductions (LDS trees), attn_common.slang `wg_reduce` / `wg_max` -------
groupshared float gRed[256];
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
