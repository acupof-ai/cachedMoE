// Shared pieces of the DSpark draft path (design §7.12).
//
// What DSpark is, in one paragraph
// -------------------------------
// A draft cycle runs three `mtp` stages over M = 5 candidate positions, using
// the same template as the main model with the batch dimension switched on. The
// numbers are in design §7.12: `main_proj [5120 x 15360]` fp8 (79 MB) plus a
// norm produces `main_x`, each of the three stages runs Mega-mHC / Q / KV /
// window attention (KV = the 128-slot ring plus the five in-block positions) /
// O / gate(128, top-3) / FP4 MoE at M = 5, and the cycle closes with the 1.32 GB
// head read once for all five positions, five sequential Markov-head steps and
// one confidence projection. About 2.2 GB and 11 ms a cycle.
//
// Why this file is separate from attn_common.slang and fp8_gemv.slang
// ------------------------------------------------------------------
// It is not: it INCLUDES attn_common.slang and reuses everything there -- the
// fp8 E4M3 decode table, `act_scale`, `fp8_round`, `bf16_round`, `load16` /
// `load_u8` / `load_u16` / `load_u32` / `load_f32`, `row_reduce`, `wg_reduce`,
// `wg_max`, `rope_pair` -- because DSpark computes the same functions the decode
// path does, only with five activation columns instead of one. What lives here
// is the little that is genuinely DSpark-specific:
//
//   1. **M-strided activation addressing.** Every activation the draft path
//      touches is `[M][n]` row-major with an explicit element stride, so one
//      dispatch can point at a column of a wider tensor without a copy.
//   2. **A dtype flag on both ends of every kernel.** The rest of gpu/shaders
//      keeps the residual stream in fp32 and reads/writes fp32 activations. The
//      draft path cannot: the reference hands `sparse_attn` a bf16 q and a bf16
//      KV ring, `act_quant(..., inplace=True)` writes fp8-rounded values back
//      into a bf16 tensor, and `linear()` returns bf16. Rather than pick one and
//      lose either agreement or generality, every DSpark kernel takes
//      `kFlagInBf16` / `kFlagOutBf16` in its push constant and the host says
//      which buffer is which. fp32 in / fp32 out is the house convention and the
//      default; bf16 is what the attention family wants.
//
// What this file deliberately does NOT contain: the LDS staging arrays. LDS is
// the union over every pipeline built from a file, so a `groupshared` array
// declared here would be charged to dspark_attn.slang and dspark_head.slang too,
// neither of which stages an activation that way. The K-tile staging lives in
// dspark_gemv.slang, next to its only two users.
//
// Specialisation constants: ids 0..3 are the fixed contract of
// gpu/vulkan/pipeline.h's PipelineSpec and are declared by attn_common.slang
// lines 42-45 (`M`, `LanesPerRow`, `RowsPerWgUnused`, `SubgroupSize`); ids 4
// onwards are `PipelineSpec::extra` and attn_common.slang already names the four
// this family needs -- `Stage` (4), `ActQuant` (5), `RowsPerLane` (6),
// `HeadsPerWg` (7). Redeclaring them here would be a duplicate definition, and
// inventing a second numbering for the same four knobs would mean one
// `Pipeline::create` could not serve both kernel families, which is the whole
// point of the shared contract. So DSpark inherits 0..7 verbatim and adds
// nothing of its own; everything else it needs varies per dispatch rather than
// per pipeline and is therefore a push constant, not a specialisation constant.
//
// **`M` is the CEILING, `pc.m` is the live count.** Some of the draft path runs
// at M = 1 (`main_proj` + `main_norm`, and the `main_kv` half of the draft
// attention, both of which see only the committed main positions) and the rest
// at M = 5. `M` is baked into the pipeline, so honouring that literally would
// mean two pipelines for every affected stage. Instead every loop over the batch
// is bounded by `mlive()` = min(M, pc.m) and the host pushes the live count: one
// pipeline per stage, and the M = 1 dispatches waste nothing but a comparison,
// because at M = 1 the wasted work is proportional to the four columns that are
// never read -- which is zero, the loop simply stops.

#include "attn_common.slang"

// The largest batch a DSpark pipeline may be specialised to. design §1.2 puts M
// at 1..6 (5 draft positions, or 6 for a verify batch of k + 1), and the LDS and
// register budgets of dspark_gemv.slang are sized against this, so it is a hard
// compile-time bound rather than a hint: `DsparkRunner::create` rejects a spec
// whose `m` exceeds it.
static const uint kMaxDraftM = 6;

// Live activation columns this dispatch covers. See the header: `M` is what the
// pipeline was specialised to and `pc.m` is what the caller actually has.
uint mlive(uint pc_m) { return min(M, pc_m); }

// --- dtype flags ------------------------------------------------------------
// Bits of the `flags` field every DSpark push constant carries. Kept in one
// place because the same two bits mean the same two things in all four shaders.
static const uint kFlagInBf16  = 1u;   // the input activation is bf16, not fp32
static const uint kFlagOutBf16 = 2u;   // write the output as bf16, not fp32
// dspark_gemv stage 1 only: round an fp32 input onto the bf16 grid BEFORE the
// RMS statistic. The reference's `linear()` returns bf16, so `RMSNorm.forward`
// squares bf16 values; the rest of gpu/shaders keeps the GEMV output in fp32 and
// measures cos 0.999998 against the oracle either way (docs/p2_attention.md), so
// this is offered rather than imposed.
static const uint kFlagRoundIn = 4u;
// dspark_head stage 0 only: take the token id from `kMkTokenIn[pos]` instead of
// the push constant, so the five sequential Markov steps of design §7.12 can be
// recorded as five dispatches with no host round trip between them.
static const uint kFlagTokenFromBuf = 8u;

bool in_bf16(uint flags)  { return (flags & kFlagInBf16)  != 0u; }
bool out_bf16(uint flags) { return (flags & kFlagOutBf16) != 0u; }

// --- M-strided activation addressing ----------------------------------------
// An activation is `[M][n]` row-major with `stride` ELEMENTS between columns --
// not bytes, because the element width is the caller's dtype flag and the index
// arithmetic must not have to know it. `stride` is normally `n`; it is larger
// when the dispatch is pointing at a slice of a wider tensor (wo_a's group `g`
// is the 4096-wide slice at offset `g * 4096` of a 32768-wide `o`).
uint64_t act_index(uint m, uint stride, uint i) {
    return (uint64_t)m * (uint64_t)stride + (uint64_t)i;
}

float act_load(uint64_t base, uint64_t idx, bool bf16) {
    return bf16 ? bf16_decode(load_u16(base + idx * 2)) : load_f32(base + idx * 4);
}

void act_store(uint64_t base, uint64_t idx, bool bf16, float v) {
    if (bf16) *(uint16_t*)(base + idx * 2) = (uint16_t)bf16_encode(v);
    else      *(float*)(base + idx * 4)    = v;
}

// 32 consecutive activation elements -- one UE8M0 block -- into registers.
//
// Vectorised rather than looped because this is the single hottest load in the
// family: dspark_gemv stages `mlive * 32` of these per K-tile per workgroup, and
// fp8_gemv.slang's P2 pass (docs/p2_attention.md §7 item 1) found the staging,
// not the weight stream, to be what the GEMVs were spending their time on. Four
// `uint4` loads for bf16 and eight for fp32, against 32 scalar loads.
//
// `idx` is a multiple of 32 at every call site, so `base + idx * 2` is 64-byte
// aligned and `base + idx * 4` 128-byte aligned: every `load16` is a natural
// `global_load_dwordx4`.
void act_load32(uint64_t base, uint64_t idx, bool bf16, out float v[32]) {
    if (bf16) {
        const uint64_t a = base + idx * 2;
        [unroll] for (uint w = 0; w < 4; ++w) {
            const uint4 q = load16(a + (uint64_t)w * 16);
            [unroll] for (uint j = 0; j < 4; ++j) {
                v[w * 8 + j * 2]     = bf16_decode(q[j] & 0xFFFFu);
                v[w * 8 + j * 2 + 1] = bf16_decode(q[j] >> 16);
            }
        }
    } else {
        const uint64_t a = base + idx * 4;
        [unroll] for (uint w = 0; w < 8; ++w) {
            const uint4 q = load16(a + (uint64_t)w * 16);
            [unroll] for (uint j = 0; j < 4; ++j) v[w * 4 + j] = asfloat(q[j]);
        }
    }
}

// --- RoPE -------------------------------------------------------------------
// attn_common.slang's `rope_pair` does the rotation itself; this is the table
// lookup around it, which DSpark needs a position index for that the M = 1
// decode path did not.
//
// **RoPE pairs ADJACENT elements.** `apply_rotary_emb` is
// `view_as_complex(x.unflatten(-1, (-1, 2)))`, so within the last
// `rope_head_dim` = 64 dims of a row the pairs are (448 + 2j, 448 + 2j + 1) and
// frequency j drives pair j. design §7.3's "rows (d, d+32) rotate in pairs" is
// wrong and following it rotates the wrong pairs (attn_common.slang, point 2).
//
// The table is `[positions][rope_dim / 2]` of (cos, sin) fp32, built on the
// host, so nothing transcendental runs here and YaRN -- off for every DSpark
// stage, all three of which have `compress_ratio == 0` and rope_theta 10000 --
// stays entirely a host concern (design §2.1).
float2 rope_cs(uint64_t tab, uint pos, uint half_dim, uint j) {
    const uint64_t a = tab + ((uint64_t)pos * half_dim + j) * 8;
    return float2(load_f32(a), load_f32(a + 4));
}

// --- act_quant over one 32-element block, values only ------------------------
// `act_quant(x, 32, "ue8m0", e8m0, inplace=True)` on a BF16 tensor, which is
// what `DSparkAttention.forward` runs on both `main_kv` and the draft `kv`
// before they reach the ring. Unlike wkv.slang's version there is no byte plane:
// the reference's window cache is a plain bf16 tensor and `inplace=True` writes
// the DEQUANTISED value back into it. That is exact -- an E4M3 number has three
// mantissa bits and the scale is a power of two, so the product always fits
// bf16's eight -- so the ring stores bf16 and the attention reads bf16.
//
// The scale comes off the fp32 bit pattern via `act_scale`, never via
// `ceil(log2(x))`, which gets exact powers of two wrong (design §2.4).
void dspark_act_quant32(inout float v[32]) {
    float amax = 0.0f;
    [unroll] for (uint i = 0; i < 32; ++i) amax = max(amax, abs(v[i]));
    const float s   = act_scale(amax);
    const float inv = 1.0f / s;
    [unroll] for (uint i = 0; i < 32; ++i)
        v[i] = fp8_round(clamp(v[i] * inv, -kFp8MaxF, kFp8MaxF)) * s;
}
