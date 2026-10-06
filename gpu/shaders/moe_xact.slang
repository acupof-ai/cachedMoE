// moe_xact -- the MoE input's `act_quant(x, 32, ue8m0)` on the GPU, straight
// from the fp32 ffn_norm output into column 0 of the runner's fp16 x.
//
// This is runtime/moe_bridge.cpp's host `act_quant_to_fp16` transcribed, and it
// must stay BIT-IDENTICAL to it (tests/test_gpu_moe.cpp checks): the block
// amax, `act_scale` = the smallest power of two >= amax/448 read off the fp32
// bit pattern (amax floored at 1e-4), `fp8_round` = round-to-nearest-even onto
// the E4M3 grid (attn_common.slang's, the same function the host copied), and
// then fp32 -> fp16 with cpu::float_to_fp16's integer RNE, written out here so
// the answer does not depend on how the driver lowers f32tof16 (RDNA has a
// round-toward-zero pack instruction).
//
// It exists so the shared expert's dispatch A can run while the host is still
// waking up for the gate (Track SE, docs/STATUS.md §7 0h): once x is on the GPU
// nothing about the shared expert waits for routing. 160 blocks, one thread
// each, one workgroup.

#include "moe_common.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> XU;   // the x allocation, raw words

struct XActPush {
    uint64_t src;   // device address of the fp32 [k] ffn_norm output
    uint     k;     // 5120
    uint     pad;
};
[[vk::push_constant]] XActPush pc;

static const float kXaAmaxMin  = 1e-4f;
static const float kXaFp8Max   = 448.0f;
static const float kXaE4M3MinN = 0.015625f;    // 2^-6, smallest normal E4M3

float xa_scale(float amax) {
    const float v = max(amax, kXaAmaxMin) * (1.0f / kXaFp8Max);
    const uint  b = asuint(v);
    const int   e = int((b >> 23) & 0xFFu) - 127 + (((b & 0x7FFFFFu) != 0u) ? 1 : 0);
    return ldexp(1.0f, e);
}

// The subnormal branch is NOT attn_common's `(a + 1.5*2^14) - 1.5*2^14`:
// Mesa's NIR folds `(a + c) - c` to `a` (an inexact algebraic rule the SPIR-V
// here does not forbid), which on RADV returned the unrounded value -- measured
// by CACHEDMOE_SE_CHECK against the host, e.g. x = 2.0266e-5 -> 0x0154 instead of
// 0x0180. Every step below is exact: a * 2^9 is a power-of-two scale, the
// integer part and the remainder of a number < 8 are exact, and the tie rule
// is written out.
float xa_fp8_round(float v) {
    const float a = abs(v);
    float mag;
    if (a < kXaE4M3MinN) {
        const float m = a * 512.0f;                 // units of 2^-9, < 8
        uint n = uint(m);                            // truncation, exact here
        const float f = m - float(n);
        if (f > 0.5f || (f == 0.5f && (n & 1u) != 0u)) ++n;
        mag = float(n) * (1.0f / 512.0f);
    } else {
        const uint b = asuint(a);
        mag = asfloat((b + 0x7FFFFu + ((b >> 20) & 1u)) & 0xFFF00000u);
    }
    return asfloat(asuint(mag) | (asuint(v) & 0x80000000u));
}

// cpu::float_to_fp16 (cpu/dequant.cpp), line for line.
uint xa_f16(float f) {
    const uint x    = asuint(f);
    const uint sign = (x >> 16) & 0x8000u;
    const uint be   = (x >> 23) & 0xFFu;
    uint man = x & 0x7FFFFFu;
    if (be == 0xFFu) return sign | ((man != 0u) ? 0x7E00u : 0x7C00u);
    const int exp = int(be) - 127 + 15;
    if (exp >= 0x1F) return sign | 0x7C00u;
    if (exp <= 0) {
        if (exp < -10) return sign;
        man |= 0x800000u;
        const uint shift = uint(14 - exp);
        uint sub = man >> shift;
        const uint rem  = man & ((1u << shift) - 1u);
        const uint half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (sub & 1u) != 0u)) ++sub;
        return sign | sub;
    }
    uint h = (uint(exp) << 10) | (man >> 13);
    const uint rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) ++h;
    return sign | h;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint blocks = max(1u,pc.pad) * pc.k / kBlockK;
    const uint blk    = gid.x * kWgSize + tid;
    if (blk >= blocks) return;

    float v[kBlockK];
    const uint64_t base = pc.src + uint64_t(blk) * uint64_t(kBlockK * 4u);
    [unroll] for (uint i = 0; i < kBlockK; ++i) v[i] = asfloat(*(uint*)(base + uint64_t(i * 4u)));
    float amax = 0.0f;
    [unroll] for (uint i = 0; i < kBlockK; ++i) amax = max(amax, abs(v[i]));
    const float s   = xa_scale(amax);
    const float inv = 1.0f / s;
    [unroll] for (uint i = 0; i < kBlockK / 2; ++i) {
        const float a = xa_fp8_round(clamp(v[2 * i + 0] * inv, -kXaFp8Max, kXaFp8Max)) * s;
        const float b = xa_fp8_round(clamp(v[2 * i + 1] * inv, -kXaFp8Max, kXaFp8Max)) * s;
        XU[blk * (kBlockK / 2) + i] = xa_f16(a) | (xa_f16(b) << 16);
    }
}
