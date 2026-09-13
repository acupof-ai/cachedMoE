// CPU GEMV: the scalar fp32 reference (implemented, this is the L1 oracle
// comparand of design §12) and the AVX-512/VNNI paths of design §8.5 (stubs).
//
// The CPU's job in deepMoE is I/O and prediction, not bulk GEMV (design §8);
// these kernels exist as the A/B reference for the Vulkan kernels, as the
// lookahead gate evaluator (§9.4), and as the P6 "CPU takes some experts"
// execution body.
//
// Ownership/threading: pure functions over caller-owned spans. No allocation,
// no global state; a caller may run several rows in parallel.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "core/status.h"
#include "core/types.h"

namespace deepmoe::cpu {

// Row-major [rows, K] weights with the checkpoint's native quantisation, a
// dense fp32 activation, and an fp32 accumulator per row (design §6: decode to
// float, accumulate in fp32).
struct GemvShape {
    size_t rows = 0;
    size_t K    = 0;
    size_t M    = 1;   // batch: 1 for decode, up to 6 for speculative verify
};

// --- scalar reference (implemented) -----------------------------------------

// y[r] = sum_k dequant(W[r][k]) * x[k]. FP4 weights with [rows][K/32] E8M0
// scales, exactly the routed-expert layout of design §5.1.
Result<void> gemv_fp4_ref(std::span<const uint8_t> weights,
                          std::span<const uint8_t> scales,
                          std::span<const float>   x,
                          GemvShape shape,
                          std::span<float> y);

// FP8 E4M3 weights with a [rows/32][K/32] E8M0 tile-scale plane (attention,
// shared expert, engram wkv, main_proj).
Result<void> gemv_fp8_ref(std::span<const uint8_t> weights,
                          std::span<const uint8_t> scales,
                          std::span<const float>   x,
                          GemvShape shape,
                          std::span<float> y);

// BF16 weights, no scales (router W, embed, head).
Result<void> gemv_bf16_ref(std::span<const uint16_t> weights,
                           std::span<const float>    x,
                           GemvShape shape,
                           std::span<float> y);

// --- AVX-512 paths (design §8.5) ---------------------------------------------
// TODO(design §8.5): FP4 nibble decode fused with _mm512_dpbusd_epi32 over
// symmetric int8-quantised activations. E2M1 maps losslessly onto
// {0,1,2,3,4,6,8,12} (design §6), so the int8 path is exact on the weight side.
// TODO(design §8.5): FP8 path decoding through a 256-entry table with
// _mm512_permutexvar_ps pairs, fp32 FMA accumulate.
Result<void> gemv_fp4_avx512(std::span<const uint8_t> weights,
                             std::span<const uint8_t> scales,
                             std::span<const float>   x,
                             GemvShape shape,
                             std::span<float> y);

Result<void> gemv_fp8_avx512(std::span<const uint8_t> weights,
                             std::span<const uint8_t> scales,
                             std::span<const float>   x,
                             GemvShape shape,
                             std::span<float> y);

Result<void> gemv_bf16_avx512(std::span<const uint16_t> weights,
                              std::span<const float>    x,
                              GemvShape shape,
                              std::span<float> y);

// True when the running CPU has AVX-512F + VNNI (checked once via cpuid).
bool has_avx512_vnni();

// Dispatches to the AVX-512 path when available and to the reference otherwise.
// Both must agree inside the L1 tolerance of design §12 (1e-3 relative).
Result<void> gemv_fp4(std::span<const uint8_t> weights,
                      std::span<const uint8_t> scales,
                      std::span<const float>   x,
                      GemvShape shape,
                      std::span<float> y);

}  // namespace deepmoe::cpu
