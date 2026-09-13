// Reference decoders for every stored weight format. This is the L0 oracle of
// design §12: these functions must agree bit-for-bit with the numpy reference,
// so they are exhaustive lookup tables rather than arithmetic conversions.
//
// Packing conventions (design appendix A):
//   FP4 E2M1  packed 2 per byte, LOW nibble = the EVEN element (PyTorch
//             float4_e2m1fn_x2). Scales are E8M0, one per 32 elements along K.
//   FP8 E4M3  one byte per element, scales E8M0 per 32x32 tile.
//   E8M0      the scale itself: value = 2^(e - 127); e == 255 is NaN.
//
// Ownership/threading: pure functions over caller-owned spans. No state, safe
// from any thread.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "core/status.h"

namespace deepmoe::cpu {

// --- element decoders --------------------------------------------------------

constexpr float fp4_e2m1_to_float(uint8_t nibble) noexcept {
    // Values {0, .5, 1, 1.5, 2, 3, 4, 6} with a sign bit; index 8 is negative
    // zero and stays -0.0f so the table round-trips.
    constexpr float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = mag[nibble & 0x7u];
    return (nibble & 0x8u) ? -v : v;
}

// E4M3 in the OCP "fn" flavour PyTorch and the checkpoint use: 1-4-3, bias 7,
// no infinities, 0x7F / 0xFF are NaN, max magnitude 448, denormals exact.
constexpr float fp8_e4m3_to_float(uint8_t b) noexcept {
    const uint32_t sign = b & 0x80u;
    const uint32_t e    = (b >> 3) & 0x0Fu;
    const uint32_t m    = b & 0x07u;
    float v;
    if (e == 0) {
        v = static_cast<float>(m) * 0.001953125f;          // m/8 * 2^-6 == m * 2^-9
    } else if (e == 0x0Fu && m == 0x07u) {
        return std::numeric_limits<float>::quiet_NaN();    // the only NaN pattern
    } else {
        const float mant = 1.0f + static_cast<float>(m) * 0.125f;
        const int   exp  = static_cast<int>(e) - 7;        // (1 + m/8) * 2^(e-7)
        v = exp >= 0 ? mant * static_cast<float>(1u << exp)
                     : mant / static_cast<float>(1u << static_cast<unsigned>(-exp));
    }
    return sign ? -v : v;
}

namespace detail {
template <size_t N, class F>
constexpr std::array<float, N> build_table(F f) {
    std::array<float, N> t{};
    for (size_t i = 0; i < N; ++i) t[i] = f(static_cast<uint8_t>(i));
    return t;
}
}  // namespace detail

// The tables the GEMV inner loops index. Materialised at compile time so the
// L0 oracle and the shader-side constants come from one definition.
inline constexpr std::array<float, 16>  kFp4E2M1Table =
    detail::build_table<16>([](uint8_t n) { return fp4_e2m1_to_float(n); });
inline constexpr std::array<float, 256> kFp8E4M3Table =
    detail::build_table<256>([](uint8_t b) { return fp8_e4m3_to_float(b); });

// E8M0 scale: 2^(e-127). e == 0xFF is NaN by the OCP spec; callers treat a NaN
// scale as corrupt data rather than propagating it.
float e8m0_to_float(uint8_t e) noexcept;
bool  e8m0_is_nan(uint8_t e) noexcept;

// Applies an E8M0 exponent without a multiply: ldexp is exact for a power of two.
float e8m0_scale(float v, uint8_t e) noexcept;

// bf16 and fp16 helpers, used by the head/embed and activation paths.
float    bf16_to_float(uint16_t h) noexcept;
uint16_t float_to_bf16(float f) noexcept;   // round-to-nearest-even
float    fp16_to_float(uint16_t h) noexcept;
uint16_t float_to_fp16(float f) noexcept;

// Nibble extraction. `i` is the element index along K; even elements are in the
// low nibble of byte i/2.
inline uint8_t fp4_nibble(std::span<const uint8_t> packed, size_t i) noexcept {
    uint8_t b = packed[i >> 1];
    return (i & 1) ? static_cast<uint8_t>(b >> 4) : static_cast<uint8_t>(b & 0x0F);
}

// --- row decoders ------------------------------------------------------------

// Decodes `n` FP4 elements from `packed` (n/2 bytes) into `out`, applying the
// E8M0 scale of each 32-element block from `scales` (n/32 bytes).
// Returns InvalidArgument when the sizes do not line up.
Result<void> dequant_fp4_row(std::span<const uint8_t> packed,
                             std::span<const uint8_t> scales,
                             size_t n,
                             std::span<float> out);

// Decodes `n` FP8 E4M3 elements. `scales` holds one E8M0 exponent per
// `scale_block` elements along K (32 in the checkpoint); `scale_stride` lets a
// caller point into a 32x32 tiled scale plane without copying.
Result<void> dequant_fp8_row(std::span<const uint8_t> bytes,
                             std::span<const uint8_t> scales,
                             size_t n,
                             std::span<float> out,
                             uint32_t scale_block = 32);

Result<void> dequant_bf16_row(std::span<const uint16_t> src, std::span<float> out);

// --- matrix helper -----------------------------------------------------------

// Decodes one row of a [rows, K] FP4 matrix stored as `[rows][K/2]` packed
// bytes with scales `[rows][K/32]` (the routed-expert layout of design §5.1).
Result<void> dequant_fp4_matrix_row(std::span<const uint8_t> weights,
                                    std::span<const uint8_t> scales,
                                    size_t row, size_t rows, size_t K,
                                    std::span<float> out);

// Same for a [rows, K] FP8 matrix whose scales are a [rows/32, K/32] tile plane.
Result<void> dequant_fp8_matrix_row(std::span<const uint8_t> weights,
                                    std::span<const uint8_t> scales,
                                    size_t row, size_t rows, size_t K,
                                    std::span<float> out);

}  // namespace deepmoe::cpu
