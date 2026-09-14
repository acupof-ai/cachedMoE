#include "cpu/dequant.h"

#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>

namespace deepmoe::cpu {

bool e8m0_is_nan(uint8_t e) noexcept { return e == 0xFF; }

float e8m0_to_float(uint8_t e) noexcept {
    if (e == 0xFF) return std::numeric_limits<float>::quiet_NaN();
    return std::ldexp(1.0f, static_cast<int>(e) - 127);
}

float e8m0_scale(float v, uint8_t e) noexcept {
    if (e == 0xFF) return std::numeric_limits<float>::quiet_NaN();
    return std::ldexp(v, static_cast<int>(e) - 127);
}

float fp8_round_scale(float amax) noexcept {
    // `fast_round_scale(amax, 1/448)`: 2^ceil(log2(v)) read off the exponent
    // field, with the amax floor act_quant_kernel applies.
    const float v = (amax < 1e-4f ? 1e-4f : amax) * (1.0f / 448.0f);
    uint32_t b;
    std::memcpy(&b, &v, 4);
    const int e = static_cast<int>((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) ? 1 : 0);
    return std::ldexp(1.0f, e);
}

uint8_t e8m0_encode(float pow2) noexcept {
    uint32_t b;
    std::memcpy(&b, &pow2, 4);
    return static_cast<uint8_t>((b >> 23) & 0xFFu);
}

uint8_t fp8_encode_rn(float v) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    const uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
    const float a = std::fabs(v);
    uint32_t mag;
    if (a < 0.015625f) {                       // below 2^-6: the subnormal grid
        const float t = a * 512.0f;            // multiples of 2^-9
        const float f = std::floor(t);
        const float d = t - f;
        const float n = (d > 0.5f) ? f + 1.0f
                      : (d < 0.5f ? f : (std::fmod(f, 2.0f) == 0.0f ? f : f + 1.0f));
        mag = static_cast<uint32_t>(n);        // 8 carries into exponent 1, i.e. 2^-6
    } else {
        uint32_t b;
        std::memcpy(&b, &a, 4);
        int      e    = static_cast<int>((b >> 23) & 0xFFu);
        uint32_t keep = (b & 0x7FFFFFu) >> 20;
        const uint32_t rem = b & 0xFFFFFu;
        if (rem > 0x80000u || (rem == 0x80000u && (keep & 1u))) {
            if (++keep == 8u) { keep = 0; ++e; }
        }
        int ee = e - 127 + 7;
        if (ee > 15 || (ee == 15 && keep == 7)) { ee = 15; keep = 6; }   // clamp at 448
        if (ee < 0) { ee = 0; keep = 0; }
        mag = (static_cast<uint32_t>(ee) << 3) | keep;
    }
    return static_cast<uint8_t>(sign | mag);
}

float act_quant_block(const float* v, size_t n, uint8_t* bytes, float* out_dequant) {
    float amax = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float a = std::fabs(v[i]);
        if (a > amax) amax = a;
    }
    const float s = fp8_round_scale(amax);
    const float inv = 1.0f / s;
    for (size_t i = 0; i < n; ++i) {
        float q = v[i] * inv;
        q = q < -448.0f ? -448.0f : (q > 448.0f ? 448.0f : q);
        bytes[i] = fp8_encode_rn(q);
        if (out_dequant) out_dequant[i] = fp8_e4m3_to_float(bytes[i]) * s;
    }
    return s;
}

float bf16_to_float(uint16_t h) noexcept {
    uint32_t x = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

uint16_t float_to_bf16(float f) noexcept {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    if (((x >> 23) & 0xFF) == 0xFF && (x & 0x7FFFFF))    // NaN: keep it quiet
        return static_cast<uint16_t>((x >> 16) | 0x0040);
    uint32_t bias = ((x >> 16) & 1u) + 0x7FFFu;          // round-to-nearest-even
    return static_cast<uint16_t>((x + bias) >> 16);
}

float fp16_to_float(uint16_t h) noexcept {
    const uint32_t sign = (h & 0x8000u) ? 0x80000000u : 0u;
    const uint32_t e    = (h >> 10) & 0x1Fu;
    const uint32_t m    = h & 0x3FFu;
    uint32_t out;
    if (e == 0) {
        if (m == 0) { out = sign; }
        else {
            float v = static_cast<float>(m) * 5.960464477539063e-8f;  // m * 2^-24
            if (sign) v = -v;
            return v;
        }
    } else if (e == 0x1F) {
        out = sign | 0x7F800000u | (m << 13);
    } else {
        out = sign | ((e + 112u) << 23) | (m << 13);     // 112 = 127 - 15
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

uint16_t float_to_fp16(float f) noexcept {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t be   = (x >> 23) & 0xFFu;
    uint32_t man = x & 0x7FFFFFu;
    if (be == 0xFF)
        return static_cast<uint16_t>(sign | (man ? 0x7E00u : 0x7C00u));
    const int32_t exp = static_cast<int32_t>(be) - 127 + 15;
    if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);   // overflow -> inf
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);           // underflow -> +-0
        man |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);      // 14..24
        uint32_t sub  = man >> shift;
        const uint32_t rem  = man & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (sub & 1u))) ++sub;
        return static_cast<uint16_t>(sign | sub);
    }
    uint32_t h = (static_cast<uint32_t>(exp) << 10) | (man >> 13);
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;          // a carry into the exponent is correct
    return static_cast<uint16_t>(sign | h);
}

Result<void> dequant_fp4_row(std::span<const uint8_t> packed,
                             std::span<const uint8_t> scales,
                             size_t n,
                             std::span<float> out) {
    if (n % 32 != 0)
        return fail(Err::InvalidArgument, std::format("fp4 row length {} is not a multiple of 32", n));
    if (packed.size() < n / 2)
        return fail(Err::InvalidArgument, std::format("fp4 row needs {} packed bytes, got {}", n / 2, packed.size()));
    if (scales.size() < n / 32)
        return fail(Err::InvalidArgument, std::format("fp4 row needs {} scales, got {}", n / 32, scales.size()));
    if (out.size() < n)
        return fail(Err::InvalidArgument, std::format("fp4 output needs {} floats, got {}", n, out.size()));

    for (size_t blk = 0; blk < n / 32; ++blk) {
        const uint8_t e = scales[blk];
        if (e8m0_is_nan(e))
            return fail(Err::Corrupt, std::format("fp4 block {} has a NaN E8M0 scale", blk));
        const int shift = static_cast<int>(e) - 127;
        const size_t base = blk * 32;
        for (size_t j = 0; j < 32; ++j) {
            const size_t i = base + j;
            const uint8_t b = packed[i >> 1];
            const uint8_t nib = (i & 1) ? static_cast<uint8_t>(b >> 4) : static_cast<uint8_t>(b & 0x0F);
            out[i] = std::ldexp(kFp4E2M1Table[nib], shift);
        }
    }
    return {};
}

Result<void> dequant_fp8_row(std::span<const uint8_t> bytes,
                             std::span<const uint8_t> scales,
                             size_t n,
                             std::span<float> out,
                             uint32_t scale_block) {
    if (scale_block == 0 || n % scale_block != 0)
        return fail(Err::InvalidArgument,
                    std::format("fp8 row length {} is not a multiple of block {}", n, scale_block));
    if (bytes.size() < n)
        return fail(Err::InvalidArgument, std::format("fp8 row needs {} bytes, got {}", n, bytes.size()));
    const size_t nblocks = n / scale_block;
    if (scales.size() < nblocks)
        return fail(Err::InvalidArgument, std::format("fp8 row needs {} scales, got {}", nblocks, scales.size()));
    if (out.size() < n)
        return fail(Err::InvalidArgument, std::format("fp8 output needs {} floats, got {}", n, out.size()));

    for (size_t blk = 0; blk < nblocks; ++blk) {
        const uint8_t e = scales[blk];
        if (e8m0_is_nan(e))
            return fail(Err::Corrupt, std::format("fp8 block {} has a NaN E8M0 scale", blk));
        const int shift = static_cast<int>(e) - 127;
        const size_t base = blk * scale_block;
        for (size_t j = 0; j < scale_block; ++j)
            out[base + j] = std::ldexp(kFp8E4M3Table[bytes[base + j]], shift);
    }
    return {};
}

Result<void> dequant_bf16_row(std::span<const uint16_t> src, std::span<float> out) {
    if (out.size() < src.size())
        return fail(Err::InvalidArgument, "bf16 output span is too small");
    for (size_t i = 0; i < src.size(); ++i) out[i] = bf16_to_float(src[i]);
    return {};
}

Result<void> dequant_fp4_matrix_row(std::span<const uint8_t> weights,
                                    std::span<const uint8_t> scales,
                                    size_t row, size_t rows, size_t K,
                                    std::span<float> out) {
    if (row >= rows) return fail(Err::OutOfRange, std::format("row {} >= rows {}", row, rows));
    if (K % 32 != 0) return fail(Err::InvalidArgument, "fp4 K must be a multiple of 32");
    const size_t row_bytes  = K / 2;
    const size_t row_scales = K / 32;
    if (weights.size() < rows * row_bytes)  return fail(Err::InvalidArgument, "fp4 weight span too small");
    if (scales.size()  < rows * row_scales) return fail(Err::InvalidArgument, "fp4 scale span too small");
    return dequant_fp4_row(weights.subspan(row * row_bytes, row_bytes),
                           scales.subspan(row * row_scales, row_scales), K, out);
}

Result<void> dequant_fp8_matrix_row(std::span<const uint8_t> weights,
                                    std::span<const uint8_t> scales,
                                    size_t row, size_t rows, size_t K,
                                    std::span<float> out) {
    if (row >= rows) return fail(Err::OutOfRange, std::format("row {} >= rows {}", row, rows));
    if (K % 32 != 0) return fail(Err::InvalidArgument, "fp8 K must be a multiple of 32");
    // Scales are a [rows/32, K/32] tile plane: all 32 rows of a tile share one
    // exponent per 32-wide K block (design appendix A).
    const size_t kblocks   = K / 32;
    const size_t scale_row = row / 32;
    if (weights.size() < rows * K) return fail(Err::InvalidArgument, "fp8 weight span too small");
    if (scales.size() < (scale_row + 1) * kblocks) return fail(Err::InvalidArgument, "fp8 scale span too small");
    return dequant_fp8_row(weights.subspan(row * K, K),
                           scales.subspan(scale_row * kblocks, kblocks), K, out, 32);
}

}  // namespace deepmoe::cpu
