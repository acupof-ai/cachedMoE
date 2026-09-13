#include "cpu/gemv_avx512.h"

#include <cmath>
#include <format>
#include <vector>

#include "cpu/dequant.h"

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace deepmoe::cpu {
namespace {

Result<void> check_shape(GemvShape s, size_t x_len, size_t y_len) {
    if (s.M == 0) return fail(Err::InvalidArgument, "M must be >= 1");
    if (s.K == 0 || s.K % 32 != 0)
        return fail(Err::InvalidArgument, std::format("K={} must be a positive multiple of 32", s.K));
    if (x_len < s.K * s.M) return fail(Err::InvalidArgument, "activation span too small");
    if (y_len < s.rows * s.M) return fail(Err::InvalidArgument, "output span too small");
    return {};
}

}  // namespace

Result<void> gemv_fp4_ref(std::span<const uint8_t> weights,
                          std::span<const uint8_t> scales,
                          std::span<const float>   x,
                          GemvShape shape,
                          std::span<float> y) {
    if (auto ok = check_shape(shape, x.size(), y.size()); !ok) return ok;
    const size_t row_bytes  = shape.K / 2;
    const size_t row_scales = shape.K / 32;
    if (weights.size() < shape.rows * row_bytes)
        return fail(Err::InvalidArgument, "fp4 weight span too small");
    if (scales.size() < shape.rows * row_scales)
        return fail(Err::InvalidArgument, "fp4 scale span too small");

    for (size_t r = 0; r < shape.rows; ++r) {
        const uint8_t* w = weights.data() + r * row_bytes;
        const uint8_t* s = scales.data() + r * row_scales;
        for (size_t m = 0; m < shape.M; ++m) {
            const float* xm = x.data() + m * shape.K;
            float acc = 0.0f;
            for (size_t blk = 0; blk < row_scales; ++blk) {
                const uint8_t e = s[blk];
                if (e8m0_is_nan(e))
                    return fail(Err::Corrupt, std::format("row {} block {} has a NaN scale", r, blk));
                // Accumulate the block in fp32, then apply the shared exponent
                // with ldexp -- exactly what the GPU kernel does (design §7.9).
                float part = 0.0f;
                const size_t base = blk * 32;
                for (size_t j = 0; j < 32; ++j) {
                    const size_t i = base + j;
                    const uint8_t b = w[i >> 1];
                    const uint8_t nib = (i & 1) ? static_cast<uint8_t>(b >> 4)
                                                : static_cast<uint8_t>(b & 0x0F);
                    part += kFp4E2M1Table[nib] * xm[i];
                }
                acc += std::ldexp(part, static_cast<int>(e) - 127);
            }
            y[m * shape.rows + r] = acc;
        }
    }
    return {};
}

Result<void> gemv_fp8_ref(std::span<const uint8_t> weights,
                          std::span<const uint8_t> scales,
                          std::span<const float>   x,
                          GemvShape shape,
                          std::span<float> y) {
    if (auto ok = check_shape(shape, x.size(), y.size()); !ok) return ok;
    const size_t kblocks = shape.K / 32;
    if (weights.size() < shape.rows * shape.K)
        return fail(Err::InvalidArgument, "fp8 weight span too small");
    const size_t scale_rows = (shape.rows + 31) / 32;
    if (scales.size() < scale_rows * kblocks)
        return fail(Err::InvalidArgument, "fp8 scale span too small");

    for (size_t r = 0; r < shape.rows; ++r) {
        const uint8_t* w = weights.data() + r * shape.K;
        const uint8_t* s = scales.data() + (r / 32) * kblocks;
        for (size_t m = 0; m < shape.M; ++m) {
            const float* xm = x.data() + m * shape.K;
            float acc = 0.0f;
            for (size_t blk = 0; blk < kblocks; ++blk) {
                const uint8_t e = s[blk];
                if (e8m0_is_nan(e))
                    return fail(Err::Corrupt, std::format("row {} block {} has a NaN scale", r, blk));
                float part = 0.0f;
                const size_t base = blk * 32;
                for (size_t j = 0; j < 32; ++j)
                    part += kFp8E4M3Table[w[base + j]] * xm[base + j];
                acc += std::ldexp(part, static_cast<int>(e) - 127);
            }
            y[m * shape.rows + r] = acc;
        }
    }
    return {};
}

Result<void> gemv_bf16_ref(std::span<const uint16_t> weights,
                           std::span<const float>    x,
                           GemvShape shape,
                           std::span<float> y) {
    if (auto ok = check_shape(shape, x.size(), y.size()); !ok) return ok;
    if (weights.size() < shape.rows * shape.K)
        return fail(Err::InvalidArgument, "bf16 weight span too small");
    for (size_t r = 0; r < shape.rows; ++r) {
        const uint16_t* w = weights.data() + r * shape.K;
        for (size_t m = 0; m < shape.M; ++m) {
            const float* xm = x.data() + m * shape.K;
            float acc = 0.0f;
            for (size_t k = 0; k < shape.K; ++k) acc += bf16_to_float(w[k]) * xm[k];
            y[m * shape.rows + r] = acc;
        }
    }
    return {};
}

bool has_avx512_vnni() {
    static const bool yes = [] {
        uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
#if defined(_MSC_VER)
        int regs[4];
        __cpuidex(regs, 0, 0);
        if (regs[0] < 7) return false;
        __cpuidex(regs, 7, 0);
        ebx = static_cast<uint32_t>(regs[1]);
        ecx = static_cast<uint32_t>(regs[2]);
#else
        if (__get_cpuid_max(0, nullptr) < 7) return false;
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
#endif
        const bool avx512f = (ebx & (1u << 16)) != 0;
        const bool vnni    = (ecx & (1u << 11)) != 0;
        (void)eax; (void)edx;
        return avx512f && vnni;
    }();
    return yes;
}

// TODO(design §8.5): implement. The int8 path quantises x per 32-element block
// and uses _mm512_dpbusd_epi32 against the losslessly remapped FP4 magnitudes
// {0,1,2,3,4,6,8,12}; the sign is folded by splitting into two dpbusd passes.
Result<void> gemv_fp4_avx512(std::span<const uint8_t>, std::span<const uint8_t>,
                             std::span<const float>, GemvShape, std::span<float>) {
    return unimplemented("cpu::gemv_fp4_avx512 (design §8.5)");
}

// TODO(design §8.5): implement with a permutexvar-based E4M3 table lookup.
Result<void> gemv_fp8_avx512(std::span<const uint8_t>, std::span<const uint8_t>,
                             std::span<const float>, GemvShape, std::span<float>) {
    return unimplemented("cpu::gemv_fp8_avx512 (design §8.5)");
}

// TODO(design §8.5): bf16 -> fp32 by a 16-bit left shift, then fp32 FMA; the
// lookahead gate GEMV of design §9.4 is the first caller.
Result<void> gemv_bf16_avx512(std::span<const uint16_t>, std::span<const float>,
                              GemvShape, std::span<float>) {
    return unimplemented("cpu::gemv_bf16_avx512 (design §8.5)");
}

Result<void> gemv_fp4(std::span<const uint8_t> weights,
                      std::span<const uint8_t> scales,
                      std::span<const float>   x,
                      GemvShape shape,
                      std::span<float> y) {
    if (has_avx512_vnni()) {
        auto r = gemv_fp4_avx512(weights, scales, x, shape, y);
        if (r) return r;
        if (r.error().code != Err::Unimplemented) return r;
    }
    return gemv_fp4_ref(weights, scales, x, shape, y);
}

}  // namespace deepmoe::cpu
