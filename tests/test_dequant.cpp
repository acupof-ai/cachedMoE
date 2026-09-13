// L0 oracle (design §12): the stored-format decoders must be exact, not close.
// These tests pin every one of the 16 FP4 codes and spot-check the FP8 E4M3
// table against values computed from the OCP spec by hand, plus the packing
// convention (low nibble = even element) and the E8M0 scale.
#include "cpu/dequant.h"

#include "cpu/gate.h"
#include "cpu/gemv_avx512.h"

#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "tests/test_framework.h"

// Set by tests/CMakeLists.txt so the test finds its data whatever the cwd is.
#ifndef DEEPMOE_TEST_DATA_DIR
#define DEEPMOE_TEST_DATA_DIR "tests/data"
#endif

using namespace deepmoe;
using namespace deepmoe::cpu;

DEEPMOE_TEST(dequant, fp4_e2m1_table_is_exact) {
    // design appendix A: E2M1 takes {0, .5, 1, 1.5, 2, 3, 4, 6} with a sign bit.
    const float want[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    for (int i = 0; i < 16; ++i) {
        CHECK_EQ(kFp4E2M1Table[static_cast<size_t>(i)], want[i]);
        CHECK_EQ(fp4_e2m1_to_float(static_cast<uint8_t>(i)), want[i]);
    }
    // Index 8 must be negative zero, not positive zero, so the table round-trips.
    CHECK(std::signbit(kFp4E2M1Table[8]));
    CHECK(!std::signbit(kFp4E2M1Table[0]));
}

DEEPMOE_TEST(dequant, fp8_e4m3_table_matches_the_ocp_spec) {
    // 1-4-3, bias 7, no infinities, max magnitude 448, 0x7F/0xFF are the only NaNs.
    CHECK_EQ(fp8_e4m3_to_float(0x00), 0.0f);
    CHECK(std::signbit(fp8_e4m3_to_float(0x80)));
    CHECK_EQ(fp8_e4m3_to_float(0x38), 1.0f);          // e=7 (bias), m=0
    CHECK_EQ(fp8_e4m3_to_float(0xB8), -1.0f);
    CHECK_EQ(fp8_e4m3_to_float(0x3C), 1.5f);          // m=4 -> 1 + 4/8
    CHECK_EQ(fp8_e4m3_to_float(0x40), 2.0f);
    CHECK_EQ(fp8_e4m3_to_float(0x7E), 448.0f);        // e=15, m=6: the max
    CHECK(std::isnan(fp8_e4m3_to_float(0x7F)));
    CHECK(std::isnan(fp8_e4m3_to_float(0xFF)));
    // Denormals: m * 2^-9, exact in fp32.
    CHECK_EQ(fp8_e4m3_to_float(0x01), 0.001953125f);
    CHECK_EQ(fp8_e4m3_to_float(0x07), 7.0f * 0.001953125f);
    // Smallest normal: e=1, m=0 -> 2^-6.
    CHECK_EQ(fp8_e4m3_to_float(0x08), 0.015625f);
    // Nothing but 0x7F/0xFF may be NaN or infinite.
    int nans = 0;
    for (int i = 0; i < 256; ++i) {
        const float v = kFp8E4M3Table[static_cast<size_t>(i)];
        if (std::isnan(v)) ++nans;
        CHECK(!std::isinf(v));
    }
    CHECK_EQ(nans, 2);
}

DEEPMOE_TEST(dequant, e8m0_scale) {
    CHECK_EQ(e8m0_to_float(127), 1.0f);
    CHECK_EQ(e8m0_to_float(128), 2.0f);
    CHECK_EQ(e8m0_to_float(126), 0.5f);
    CHECK_EQ(e8m0_to_float(135), 256.0f);
    CHECK(e8m0_is_nan(255));
    CHECK(std::isnan(e8m0_to_float(255)));
    CHECK_EQ(e8m0_scale(3.0f, 129), 12.0f);
    CHECK_EQ(e8m0_scale(3.0f, 125), 0.75f);
}

DEEPMOE_TEST(dequant, fp4_low_nibble_is_the_even_element) {
    // design appendix A / PyTorch float4_e2m1fn_x2.
    std::vector<uint8_t> packed(16, 0);
    std::vector<uint8_t> scales(1, 127);           // 2^0
    // byte 0: low nibble = 2 (value 1.0), high nibble = 5 (value 3.0)
    packed[0] = static_cast<uint8_t>((5u << 4) | 2u);
    std::vector<float> out(32, -1.0f);
    REQUIRE_OK(dequant_fp4_row(packed, scales, 32, out));
    CHECK_EQ(out[0], 1.0f);     // even element -> low nibble
    CHECK_EQ(out[1], 3.0f);     // odd element  -> high nibble
    for (size_t i = 2; i < 32; ++i) CHECK_EQ(out[i], 0.0f);

    CHECK_EQ(fp4_nibble(packed, 0), 2u);
    CHECK_EQ(fp4_nibble(packed, 1), 5u);
}

DEEPMOE_TEST(dequant, fp4_row_applies_the_block_scale) {
    std::vector<uint8_t> packed(32, 0);            // 64 elements
    std::vector<uint8_t> scales{129, 125};         // 2^2 for block 0, 2^-2 for block 1
    for (auto& b : packed) b = static_cast<uint8_t>((2u << 4) | 2u);   // every element 1.0
    std::vector<float> out(64);
    REQUIRE_OK(dequant_fp4_row(packed, scales, 64, out));
    for (size_t i = 0; i < 32; ++i) CHECK_EQ(out[i], 4.0f);
    for (size_t i = 32; i < 64; ++i) CHECK_EQ(out[i], 0.25f);
}

DEEPMOE_TEST(dequant, fp4_row_rejects_bad_shapes) {
    std::vector<uint8_t> packed(16), scales(1);
    std::vector<float> out(32);
    CHECK_ERR(dequant_fp4_row(packed, scales, 31, out), Err::InvalidArgument);  // not a multiple of 32
    CHECK_ERR(dequant_fp4_row(std::span(packed).first(4), scales, 32, out), Err::InvalidArgument);
    CHECK_ERR(dequant_fp4_row(packed, std::span<const uint8_t>{}, 32, out), Err::InvalidArgument);
    CHECK_ERR(dequant_fp4_row(packed, scales, 32, std::span(out).first(8)), Err::InvalidArgument);
    // A NaN E8M0 exponent is corrupt data, not something to propagate.
    std::vector<uint8_t> bad_scales{255};
    CHECK_ERR(dequant_fp4_row(packed, bad_scales, 32, out), Err::Corrupt);
}

DEEPMOE_TEST(dequant, fp8_row_uses_tiled_scales) {
    std::vector<uint8_t> bytes(64, 0x38);          // every element 1.0
    std::vector<uint8_t> scales{128, 127};         // block 0 -> x2, block 1 -> x1
    std::vector<float> out(64);
    REQUIRE_OK(dequant_fp8_row(bytes, scales, 64, out, 32));
    for (size_t i = 0; i < 32; ++i) CHECK_EQ(out[i], 2.0f);
    for (size_t i = 32; i < 64; ++i) CHECK_EQ(out[i], 1.0f);
}

DEEPMOE_TEST(dequant, fp8_matrix_row_shares_a_scale_across_32_rows) {
    // A [64, 32] fp8 matrix has a [2, 1] scale plane: rows 0..31 share one
    // exponent, rows 32..63 the other (design appendix A).
    const size_t rows = 64, K = 32;
    std::vector<uint8_t> w(rows * K, 0x38);        // 1.0 everywhere
    std::vector<uint8_t> s{128, 126};              // x2 and x0.5
    std::vector<float> out(K);
    REQUIRE_OK(dequant_fp8_matrix_row(w, s, 0, rows, K, out));
    CHECK_EQ(out[0], 2.0f);
    REQUIRE_OK(dequant_fp8_matrix_row(w, s, 31, rows, K, out));
    CHECK_EQ(out[0], 2.0f);
    REQUIRE_OK(dequant_fp8_matrix_row(w, s, 32, rows, K, out));
    CHECK_EQ(out[0], 0.5f);
    CHECK_ERR(dequant_fp8_matrix_row(w, s, 64, rows, K, out), Err::OutOfRange);
}

DEEPMOE_TEST(dequant, bf16_and_fp16_round_trip) {
    const float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, 3.14159f, -2.71828f, 65504.0f, 1e-4f};
    for (float v : vals) {
        // bf16 keeps 8 mantissa bits: ~1e-2 relative.
        CHECK_CLOSE(bf16_to_float(float_to_bf16(v)), v, 1e-2);
        // fp16 keeps 11: ~1e-3 relative, and these all fit its range.
        CHECK_CLOSE(fp16_to_float(float_to_fp16(v)), v, 1e-3);
    }
    CHECK_EQ(bf16_to_float(float_to_bf16(0.0f)), 0.0f);
    CHECK_EQ(fp16_to_float(float_to_fp16(0.0f)), 0.0f);
    CHECK_EQ(fp16_to_float(float_to_fp16(-0.0f)), 0.0f);
    CHECK(std::signbit(fp16_to_float(float_to_fp16(-0.0f))));
    // fp16 overflow saturates to infinity, underflow to zero.
    CHECK(std::isinf(fp16_to_float(float_to_fp16(1e10f))));
    CHECK_EQ(fp16_to_float(float_to_fp16(1e-12f)), 0.0f);
    // bf16 exponent is the fp32 one, so a huge value survives.
    CHECK_CLOSE(bf16_to_float(float_to_bf16(1e10f)), 1e10f, 1e-2);
}

DEEPMOE_TEST(dequant, gemv_fp4_reference_matches_a_hand_sum) {
    // 2 rows x 64 K. Row 0: all 1.0 at scale 2^0. Row 1: all 2.0 at scale 2^1.
    const size_t rows = 2, K = 64;
    std::vector<uint8_t> w(rows * K / 2);
    std::vector<uint8_t> s(rows * K / 32);
    for (size_t i = 0; i < K / 2; ++i) w[i] = static_cast<uint8_t>((2u << 4) | 2u);        // 1.0
    for (size_t i = 0; i < K / 2; ++i) w[K / 2 + i] = static_cast<uint8_t>((4u << 4) | 4u); // 2.0
    s[0] = s[1] = 127;                 // row 0: x1
    s[2] = s[3] = 128;                 // row 1: x2
    std::vector<float> x(K, 0.5f), y(rows, 0.0f);

    GemvShape shape{rows, K, 1};
    REQUIRE_OK(gemv_fp4_ref(w, s, x, shape, y));
    CHECK_CLOSE(y[0], 64 * 1.0 * 0.5, 1e-6);          // 32
    CHECK_CLOSE(y[1], 64 * 2.0 * 2.0 * 0.5, 1e-6);    // 128
}

DEEPMOE_TEST(dequant, gemv_dispatcher_falls_back_to_the_reference) {
    // gemv_fp4 tries AVX-512 first; that path is a documented stub, so the
    // dispatcher must silently produce the reference result rather than fail.
    const size_t rows = 1, K = 32;
    std::vector<uint8_t> w(K / 2, static_cast<uint8_t>((2u << 4) | 2u));
    std::vector<uint8_t> s(1, 127);
    std::vector<float> x(K, 1.0f), y(1), y_ref(1);
    REQUIRE_OK(gemv_fp4(w, s, x, GemvShape{rows, K, 1}, y));
    REQUIRE_OK(gemv_fp4_ref(w, s, x, GemvShape{rows, K, 1}, y_ref));
    CHECK_EQ(y[0], y_ref[0]);
    CHECK_CLOSE(y[0], 32.0, 1e-6);
    // The AVX-512 path itself must still report Unimplemented, not pretend.
    CHECK_ERR(gemv_fp4_avx512(w, s, x, GemvShape{rows, K, 1}, y), Err::Unimplemented);
}

DEEPMOE_TEST(dequant, gate_reproduces_the_reference_router_maths) {
    // design §2.4: scores = sqrt(softplus(xW)); top-k over (scores + bias);
    // weights normalise the *unbiased* scores and scale by 1.5.
    const uint32_t n = 4, K = 32;
    std::vector<uint16_t> w(n * K);
    for (uint32_t r = 0; r < n; ++r)
        for (uint32_t k = 0; k < K; ++k)
            w[r * K + k] = float_to_bf16(static_cast<float>(r + 1) / static_cast<float>(K));
    std::vector<float> x(K, 1.0f);
    std::vector<float> scores(n);
    REQUIRE_OK(gate_scores(w, x, n, K, scores));
    for (uint32_t r = 0; r < n; ++r)
        CHECK_CLOSE(scores[r], sqrt_softplus(static_cast<float>(r + 1)), 1e-2);

    // Bias flips the selection without changing the weights.
    std::vector<float> bias{10.0f, 0.0f, 0.0f, 0.0f};
    auto g = gate_topk(scores, bias, 2, 4);
    REQUIRE_OK(g);
    CHECK_EQ(g->ids[0], 0);          // biased winner
    CHECK_EQ(g->ids[1], 3);          // then the genuinely highest score
    const float denom = scores[0] + scores[3] + 1e-20f;
    CHECK_CLOSE(g->weights[0], scores[0] / denom * 1.5f, 1e-5);
    CHECK_CLOSE(g->weights[1], scores[3] / denom * 1.5f, 1e-5);
    CHECK_CLOSE(g->weights[0] + g->weights[1], 1.5f, 1e-5);
    // raw_scores carry the unbiased value, which is what the heat EWMA wants.
    CHECK_CLOSE(g->raw_scores[0], scores[0], 1e-6);
    CHECK_EQ(g->k, 4u);              // `report` widened the result to top-4
}

DEEPMOE_TEST(dequant, gate_topk_is_deterministic_on_ties) {
    // The §10.2 speculation invariant needs bit-identical routing on re-run.
    std::vector<float> scores{1.0f, 1.0f, 1.0f, 1.0f};
    auto a = gate_topk(scores, {}, 2);
    auto b = gate_topk(scores, {}, 2);
    REQUIRE_OK(a);
    REQUIRE_OK(b);
    CHECK_EQ(a->ids[0], 0);          // lowest id wins a tie
    CHECK_EQ(a->ids[1], 1);
    CHECK_EQ(a->ids[0], b->ids[0]);
    CHECK_EQ(a->ids[1], b->ids[1]);
}

// --- L0 golden vectors (design §12 L0) ---------------------------------------
// tests/data/l0_dequant.bin is produced by `tools/oracle.py --level l0`. The FP8
// and UE8M0 tables in it come straight from torch (torch.float8_e4m3fn /
// torch.float8_e8m0fnu), which is what inference/kernel.py casts through, and
// the FP4 table is cross-checked there against ml_dtypes.float4_e2m1fn. This
// test is therefore pinned to the checkpoint's semantics rather than to our own
// reading of the OCP spec -- which is the entire point of design §12's L0.
DEEPMOE_TEST(dequant, matches_the_oracle_golden_tables) {
    const std::string path = std::string(DEEPMOE_TEST_DATA_DIR) + "/l0_dequant.bin";
    std::FILE* f = std::fopen(path.c_str(), "rb");
    REQUIRE(f != nullptr);

    char magic[4] = {};
    uint32_t header[4] = {};
    const bool head_ok = std::fread(magic, 1, 4, f) == 4 &&
                         std::fread(header, sizeof(uint32_t), 4, f) == 4;
    if (!head_ok) { std::fclose(f); REQUIRE(head_ok); }
    CHECK_EQ(std::string(magic, 4), std::string("DMQ0"));
    CHECK_EQ(header[0], 1u);          // version
    REQUIRE_EQ(header[1], 16u);       // fp4 codes
    REQUIRE_EQ(header[2], 256u);      // fp8 codes
    REQUIRE_EQ(header[3], 256u);      // e8m0 codes

    std::vector<float> fp4(16), fp8(256), e8m0(256);
    const bool body_ok = std::fread(fp4.data(), 4, 16, f) == 16 &&
                         std::fread(fp8.data(), 4, 256, f) == 256 &&
                         std::fread(e8m0.data(), 4, 256, f) == 256;
    std::fclose(f);
    REQUIRE(body_ok);

    // Bit-for-bit, not "close": a decode table that is 1 ULP out is a wrong
    // table, and -0.0 must stay negative so the FP4 codes round-trip.
    for (size_t i = 0; i < 16; ++i) {
        CHECK_EQ(kFp4E2M1Table[i], fp4[i]);
        CHECK_EQ(std::signbit(kFp4E2M1Table[i]), std::signbit(fp4[i]));
    }
    for (size_t i = 0; i < 256; ++i) {
        const float got = kFp8E4M3Table[i];
        if (std::isnan(fp8[i])) {
            CHECK(std::isnan(got));
        } else {
            CHECK_EQ(got, fp8[i]);
            CHECK_EQ(std::signbit(got), std::signbit(fp8[i]));
        }
        const float scale = e8m0_to_float(static_cast<uint8_t>(i));
        if (std::isnan(e8m0[i])) CHECK(std::isnan(scale));
        else                     CHECK_EQ(scale, e8m0[i]);
    }
}
