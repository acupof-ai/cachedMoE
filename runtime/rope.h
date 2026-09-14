// Rotary frequencies, host side (design §2.1, §7.3).
//
// `precompute_freqs_cis` from inference/model.py, transcribed. The kernels
// never evaluate a transcendental: they read a small (cos, sin) table for the
// positions a dispatch touches, so YaRN and the two different rope bases live
// entirely here.
//
// Two configurations, chosen per layer by `compress_ratios[layer]`:
//
//   ratio == 0  a window-only layer (0, 1, and the DSpark blocks): YaRN OFF
//               (`original_seq_len = 0`) and base `rope_theta` = 10000.
//   ratio != 0  base `compress_rope_theta` = 160000 with YaRN over
//               `original_seq_len` = 65536.
//
// Getting that wrong is invisible at position 0 and grows with the context, so
// it is a per-layer decision the caller must make explicitly.
//
// Pairing: `apply_rotary_emb` views the last `rope_head_dim` values as
// `rope_head_dim / 2` complex numbers of ADJACENT elements, so frequency j
// drives the pair (lo + 2j, lo + 2j + 1). design §7.3's "(d, d+32)" is wrong.
//
// Ownership/threading: pure functions over caller-owned vectors.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace deepmoe::runtime {

struct RopeConfig {
    uint32_t rope_head_dim = 64;
    double   base          = 10000.0;    // rope_theta or compress_rope_theta
    uint32_t original_seq_len = 0;       // 0 disables YaRN
    double   factor      = 16.0;
    double   beta_fast   = 32.0;
    double   beta_slow   = 1.0;
};

// The half-dimension frequencies, after YaRN.
inline std::vector<double> rope_freqs(const RopeConfig& c) {
    const uint32_t half = c.rope_head_dim / 2;
    std::vector<double> f(half);
    for (uint32_t i = 0; i < half; ++i)
        f[i] = 1.0 / std::pow(c.base, (2.0 * i) / c.rope_head_dim);
    if (c.original_seq_len == 0) return f;

    // The dimension whose wavelength completes `rotations` turns over the
    // training context; between `low` and `high` the two bases are blended.
    auto corrected = [&](double rot) {
        return c.rope_head_dim * std::log(c.original_seq_len / (rot * 2.0 * std::numbers::pi)) /
               (2.0 * std::log(c.base));
    };
    const double low  = std::max(std::floor(corrected(c.beta_fast)), 0.0);
    const double high = std::min(std::ceil(corrected(c.beta_slow)),
                                 static_cast<double>(c.rope_head_dim - 1));
    for (uint32_t i = 0; i < half; ++i) {
        double ramp = (i - low) / std::max(high - low, 1e-3);
        ramp = std::clamp(ramp, 0.0, 1.0);
        const double smooth = 1.0 - ramp;
        f[i] = f[i] / c.factor * (1.0 - smooth) + f[i] * smooth;
    }
    return f;
}

// (cos, sin) interleaved for one position: what the kernels' `RopeTab` holds.
inline std::vector<float> rope_table(const RopeConfig& c, uint32_t position) {
    const std::vector<double> f = rope_freqs(c);
    std::vector<float> out(f.size() * 2);
    for (size_t i = 0; i < f.size(); ++i) {
        const double a = static_cast<double>(position) * f[i];
        out[i * 2 + 0] = static_cast<float>(std::cos(a));
        out[i * 2 + 1] = static_cast<float>(std::sin(a));
    }
    return out;
}

// The configuration design §2.1 gives a layer with this compression ratio.
inline RopeConfig rope_for_layer(uint32_t compress_ratio, uint32_t rope_head_dim = 64,
                                 double rope_theta = 10000.0,
                                 double compress_rope_theta = 160000.0,
                                 uint32_t original_seq_len = 65536,
                                 double factor = 16.0) {
    RopeConfig c;
    c.rope_head_dim = rope_head_dim;
    if (compress_ratio == 0) {
        c.base = rope_theta;
        c.original_seq_len = 0;
    } else {
        c.base = compress_rope_theta;
        c.original_seq_len = original_seq_len;
    }
    c.factor = factor;
    return c;
}

}  // namespace deepmoe::runtime
