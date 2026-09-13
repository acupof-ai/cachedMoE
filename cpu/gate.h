// Router (gate) maths, design §2.4 and §7.8:
//
//   scores  = sqrt(softplus(x_f32 . W))          W is bf16 [384, 5120]
//   chosen  = top-k over (scores + bias)         bias is fp32, `noaux_tc`
//   weights = normalise(scores[chosen]) * 1.5    the *unbiased* scores
//
// The CPU evaluates this twice: once as the L1/L2 oracle for the GPU gate
// kernel, and once per predicted layer for the lookahead prefetcher of §9.4.
//
// Ownership/threading: pure functions over caller-owned spans. GateResult is a
// small fixed-size value, no allocation on the prediction path.
#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "core/status.h"
#include "core/types.h"
#include "model/layout.h"

namespace deepmoe::cpu {

inline constexpr uint32_t kMaxTopK  = 16;   // top-16 is what the planner wants for heat (design §7.8)
inline constexpr uint32_t kMaxHeat  = 16;

struct GateResult {
    uint32_t k = 0;                                  // how many entries are valid
    std::array<uint16_t, kMaxTopK> ids{};            // expert ids, best first
    std::array<float,    kMaxTopK> weights{};        // normalised, already x routed_scaling_factor
    std::array<float,    kMaxTopK> raw_scores{};     // unbiased sqrt(softplus(.)), for the planner's heat EWMA
};

// softplus then sqrt, guarded the way model.py is: log1p(exp(-|v|)) + max(v,0).
float sqrt_softplus(float v) noexcept;

// Scores every expert. `w` is [n_experts, K] bf16 row-major, `x` is K fp32.
Result<void> gate_scores(std::span<const uint16_t> w,
                         std::span<const float>    x,
                         uint32_t n_experts, uint32_t K,
                         std::span<float> scores_out);

// Selects top-k of (scores + bias) but reports the *unbiased* scores as
// weights, normalised and scaled (design §2.4). `bias` may be empty.
// `report` >= k additionally fills ids/raw_scores with the next best experts so
// the planner sees the "near miss" set of design §9.3.
Result<GateResult> gate_topk(std::span<const float> scores,
                             std::span<const float> bias,
                             uint32_t k,
                             uint32_t report = 0,
                             float scaling = layout::kRoutedScaling);

// Convenience: scores + top-k in one call.
Result<GateResult> gate(std::span<const uint16_t> w,
                        std::span<const float>    bias,
                        std::span<const float>    x,
                        uint32_t n_experts, uint32_t K, uint32_t k,
                        uint32_t report = 0);

// --- lookahead (design §9.4) -------------------------------------------------
// TODO(design §9.4): the predictor feeds the layer-L residual stream into the
// gate of layer L+d. Two input approximations are on the table
// (ffn_norm(mean_hc(x)) and ffn_norm(hc_pre(x, identity))); tools/route_trace.py
// picks the one with higher recall before this is written.
Result<GateResult> gate_lookahead(std::span<const uint16_t> w,
                                  std::span<const float>    bias,
                                  std::span<const float>    residual_hc,  // [hc_mult][hidden]
                                  uint32_t n_experts, uint32_t K, uint32_t width);

}  // namespace deepmoe::cpu
