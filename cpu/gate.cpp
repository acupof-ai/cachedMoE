#include "cpu/gate.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>
#include <vector>

#include "cpu/dequant.h"
#include "cpu/gemv_avx512.h"

namespace deepmoe::cpu {

float sqrt_softplus(float v) noexcept {
    // softplus(v) computed the stable way, then sqrt. Negative inputs give a
    // small positive score, so every expert stays rankable.
    const float sp = std::log1p(std::exp(-std::fabs(v))) + std::max(v, 0.0f);
    return std::sqrt(sp);
}

Result<void> gate_scores(std::span<const uint16_t> w,
                         std::span<const float>    x,
                         uint32_t n_experts, uint32_t K,
                         std::span<float> scores_out) {
    if (scores_out.size() < n_experts)
        return fail(Err::InvalidArgument, "gate score span too small");
    GemvShape shape{n_experts, K, 1};
    if (auto r = gemv_bf16_ref(w, x, shape, scores_out); !r) return r;
    for (uint32_t i = 0; i < n_experts; ++i) scores_out[i] = sqrt_softplus(scores_out[i]);
    return {};
}

Result<GateResult> gate_topk(std::span<const float> scores,
                             std::span<const float> bias,
                             uint32_t k,
                             uint32_t report,
                             float scaling) {
    const uint32_t n = static_cast<uint32_t>(scores.size());
    if (k == 0 || k > n)
        return fail(Err::InvalidArgument, std::format("top-k {} out of range for {} experts", k, n));
    if (!bias.empty() && bias.size() < n)
        return fail(Err::InvalidArgument, "gate bias shorter than the expert count");
    const uint32_t want = std::min(std::max(report, k), kMaxTopK);

    // Rank by (score + bias); ties broken by the lower expert id so the result
    // is deterministic, which the §10.2 speculation invariant depends on.
    std::vector<uint32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0u);
    const auto biased = [&](uint32_t i) { return bias.empty() ? scores[i] : scores[i] + bias[i]; };
    std::partial_sort(idx.begin(), idx.begin() + want, idx.end(),
                      [&](uint32_t a, uint32_t b) {
                          const float fa = biased(a), fb = biased(b);
                          return fa != fb ? fa > fb : a < b;
                      });

    GateResult g;
    g.k = want;
    float sum = 0.0f;
    for (uint32_t i = 0; i < want; ++i) {
        g.ids[i]        = static_cast<uint16_t>(idx[i]);
        g.raw_scores[i] = scores[idx[i]];
        if (i < k) sum += scores[idx[i]];       // normalisation uses the *unbiased* scores
    }
    // model.py adds 1e-20 before dividing (design §2.4).
    const float denom = sum + 1e-20f;
    for (uint32_t i = 0; i < want; ++i)
        g.weights[i] = (i < k) ? (scores[idx[i]] / denom) * scaling : 0.0f;
    return g;
}

Result<GateResult> gate(std::span<const uint16_t> w,
                        std::span<const float>    bias,
                        std::span<const float>    x,
                        uint32_t n_experts, uint32_t K, uint32_t k,
                        uint32_t report) {
    std::vector<float> scores(n_experts);
    if (auto r = gate_scores(w, x, n_experts, K, scores); !r) return std::unexpected(r.error());
    return gate_topk(scores, bias, k, report);
}

// TODO(design §9.4): implement once tools/route_trace.py has answered Q4
// (which residual approximation, which d, which K). Design §16 forbids writing
// the predictor policy before the measurement exists.
Result<GateResult> gate_lookahead(std::span<const uint16_t>, std::span<const float>,
                                  std::span<const float>, uint32_t, uint32_t, uint32_t) {
    return unimplemented("cpu::gate_lookahead (design §9.4, gated on Q4)");
}

}  // namespace deepmoe::cpu
