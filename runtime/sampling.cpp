#include "runtime/sampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace deepmoe::runtime {
namespace {

// Sorted (logit desc, id asc) candidates with their unnormalised weights ->
// the nucleus. `S` is the full-vocabulary normaliser.
Nucleus cut(std::vector<std::pair<float, uint32_t>>& by, const std::vector<double>& w, double S,
            double retained, double lse, float top_p, bool whole_vocab) {
    Nucleus n;
    n.retained  = retained;
    n.logsumexp = lse;
    double cum = 0.0;
    for (size_t i = 0; i < by.size(); ++i) {
        const double p = w[i] / S;
        n.ids.push_back(by[i].second);
        n.p.push_back(p);
        cum += p;
        if (top_p < 1.0f && cum >= double(top_p)) break;
    }
    n.kept  = cum;
    n.exact = (top_p < 1.0f && cum >= double(top_p)) || whole_vocab;
    n.cdf.resize(n.p.size());
    double c = 0.0;
    for (size_t i = 0; i < n.p.size(); ++i) {
        c += n.p[i];
        n.cdf[i] = c / cum;
    }
    if (!n.cdf.empty()) n.cdf.back() = 1.0;
    return n;
}

bool by_logit(const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
    return a.first != b.first ? a.first > b.first : a.second < b.second;
}

uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

}  // namespace

Nucleus nucleus_from_topk(const TopKLogits& tk, float temperature, float top_p) {
    if (tk.overflow || tk.cand.empty() || !(temperature > 0.0f)) return Nucleus{};
    std::vector<std::pair<float, uint32_t>> by;
    by.reserve(tk.cand.size());
    for (const TopKLogits::Cand& c : tk.cand) by.emplace_back(c.logit, c.id);
    std::sort(by.begin(), by.end(), by_logit);
    const double inv_t = 1.0 / double(temperature);
    const double M = double(tk.max_logit);
    std::vector<double> w(by.size());
    double sum = 0.0;
    for (size_t i = 0; i < by.size(); ++i) {
        w[i] = std::exp((double(by[i].first) - M) * inv_t);
        sum += w[i];
    }
    const double S = sum + std::max(0.0, tk.tail);
    return cut(by, w, S, sum / S, M * inv_t + std::log(S), top_p,
               tk.rows != 0 && by.size() == tk.rows);
}

Nucleus nucleus_from_full(std::span<const float> logits, float temperature, float top_p) {
    if (logits.empty() || !(temperature > 0.0f)) return Nucleus{};
    const double inv_t = 1.0 / double(temperature);
    float M = -std::numeric_limits<float>::infinity();
    for (float v : logits) M = std::max(M, v);
    double S = 0.0;
    for (float v : logits) S += std::exp((double(v) - double(M)) * inv_t);
    // Only tokens that can be in a nucleus need sorting; everything whose
    // weight is below 1e-12 of the total is kept out of the sort unless top_p
    // asks for the whole vocabulary.
    std::vector<std::pair<float, uint32_t>> by;
    const bool all = top_p >= 1.0f;
    for (uint32_t i = 0; i < logits.size(); ++i) {
        const double w = std::exp((double(logits[i]) - double(M)) * inv_t);
        if (all || w >= S * 1e-12) by.emplace_back(logits[i], i);
    }
    std::sort(by.begin(), by.end(), by_logit);
    std::vector<double> w(by.size());
    for (size_t i = 0; i < by.size(); ++i) w[i] = std::exp((double(by[i].first) - double(M)) * inv_t);
    Nucleus n = cut(by, w, S, 1.0, double(M) * inv_t + std::log(S), top_p, true);
    n.exact = true;
    return n;
}

Nucleus nucleus_from_partial(std::span<const uint32_t> ids, std::span<const float> logits,
                             double logsumexp, float temperature, float top_p) {
    if (ids.empty() || ids.size() != logits.size() || !(temperature > 0.0f)) return Nucleus{};
    std::vector<std::pair<float, uint32_t>> by;
    for (size_t i = 0; i < ids.size(); ++i) by.emplace_back(logits[i], ids[i]);
    std::sort(by.begin(), by.end(), by_logit);
    const double inv_t = 1.0 / double(temperature);
    // Weights relative to exp(logsumexp): p_i = exp(l_i / T - lse).
    std::vector<double> w(by.size());
    double sum = 0.0;
    for (size_t i = 0; i < by.size(); ++i) {
        w[i] = std::exp(double(by[i].first) * inv_t - logsumexp);
        sum += w[i];
    }
    return cut(by, w, 1.0, sum, logsumexp, top_p, false);
}

uint32_t sample_nucleus(const Nucleus& n, double u) {
    if (n.ids.empty()) return 0;
    auto it = std::upper_bound(n.cdf.begin(), n.cdf.end(), u);
    if (it == n.cdf.end()) --it;
    return n.ids[size_t(it - n.cdf.begin())];
}

double uniform01(uint64_t seed, uint64_t counter) {
    const uint64_t x = splitmix64(seed ^ splitmix64(counter + 0x632BE59BD9B4E019ull));
    return double(x >> 11) * (1.0 / 9007199254740992.0);
}

TopKLogits emulate_topk(std::span<const float> logits, uint32_t k, float temperature) {
    TopKLogits tk;
    const uint32_t rows = static_cast<uint32_t>(logits.size());
    tk.rows = rows;
    const uint32_t T = kTopKThreads;
    // phase 1-2: the maximum
    float M = -3.0e38f;
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t i = t; i < rows; i += T) M = std::max(M, logits[i]);
    tk.max_logit = M;
    // phase 3: the histogram
    std::vector<uint32_t> hist(kTopKBins, 0);
    for (uint32_t i = 0; i < rows; ++i) {
        const float fb = (M - logits[i]) * kTopKBinsPerLogit;
        if (fb < float(kTopKBins)) ++hist[uint32_t(fb)];
    }
    // phase 4: the bin
    uint32_t cum = 0, J = kTopKBins - 1;
    for (uint32_t b = 0; b < kTopKBins; ++b) {
        cum += hist[b];
        if (cum >= k) { J = b; break; }
    }
    tk.bin = J;
    // phase 5: collect, and the tail per thread
    const float inv_t = 1.0f / temperature;
    float tail = 0.0f;
    for (uint32_t t = 0; t < T; ++t) {
        float acc = 0.0f;
        uint32_t n = 0;
        for (uint32_t i = t; i < rows; i += T) {
            const float d = M - logits[i];
            if (d * kTopKBinsPerLogit < float(J + 1)) {
                if (n < kTopKCapPerThread) tk.cand.push_back({i, logits[i]});
                else tk.overflow = true;
                ++n;
            } else {
                acc += std::exp(-d * inv_t);
            }
        }
        tail += acc;
    }
    tk.tail = tail;
    return tk;
}

}  // namespace deepmoe::runtime
