#pragma once
#include <algorithm>
#include <array>
#include <span>
#include <vector>

namespace deepmoe::runtime {
// Original gate weights, no renormalisation. Return the smallest descending
// miss prefix that bounds the lost mass. tau=1 is the existing all-mask mode.
inline std::vector<uint32_t> mask_wait_candidates(std::span<const float> weights,
                                                 std::span<const float> effective,
                                                 double tau) {
    std::vector<uint32_t> miss;
    double total = 0, lost = 0;
    for (uint32_t i = 0; i < weights.size(); ++i) {
        total += weights[i];
        if (weights[i] > 0 && effective[i] == 0) { miss.push_back(i); lost += weights[i]; }
    }
    if (tau >= 1 || total <= 0) return {};
    const bool shared_only = miss.size() == weights.size();
    std::stable_sort(miss.begin(), miss.end(), [&](auto a, auto b) {return weights[a] > weights[b];});
    size_t n = 0;
    for (uint32_t i : miss) {
        if (lost <= tau * total && !(shared_only && n == 0)) break;
        lost -= weights[i];
        ++n;
    }
    miss.resize(n);
    return miss;
}
} // namespace deepmoe::runtime
