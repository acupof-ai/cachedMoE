#include "runtime/speculate.h"

#include <algorithm>
#include <cmath>
#include <format>

namespace deepmoe::runtime {

namespace {
constexpr uint32_t kWg = 256;   // gpu/shaders/attn_common.slang's kWg
}  // namespace

const char* spec_mode_name(SpecMode m) {
    switch (m) {
        case SpecMode::Off:    return "off";
        case SpecMode::Greedy: return "greedy";
        case SpecMode::Sample: return "sample";
    }
    return "?";
}

Result<SpecMode> parse_spec_mode(std::string_view s) {
    if (s == "off")    return SpecMode::Off;
    if (s == "greedy") return SpecMode::Greedy;
    if (s == "sample") return SpecMode::Sample;
    return fail(Err::InvalidArgument,
                std::format("--spec takes off, greedy or sample, not '{}'", s));
}

void SpecStats::add(const SpecCycle& c) {
    ++cycles;
    tokens        += c.emitted;
    accepted      += c.accepted;
    verified      += c.k;
    draft_ms      += c.draft_ms;
    verify_ms     += c.verify_ms;
    cpu_ms        += c.cpu_ms;
    rollback_ms   += c.rollback_ms;
    stall_ms      += c.stall_ms;
    union_experts += c.union_experts;
    miss_bytes    += c.miss_bytes;
}

std::string SpecStats::to_string() const {
    if (!cycles) return "no speculative cycles";
    const double n = double(cycles);
    return std::format(
        "{} cycles, {} tokens: accepted {:.2f}/verify of {:.2f} drafted, "
        "{:.2f} tokens a cycle | draft {:.1f} ms, verify {:.1f} ms (stall {:.1f}), "
        "cpu {:.2f} ms, rollback {:.2f} ms -> {:.1f} ms a cycle, {:.2f} tok/s | "
        "expert union {:.1f} a cycle, {:.0f} MB missed a cycle",
        cycles, tokens, accepted_per_verify(), double(verified) / n, tokens_per_cycle(),
        draft_ms / n, verify_ms / n, stall_ms / n, cpu_ms / n, rollback_ms / n,
        cycle_ms(), cycle_ms() > 0 ? tokens_per_cycle() * 1000.0 / cycle_ms() : 0.0,
        double(union_experts) / n, double(miss_bytes) / n / (1024.0 * 1024.0));
}

// --- the host mirror of dspark_verify.slang ---------------------------------

uint32_t verify_hash3(uint32_t a, uint32_t b, uint32_t c) {
    auto mix32 = [](uint32_t z) {
        z ^= z >> 16; z *= 0x7FEB352Du;
        z ^= z >> 15; z *= 0x846CA68Bu;
        z ^= z >> 16;
        return z;
    };
    return mix32(mix32(a ^ 0x9E3779B9u) ^ mix32(b + 0x85EBCA6Bu) ^ (c * 0xC2B2AE35u));
}

float verify_u01(uint32_t h) {
    return (float(h >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

float verify_gumbel(uint32_t seed, uint32_t row, uint32_t id) {
    return -std::log(-std::log(verify_u01(verify_hash3(seed, row, id))));
}

void emulate_verify_row(std::span<const float> logits, std::span<const int32_t> cand,
                        uint32_t seed, uint32_t row, float inv_t,
                        gpu::DsparkVerifyRow& out) {
    out = gpu::DsparkVerifyRow{};
    const uint32_t rows = static_cast<uint32_t>(logits.size());
    out.rows = rows;

    // The candidate list, ascending, exactly as the kernel's thread 0 builds it.
    std::vector<int32_t> sorted;
    for (int32_t v : cand) {
        if (v < 0) break;
        if (sorted.size() >= gpu::kDsVerifyMaxCand) break;
        sorted.insert(std::upper_bound(sorted.begin(), sorted.end(), v), v);
    }
    const uint32_t n_cand = static_cast<uint32_t>(sorted.size());
    out.n_cand = n_cand;
    auto is_cand = [&](int32_t id) {
        return std::binary_search(sorted.begin(), sorted.end(), id);
    };

    // 1. the maximum: per-lane over the stride, then the fixed-shape tree.
    std::vector<float> red(kWg, -3.0e38f);
    for (uint32_t t = 0; t < kWg; ++t) {
        float m = -3.0e38f;
        for (uint32_t i = t; i < rows; i += kWg) m = std::max(m, logits[i]);
        red[t] = m;
    }
    for (uint32_t s = kWg >> 1; s > 0; s >>= 1)
        for (uint32_t t = 0; t < s; ++t) red[t] = std::max(red[t], red[t + s]);
    const float mx = red[0];
    out.max_logit = mx;

    // 2. the shifted sum, the same tree.
    std::vector<float> sum(kWg, 0.0f);
    for (uint32_t t = 0; t < kWg; ++t) {
        float acc = 0.0f;
        for (uint32_t i = t; i < rows; i += kWg) acc += std::exp(logits[i] - mx);
        sum[t] = acc;
    }
    for (uint32_t s = kWg >> 1; s > 0; s >>= 1)
        for (uint32_t t = 0; t < s; ++t) sum[t] = sum[t] + sum[t + s];
    out.lse = mx + std::log(sum[0]);

    // 3. the two Gumbel-max draws, one pass, ties to the lowest id.
    std::vector<float> key_all(kWg, -3.0e38f), key_msk(kWg, -3.0e38f);
    std::vector<uint32_t> id_all(kWg, 0), id_msk(kWg, 0);
    for (uint32_t t = 0; t < kWg; ++t) {
        for (uint32_t i = t; i < rows; i += kWg) {
            const float key = logits[i] * inv_t + verify_gumbel(seed, row, i);
            if (key > key_all[t]) { key_all[t] = key; id_all[t] = i; }
            if (!is_cand(int32_t(i)) && key > key_msk[t]) { key_msk[t] = key; id_msk[t] = i; }
        }
    }
    auto reduce_argmax = [&](std::vector<float>& k, std::vector<uint32_t>& id) {
        for (uint32_t s = kWg >> 1; s > 0; s >>= 1)
            for (uint32_t t = 0; t < s; ++t)
                if (k[t + s] > k[t] || (k[t + s] == k[t] && id[t + s] < id[t])) {
                    k[t] = k[t + s];
                    id[t] = id[t + s];
                }
    };
    reduce_argmax(key_all, id_all);
    reduce_argmax(key_msk, id_msk);
    out.full_token   = id_all[0];
    out.full_key     = key_all[0];
    out.masked_token = id_msk[0];
    out.masked_key   = key_msk[0];

    // 4. the candidates' logits, in candidate order.
    const uint32_t n = std::min<uint32_t>(static_cast<uint32_t>(cand.size()),
                                          gpu::kDsVerifyMaxCand);
    for (uint32_t i = 0; i < n; ++i) {
        const int32_t v = cand[i];
        out.cand_logit[i] = (v >= 0 && uint32_t(v) < rows) ? logits[uint32_t(v)] : -3.0e38f;
    }
}

}  // namespace deepmoe::runtime
