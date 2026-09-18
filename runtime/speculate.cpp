#include "runtime/speculate.h"

#include "core/profiler.h"

#include <algorithm>
#include <cmath>
#include <array>
#include <cstring>
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

// --- the cycle ---------------------------------------------------------------

namespace {
double ms_between(TimePoint a, TimePoint b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
}  // namespace

double Speculator::next_u() {
    // splitmix64, so the uniform stream is reproducible from SpecConfig::seed
    // alone and a speculative run can be replayed.
    rng_ += 0x9E3779B97F4A7C15ull;
    uint64_t z = rng_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    // (0, 1): 53 bits, offset so neither endpoint is attainable.
    return (double(z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

Result<SpecCycle> Speculator::cycle(uint32_t p0, uint32_t last_token,
                                    std::vector<uint32_t>& out) {
    using cpu::dspark::kPositions;
    if (!model_) return fail(Err::FailedPrecondition, "Speculator has no model");
    if (cfg_.mode == SpecMode::Off)
        return fail(Err::FailedPrecondition, "Speculator::cycle with --spec off");
    if (cfg_.lattice_k == 0 || cfg_.lattice_k > gpu::kDsVerifyMaxCand)
        return fail(Err::InvalidArgument,
                    std::format("lattice K must be 1..{}, not {}", gpu::kDsVerifyMaxCand,
                                cfg_.lattice_k));
    if (rng_ == 0) rng_ = cfg_.seed;

    SpecCycle c;
    const TimePoint t0 = Clock::now();

    // 1. the draft: one forward, M = 5.
    std::array<float, kPositions> conf{};
    if (auto r = model_->draft_forward(p0, last_token, cfg_.lattice_k, lat_,
                                       std::span<float, kPositions>(conf));
        !r)
        return std::unexpected(r.error());
    const TimePoint t1 = Clock::now();
    c.draft_ms = ms_between(t0, t1);

    // 2. k. The confidence rule is a PREFIX rule -- whether position j is
    //    verified depends only on positions < j -- which is what makes the
    //    temperature-1 coupling exact (docs/p3_dspark.md §3.2).
    uint32_t k = cfg_.k ? std::min<uint32_t>(cfg_.k, kPositions)
                        : cpu::dspark::k_from_confidence(std::span<const float, kPositions>(conf),
                                                         cfg_.theta);
    // Never verify nothing: M = k + 1 >= 2 means a cycle always commits at
    // least the token the model would have produced anyway, so speculation off
    // and speculation on with k = 1 emit the same stream at the same cost plus
    // one row.
    if (k == 0) k = 1;
    last_k_ = k;
    c.k = k;

    // 3. the path. Greedy takes the objective's path; temperature 1 takes an
    //    ancestral sample, which is the q the acceptance rule is coupled to.
    cpu::dspark::Path path{};
    std::array<double, kPositions> u_path{};
    if (cfg_.mode == SpecMode::Sample) {
        for (uint32_t i = 0; i < kPositions; ++i) u_path[i] = next_u();
        path = lat_.sample(std::span<const double, kPositions>(u_path));
    } else {
        path = lat_.path(cfg_.objective);
    }
    const auto path_tokens = lat_.tokens(path);

    // 4. the verify batch: [last, path[:k]] at p0 .. p0 + k.
    const uint32_t M = k + 1;
    batch_.assign(1, last_token);
    for (uint32_t j = 0; j < k; ++j) batch_.push_back(uint32_t(path_tokens[j]));

    const bool exact = cfg_.mode == SpecMode::Sample;
    const uint32_t K = cfg_.lattice_k;
    const uint32_t stride = gpu::kDsVerifyMaxCand;
    cand_.clear();
    if (exact) {
        // Row j scores the candidates of DRAFT position j, which is what
        // `accept_sampling_exact` indexes: row 0 of the batch is the last
        // accepted token and its answer is position 0's, row 1 is position 1's.
        cand_.assign(size_t(M) * stride, -1);
        for (uint32_t j = 0; j < M && j < kPositions; ++j)
            for (uint32_t c2 = 0; c2 < K; ++c2)
                cand_[size_t(j) * stride + c2] = lat_.token(j, c2);
    }
    rows_.assign(M, VerifyRow{});

    if (auto r = model_->snapshot_ring(p0, M); !r) return std::unexpected(r.error());
    const TimePoint t2 = Clock::now();
    c.cpu_ms += ms_between(t1, t2);

    if (auto r = model_->verify_forward(p0, batch_, cand_, stride, exact, rows_); !r)
        return std::unexpected(r.error());
    const TimePoint t3 = Clock::now();
    c.verify_ms = ms_between(t2, t3);
    c.stall_ms = model_->last_stall_ms();
    c.union_experts = model_->last_union_experts();
    c.miss_bytes = model_->last_miss_bytes();

    // 5. acceptance.
    cpu::dspark::Accept acc{};
    if (cfg_.mode == SpecMode::Greedy) {
        std::array<int32_t, kPositions + 1> argmax{};
        for (uint32_t j = 0; j < M; ++j) argmax[j] = int32_t(rows_[j].argmax);
        acc = cpu::dspark::accept_greedy(
            std::span<const int32_t>(path_tokens.data(), k),
            std::span<const int32_t>(argmax.data(), M), k);
    } else {
        cand_logit_.assign(size_t(kPositions) * stride, 0.0f);
        lse_.assign(size_t(k) + 1, 0.0f);
        masked_.assign(k ? k : 1, 0);
        full_.assign(size_t(k) + 1, 0);
        for (uint32_t j = 0; j < M; ++j) {
            lse_[j] = rows_[j].exact.lse;
            full_[j] = int32_t(rows_[j].exact.full_token);
            if (j < k) masked_[j] = int32_t(rows_[j].exact.masked_token);
            if (j < kPositions)
                std::memcpy(cand_logit_.data() + size_t(j) * stride,
                            rows_[j].exact.cand_logit, size_t(K) * sizeof(float));
        }
        std::array<double, kPositions> u_acc{};
        std::array<double, kPositions + 1> u_res{};
        for (uint32_t i = 0; i < kPositions; ++i) u_acc[i] = next_u();
        for (uint32_t i = 0; i <= kPositions; ++i) u_res[i] = next_u();
        acc = cpu::dspark::accept_sampling_exact(
            lat_, path, k, cand_logit_, stride, lse_, masked_, full_,
            std::span<const double, kPositions>(u_acc),
            std::span<const double, kPositions + 1>(u_res));
    }
    const TimePoint t4 = Clock::now();
    c.cpu_ms += ms_between(t3, t4);
    c.accepted = acc.accepted;
    c.emitted  = acc.n_emitted;

    // 6. the rollback, then the commit. Only the rejected positions' ring slots:
    //    p0 + a + 1 .. p0 + k. Position p0 + a + 1 is among them because the
    //    correction token is not the drafted one -- the next cycle's row 0
    //    rewrites that slot with the right value.
    if (acc.accepted < k) {
        if (auto r = model_->restore_ring(p0, acc.accepted, M); !r)
            return std::unexpected(r.error());
    }
    std::vector<uint32_t> emitted;
    emitted.reserve(acc.n_emitted);
    for (uint32_t j = 0; j < acc.n_emitted; ++j) emitted.push_back(uint32_t(acc.tokens[j]));
    if (auto r = model_->commit(p0, emitted); !r) return std::unexpected(r.error());
    out.insert(out.end(), emitted.begin(), emitted.end());
    c.rollback_ms = ms_between(t4, Clock::now());

    stats_.add(c);
    return c;
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
