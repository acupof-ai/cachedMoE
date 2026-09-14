#include "cpu/dspark_tree.h"

// Bit-exactness with tools/dspark_tree.py requires that a*b + c is never fused
// into one FMA instruction (-march=znver5 has FMA).
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif
#pragma STDC FP_CONTRACT OFF

#include <array>
#include <cmath>
#include <format>
#include <limits>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace deepmoe::cpu::dspark {
namespace {

constexpr double kLn2      = 0.6931471805599453;
constexpr double kSqrtHalf = 0.7071067811865476;
constexpr int    kExpTerms = 16;
constexpr int    kLogTerms = 14;
constexpr double kExpLo    = -745.0;
constexpr double kNegInf   = -std::numeric_limits<double>::infinity();

uint32_t first_argmax(const double* v, uint32_t n) noexcept {
    double best = v[0];
    uint32_t arg = 0;
    for (uint32_t i = 1; i < n; ++i)
        if (v[i] > best) { best = v[i]; arg = i; }
    return arg;
}

// tools/dspark_tree.py::_sample_index
uint32_t sample_index(const double* w, uint32_t n, double u) noexcept {
    double total = 0.0;
    for (uint32_t i = 0; i < n; ++i) total = total + w[i];
    const double t = u * total;
    double cum = 0.0;
    uint32_t last = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (w[i] > 0.0) last = i;
        cum = cum + w[i];
        if (cum > t) return i;
    }
    return last;
}

// tools/dspark_tree.py::lse_seq for one row, optional extra term.
double lse_seq(const double* x, uint32_t n, bool has_extra, double extra) noexcept {
    double m = x[0];
    for (uint32_t i = 1; i < n; ++i)
        if (x[i] > m) m = x[i];
    if (has_extra && extra > m) m = extra;
    if (!std::isfinite(m)) return m;
    double s = 0.0;
    for (uint32_t i = 0; i < n; ++i) s = s + dm_exp(x[i] - m);
    if (has_extra) s = s + dm_exp(extra - m);
    return m + dm_log(s);
}

}  // namespace

double dm_exp(double x) noexcept {
    if (!(x >= kExpLo)) return 0.0;
    const double k = std::floor(x / kLn2 + 0.5);
    const double r = x - k * kLn2;
    double p = 1.0;
    // reciprocals, not divisions: 1.0 / n is the same correctly rounded double on both sides
    static constexpr auto inv = [] {
        std::array<double, kExpTerms + 1> t{};
        for (int n = 1; n <= kExpTerms; ++n) t[n] = 1.0 / static_cast<double>(n);
        return t;
    }();
    for (int n = kExpTerms; n >= 1; --n) p = 1.0 + (p * r) * inv[n];
    return std::ldexp(p, static_cast<int>(k));
}

double dm_log(double y) noexcept {
    if (!(y > 0.0)) return kNegInf;
    int e = 0;
    double m = std::frexp(y, &e);
    if (m < kSqrtHalf) { m = m * 2.0; e -= 1; }
    const double f  = (m - 1.0) / (m + 1.0);
    const double f2 = f * f;
    static constexpr auto c = [] {
        std::array<double, kLogTerms + 1> t{};
        for (int n = 0; n <= kLogTerms; ++n) t[n] = 2.0 / static_cast<double>(2 * n + 1);
        return t;
    }();
    double s = c[kLogTerms];
    for (int n = kLogTerms - 1; n >= 0; --n) s = c[n] + f2 * s;
    return f * s + static_cast<double>(e) * kLn2;
}

float dot16(const float* a, const float* b, size_t n) noexcept {
    float acc[16];
#if defined(__AVX512F__)
    __m512 v = _mm512_setzero_ps();
    for (size_t r = 0; r < n; r += 16)
        v = _mm512_add_ps(v, _mm512_mul_ps(_mm512_loadu_ps(a + r), _mm512_loadu_ps(b + r)));
    _mm512_storeu_ps(acc, v);
#else
    for (int l = 0; l < 16; ++l) acc[l] = 0.0f;
    for (size_t r = 0; r < n; r += 16)
        for (int l = 0; l < 16; ++l) {
            const float prod = a[r + l] * b[r + l];
            acc[l] = acc[l] + prod;
        }
#endif
    float s = 0.0f;
    for (int l = 0; l < 16; ++l) s = s + acc[l];
    return s;
}

Result<void> Lattice::build(const LatticeInput& in) {
    const uint32_t K = in.K, P = kPositions;
    if (K == 0 || K > kMaxK) return fail(Err::InvalidArgument, std::format("K={} out of range", K));
    if (in.cand_ids.size() < P * K || in.cand_logit.size() < P * K)
        return fail(Err::InvalidArgument, "candidate spans too small");
    if (in.e_in.size() < kRank || in.e_cand.size() < (P - 1) * K * kRank ||
        in.h_cand.size() < P * K * kRank)
        return fail(Err::InvalidArgument, "markov row spans too small");
    if (!in.lse_anchor.empty() && in.lse_anchor.size() < P)
        return fail(Err::InvalidArgument, "lse_anchor span too small");

    K_ = K;
    ids_.assign(in.cand_ids.begin(), in.cand_ids.begin() + P * K);
    e_in_.assign(in.e_in.begin(), in.e_in.begin() + kRank);
    e_cand_.assign(in.e_cand.begin(), in.e_cand.begin() + (P - 1) * K * kRank);
    bias_.assign(P * K * K, 0.0f);
    logq_.assign(P * K * K, 0.0);

    const bool has_tail = !in.lse_anchor.empty();
    std::array<double, kMaxK> score{};
    for (uint32_t i = 0; i < P; ++i) {
        const uint32_t n_prev = i == 0 ? 1 : K;
        for (uint32_t p = 0; p < n_prev; ++p) {
            const float* prev = i == 0 ? e_in_.data() : e_cand_.data() + ((i - 1) * K + p) * kRank;
            const size_t row = (i * K + p) * K;
            for (uint32_t c = 0; c < K; ++c)
                bias_[row + c] = dot16(prev, in.h_cand.data() + (i * K + c) * kRank, kRank);
        }
        // tail mass outside the candidates, exact for the anchor (prev row 0)
        double tail = kNegInf;
        if (has_tail) {
            const double la = static_cast<double>(in.lse_anchor[i]);
            const size_t row0 = (i * K) * K;
            double s = 0.0;
            for (uint32_t c = 0; c < K; ++c) {
                const double sc = static_cast<double>(in.cand_logit[i * K + c]) +
                                  static_cast<double>(bias_[row0 + c]);
                s = s + dm_exp(sc - la);
            }
            const double rest = 1.0 - s;
            tail = rest > 0.0 ? la + dm_log(rest) : kNegInf;
        }
        for (uint32_t p = 0; p < n_prev; ++p) {
            const size_t row = (i * K + p) * K;
            for (uint32_t c = 0; c < K; ++c)
                score[c] = static_cast<double>(in.cand_logit[i * K + c]) +
                           static_cast<double>(bias_[row + c]);
            const double lse = lse_seq(score.data(), K, has_tail, tail);
            for (uint32_t c = 0; c < K; ++c) logq_[row + c] = score[c] - lse;
        }
    }
    return {};
}

const float* Lattice::prev_embed(const Path& path, uint32_t i) const noexcept {
    return i == 0 ? e_in_.data() : e_cand_.data() + ((i - 1) * K_ + path[i - 1]) * kRank;
}

Path Lattice::path(Objective obj) const noexcept {
    const uint32_t K = K_, P = kPositions;
    Path out{};
    switch (obj) {
    case Objective::Chain: {
        uint32_t prev = 0;
        for (uint32_t i = 0; i < P; ++i) {
            out[i] = first_argmax(&logq_[(i * K + prev) * K], K);
            prev = out[i];
        }
        return out;
    }
    case Objective::Viterbi: {
        std::array<double, kMaxK> v{}, best{};
        std::array<std::array<uint32_t, kMaxK>, kPositions> back{};
        for (uint32_t c = 0; c < K; ++c) v[c] = logq_[c];
        for (uint32_t i = 1; i < P; ++i) {
            for (uint32_t c = 0; c < K; ++c) {
                best[c] = v[0] + logq_[(i * K + 0) * K + c];
                back[i][c] = 0;
            }
            for (uint32_t p = 1; p < K; ++p)
                for (uint32_t c = 0; c < K; ++c) {
                    const double t = v[p] + logq_[(i * K + p) * K + c];
                    if (t > best[c]) { best[c] = t; back[i][c] = p; }
                }
            v = best;
        }
        uint32_t c = first_argmax(v.data(), K);
        out[P - 1] = c;
        for (uint32_t i = P - 1; i > 0; --i) {
            c = back[i][c];
            out[i - 1] = c;
        }
        return out;
    }
    case Objective::Eal: {
        std::array<double, kMaxK> g{}, g_next{}, val{};
        std::array<std::array<uint32_t, kMaxK>, kPositions> choice{};
        for (uint32_t ii = P; ii-- > 0;) {
            const uint32_t n_prev = ii == 0 ? 1 : K;
            for (uint32_t p = 0; p < n_prev; ++p) {
                const size_t row = (ii * K + p) * K;
                for (uint32_t c = 0; c < K; ++c) val[c] = dm_exp(logq_[row + c]) * (1.0 + g[c]);
                const uint32_t a = first_argmax(val.data(), K);
                choice[ii][p] = a;
                g_next[p] = val[a];
            }
            g = g_next;
        }
        uint32_t prev = 0;
        for (uint32_t i = 0; i < P; ++i) {
            out[i] = choice[i][prev];
            prev = out[i];
        }
        return out;
    }
    }
    return out;
}

Path Lattice::sample(std::span<const double, kPositions> u) const noexcept {
    const uint32_t K = K_;
    Path out{};
    std::array<double, kMaxK> w{};
    uint32_t prev = 0;
    for (uint32_t i = 0; i < kPositions; ++i) {
        const size_t row = (i * K + prev) * K;
        for (uint32_t c = 0; c < K; ++c) w[c] = dm_exp(logq_[row + c]);
        out[i] = sample_index(w.data(), K, u[i]);
        prev = out[i];
    }
    return out;
}

double Lattice::eal_value(const Path& path, uint32_t k) const noexcept {
    double tot = 0.0, run = 1.0;
    uint32_t prev = 0;
    for (uint32_t i = 0; i < k; ++i) {
        run = run * dm_exp(logq(i, prev, path[i]));
        tot = tot + run;
        prev = path[i];
    }
    return tot;
}

std::array<int32_t, kPositions> Lattice::tokens(const Path& path) const noexcept {
    std::array<int32_t, kPositions> t{};
    for (uint32_t i = 0; i < kPositions; ++i) t[i] = token(i, path[i]);
    return t;
}

float confidence(const float* x, const float* e, const float* w, float bias) noexcept {
    // dot16 over cat(x, e): 5120 = 320 x 16, so the lanes continue across the seam.
    float acc[16];
#if defined(__AVX512F__)
    __m512 v = _mm512_setzero_ps();
    for (size_t r = 0; r < kHidden; r += 16)
        v = _mm512_add_ps(v, _mm512_mul_ps(_mm512_loadu_ps(x + r), _mm512_loadu_ps(w + r)));
    for (size_t r = 0; r < kRank; r += 16)
        v = _mm512_add_ps(v, _mm512_mul_ps(_mm512_loadu_ps(e + r), _mm512_loadu_ps(w + kHidden + r)));
    _mm512_storeu_ps(acc, v);
#else
    for (int l = 0; l < 16; ++l) acc[l] = 0.0f;
    for (size_t r = 0; r < kHidden; r += 16)
        for (int l = 0; l < 16; ++l) { const float p = x[r + l] * w[r + l]; acc[l] = acc[l] + p; }
    for (size_t r = 0; r < kRank; r += 16)
        for (int l = 0; l < 16; ++l) { const float p = e[r + l] * w[kHidden + r + l]; acc[l] = acc[l] + p; }
#endif
    float s = 0.0f;
    for (int l = 0; l < 16; ++l) s = s + acc[l];
    return s + bias;
}

double sigmoid(double c) noexcept { return 1.0 / (1.0 + dm_exp(-c)); }

uint32_t k_from_confidence(std::span<const float, kPositions> conf, double theta) noexcept {
    double run = 1.0;
    uint32_t k = 0;
    for (uint32_t j = 0; j < kPositions; ++j) {
        run = run * sigmoid(static_cast<double>(conf[j]));
        if (run < theta) break;
        k = j + 1;
    }
    return k;
}

Accept accept_greedy(std::span<const int32_t> path_tokens, std::span<const int32_t> argmax,
                     uint32_t k) noexcept {
    Accept r;
    uint32_t a = 0;
    while (a < k && path_tokens[a] == argmax[a]) {
        r.tokens[a] = path_tokens[a];
        ++a;
    }
    r.tokens[a] = argmax[a];
    r.accepted = a;
    r.n_emitted = a + 1;
    return r;
}

Accept accept_sampling(const Lattice& lat, const Path& path, uint32_t k,
                       std::span<const int32_t> ver_ids, std::span<const float> ver_logit,
                       uint32_t Kv, std::span<const double, kPositions> u_acc,
                       std::span<const double, kPositions + 1> u_res) noexcept {
    const uint32_t K = lat.K();
    Accept r;
    std::array<double, kMaxKv> pl{}, pv{}, res{};
    std::array<double, kMaxK> qrow{};
    auto target = [&](uint32_t j) {
        for (uint32_t v = 0; v < Kv; ++v) pl[v] = static_cast<double>(ver_logit[j * Kv + v]);
        const double lse = lse_seq(pl.data(), Kv, false, 0.0);
        for (uint32_t v = 0; v < Kv; ++v) pv[v] = dm_exp(pl[v] - lse);
    };
    for (uint32_t j = 0; j < k; ++j) {
        target(j);
        const uint32_t prev = j == 0 ? 0 : path[j - 1];
        for (uint32_t c = 0; c < K; ++c) qrow[c] = dm_exp(lat.logq(j, prev, c));
        const uint32_t c = path[j];
        const int32_t x = lat.token(j, c);
        const double qx = qrow[c];
        double px = 0.0;
        for (uint32_t v = 0; v < Kv; ++v)
            if (ver_ids[j * Kv + v] == x) { px = pv[v]; break; }
        if (u_acc[j] * qx < px) {
            r.tokens[j] = x;
            continue;
        }
        for (uint32_t v = 0; v < Kv; ++v) {
            double qv = 0.0;
            for (uint32_t cc = 0; cc < K; ++cc)
                if (lat.token(j, cc) == ver_ids[j * Kv + v]) { qv = qrow[cc]; break; }
            const double d = pv[v] - qv;
            res[v] = d > 0.0 ? d : 0.0;
        }
        double total = 0.0;
        for (uint32_t v = 0; v < Kv; ++v) total = total + res[v];
        r.tokens[j] = total > 0.0 ? ver_ids[j * Kv + sample_index(res.data(), Kv, u_res[j])]
                                  : ver_ids[j * Kv + first_argmax(pv.data(), Kv)];
        r.accepted = j;
        r.n_emitted = j + 1;
        return r;
    }
    target(k);
    r.tokens[k] = ver_ids[k * Kv + sample_index(pv.data(), Kv, u_res[k])];
    r.accepted = k;
    r.n_emitted = k + 1;
    return r;
}

Accept accept_sampling_exact(const Lattice& lat, const Path& path, uint32_t k,
                             std::span<const float> cand_logit, uint32_t stride,
                             std::span<const float> lse, std::span<const int32_t> masked,
                             std::span<const int32_t> full,
                             std::span<const double, kPositions> u_acc,
                             std::span<const double, kPositions + 1> u_res) noexcept {
    const uint32_t K = lat.K();
    Accept r;
    std::array<double, kMaxK> pc{}, qrow{}, w{};
    for (uint32_t j = 0; j < k; ++j) {
        const double la = static_cast<double>(lse[j]);
        for (uint32_t c = 0; c < K; ++c) pc[c] = dm_exp(static_cast<double>(cand_logit[j * stride + c]) - la);
        const uint32_t prev = j == 0 ? 0 : path[j - 1];
        for (uint32_t c = 0; c < K; ++c) qrow[c] = dm_exp(lat.logq(j, prev, c));
        const uint32_t c = path[j];
        if (u_acc[j] * qrow[c] < pc[c]) {
            r.tokens[j] = lat.token(j, c);
            continue;
        }
        double sw = 0.0, sp = 0.0;
        for (uint32_t i = 0; i < K; ++i) {
            const double d = pc[i] - qrow[i];
            w[i] = d > 0.0 ? d : 0.0;
            sw = sw + w[i];
            sp = sp + pc[i];
        }
        double rest = 1.0 - sp;
        if (!(rest > 0.0)) rest = 0.0;
        const double t = u_res[j] * (sw + rest);
        if (t < sw) {
            double cum = 0.0;
            int pick = -1;
            uint32_t last = 0;
            for (uint32_t i = 0; i < K; ++i) {
                if (w[i] > 0.0) last = i;
                cum = cum + w[i];
                if (cum > t) { pick = static_cast<int>(i); break; }
            }
            r.tokens[j] = lat.token(j, pick < 0 ? last : static_cast<uint32_t>(pick));
        } else {
            r.tokens[j] = masked[j];
        }
        r.accepted = j;
        r.n_emitted = j + 1;
        return r;
    }
    r.tokens[k] = full[k];
    r.accepted = k;
    r.n_emitted = k + 1;
    return r;
}

}  // namespace deepmoe::cpu::dspark
