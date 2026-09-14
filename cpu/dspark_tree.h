// DSpark tree sampling over the draft matrix and top-K acceptance on the verify
// matrix (docs/p3_dspark.md §3, Track K2). The CPU half of the single-pass,
// single-path speculation scheme:
//
//   draft forward  -> base logits B[5, V] (token-independent) -> GPU top-K
//   CPU (here)     -> K x K Markov bias between consecutive positions from
//                     gathered 256-d rows, a 5-level lattice, ONE path
//   verify forward -> [last, path[:k]], M = k + 1 -> GPU top-Kv per row
//   CPU (here)     -> greedy prefix match, or lossless top-Kv speculative sampling
//
// Bit-exact with tools/dspark_tree.py (tests/test_dspark_tree.cpp compares with
// ==): float32 dot products accumulate in 16 lanes (one AVX-512 register per row)
// and sum the lanes in order; exp / log are float64 series built from + - * /
// floor frexp ldexp only; every logsumexp and cumulative sum is sequential. The
// .cpp is compiled with floating-point contraction off.
//
// Ownership/threading: Lattice copies what it needs from caller-owned spans in
// build() and is then immutable; all functions are reentrant. No allocation after
// build() on the path/accept calls.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/status.h"

namespace deepmoe::cpu::dspark {

inline constexpr uint32_t kPositions = 5;       // dspark_block_size
inline constexpr uint32_t kRank      = 256;     // dspark_markov_rank
inline constexpr uint32_t kHidden    = 5120;    // dim
inline constexpr uint32_t kMaxK      = 64;
inline constexpr uint32_t kMaxKv     = 256;

// float64 exp / log without libm (see the header comment).
double dm_exp(double x) noexcept;
double dm_log(double y) noexcept;

// float32 dot product over n elements (n % 16 == 0), 16-lane order.
float dot16(const float* a, const float* b, size_t n) noexcept;

enum class Objective : uint8_t { Viterbi, Eal, Chain };

using Path = std::array<uint32_t, kPositions>;   // candidate index per position

// Candidates per position and the rows gathered for them. The anchor of position i
// is the prev the GPU biased that position's full row with (the input token at
// i = 0, candidate 0 of position i-1 after it -- the reference's greedy chain),
// so candidate 0 of every position is the chain token.
struct LatticeInput {
    uint32_t K = 0;
    std::span<const int32_t> cand_ids;    // [P*K] anchor row's top-K, best first
    std::span<const float>   cand_logit;  // [P*K] BASE logits at the candidates (no bias)
    std::span<const float>   e_in;        // [256]      markov embed of the draft input token
    std::span<const float>   e_cand;      // [(P-1)*K*256] markov embed of candidates 0..P-2
    std::span<const float>   h_cand;      // [P*K*256]  markov head rows of the candidates
    std::span<const float>   lse_anchor;  // [P] full-vocab logsumexp of the anchor rows;
                                          // empty = normalise over the K candidates only
};

class Lattice {
public:
    Result<void> build(const LatticeInput& in);

    uint32_t K() const noexcept { return K_; }
    // log q_i(c | prev); prev is ignored (0) at i == 0.
    double logq(uint32_t i, uint32_t prev, uint32_t c) const noexcept {
        return logq_[(i * K_ + (i == 0 ? 0 : prev)) * K_ + c];
    }
    float bias(uint32_t i, uint32_t prev, uint32_t c) const noexcept {
        return bias_[(i * K_ + (i == 0 ? 0 : prev)) * K_ + c];
    }
    int32_t token(uint32_t i, uint32_t c) const noexcept { return ids_[i * K_ + c]; }
    // The markov embed the confidence head pairs with position i on `path`.
    const float* prev_embed(const Path& path, uint32_t i) const noexcept;

    Path path(Objective obj) const noexcept;
    Path sample(std::span<const double, kPositions> u) const noexcept;
    double eal_value(const Path& path, uint32_t k = kPositions) const noexcept;
    std::array<int32_t, kPositions> tokens(const Path& path) const noexcept;

    // Every value, row order [i][prev][c]; i == 0 uses row prev = 0, the rest are 0.
    std::span<const double> logq_all() const noexcept { return logq_; }
    std::span<const float>  bias_all() const noexcept { return bias_; }

private:
    uint32_t K_ = 0;
    std::vector<int32_t> ids_;
    std::vector<float>   e_in_, e_cand_;
    std::vector<float>   bias_;     // [P*K*K]
    std::vector<double>  logq_;     // [P*K*K]
};

// DSparkConfidenceHead: proj(cat(x_i, E[prev_i])), raw score. x [5120], e [256], w [5376].
float confidence(const float* x, const float* e, const float* w, float bias = 0.0f) noexcept;
double sigmoid(double c) noexcept;
// Longest prefix whose cumulative sigmoid(conf) stays >= theta.
uint32_t k_from_confidence(std::span<const float, kPositions> conf, double theta) noexcept;

struct Accept {
    uint32_t accepted = 0;                       // a
    uint32_t n_emitted = 0;                      // a + 1
    std::array<int32_t, kPositions + 1> tokens{};
};

// Greedy: argmax[j] is row j's top-1 (rows 0..k).
Accept accept_greedy(std::span<const int32_t> path_tokens, std::span<const int32_t> argmax,
                     uint32_t k) noexcept;

// Speculative sampling against top-Kv truncated verify rows. `lat` must be the
// lattice the path was sampled from (normalised over its K candidates).
// ver_ids / ver_logit: [(k+1)*Kv], each row best first.
Accept accept_sampling(const Lattice& lat, const Path& path, uint32_t k,
                       std::span<const int32_t> ver_ids, std::span<const float> ver_logit,
                       uint32_t Kv, std::span<const double, kPositions> u_acc,
                       std::span<const double, kPositions + 1> u_res) noexcept;

// Speculative sampling exactly lossless against plain temperature-1 sampling.
// Per verify row j (one pass over the row the head already produced):
//   cand_logit [P*stride]  main logits at C_j (the draft candidates, path-independent),
//                          row j at offset j*stride, the first K entries used
//   lse        [k+1]       full-vocab logsumexp of each row
//   masked     [k]         a sample of row j with C_j masked out
//   full       [k+1]       a sample of the whole row (only row k is used: the bonus)
// See tools/dspark_tree.py::accept_sampling_exact for the rule.
Accept accept_sampling_exact(const Lattice& lat, const Path& path, uint32_t k,
                             std::span<const float> cand_logit, uint32_t stride,
                             std::span<const float> lse, std::span<const int32_t> masked,
                             std::span<const int32_t> full,
                             std::span<const double, kPositions> u_acc,
                             std::span<const double, kPositions + 1> u_res) noexcept;

}  // namespace deepmoe::cpu::dspark
