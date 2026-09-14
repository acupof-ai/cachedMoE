// Temperature + top-p sampling from a GPU top-k (design §7.11, docs/p3_chat.md §2).
//
// The head writes 129,280 fp32 logits into GPU-visible memory. Copying them to
// the host is a write-combining read (~0.5 ms), so the sampled path does what
// the greedy one does and reduces them on the GPU first:
// `gpu/shaders/sample_topk.slang` finds the maximum M, histograms `M - l` in
// bins of 1/16 logit, picks the smallest bin J whose cumulative count reaches
// k, and returns every (id, logit) in bins 0..J -- a true top set, never a
// sample of one, so the candidates are exactly the tokens with the largest
// logits -- plus
//
//     tail = sum over every OTHER token of exp((l - M) / T)
//
// With the candidates' own weights summed in double on the host,
//
//     S   = sum_cand exp((l - M) / T) + tail
//     p_i = exp((l_i - M) / T) / S                   (the full-vocab softmax)
//
// so the nucleus built from the candidates is the full-vocabulary nucleus
// EXACTLY whenever the candidates hold at least top_p of the mass: sorting by
// (logit desc, id asc) and cutting at the first cumulative mass >= top_p never
// needs a token the GPU did not return. When they do not (a flat distribution,
// top_p = 1, or a per-thread candidate segment overflowed), the caller copies
// the whole logit vector and `nucleus_from_full` does the same thing on the
// host. Either way the dropped mass (1 - retained) is reported.
//
// The RNG is counter-based -- `uniform01(seed, position)` -- so a sampled run
// is reproducible from its seed and does not depend on how many draws came
// before (a KV-continued turn samples exactly what a fresh run would).
//
// Ownership/threading: pure functions over caller-owned spans.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/status.h"

namespace deepmoe::runtime {

struct SamplingParams {
    float    temperature = 1.0f;    // <= 0 means greedy (argmax)
    float    top_p       = 0.95f;   // model README: 0.95 (or 1.0)
    uint64_t seed        = 0;
    bool greedy() const { return !(temperature > 0.0f); }
};

// What gpu/shaders/sample_topk.slang writes, decoded (or what
// `emulate_topk` computes the same way on the host).
struct TopKLogits {
    struct Cand { uint32_t id; float logit; };
    std::vector<Cand> cand;   // every token in bins 0..J, unordered
    float    max_logit = 0.0f;
    double   tail = 0.0;      // sum of exp((l - max) / T) over the non-candidates
    uint32_t bin = 0;         // J
    uint32_t rows = 0;        // the vocabulary the kernel scanned
    bool     overflow = false;
};

// The distribution a draw is taken from.
struct Nucleus {
    std::vector<uint32_t> ids;     // sorted by (logit desc, id asc)
    std::vector<double>   p;       // full-softmax probabilities of `ids`
    std::vector<double>   cdf;     // cumulative over `ids`, renormalised: cdf.back() == 1
    double retained = 0.0;         // softmax mass of the candidate set (1 for a full vocab)
    double kept     = 0.0;         // softmax mass of `ids` (>= top_p unless the vocab ran out)
    double logsumexp = 0.0;        // of logit / T over the whole vocabulary
    bool   exact = false;          // the nucleus provably equals the full-vocab one
};

// From a GPU (or emulated) top set. Returns a nucleus with exact == false when
// the candidates cannot prove it -- the caller falls back to the full copy.
Nucleus nucleus_from_topk(const TopKLogits& tk, float temperature, float top_p);

// From the whole logit vector (the fallback, and the reference in tests).
Nucleus nucleus_from_full(std::span<const float> logits, float temperature, float top_p);

// From a partial list plus the full-vocab logsumexp of (logit / T) -- what the
// L3 export records (`top_logits`, `logit_stats[1]` at T = 1). exact is true
// iff the listed mass reaches top_p.
Nucleus nucleus_from_partial(std::span<const uint32_t> ids, std::span<const float> logits,
                             double logsumexp, float temperature, float top_p);

// Inverse-CDF draw: the first index whose cdf exceeds u, u in [0, 1).
uint32_t sample_nucleus(const Nucleus& n, double u);

// A [0, 1) double from (seed, counter): two rounds of splitmix64, 53 bits.
double uniform01(uint64_t seed, uint64_t counter);

// sample_topk.slang's arithmetic on the host, float for float: used by the
// test and by `--check-topk` to check the kernel's candidate set.
TopKLogits emulate_topk(std::span<const float> logits, uint32_t k, float temperature);

// The kernel's constants, shared with gpu/shaders/sample_topk.slang.
inline constexpr uint32_t kTopKBins          = 256;
inline constexpr float    kTopKBinsPerLogit  = 16.0f;
inline constexpr uint32_t kTopKCapPerThread  = 32;
inline constexpr uint32_t kTopKThreads       = 256;
inline constexpr uint32_t kTopKHeaderWords   = 8;
inline constexpr uint32_t kTopKDefaultK      = 1024;

}  // namespace deepmoe::runtime
