// Sampling (design §7.11).
//
// Sampling happens on the GPU: temperature scaling plus Gumbel-max (matching
// the reference) or plain argmax, with a counter-based Philox stream so a run
// is reproducible from (seed, position). Only the top-k and the chosen token
// come back across PCIe -- never the 129,280-wide logit vector.
//
// Ownership/threading: stateless apart from the counter. The counter is
// advanced once per sampled position on the engine thread.
#pragma once

#include <cstdint>
#include <limits>
#include <span>

#include "core/status.h"
#include "core/types.h"

namespace deepmoe::runtime {

struct SamplerConfig {
    float    temperature = 0.0f;   // 0 = argmax; the speculation invariant of §10.2
    uint32_t top_k       = 0;      // 0 = disabled
    float    top_p       = 1.0f;
    uint64_t seed        = 0;
};

struct SampleResult {
    uint32_t token = 0;
    float    logprob = 0.0f;
    float    margin = 0.0f;   // top1 - top2; design §12 L3 records this at a divergence
};

// Philox 4x32-10 counter-based RNG: the same (seed, counter) gives the same
// stream on CPU and GPU, which is what makes a sampled run reproducible and a
// CPU/GPU A/B meaningful.
struct Philox {
    uint64_t seed    = 0;
    uint64_t counter = 0;
    // TODO(design §7.11): the 10-round implementation, shared with the shader.
    Result<void> fill_uniform(std::span<float> out) {
        (void)out;
        return unimplemented("runtime::Philox::fill_uniform (design §7.11)");
    }
};

class Sampler {
public:
    virtual ~Sampler() = default;

    // CPU reference path, used by the oracle of design §12 and by the CLI
    // before the GPU sampler exists.
    // TODO(design §7.11): implement alongside the CPU forward pass (P0).
    virtual Result<SampleResult> sample(std::span<const float> logits,
                                        const SamplerConfig& cfg,
                                        uint64_t position) = 0;
};

// Greedy argmax over an fp32 logit vector. Implemented: it is the whole of
// temperature-0 decoding, which is what every correctness test uses. `margin`
// is top1 - top2, the quantity design §12 L3 records when deepMoE and the fp32
// oracle disagree.
inline Result<SampleResult> argmax(std::span<const float> logits) {
    if (logits.empty()) return fail(Err::InvalidArgument, "empty logit vector");
    uint32_t best = 0;
    float best_v = logits[0];
    float second = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > best_v) { second = best_v; best_v = logits[i]; best = i; }
        else if (logits[i] > second) { second = logits[i]; }
    }
    SampleResult r;
    r.token   = best;
    r.logprob = best_v;
    r.margin  = (logits.size() > 1) ? best_v - second : 0.0f;
    return r;
}

}  // namespace deepmoe::runtime
