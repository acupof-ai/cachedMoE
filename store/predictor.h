// Lookahead router predictor (design §9.4).
//
// The idea: after layer L's FFN updates the residual stream, the CPU evaluates
// the *gate weights of layers L+1..L+d* against an approximation of what those
// layers will see, takes top-K per layer, and hands the not-yet-resident ones
// to the Planner as P1 work. With T_layer ~= 1.3 ms and T_io(18.8 MB) ~= 3.8 ms,
// d must be at least 3-4 for a miss to be fully hidden.
//
// Nothing here is implemented, by design. Design §16: "no Planner policy code
// before the P-1 and P1 measurements". The open questions this interface exists
// to be filled in against are:
//
//   Q4  recall@K of the L-d prediction for L's true top-6, over d=1..8, K=6..16
//   Q5  T_layer / T_io, which sets the minimum useful d
//
// tools/route_trace.py produces the trace; tools/cache_sim.py evaluates the
// (d, K, input approximation) grid and writes the answer back into design §9.
//
// Ownership/threading: a Predictor is owned by the Planner and called from the
// planner thread only. It borrows the gate weights (which live in the pinned
// pinned region) and never copies them.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/config.h"
#include "core/status.h"
#include "core/types.h"

namespace deepmoe::store {

// One layer's prediction: `width` expert ids with their predicted scores,
// best first.
struct Prediction {
    uint32_t layer = 0;
    std::vector<uint16_t> ids;
    std::vector<float>    scores;
};

// Which approximation of the layer-L input to feed the layer-(L+d) gate.
// design §9.4 lists two candidates; route_trace picks the one with better
// recall.
enum class LookaheadInput : uint8_t {
    MeanHc = 0,     // ffn_norm(mean over the 4 hc copies)
    PreIdentity,    // ffn_norm(hc_pre(x, identity))
};

// Online precision accounting: a prefetched expert counts as "used" if a layer
// within the next `horizon` layers actually routes to it (design §9.4).
struct PredictorAccuracy {
    uint64_t predicted = 0;
    uint64_t used      = 0;
    double precision() const { return predicted ? static_cast<double>(used) / predicted : 0.0; }
};

class Predictor {
public:
    virtual ~Predictor() = default;
    virtual const char* name() const = 0;

    // Predicts the top-`width` experts for layers `from_layer+1 .. +depth`.
    // `residual_hc` is the current [hc_mult][hidden] fp32 residual stream.
    virtual Result<std::vector<Prediction>>
    predict(std::span<const float> residual_hc, uint32_t from_layer,
            uint32_t depth, uint32_t width) = 0;

    // Reports whether a previously predicted expert was actually routed to, so
    // the Planner can widen or narrow K online (design §9.4).
    virtual void observe(uint32_t layer, std::span<const uint16_t> actual_top6) = 0;
    virtual PredictorAccuracy accuracy(uint32_t layer) const = 0;
};

// Gate weights for one layer, borrowed from the pinned region.
struct GateWeights {
    std::span<const uint16_t> w;      // [n_experts, hidden] bf16
    std::span<const float>    bias;   // [n_experts] fp32, `noaux_tc`
    uint32_t n_experts = 0;
    uint32_t hidden    = 0;
};

// TODO(design §9.4): the real predictor -- an AVX-512 bf16 GEMV per predicted
// layer (3.9 MB each, ~157 MB/token over 40 layers) overlapped with the GPU.
// Returns Unimplemented until Q4 has picked LookaheadInput, d and K.
Result<std::unique_ptr<Predictor>> make_gate_predictor(std::span<const GateWeights> per_layer_gates,
                                                       LookaheadInput input,
                                                       const PrefetchConfig& cfg);

// A predictor that predicts nothing. This is what the runtime uses today, and
// it is the honest baseline: demand-only paging, the h=0.30 row of design §3.1.
std::unique_ptr<Predictor> make_null_predictor();

}  // namespace deepmoe::store
