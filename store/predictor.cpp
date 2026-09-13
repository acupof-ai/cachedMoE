#include "store/predictor.h"

#include <memory>

namespace deepmoe::store {
namespace {

class NullPredictor final : public Predictor {
public:
    const char* name() const override { return "null(demand-only)"; }
    Result<std::vector<Prediction>> predict(std::span<const float>, uint32_t,
                                            uint32_t, uint32_t) override {
        return std::vector<Prediction>{};
    }
    void observe(uint32_t, std::span<const uint16_t>) override {}
    PredictorAccuracy accuracy(uint32_t) const override { return {}; }
};

}  // namespace

std::unique_ptr<Predictor> make_null_predictor() { return std::make_unique<NullPredictor>(); }

// TODO(design §9.4): implement once route_trace/cache_sim answer Q4 and Q5.
Result<std::unique_ptr<Predictor>> make_gate_predictor(std::span<const GateWeights>,
                                                       LookaheadInput,
                                                       const PrefetchConfig&) {
    return unimplemented("store::make_gate_predictor (design §9.4, gated on Q4/Q5)");
}

}  // namespace deepmoe::store
