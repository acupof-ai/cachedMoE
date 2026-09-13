// MoE sublayer interface (design §7.8, §7.9).
//
// Per layer: gate -> top-6 -> [CPU checks residency, issues P0 I/O, host-signals
// the timeline] -> fused FP4 gate/up+SwiGLU over 7 experts -> down + reduction.
// The 6 routed experts plus the shared expert are 112.8 + 36.5 MB per layer,
// the 4.5 GB per token that design §3.1 says decides everything.
//
// The GPU never learns which experts are resident. It reads the pointer table
// of design §5.3 and trusts the timeline wait; the residency decision is
// entirely store/planner.h's.
//
// Ownership/threading: Moe borrows the ExpertStore, Planner and Timeline. The
// routing readback is host-coherent memory written by the gate kernel and spun
// on by the submit thread (design §7.8 -- a counter, not a fence).
#pragma once

#include <cstdint>
#include <span>

#include "core/status.h"
#include "core/types.h"
#include "gpu/vulkan/timeline.h"
#include "store/planner.h"

namespace deepmoe::runtime {

// The host-coherent block the gate kernel writes (design §7.8).
struct RouterReadback {
    uint32_t ids[layout::kExpertsPerTok]     = {};
    float    weights[layout::kExpertsPerTok] = {};
    uint32_t top16_ids[16]    = {};
    float    top16_scores[16] = {};
    uint32_t layer_done = 0;   // atomic counter the CPU spins on
};

class Moe {
public:
    virtual ~Moe() = default;

    // Records dispatches #9..#11 of design §7.14.
    // TODO(design §7.8, §7.9): implement in P2.
    virtual Result<void> record_decode(uint32_t layer, uint32_t batch_m) = 0;

    // Expert-major streaming prefill: iterate experts in shard order,
    // gather the tokens routed to each, one GEMM, scatter-add back
    // (design §7.13, §9.7).
    // TODO(design §7.13, §9.7): implement in P5.
    virtual Result<void> record_prefill(uint32_t layer, std::span<const uint32_t> token_expert_map) = 0;

    // Blocks until the gate kernel has published `layer`'s routing, then runs
    // the planner and signals the timeline value the MoE dispatch waits on.
    // This is the only CPU-in-the-loop step of a decode layer (design §7.8).
    // TODO(design §7.8): implement in P3, once the gate kernel exists.
    virtual Result<store::LayerPlan> resolve_routing(uint32_t layer, TokenIndex token) = 0;
};

}  // namespace deepmoe::runtime
