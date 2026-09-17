// DSpark speculative decoding in the runtime: the cycle, its accounting, and
// the host mirror of the verify-row kernel (design §7.12 / §10,
// docs/p3_dspark.md §3, docs/p4_dspark_runtime.md).
//
// One cycle
// ---------
//   draft   one forward at M = 5 over the main model's layer-37/38/39 hc means
//           -> base logits, then five sequential Markov steps -> a K = 16
//           candidate lattice (cpu::dspark::Lattice)
//   path    one path through the lattice: `eal` at temperature 0, ancestral
//           `sample` at temperature 1
//   k       the longest prefix whose cumulative sigmoid(confidence) stays above
//           theta, or a fixed k
//   verify  ONE forward at M = k + 1 over [last accepted, path[:k]]
//   accept  greedy prefix match, or the exactly lossless sampling rule
//   roll    restore the window-ring slots the rejected positions wrote
//
// The invariant (design §10.2)
// ---------------------------
// At temperature 0 the token stream with speculation on must be the token
// stream with it off, token for token. That is what `SpecMode::Greedy` is
// tested against, and it is the reason `Engine::forward_batch`'s per-row argmax
// has to agree with M = 1 decode: a batch-boundary difference in the logits is
// an output difference, not a rounding detail.
//
// What this file does NOT own: the draft forward and the verify forward are
// Engine's (`Engine::forward_batch`), the lattice and the acceptance rules are
// `cpu/dspark_tree.h`'s, and the ring rollback is `KvStore::snapshot_ring` /
// `restore_ring`. This is the cycle that puts them in order plus the numbers a
// go / no-go needs.
//
// Ownership/threading: single-threaded, on the engine thread.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/status.h"
#include "cpu/dspark_tree.h"
#include "gpu/vulkan/dspark_kernels.h"

namespace deepmoe::runtime {

enum class SpecMode : uint8_t {
    Off = 0,
    // Temperature 0. `accept_greedy`; the hard invariant applies.
    Greedy,
    // Temperature 1. `accept_sampling_exact`; lossless against plain sampling,
    // so the token stream is NOT expected to match a non-speculative run --
    // only its distribution is (docs/p3_dspark.md §13.3).
    Sample,
};

const char* spec_mode_name(SpecMode m);
// "off" / "greedy" / "sample"; anything else is an error.
Result<SpecMode> parse_spec_mode(std::string_view s);

struct SpecConfig {
    SpecMode mode = SpecMode::Off;
    // Draft positions to verify. 0 = choose per cycle from the confidence head
    // (the prefix rule of docs/p3_dspark.md §3.2); 1..5 = fixed.
    uint32_t k = 0;
    // The confidence threshold when `k == 0`. K2 swept {0.3, 0.5, 0.7}.
    double   theta = 0.5;
    // Lattice width. 16 is K2's; the kernel's candidate bound is
    // gpu::kDsVerifyMaxCand.
    uint32_t lattice_k = 16;
    // Path objective at temperature 0. `Eal` is the default: it maximises the
    // expected accepted length rather than the joint probability.
    cpu::dspark::Objective objective = cpu::dspark::Objective::Eal;
    uint64_t seed = 0x5DEEC'E66Dull;
};

// One cycle's accounting, so a go / no-go has the terms of
// `E[accept] / T_cycle(k)` measured rather than projected.
struct SpecCycle {
    uint32_t k = 0;             // draft positions verified
    uint32_t accepted = 0;      // a in [0, k]
    uint32_t emitted = 0;       // a + 1 tokens committed
    double   draft_ms = 0.0;
    double   verify_ms = 0.0;
    double   cpu_ms = 0.0;      // lattice + acceptance
    double   rollback_ms = 0.0;
    double   stall_ms = 0.0;    // of verify_ms: waiting on expert I/O
    uint32_t union_experts = 0; // the verify batch's expert union, summed over layers
    uint64_t miss_bytes = 0;
};

struct SpecStats {
    uint64_t cycles = 0;
    uint64_t tokens = 0;        // emitted
    uint64_t accepted = 0;      // sum of a
    uint64_t verified = 0;      // sum of k
    double   draft_ms = 0.0, verify_ms = 0.0, cpu_ms = 0.0, rollback_ms = 0.0, stall_ms = 0.0;
    uint64_t union_experts = 0;
    uint64_t miss_bytes = 0;

    void add(const SpecCycle& c);
    double accepted_per_verify() const { return cycles ? double(accepted) / double(cycles) : 0.0; }
    double tokens_per_cycle() const { return cycles ? double(tokens) / double(cycles) : 0.0; }
    double cycle_ms() const {
        return cycles ? (draft_ms + verify_ms + cpu_ms + rollback_ms) / double(cycles) : 0.0;
    }
    std::string to_string() const;
};

// --- the host mirror of gpu/shaders/dspark_verify.slang ----------------------
//
// Float for float, including the 256-lane strided partition and the fixed-shape
// reduction tree, so a test can compare the kernel's record word by word rather
// than "about right". The one thing it cannot reproduce exactly is `exp` and
// `log`: the GPU's are its own, so `lse` and the Gumbel keys agree to a few ULP
// and the SAMPLED TOKEN can differ when two keys are within that -- which is a
// near-tie, and the test reports the margin rather than pretending otherwise.
//
// `logits` is the row (vocab entries), `cand` the row's candidate ids with a
// negative entry ending the list, `row` the verify row index (part of the RNG
// counter, so row j and row j' draw independently from the same seed).
void emulate_verify_row(std::span<const float> logits, std::span<const int32_t> cand,
                        uint32_t seed, uint32_t row, float inv_t,
                        gpu::DsparkVerifyRow& out);

// The uniforms the kernel draws, exposed so a test can check the hash itself
// rather than only its consequences.
uint32_t verify_hash3(uint32_t a, uint32_t b, uint32_t c);
float    verify_u01(uint32_t h);
float    verify_gumbel(uint32_t seed, uint32_t row, uint32_t id);

}  // namespace deepmoe::runtime
