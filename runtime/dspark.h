// DSpark speculative decoding (design §7.12, §10).
//
// The cycle:
//   1. main forward at M = k+1 (last accepted token + k drafts); record the hc
//      mean of the *inputs* to layers 37/38/39's attention
//   2. verify position by position -> accept a prefix of length a in [0, k]
//   3. draft: main_proj(15360 -> 5120) feeds three DSparkBlocks at M=5, then
//      head, then the Markov head steps 5 positions, then the confidence head
//   4. the scheduler picks the next k from the confidence outputs
//
// The invariant that makes this safe to leave on: at temperature 0, output with
// speculation enabled must be token-for-token identical to output without it
// (design §10.2). That is a test, not a hope.
//
// The draft blocks' 128 experts x 3 blocks (7.2 GB) are pinned, so a draft
// cycle never waits on NVMe -- about 2.2 GB of traffic, ~11 ms.
//
// Ownership/threading: Dspark borrows the model runtime and the KV cache. It
// runs on the engine thread; its expert set is pinned at startup so the planner
// never sees it.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/config.h"
#include "core/status.h"
#include "core/types.h"
#include "runtime/kvcache.h"

namespace deepmoe::runtime {

// One draft block's output: kDsparkBlockSize candidate tokens with the
// per-position acceptance probability from the confidence head.
struct DraftBlock {
    std::vector<uint32_t> tokens;      // <= dspark_block_size (5)
    std::vector<float>    confidence;  // c_j, conditional acceptance probability
};

struct VerifyResult {
    uint32_t accepted = 0;             // a in [0, k]
    std::vector<uint32_t> tokens;      // a + 1 tokens: the accepted prefix plus the correction
    bool diverged_at_low_margin = false;  // recorded, not fatal (design §12 L3)
};

class Dspark {
public:
    virtual ~Dspark() = default;

    // Produces the next draft block from the main model's layer-37/38/39 hc
    // means and the last accepted token (design §7.12).
    // TODO(design §7.12): implement in P4.
    virtual Result<DraftBlock> draft(std::span<const float> main_hidden,
                                     uint32_t last_token) = 0;

    // Greedy verification: compare argmax position by position. Sampling-mode
    // rejection sampling is the second half of P4 (design §10.2).
    // TODO(design §10.2): implement in P4.
    virtual Result<VerifyResult> verify(std::span<const uint32_t> drafted,
                                        std::span<const float> target_logits) = 0;

    // Confidence-scheduled verify length: maximise E[accepted] / T_cycle(k)
    // using the profiler's live T_hot(M), T_nvme(k) and T_draft curves
    // (design §10.3).
    // TODO(design §10.3): implement in P4, needs the Q3 Jaccard statistics.
    virtual Result<uint32_t> schedule_next_k(const DraftBlock& last) = 0;

    // Undo the main model's KV writes for rejected positions (design §10.2).
    // TODO(design §10.2): implement in P4.
    virtual Result<void> rollback(KvCache& kv, const KvSnapshot& snap) = 0;
};

}  // namespace deepmoe::runtime
