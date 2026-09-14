// A conversation over one long-lived Engine (Track P, docs/p3_chat.md §1).
//
// What a turn costs is dominated by what the process already has: the pinned
// set (9 GiB, ~10 s to load) and the routed-expert cache (80 GiB, which takes
// hundreds of tokens to fill and is worth 1 -> 12 tok/s once warm). So the
// session lives as long as the process, and a request whose prompt EXTENDS the
// tokens already in the KV store only pays for the new ones.
//
// The rule: the KV store holds `engine.history()`. A prompt p with
// history == p[0..h) for h < |p| feeds p[h..) and generates; anything else
// resets to position 0 and feeds all of p. A generated token is fed only when
// the next one is wanted, so after a turn that ended on a stop id the history is
// prompt + generated[..-1] and the next rendered prompt -- which contains the
// stop id -- extends it.
//
// Ownership/threading: borrows the Engine and the Tokenizer; one thread.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "core/status.h"
#include "runtime/engine.h"
#include "runtime/sampling.h"
#include "text/tokenizer.h"

namespace deepmoe::runtime {

struct GenerateRequest {
    std::vector<uint32_t> prompt_ids;
    uint32_t              max_tokens = 256;
    SamplingParams        sampling{};                // temperature <= 0: greedy
    std::vector<uint32_t> stop_ids{1};               // <｜end▁of▁sentence｜>
    bool                  reuse = true;              // allow KV continuation
};

struct TokenEvent {
    uint32_t    id = 0;
    std::string text;          // the UTF-8 this token completed (may be empty)
    double      t_ms = 0.0;    // since the request arrived
    double      step_ms = 0.0; // the decode step that produced it
    double      p = 0.0;       // softmax probability (sampled steps)
    float       margin = 0.0f; // top1 - top2 of the step's logits
    uint32_t    nucleus = 0;
    bool        fallback = false;
    double      hit_rate = 0.0;   // of the step that produced it
};

struct GenerateStats {
    uint32_t prompt_tokens = 0, reused_tokens = 0, prefilled_tokens = 0, generated = 0;
    std::string prefill_mode = "none";     // none | decode | gpu
    std::string finish = "length";         // stop | length | context
    double   prefill_ms = 0.0, ttft_ms = 0.0, decode_ms = 0.0, total_ms = 0.0;
    // expert cache over the prefill steps and over the decode steps
    uint64_t prefill_requests = 0, prefill_hits = 0, prefill_nvme_bytes = 0;
    uint64_t decode_requests = 0, decode_hits = 0, decode_nvme_bytes = 0;
    // design 13.1, summed over the decode steps (divide by decode_steps)
    StepBreakdown decode_sum{};
    uint32_t decode_steps = 0;
    uint32_t sampled_steps = 0, topk_fallbacks = 0, topk_checked = 0, topk_mismatches = 0;
    uint64_t nucleus_sum = 0;
    double   sample_ms_sum = 0.0;
    uint32_t context_after = 0;

    double decode_tok_s() const { return decode_ms > 0 ? decode_steps * 1e3 / decode_ms : 0.0; }
    double prefill_tok_s() const {
        return prefill_ms > 0 ? prefilled_tokens * 1e3 / prefill_ms : 0.0;
    }
    double decode_hit_rate() const {
        return decode_requests ? double(decode_hits) / double(decode_requests) : 0.0;
    }
    double prefill_hit_rate() const {
        return prefill_requests ? double(prefill_hits) / double(prefill_requests) : 0.0;
    }
    // One line of JSON fields (no braces), for `deepmoe serve`'s done event.
    std::string json_fields() const;
};

struct SessionOptions {
    // Prompts (fed from position 0) at least this long go through Track L's
    // GPU prefill; 0 = never. Shorter ones, and every continuation, go through
    // the decode path, which also warms the expert cache for the reply.
    uint32_t gpu_prefill_min = 0;
    uint32_t replay = 128;
    // A progress callback every this many prefill tokens.
    uint32_t progress_every = 16;
};

class Session {
public:
    Session(Engine& engine, const text::Tokenizer& tok, SessionOptions opt = {})
        : engine_(&engine), tok_(&tok), opt_(opt) {}

    Result<GenerateStats> generate(
        const GenerateRequest& req, const std::function<void(const TokenEvent&)>& on_token,
        const std::function<void(uint32_t done, uint32_t total)>& on_prefill = {});

    void reset() { engine_->reset_context(); }

private:
    Engine*                engine_;
    const text::Tokenizer* tok_;
    SessionOptions         opt_;
};

}  // namespace deepmoe::runtime
