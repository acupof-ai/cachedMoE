// A conversation over one long-lived Engine (Track P, docs/p3_chat.md §1), and
// what Track R2 adds to it (docs/p4_kv_ux.md): KV rollback, window replay,
// parked named sessions, and cancellation.
//
// What a turn costs is dominated by what the process already has: the pinned
// set (9 GiB, ~10 s to load) and the routed-expert cache (80 GiB, which takes
// hundreds of tokens to fill and is worth 1 -> 12 tok/s once warm). So the
// session lives as long as the process, and a request whose prompt shares a
// prefix with the tokens already in the KV store only pays for the rest.
//
// The rule: the KV store holds `engine.history()`. A prompt p that strictly
// extends the history feeds p[h..). One that diverges at position c ROLLS BACK
// to k = c rounded down to even (never past |p| - 1): the compressed rows,
// index keys and counts are truncated to k, the window ring is rebuilt by
// replaying at most 128 tokens, and p[k..) is fed. Rolling back is chosen when
// k is larger than the replay it needs; otherwise the context resets to 0. A
// generated token is fed only when the next one is wanted, so after a turn
// that ended on a stop id the history is prompt + generated[..-1].
//
// The window ring is never saved (design §11.2)
// ---------------------------------------------
// Everything that restores a context -- a rollback, a parked session coming
// back -- keeps only the non-SWA state: compressed KV rows, index keys, the
// compressor's carried group and the token ids. The ring a step at k needs
// holds positions [k - 127, k); whichever of them the store no longer holds
// (`KvStore::ring_holds`) are recomputed by running those positions through the
// decode path again, with
//   * the window floor at the first replayed position, so a replayed query
//     does not attend to slots holding other positions;
//   * the replay starting on an even position, so no ratio-2 group completes
//     on a carry the replay did not write;
//   * after every replayed step, the compressed row and index key it completed
//     put back from a bf16 copy taken before the replay, and after the last
//     one the carried state -- so the non-SWA state stays exactly what the
//     original run wrote and only the ring is the replay's.
// A ring slot is only lost when a later position overwrote it, so rolling back
// d < 128 tokens replays d positions, not 128.
//
// Why k is even. A ratio-2 source's carry holds the first half of an open
// group, which is overwritten one step later and is not kept per group. At an
// even k no group is open, the carry is dead state (both slots are rewritten
// before the next pool), and the token at k - 1 of an odd divergence point is
// simply fed again. That is how "a partially filled group" is handled exactly:
// it never has to be reconstructed.
//
// Ownership/threading: borrows the Engine and the Tokenizer; one thread. A
// cancel callback may be backed by an atomic set from another thread.
#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <string>
#include <vector>

#include "core/status.h"
#include "runtime/engine.h"
#include "runtime/kvstore.h"
#include "runtime/sampling.h"
#include "text/tokenizer.h"

namespace deepmoe::runtime {

// --- window replay, rollback, parking (Track R2) ------------------------------------

struct ReplayPlan {
    uint32_t keep  = 0;    // context length after the operation
    uint32_t first = 0;    // positions [first, end) are replayed
    uint32_t end   = 0;
    uint32_t steps() const { return end > first ? end - first : 0; }
};

struct ReplayStats {
    ReplayPlan plan;
    uint32_t   dropped = 0;       // tokens the rollback removed
    double     ms = 0.0;          // the replay steps
    double     backup_ms = 0.0;   // copying and restoring rows
    uint64_t   backup_bytes = 0;
    bool       cancelled = false;
};

// What rolling the engine's context back to `keep` tokens would replay.
// `replay_max` bounds it (<= window); fewer replayed positions leave the
// oldest part of the next steps' window masked instead.
ReplayPlan plan_rollback(Engine& e, uint32_t keep, uint32_t replay_max = 128);

// Keeps the first `keep` tokens (even, <= context_length()) and leaves the
// store ready for a step at `keep`.
Result<ReplayStats> rollback_context(Engine& e, uint32_t keep, uint32_t replay_max = 128);

// A context without its window: the token ids and the packed non-SWA state.
struct ParkedContext {
    std::vector<uint32_t> tokens;
    KvPacked              kv;
    uint64_t bytes() const { return kv.bytes() + tokens.size() * sizeof(uint32_t); }
};

Result<ParkedContext> park_context(Engine& e);
// Replaces the engine's context with `p`: store cleared, non-SWA state
// unpacked, window rebuilt by replaying the last <= replay_max tokens.
Result<ReplayStats> restore_context(Engine& e, const ParkedContext& p, uint32_t replay_max = 128);

// --- disk session/prefix cache (design §11.4) --------------------------------
// Saves a parked context's non-SWA state (token ids + KvPacked) under
// `<dir>/<session>.pkv`. The window ring is NEVER written: a hit loads the rows
// and then restore_context() rebuilds the window with a bounded replay. The
// header carries a hash of `model_tag`; a mismatch or a corrupt file is a miss,
// not a hard failure. Format version 1.
struct KvDiskOptions {
    std::string dir;                     // empty = disabled
    std::string model_tag;               // model identity, e.g. the model directory
    uint64_t    max_bytes = 4ull << 30;  // evict least-recently-written *.pkv past this
};

// Writes `p` atomically (`<name>.pkv.tmp` -> rename) and evicts old files past
// `opt.max_bytes`. Missing/empty dir is InvalidArgument; IO failures are Io.
Result<void> save_parked_context(const ParkedContext& p, const KvDiskOptions& opt,
                                 const std::string& name);
// Missing file, model-tag mismatch or bad version returns NotFound/Corrupt.
Result<ParkedContext> load_parked_context(const KvDiskOptions& opt, const std::string& name);
// Removes the file if present; true when a file was removed.
bool drop_parked_context(const KvDiskOptions& opt, const std::string& name);


// --- generation ------------------------------------------------------------------------

struct GenerateRequest {
    std::vector<uint32_t> prompt_ids;
    uint32_t              max_tokens = 256;
    SamplingParams        sampling{};                // temperature <= 0: greedy
    std::vector<uint32_t> stop_ids{1};               // <｜end▁of▁sentence｜>
    bool                  reuse = true;              // allow KV continuation / rollback
    // Polled between tokens (and between prefill chunks); true stops the
    // request with finish "cancel" and the KV store consistent with history().
    std::function<bool()> cancel;
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
    std::string finish = "length";         // stop | length | context | cancel
    double   prefill_ms = 0.0, ttft_ms = 0.0, decode_ms = 0.0, total_ms = 0.0;
    // Track R2: a rollback in front of the prefill.
    uint32_t rollback_dropped = 0, replay_steps = 0;
    double   replay_ms = 0.0;
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
    // Track R1 round 2 (§7): what the turn's reheat pass was handed. Zero when
    // reheating is off.
    uint32_t reheat_turn = 0, reheat_keys = 0, reheat_free_slots = 0;

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
    // A progress callback (and a cancel check) every this many prefill tokens.
    uint32_t progress_every = 16;
    // Track R2: roll back to a diverging prompt's common prefix instead of
    // re-prefilling from 0, when that is cheaper.
    bool     rollback = true;
    // Track R1 round 2 (docs/p4_hitrate.md §7): reheat the expert cache at every
    // turn boundary -- decay all slot heat by `reheat_decay`, re-rank the
    // resident experts, and run the P3 backfill again on that order. The read is
    // issued after the reply is finished and runs behind the next turn's demand
    // traffic, so it costs the turn nothing.
    bool     reheat = false;
    float    reheat_decay = 0.5f;
};

class Session {
public:
    Session(Engine& engine, const text::Tokenizer& tok, SessionOptions opt = {})
        : engine_(&engine), tok_(&tok), opt_(opt) {}

    Result<GenerateStats> generate(
        const GenerateRequest& req, const std::function<void(const TokenEvent&)>& on_token,
        const std::function<void(uint32_t done, uint32_t total)>& on_prefill = {});

    void reset() { engine_->reset_context(); }
    const SessionOptions& options() const { return opt_; }

private:
    Engine*                engine_;
    const text::Tokenizer* tok_;
    SessionOptions         opt_;
};

// --- named sessions -------------------------------------------------------------------

struct SessionPoolOptions {
    uint32_t      max_parked       = 8;
    uint64_t      max_parked_bytes = 4ull << 30;
    KvDiskOptions disk{};            // non-empty dir: save/load parked contexts
};

// Several conversations over one Engine. One is live in the KV store; the others
// are parked as `ParkedContext`s (their non-SWA state, packed, and their ids).
// The expert cache, the pinned set and the planner are shared by all of them.
// Switching parks the live one and restores the target, replaying its window;
// past `max_parked` or `max_parked_bytes` the least recently used parked
// session is dropped.
class SessionPool {
public:
    SessionPool(Engine& engine, const text::Tokenizer& tok, SessionOptions opt = {},
                SessionPoolOptions pool = {});

    // Makes `name` live (creating it empty if it does not exist).
    Result<ReplayStats> activate(const std::string& name);
    // Track R2 SSD cache: park/save the live session (clean shutdown), and try
    // to load the active name from disk into an empty engine (fresh process).
    // `restore_active_from_disk` returns NotFound when there is no file.
    Result<void> park_active();
    Result<ReplayStats> restore_active_from_disk();
    Session& live() { return session_; }
    const std::string& active() const { return active_; }
    // Drops a session's state; dropping the live one resets the context.
    bool drop(const std::string& name);
    void reset_live() { session_.reset(); }

    struct Info {
        std::string name;
        uint32_t    tokens = 0;
        uint64_t    bytes = 0;
        bool        live = false;
    };
    std::vector<Info> list() const;
    uint32_t evicted() const { return evicted_; }

private:
    void enforce_budget();

    Engine*                engine_;
    Session                session_;
    SessionPoolOptions     pool_;
    std::string            active_ = "default";
    std::map<std::string, ParkedContext> parked_;
    std::list<std::string> lru_;     // most recent first, parked names only
    uint32_t               evicted_ = 0;
};

}  // namespace deepmoe::runtime
