// Engine: the object that owns everything and runs the token loop
// (design §4, §15).
//
// Ownership graph (see docs/architecture.md):
//
//   Engine
//    +- V41Config, Manifest                    parsed once, immutable
//    +- store::ShardSet         the 48 original safetensors shards (no repack)
//    +- storage::IoEngine  -> Backend          owns 1 dispatcher thread,
//    |                                         the backend owns 2 IOCP threads
//    +- gpu::Device -> MemoryAllocator x2 (path A and path B), Timeline
//    +- store::PinnedStore                     the ~17.7 GB of design §9.3
//    +- store::ExpertStore -> SlabPool -> SlabBacking   path A then path B
//    +- store::Planner     -> EvictionPolicy, Predictor
//    +- gpu::AttnRunner, gpu::DecodeRunner, gpu::GpuScratch
//    +- runtime::KvStore, DecodeLayer, GpuMoeBridge, EngramRunner
//    +- runtime::DecodeState                   the LOADED half (design §7.4)
//    +- Profiler                               JSONL sink
//
// Threading model:
//   engine/submit thread  records the token loop's command buffers -- cut at
//                         each gate, so one submit carries a layer's MoE and
//                         the next layer's attention -- submits them gated on
//                         the residency timeline, fences on a second timeline,
//                         reads the gate's ids, runs the Planner, host-signals
//                         the residency timeline, issues the engram reads at
//                         token start, samples. Vulkan queue submission happens
//                         only here.
//   io dispatcher thread  owned by IoEngine: priority, chunking, completions.
//   iocp completion x2    owned by the backend: GetQueuedCompletionStatus.
//   planner thread        background prefetch and backfill between tokens
//                         (design §9.4, §9.6 P3) -- not started; §9.4 is a net
//                         loss at every (d, K) the simulator tried.
//
// What a decode step really does, and what it is handed
// ----------------------------------------------------
// Real: the embedding lookup, forty layers of design §7.14's dispatches 1-11
// including §7.4's compressor and indexer on their source layers, the §7.1
// residency gate between 9 and 10, the engram at layers 1 and 14 including its
// NVMe row fetch, the final collapse, the 1.32 GB head and the argmax. Handed
// to it: the state the prompt left behind -- from the L3 export's prefill
// record, or from `slow_prefill`, in which case nothing at all. `status()` says
// which in one line, and docs/p2_decode.md §9 says it at length.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <cstdio>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/profiler.h"
#include "core/status.h"
#include "core/types.h"
#include "gpu/vulkan/attn_kernels.h"
#include "gpu/vulkan/decode_kernels.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/timeline.h"
#include "model/manifest.h"
#include "model/v41_config.h"
#include "runtime/decode_layer.h"
#include "runtime/decode_state.h"
#include "runtime/engram.h"
#include "runtime/kvcache.h"
#include "runtime/kvstore.h"
#include "runtime/moe_bridge.h"
#include "runtime/resident_route.h"
#include "runtime/sampler.h"
#include "runtime/sampling.h"
#include "runtime/trace.h"
#include "storage/file.h"
#include "storage/io_engine.h"
#include "store/expert_store.h"
#include "store/pinned.h"
#include "store/planner.h"
#include "store/shard_set.h"

namespace deepmoe::gpu { struct PrefillHandoff; }

namespace deepmoe::runtime {

// --- H1a: the auto cache size never exceeds the measured safe slot count -----
//
// `--cache-gb` / `--cache-slots` are explicit and own their risk. The *auto*
// budget is not: it is arithmetic over two heaps, and on this machine the
// arithmetic lands at 5,100 slots / 89.3 GiB (docs/p4_hitrate.md, F4) while the
// measured ceiling is 5,000 -- 5,100 died on the FIRST decode submit today
// (`vkQueueSubmit2 failed (-2)`, slab layout 34 A + 17 B), 5,400 died after
// twelve minutes and 5,500 lost the device on the first token (STATUS §1
// "cache 容量上限", §3 row 59). The heap arithmetic cannot see any of that, so
// the measured number is a hard cap applied AFTER it.
//
// 5,000 = 34 path-A slabs + 16 path-B slabs, which is the layout that ran four
// clean turns on 2026-09-19. `DEEPMOE_CACHE_SLOT_CAP` overrides it for a
// different machine; 0 turns the cap off and puts you back on the arithmetic.
inline constexpr uint32_t kAutoSlotCap = 5000;

// The cap in force: `DEEPMOE_CACHE_SLOT_CAP` if set (0 = no cap), else
// kAutoSlotCap. Read once per process.
uint32_t auto_slot_cap();

// Pure: the auto budget, capped. Returns `budget_bytes` unchanged when the cap
// is off (`slot_cap == 0`), when `slot_bytes == 0`, or when the budget already
// sits at or under the cap; otherwise `slot_cap * slot_bytes`. Whole slabs are
// the pool's business -- capping the byte budget is enough, because the pool
// only ever allocates slabs that fit inside it.
uint64_t cap_auto_budget(uint64_t budget_bytes, uint64_t slot_bytes, uint32_t slot_cap);

// How many whole slots a byte budget buys, for logging the before/after.
inline uint64_t budget_slots(uint64_t budget_bytes, uint64_t slot_bytes) {
    return slot_bytes ? budget_bytes / slot_bytes : 0;
}

// --- H1a: auto PROBES rather than computes ---------------------------------
//
// The cap above is a fallback -- a number measured on one machine on one day.
// The real check is empirical: build the slab pool, submit once, and see. The
// submit is where an over-sized cache is discovered (the driver accepts every
// allocation and only has to make them all resident when work is queued), so a
// trivial command buffer is a sufficient probe. A refused probe releases the
// pool, drops the slot budget by `cache_backoff_step()` and rebuilds.
//
// Only `auto` backs off. `--cache-slots` / `--cache-gb` are a deliberate
// request: they get the same probe, but a failure is fatal with the number to
// try next, because silently handing back a smaller cache would corrupt every
// A/B that pins the slot count.
inline constexpr uint32_t kCacheBackoffSlots   = 200;
inline constexpr uint32_t kCacheBackoffTries   = 5;   // the first try included
inline constexpr uint32_t kCacheBackoffFloor   = 200; // never probe below this

// `DEEPMOE_CACHE_BACKOFF_SLOTS` if set and > 0, else kCacheBackoffSlots.
uint32_t cache_backoff_step();

// Pure: the slot counts an auto-sized cache tries, in order, first one first.
// Strictly DECREASING by `step` and floored at `floor_slots`; the sequence
// stops early once `floor_slots` is reached, so it never repeats a count and
// never grows. `attempts == 0` or `start == 0` gives an empty plan.
std::vector<uint32_t> cache_backoff_slots(uint32_t start, uint32_t step, uint32_t attempts,
                                          uint32_t floor_slots = kCacheBackoffFloor);

struct GenerateOptions {
    uint32_t max_tokens = 64;
    bool     greedy     = true;     // design §10.2 invariant
    bool     speculative = false;   // P4
    // Teacher forcing: feed `forced[i]` as step i's input instead of what the
    // previous step sampled. design §12 L3 wants both -- teacher-forced isolates
    // one step's error, free-running compounds it.
    std::span<const uint32_t> forced;
};

struct GenerateResult {
    std::vector<uint32_t> tokens;
    RunSummary            summary;
};

// The §13.1 breakdown for one layer of one token.
//
// The GPU halves are GPU TIMESTAMPS, not host wall clocks around a submit: since
// a layer's MoE and the next layer's attention share one command buffer, a host
// clock can no longer split them, and it never could separate the kernel from
// the queue.
struct LayerTiming {
    double   engram_ms = 0.0;   // row fetch (host) + the two dispatches (GPU)
    double   attn_ms   = 0.0;   // dispatches 1-9 plus design 7.4's, GPU
    double   gate_ms   = 0.0;   // the §7.1 residency gate: plan + wait for NVMe
    double   moe_ms    = 0.0;   // moe_gpu_ms + moe_host_ms
    double   moe_gpu_ms = 0.0;  // dispatch A, the h quantisation, dispatch B: GPU
    double   moe_host_ms = 0.0; // x out of GPU memory, act_quant, the table rows
    uint32_t hits = 0, misses = 0;
    uint64_t miss_bytes = 0;
};

// One token, summed over its layers, in the buckets docs/p2_decode.md reports.
struct StepBreakdown {
    double   attn_ms = 0.0, moe_gpu_ms = 0.0, moe_host_ms = 0.0;
    double   gate_ms = 0.0, engram_ms = 0.0, tail_ms = 0.0;
    double   other_ms = 0.0;    // wall minus everything above: record, submit, fence, bind
    // `other_ms`, taken apart on the host clock. What is left of it after
    // these is the part of each fence wait the GPU timestamps do not cover:
    // queue latency, the wake-up, and the dispatches outside a stamped span.
    double   record_ms = 0.0;   // vkCmd* calls, every buffer of the token
    double   submit_ms = 0.0;   // vkQueueSubmit2
    double   wait_ms   = 0.0;   // host blocked on the completion fence
    double   bind_ms   = 0.0;   // slot tables, RoPE tables, design 7.4 bookkeeping
    // `moe_host_ms`, taken apart.
    double   moe_x_ms = 0.0, moe_quant_ms = 0.0, moe_table_ms = 0.0;
    uint32_t submits = 0;
    bool     gpu_timed = false; // false if the device has no timestamp queries
    // Track MS: this STEP's own expert-cache accounting, summed from the
    // layers. The store's counters are process-wide, so with two streams
    // decoding at once a delta of them is both streams' traffic; these are the
    // stream's own and are what a per-stream hit rate has to be built from.
    uint32_t requests = 0, hits = 0;
    uint64_t miss_bytes = 0;
};

struct DecodeStepResult {
    uint32_t token    = 0;
    uint32_t position = 0;
    float    top1 = 0.0f, top2 = 0.0f;
    float    margin() const { return top1 - top2; }
    double   wall_ms = 0.0;
    StepBreakdown breakdown{};
    TokenRecord record{};

    // Track P (runtime/sampling.h, docs/p3_chat.md §2). `token` is what the
    // step emits: the argmax unless the step sampled, the draw if it did.
    uint32_t greedy_token  = 0;       // head.slang's argmax, always
    bool     sampled       = false;
    bool     topk_fallback = false;   // the GPU top set could not prove the nucleus: full copy
    bool     topk_checked  = false;   // --check-topk compared the kernel with emulate_topk
    bool     topk_mismatch = false;
    uint32_t candidates    = 0;       // tokens the GPU top set returned
    uint32_t nucleus_size  = 0;
    double   p_token  = 0.0;          // full-softmax probability of the emitted token
    double   retained = 0.0;          // softmax mass of the candidate set
    double   kept     = 0.0;          // softmax mass of the nucleus
    double   sample_ms = 0.0;         // host: read the top set, build the nucleus, draw
};

// Track P: what a conversation needs that the L3 export used to supply.
struct SessionConfig {
    // The engram hash constants: a pure function of the tokenizer and
    // config.json. Empty (the default) derives them from the model directory at
    // startup (runtime/engram_tables.h, Track R2); a directory loads the tables
    // tools/oracle.py exported into it (EngramTables::load).
    std::string engram_tables_dir;
    // Positions the KV store is sized for; capped by kMaxIndexPositions.
    uint32_t    max_context = 4096;
    // design §9.6 P3 (Track R1, docs/p4_hitrate.md §5): fill the cache's free
    // slots in the background, hottest static experts first, while the drive
    // is not serving P0 misses. Also DEEPMOE_BACKFILL=1/0.
    bool        backfill = false;
};

// --- Track MS: one decode stream (docs/p4_multistream.md) -------------------
//
// Everything a SEQUENCE owns, as opposed to what the process owns. The split
// is exactly design §4's ownership graph cut in two: the expert cache, the
// planner, the pinned set, the I/O engine, the device and the resolved weight
// addresses are the process's and are shared by every stream; the KV store,
// the activations, the address tables the shaders read, the command buffer and
// the two timelines belong to the sequence being decoded.
//
// Why the address tables have to be per stream. `AttnRunner::slots(stage)` is
// host-visible memory a DISPATCH READS WHEN IT RUNS, not something baked into
// the command buffer at record time -- runtime/decode_layer.h's `bind_close`
// comment is the single-stream version of the same hazard. So a second stream
// that binds while the first stream's submitted-but-not-yet-executed
// dispatches are still outstanding would rewrite the table under them. One
// AttnRunner / DecodeRunner / MoeRunner per stream is the cheap fix: the
// pipelines are compiled SPIR-V modules and the tables are a few KB, while the
// 89 GiB expert cache -- the only thing that is actually big -- stays shared.
//
// A stream costs ~50 MB: the 32 MB GpuScratch, the KV store (~15 MB at a 4,096
// position context), the logit buffer and the runners' tables.
struct Stream {
    uint32_t             id = 0;
    bool                 made_ = false;   // its GPU resources exist
    gpu::AttnRunner      attn_;
    gpu::DecodeRunner    dec_;
    gpu::GpuScratch      scratch_;
    gpu::Timeline        timeline_;     // residency gate, host-signalled
    gpu::Timeline        fence_;        // completion, GPU-signalled
    TimelineValue        fence_value_ = 0;
    gpu::CommandPool     tok_pool_;
    gpu::CommandBuffer   tok_cmd_{};
    bool                 tok_open_ = false;
    bool                 tok_first_ = true;
    gpu::QueryPool       tsq_;
    uint32_t             tsq_used_ = 0;
    gpu::GpuBuffer       logits_{};
    gpu::GpuBuffer       sample_{};
    gpu::GpuBuffer       topk_out_{};
    gpu::GpuBuffer       topk_hist_{};
    gpu::GpuBuffer       ffn_in_buf_{};
    gpu::MemoryAllocator* ffn_in_alloc_ = nullptr;
    DecodeLayer          layer_;
    GpuMoeBridge         moe_;
    EngramRunner         engram_;
    KvStore              kvs_;
    std::vector<uint32_t> history_;
    // The stream's own token counter. It numbers the residency timeline
    // (`gpu::timeline_value(token_, L)`), which is per stream because the
    // timeline is; the cache's LRU clock is the process-wide `Engine::clock_`,
    // so eviction still ranks every stream's accesses on one axis.
    TokenIndex           token_ = 0;
    uint32_t             pub_index_k_ = 0;
    bool                 prefill_loaded_ = false;
    bool                 sample_step_ = false;
    // --- per-step scratch (reset at the top of a step) ---
    struct Stamp { uint32_t begin = ~0u, end = ~0u; };
    std::vector<LayerTiming> timings_;
    std::vector<Stamp>   ts_attn_, ts_moe_, ts_engram_, ts_moe_early_;
    Stamp                ts_tail_{};
    std::vector<double>  engram_host_ms_;
    std::vector<uint16_t> route_ids_;
    uint32_t             submits_ = 0;
    double               rec_ms_ = 0.0, sub_ms_ = 0.0, wait_ms_ = 0.0, bind_ms_ = 0.0;
    double               mx_ms_ = 0.0, mq_ms_ = 0.0, mt_ms_ = 0.0;
    uint64_t             spin_hits_ = 0, spin_misses_ = 0;
    TimelineValue        layer_guard_ = 0;
    bool                 layer_guard_pending_ = false;
    TimelineValue        open_guard_ = 0;
    TimelineValue        inflight_guard_ = 0;
    std::vector<ExpertKey> rr_pending_;
    // --- the M > 1 forward, stream 0 only (docs/p4_dspark_runtime.md) ---
    gpu::MgtRunner       mgt_;
    gpu::GpuScratch      bscratch_;
    gpu::GpuBuffer       blogits_{}, bsample_{}, btopk_out_{}, btopk_hist_{};
    // Track MS: what `run_layer` decided in its first two phases and its third
    // needs. In a single-stream step the three phases are consecutive
    // statements and this is dead bookkeeping; in an interleaved step the
    // other stream runs between them.
    struct LayerCtx {
        LayerStep st{};
        MoeCall   call{};
        store::LayerPlan plan{};
        LayerTiming* t = nullptr;
        uint32_t  ids_eff[16]{};
        float     wts_eff[16]{};
        const uint32_t* ids = nullptr;
        const float*    wts = nullptr;
        uint32_t  late[16]{};
        uint32_t  n_late = 0;
        bool      split = false;
        bool      staged = false;
        bool      engram = false;
        TimePoint g0{}, gp_w0{}, gp_w1{}, gp_p0{};
    };
    LayerCtx             lc_{};
    // Track G's probe rows are the Engine's (they aggregate); which row the
    // NEXT layer's prologue belongs to is the stream's.
    void*                gp_open_ = nullptr;
    TimePoint            gp_top_{};
};

class Engine {
public:
    // Stream 0 exists from construction (without GPU resources) so that every
    // accessor that reaches through `cur_` is safe before init_gpu().
    Engine();
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Brings up everything that does not need a GPU: config, manifest, files,
    // I/O engine, expert store (host-backed), planner, profiler. This is what
    // `deepmoe bench nvme` and the storage tests exercise.
    Result<void> init(const RuntimeConfig& cfg);

    // Adds the Vulkan device, both memory paths, the ~17.7 GB pinned set, the
    // routed-expert slab pool, every pipeline, the KV store and the engram.
    // Re-backs the ExpertStore onto GPU memory, so the host backing init()
    // installed is only ever used by a no-GPU caller.
    Result<void> init_gpu();

    void shutdown();

    // --- token loop (design §15 P2-P5) ------------------------------------

    // Loads the oracle's L3 export: the state every layer holds after the
    // prompt (window KV; on the kv sources the compressed-KV cache, the index
    // keys and the compressor state), the engram hash tables, and the
    // reference trajectory and per-step tensors to check against. Seeds the
    // prompt's state; nothing per step is seeded unless `set_produce_ced(false)`.
    Result<void> load_decode_state(const std::string& dir);
    const DecodeState* decode_state() const { return state_.get(); }
    // Track Q: puts the loaded export's prefill state back into the KV store in
    // place -- window rings, compressed and index-key caches, compressor carry
    // -- so a second trajectory can start at decode_pos() again. Re-running
    // positions on a used store is NOT that once the ring has wrapped (its
    // other 127 slots hold the first run's tokens) or a ratio-2 group was
    // left half full. The LRU clock `token_` is deliberately not rewound.
    Result<void> reseed_decode_state();

    // Encoder over the whole prompt, then decoder bounded replay over the last
    // 128 tokens (design §11.1, §11.2).
    // TODO(design §11, §9.7): implement in P5.
    Result<void> prefill(std::span<const uint32_t> prompt);

    // Prefill the slow way: the prompt through the DECODE path, one token at a
    // time, teacher-forced. Not design §11 -- it is 64 forward passes where §11
    // wants one chunked pass, so it is O(n) times too expensive and is not how
    // a prompt will ever be consumed. What it is, is a way to have the prompt's
    // state be OURS without the chunked kernels existing.
    //
    // It is exactly equivalent to the reference's chunked prefill at this
    // geometry, and the equivalence is worth stating because it is not
    // obvious:
    //   * the window ring at n <= window puts token p in slot p, and the decode
    //     top-k list marks every slot above p as -1, which is the causal window
    //     `get_window_topk_idxs` builds for query p at start_pos 0;
    //   * a ratio-2 group completes at odd p, pooling slots {0, 1} = tokens
    //     {p-1, p}, writing cache row p / 2 and rotating at p + 1 - ratio --
    //     the same group, row and position the chunked form gives group p / 2;
    //   * query p sees compress_len = (p + 1) / ratio cache rows, which is what
    //     the chunked form's per-query mask leaves it.
    // Returns the token the last prompt position predicts, i.e. the input to
    // decode step 0. `history()` and the KV store are left ready for it.
    Result<DecodeStepResult> slow_prefill(std::span<const uint32_t> prompt);

    // One decode step: forty layers, the head, the argmax. `state_step` selects
    // which of the L3 export's per-step compressed KV / top-k lists to seed
    // before running; -1 leaves whatever is in the KV store alone. It is
    // IGNORED when `produce_ced()` is on, because then there is nothing to
    // seed: design §7.4's kernels write both.
    Result<DecodeStepResult> decode_step(uint32_t in_token, uint32_t position,
                                         int32_t state_step);

    // Whether design §7.4's compressor and indexer run (the default whenever
    // the state they carry forward is available), or the compressed KV and the
    // top-k list are seeded per step from the export. The second is what
    // docs/p2_decode.md called LOADED and exists only so the two can be
    // compared; `status()` says which is in force.
    void set_produce_ced(bool on) { produce_ced_ = on; }
    bool produce_ced() const { return produce_ced_; }

    const KvStore& kv() const { return cur_->kvs_; }
    // Track R2 (runtime/session.h): rollback, parking and window replay write
    // the store between steps.
    KvStore& kv_store() { return cur_->kvs_; }
    const std::vector<uint32_t>& history() const { return cur_->history_; }

    // What layer `l` actually read at the last step: its own window ring, and
    // whichever layer's compressed plane and top-k list `shared_attn` pointed
    // it at. A validator comparing against the oracle's per-layer export needs
    // exactly this, because the oracle records what the layer SAW.
    Result<KvLayerView> effective_kv(uint32_t l) const;

    // The single-argument form of the old interface: the next step of the
    // sequence this Engine is already decoding.
    Result<SampleResult> decode_step();

    // Full loop. Without speculation this is `opts.max_tokens` decode steps
    // starting from the loaded state, greedily or teacher-forced.
    Result<GenerateResult> generate(std::span<const uint32_t> prompt,
                                    const GenerateOptions& opts);

    // --- M1: one forward over M <= 6 consecutive tokens -------------------
    //
    // docs/p4_dspark_runtime.md §6.4. `forward_batch(p0, tokens)` appends
    // `tokens` at positions p0 .. p0 + M - 1 to whatever the KV store already
    // holds and runs ONE forward over all M through the gpu/shaders/mgt1_*
    // kernels -- causal within the batch -- returning every position's logits
    // row. It is the verify half of a speculative cycle; at M = 1 it is a
    // second implementation of a decode step and the gate
    // (`spec_forward.batch_matches_m1`) is exactly that comparison.
    //
    // What it does NOT do that `decode_step` does: sampling, the profiler's
    // per-phase accounting, the MOE_OVERLAP split. Routing per position honours
    // `set_resident_only` with the block phase taken from the position INSIDE
    // the batch (mode `verify`: row 0 exact, rows 1.. resident-only), which is
    // the shape a real verify batch has.
    struct BatchRow {
        uint32_t argmax = 0;
        float    top1 = 0.0f, top2 = 0.0f;
    };
    // `logits`, when non-empty, must be [M][vocab] and receives every row.
    Result<void> forward_batch(uint32_t p0, std::span<const uint32_t> tokens,
                               std::span<BatchRow> rows, std::span<float> logits = {});
    // Builds the M > 1 runner, its scratch and its tail buffers. Called by
    // `forward_batch` on first use; separate so a caller can pay for it up front.
    Result<void> init_batch(uint32_t m_cap);
    bool batch_ready() const { return batch_ready_; }
    // The union of the last forward_batch's routed experts, summed over layers,
    // and the P0 bytes it missed.
    uint32_t last_batch_union() const { return batch_union_; }
    uint64_t last_batch_miss_bytes() const { return batch_miss_bytes_; }

    // The window ring around a verify batch (docs/p3_dspark.md §3.5). Take the
    // snapshot BEFORE `forward_batch` writes the M positions; `restore` undoes
    // positions p0 + accepted + 1 .. p0 + m - 1 and the host state that went
    // with them.
    Result<void> snapshot_batch_ring(uint32_t p0, uint32_t m);
    Result<void> restore_batch_ring(uint32_t p0, uint32_t accepted, uint32_t m);

    // --- Track P: conversations (docs/p3_chat.md) ------------------------------
    //
    // A session is the engine without an L3 export: an empty KV store sized for
    // `max_context`, the engram tables, and design 7.4's kernels producing
    // everything. `history()` is then exactly the tokens whose KV is written,
    // and `feed` appends to it -- which is what makes a second turn reuse the
    // first turn's state instead of re-reading the prompt.
    Result<void> begin_session(const SessionConfig& sc);
    // Back to position 0: KV store cleared, history empty. The expert cache,
    // the pinned set and the planner's clock are untouched (that warmth is the
    // point of a long-running process).
    void reset_context();
    // Track R2: declares that the KV store holds `tokens` -- a prefix of the
    // history after a rollback, or a parked context's ids after its non-SWA
    // state is unpacked. Runs nothing and writes nothing in the store; the
    // caller has made the store agree (runtime/session.h).
    Result<void> set_context_tokens(std::span<const uint32_t> tokens);
    uint32_t context_length() const { return static_cast<uint32_t>(cur_->history_.size()); }
    // The longest sequence this session can hold.
    uint32_t max_context() const;

    // `tokens` at positions context_length()...: one teacher-forced decode step
    // each (`slow_prefill`'s equivalence, without its 128-token limit -- a
    // decode step at p > 128 sees the ring's last 128 positions, which is the
    // band a chunked prefill's query p sees). Only the LAST step samples, per
    // `set_sampling`; the result is that step's, i.e. the next token.
    // `on_step(i, result)` runs after every step.
    Result<DecodeStepResult> feed(
        std::span<const uint32_t> tokens,
        const std::function<void(uint32_t, const DecodeStepResult&)>& on_step = {});

    // Replaces the whole KV state with a GPU prefill's (Track L,
    // gpu/vulkan/prefill_kernels.h, docs/p3_prefill.md §8.3 item 1): the window
    // rings, the kv sources' compressed cache, index keys and compressor state,
    // and the history.
    Result<void> seed_from_prefill(const gpu::PrefillHandoff& h);
    // Runs Track L's prefill over `prompt` from position 0 on this engine's
    // device, pinned set and I/O, seeds the state, and returns the first token
    // (sampled per `set_sampling` from the handoff's host logits). The prefill's
    // buffers are allocated for the call and freed after it, so path A needs
    // room beside the expert cache (serve --cache-gb).
    Result<DecodeStepResult> gpu_prefill(std::span<const uint32_t> prompt, uint32_t replay = 128);

    void set_sampling(const SamplingParams& p) { sampling_ = p; }
    const SamplingParams& sampling() const { return sampling_; }
    // Every sampled step also copies the logits and runs `emulate_topk` against
    // the kernel's answer (a validation mode; costs ~5 ms a token).
    void set_check_topk(bool on) { check_topk_ = on; }
    uint32_t topk_mismatches() const { return topk_mismatches_; }

    // --- R1 round 2: per-turn reheat -----------------------------------------
    //
    // docs/p4_hitrate.md §7. The expert cache is filled once, at begin_session,
    // from the static heat order -- a global average over a 27k-token trace --
    // and from then on only demand fills it. That is enough for a single topic
    // and wrong for a conversation: docs/p4_expert_patterns.md measures 86.6% of
    // a decode step's experts already appearing in the prompt's prefill, and a
    // topic switch dropping the hit rate from 0.94-0.96 to 0.88-0.91, because
    // the cache is full of the previous topic's experts and nothing puts the
    // new topic's in until each one misses.
    //
    // `reheat(decay)` is the turn boundary. It decays every slot's heat by
    // `decay` and re-runs the P3 backfill with the now-coldest-first order. The
    // backfill never evicts -- it only fills FREE slots, which the LRU has
    // meanwhile made out of the previous topic's leftovers -- so the pass is a
    // no-op when the cache has no free slot, and it cannot cost a hit. The read
    // is P3, i.e. it runs behind demand traffic on the IoEngine, so a turn's
    // first token does not wait for it.
    //
    // The heat itself needs no new bookkeeping: `ExpertSlot::heat` is already an
    // EWMA of the router score over the chosen top-6 and the near misses
    // (design §9.3), updated at every layer of every token, so decaying it makes
    // the current turn's routing the newest information in the order.
    //
    // `HeatOrder` reports what the pass was asked to do; `free_slots` is how
    // many slots it could fill, i.e. the ceiling on its effect.
    struct HeatOrder {
        uint32_t   slots = 0;            // resident, unpinned experts considered
        uint32_t   warm = 0;             // of those, at or above the reheat floor
        uint32_t   evicted = 0;          // the coldest tail, freed for the pass
        uint32_t   passed = 0;           // keys handed to the backfill
        uint32_t   free_slots = 0;       // free slots when the pass started
        uint32_t   turn = 0;             // reheat passes so far
        float      decay = 1.0f;
        double     ms = 0.0;             // building the order (one store scan + sort)
        std::string to_string() const;
    };
    // Safe to call whether or not reheat is enabled; `set_reheat` only decides
    // whether the serve loop calls it after every turn.
    Result<HeatOrder> reheat(float decay = 0.5f);
    void set_reheat(bool on) { reheat_on_ = on; }
    bool reheat_enabled() const { return reheat_on_; }

    // --- Track Y: resident-only routing (docs/p4_resident_routing.md) ------
    //
    // `Off` is the shipped behaviour: a layer's missing experts are fetched at
    // P0 and the step waits for them. `All` never waits -- the gate's experts
    // that are not resident are dropped, the rest are renormalised
    // (runtime/resident_route.h) and the dropped ones are handed to the
    // background fetcher at P3, which evicts the LRU to admit them so the cache
    // still tracks the conversation.
    //
    // Set by `DEEPMOE_ROUTE_RESIDENT_ONLY=off|all` at load, or by this setter
    // (the `--resident-only` CLI flag), which wins over the environment.
    // `Stall1` is the middle ground: a layer may block on at most ONE expert --
    // the highest-gate-weight missing one, a single P0 fetch of about 4 ms --
    // and skips the rest exactly as `All` does.
    // `Verify` is the DSpark block-5 shape (docs/p4_resident_routing.md §10):
    // one step in every five routes EXACTLY -- it is the block's first
    // position, whose misses are fetched at P0 and which warms the cache --
    // and the other four route resident-only exactly as `All` does. In a real
    // speculative cycle those four are the draft positions of the verify
    // batch, so the verify forward never waits on the drive; run at M = 1 it is
    // the same cache state and the same routing decision, one position at a
    // time, which is what makes it measurable on the teacher-forced harness
    // while `Engine::forward_batch` does not exist.
    enum class ResidentOnly : uint8_t { Off = 0, All = 1, Stall1 = 2, Verify = 3 };
    // DSpark's block: one verify forward over [last accepted, 4 drafts].
    // `Verify` mode routes step `token_ % kVerifyBlock == 0` exactly and the
    // other four resident-only.
    static constexpr uint64_t kVerifyBlock = 5;
    // Fills every free slot from the static heat table at P3 and blocks until
    // the cache is full (or `timeout` passes), so a measurement can start from
    // a warm cache instead of the cold one `load_state` leaves. `begin_session`
    // does the same thing for a conversation; this is the `run` path's version.
    Result<uint32_t> warm_cache_from_heat(std::chrono::seconds timeout = std::chrono::seconds(180));

    void set_resident_only(ResidentOnly m) { resident_only_ = m; }
    ResidentOnly resident_only() const { return resident_only_; }
    const ResidentRouteStats& resident_route_stats() const { return rr_; }
    void reset_resident_route_stats() { rr_ = {}; }
    // One line per counter, for the end of a run.
    std::string resident_route_report() const;

    // --- Track MS: several streams in one engine (docs/p4_multistream.md) ---
    //
    // The point is the shape of a decode step: ~97 ms of GPU compute and ~100 ms
    // of NVMe stall per token, and the two do not overlap -- the drive is idle
    // while the kernels run and the GPU is idle while the drive reads
    // (docs/p4_p0_queue.md). Two INDEPENDENT sequences have no data dependency
    // between them, so one's stall is the other's compute window.
    //
    // `set_streams(n)` creates n - 1 more Streams (each ~50 MB; the 89 GiB
    // expert cache stays shared, which is also why the hit rate has to be
    // reported: two working sets share one LRU). It may only grow, and only
    // before a step is in flight. Every existing entry point acts on the
    // SELECTED stream, so a single-stream caller is unchanged.
    Result<void> set_streams(uint32_t n);
    uint32_t     streams() const { return static_cast<uint32_t>(streams_.size()); }
    Result<void> select_stream(uint32_t i);
    uint32_t     current_stream() const { return cur_ ? cur_->id : 0; }
    // `begin_session` for one more stream: its own KV store and engram planes.
    Result<void> begin_session_on(uint32_t stream, const SessionConfig& sc);

    // One decode step on each of `steps`, layer-interleaved: layer L of every
    // stream, then layer L + 1, so that while stream A's layer-L MoE waits on
    // its P0 misses stream B's layer-L attention chain is on the GPU and
    // stream B's misses are also outstanding at the drive. `out` must be the
    // same length as `steps`. With one entry it is exactly `decode_step`.
    struct MultiStep {
        uint32_t stream = 0;
        uint32_t in_token = 0;
        uint32_t position = 0;
    };
    Result<void> decode_step_multi(std::span<const MultiStep> steps,
                                   std::span<DecodeStepResult> out);
    // `feed` for several streams at once: one token each, interleaved. Every
    // entry appends at that stream's own `context_length()`.
    Result<void> feed_multi(std::span<const MultiStep> steps,
                            std::span<DecodeStepResult> out);
    // Token-level ping-pong instead of the layer interleave (design D2, the
    // control): whole token of A, then whole token of B. No compute/stall
    // overlap -- only the drive's queue depth is shared.
    // `Pipeline` is the one that overlaps: see docs/p4_multistream.md §2. The
    // other two are the measured intermediates -- `Interleave` groups the
    // phases (every stream's attention, then every gate, then every MoE),
    // which issues both streams' misses early but leaves the GPU with nothing
    // queued while either of them waits; `PingPong` is design D2.
    enum class MsSched : uint8_t { Pipeline = 0, Interleave = 1, PingPong = 2 };
    void    set_ms_sched(MsSched s) { ms_sched_ = s; }
    MsSched ms_sched() const { return ms_sched_; }

    // --- accessors --------------------------------------------------------

    const RuntimeConfig&    config()   const { return cfg_; }
    const V41Config&        model()    const { return model_cfg_; }
    const Manifest&         manifest() const { return manifest_; }
    Profiler&               profiler()       { return profiler_; }
    store::ExpertStore&     store()          { return store_; }
    store::Planner&         planner()        { return planner_; }
    store::PinnedStore&     pinned()         { return pinned_; }
    storage::IoEngine&      io()             { return io_; }
    gpu::Device&            device()         { return device_; }
    TokenIndex              token_index() const { return cur_->token_; }
    bool                    gpu_ready()  const { return gpu_ready_; }

    // The per-layer breakdown of the last decode step (design §13.1).
    const std::vector<LayerTiming>& layer_timings() const { return cur_->timings_; }
    // Every fp32 logit of the last step, in GPU-visible memory. Reading it on
    // the host is a write-combining read and costs ~0.5 ms; the decode loop
    // never does, only the validators do.
    std::span<const float> last_logits() const;

    // Mean milliseconds of one submit + fence round trip, measured with a
    // dispatch that does essentially nothing (the argmax stage over the logit
    // buffer: 517 KB, about 2.4 us of memory time at the §5 ceiling). A decode
    // step makes ~128 of these, so this number times 128 is the floor the
    // current one-command-buffer-per-dispatch-group shape cannot go below --
    // which is what docs/p2_decode.md §5 compares the measured token against.
    Result<double> measure_submit_overhead(uint32_t iterations = 64);

    // One line per subsystem, for `deepmoe info` and the benchmark header.
    std::string status() const;

    // --- Track G: the gate host round trip, broken down (docs/plan_p5.md (g))
    //
    // The per-dispatch trace (Track W) says the GPU idles 0.40 ms/layer in
    // front of every MoE dispatch, because the command buffer is cut at the
    // router gate. That number is a GPU-side GAP; it says nothing about WHICH
    // host step inside the round trip owns it. These counters are the host
    // clock across the same interval, one accumulator per segment, split
    // between layers whose experts were all resident ("hit") and layers that
    // had to fetch ("miss") -- a miss layer's `plan_wait` is NVMe, not gate.
    //
    // Off unless DEEPMOE_GATE_PROBE=1. The probe is six Clock::now() calls a
    // layer, which is why it may be left compiled in.
    struct GateSeg {
        double   fence_us = 0.0;   // (i)   host blocked in cmd_wait
        double   ids_us   = 0.0;   // (ii)  verify_after_attention + reading top-k
        double   plan_us  = 0.0;   // (iii) Planner::plan_layer (lookup + miss issue)
        double   pwait_us = 0.0;   // (iii) Planner::wait_layer (NVMe on a miss layer)
        double   stage_us = 0.0;   // (iv)  GpuMoeBridge staging
        double   rec_us   = 0.0;   // (iv)  recording the MoE dispatches
        double   next_us  = 0.0;   // (v)   the NEXT layer's prologue, up to the submit
        double   sub_us   = 0.0;   //       of which: the vkQueueSubmit2 call itself
        uint64_t n        = 0;     // layer-steps accumulated
    };
    bool gate_probe_on() const { return gate_probe_; }
    const GateSeg& gate_probe_hit()  const { return gp_hit_; }
    const GateSeg& gate_probe_miss() const { return gp_miss_; }
    void  gate_probe_reset() { gp_hit_ = GateSeg{}; gp_miss_ = GateSeg{}; }
    std::string gate_probe_report() const;

    // Called after every layer's MoE, with the layer index and the DecodeLayer
    // holding that layer's `ffn_norm` output, gate ids and MoE output. Null on
    // a real run; the validators set it so a forty-layer disagreement can be
    // traced to the layer it starts at instead of being visible only as a
    // different token. Costs a host read of write-combining memory, so it is
    // not something to leave on.
    std::function<void(uint32_t, const DecodeLayer&)> layer_probe;
    // The same hook for the M > 1 forward, called once a layer after the batch
    // attention has landed: batch() then holds every row's ffn_norm output and
    // gate ids, which is what a first-divergent-layer bisect between the two
    // implementations needs.
    std::function<void(uint32_t, const DecodeLayer&)> batch_probe;

private:
    Result<void> open_model_files();
    // Track D2: probes each read source and hands the mirror table to the
    // IoEngine. A no-op when no mirror was given.
    Result<void> configure_io_sources();
    Result<void> load_pinned();
    Result<void> build_expert_cache();
    // H1a: one trivial submit against the freshly built slab pool. This is the
    // moment an over-sized expert cache is discovered -- allocation succeeded,
    // residency is only checked when work is queued -- so the probe costs one
    // empty command buffer and answers the question the arithmetic cannot.
    Result<void> probe_submit();
    // H1a: build_expert_cache + probe_submit, backing the slot count off by
    // `cache_backoff_step()` and rebuilding when the probe is refused. Only
    // called with `backoff = true` for an auto-sized budget; an explicit
    // --cache-slots / --cache-gb gets one attempt and a fatal message naming
    // the next count to try.
    Result<void> build_expert_cache_probed(bool backoff);
    Result<void> resolve_weights();
    Result<void> embed_token(uint32_t token);
    Result<void> run_layer(uint32_t layer, uint32_t position, bool& apply_post,
                           LayerTiming& t);
    // Track MS: `run_layer` in the three pieces the interleave needs. Every one
    // of them sets `cur_` to `s` first, so the body -- and the trace callback,
    // which reaches the engine through a void* -- acts on that stream.
    //   (1) begin  engram, bind, record the attention chain, submit it
    //   (2) gate   fence, read the gate's ids, route, plan -- ISSUES the P0
    //              misses and returns without waiting for them
    //   (3) moe    wait for residency, signal, stage and record the MoE
    // The window between (2) and (3) is the whole point: it is where another
    // stream's (1) and (2) run while this one's drive reads are in flight.
    Result<void> layer_begin(Stream& s, uint32_t L, uint32_t position, bool& apply_post,
                             LayerTiming& t);
    Result<void> layer_gate(Stream& s, uint32_t L, uint32_t position);
    Result<void> layer_moe(Stream& s, uint32_t L, bool& apply_post);
    // Everything decode_step does around the layer loop, so both schedulers
    // share it.
    Result<void> step_prologue(Stream& s, uint32_t in_token, uint32_t position,
                               int32_t state_step, double& prep_ms);
    Result<DecodeStepResult> step_epilogue(Stream& s, uint32_t position, double prep_ms,
                                           TimePoint t_start);
    // Multi-stream: submit a layer's MoE as soon as it is recorded instead of
    // letting the next layer's attention carry it. Costs one submit a layer
    // (~0.15 ms) and buys the other stream's stall window a GPU dispatch.
    static bool ms_eager_moe();
    // The M > 1 twin of `run_layer`. `st` carries everything the batch decided
    // once (p0, m, the list index, the ced sources); this fills in the layer.
    Result<void> run_layer_batch(uint32_t L, uint32_t p0, uint32_t m, bool& apply_post);
    // One position's resident-only decision, shared by run_layer and the batch:
    // `mode` is already resolved (no Verify), `ids_raw`/`wts_raw` are the gate's
    // 16, and the effective top-k lands in `eff_ids`/`eff_w`. `kept_*` is the
    // subset the planner should see. Returns false when the layer has nothing
    // resident at all and must fall through to the ordinary demand path.
    bool route_resident_only(uint32_t L, ResidentOnly mode, const uint32_t* ids_raw,
                             const float* wts_raw, uint32_t topk, uint32_t* eff_ids,
                             float* eff_w, uint16_t* kept_ids, float* kept_w,
                             uint32_t& n_kept, std::vector<ExpertKey>& dropped);
    Result<DecodeStepResult> collapse_and_sample(uint32_t position);

    // --- the token loop's command buffer (design §7.1) ----------------------
    // One buffer is open at a time. It is cut where the host HAS to look at
    // the GPU's output -- after the gate, for the ids -- so a layer's MoE and
    // the next layer's attention travel in one submit.
    Result<void> cmd_open();
    uint32_t     cmd_stamp();                           // ~0u when untimed
    Result<void> cmd_submit(TimelineValue wait_value);  // 0 = no gate
    Result<void> cmd_wait();
    // The wall-clock budget one GPU fence wait gets (DEEPMOE_GPU_WAIT_S, 900 s).
    static double gpu_wait_budget_s();
    Result<void> cmd_flush(TimelineValue wait_value) {
        if (auto r = cmd_submit(wait_value); !r) return r;
        return cmd_wait();
    }
    void         read_timestamps(DecodeStepResult& res);
    // The per-step half of design §7.4's bookkeeping: how many compressed
    // positions each layer may read, and the window half of its top-k list.
    Result<void> prepare_ced(uint32_t position);
    void         build_ced_plan();

    RuntimeConfig cfg_{};
    V41Config     model_cfg_{};
    Manifest      manifest_{};
    Profiler      profiler_;

    store::ShardSet     shards_;    // the 48 original safetensors shards (design §5.1 v0.5)
    storage::IoEngine   io_;
    store::ExpertStore  store_;
    store::Planner      planner_;
    store::PinnedStore  pinned_;

    gpu::Device          device_;
    gpu::MemoryAllocator alloc_a_, alloc_b_;
    // Track MS: the per-sequence half, one per stream. `cur_` is the stream
    // every member function acts on; single-stream callers never see it move.
    std::vector<std::unique_ptr<Stream>> streams_;
    Stream*              cur_ = nullptr;
    // The expert cache's LRU clock, process-wide: every stream's accesses are
    // ranked on one axis or eviction would prefer whichever stream ticked last.
    TokenIndex           clock_ = 0;
    Result<void>         create_stream(Stream& s);
    using Stamp = Stream::Stamp;
    // ADDITIVE (Track W): the per-dispatch trace. Idle unless
    // RuntimeConfig::trace_file is set; `trace_stamp` is the callback it
    // stamps through, which is just `cmd_stamp` behind a C pointer.
    trace::Tracer        tracer_;
    static uint32_t      trace_stamp(void* ctx) {
        return static_cast<Engine*>(ctx)->cmd_stamp();
    }
    // Track G's probe. `Stream::gp_open_` is the row the NEXT layer's prologue
    // belongs to: segment (v) is measured one layer later than the rest of the
    // round trip, because the buffer that carries layer L's MoE is not
    // submitted until layer L+1 has bound and recorded its attention chain.
    bool                 gate_probe_ = false;
    // Track G: DEEPMOE_FENCE_SPIN_US, and how often the spin caught the signal.
    static double        fence_spin_us();
    GateSeg              gp_hit_{}, gp_miss_{};
    SamplingParams       sampling_{};
    bool                 check_topk_ = false;
    uint32_t             topk_mismatches_ = 0;
    // Draws the emitted token for a sampled step from the kernel's top set (or
    // the full logits) into `res`.
    Result<void>         sample_into(DecodeStepResult& res, uint32_t position);
    // --- M1: the M > 1 forward (docs/p4_dspark_runtime.md §6.4) -----------
    // The runner, the scratch and the tail buffers are the stream's; the batch
    // is only ever run on stream 0.
    bool                 batch_ready_ = false;
    uint32_t             batch_cap_ = 0;
    std::vector<uint32_t> batch_list_;   // layer -> BatchScratch list index
    uint32_t             batch_union_ = 0;
    uint64_t             batch_miss_bytes_ = 0;
    KvStore::RingSnapshot batch_snap_{};
    uint32_t             batch_snap_p0_ = 0, batch_snap_m_ = 0;
    uint32_t             batch_snap_pub_ = 0;
    size_t               batch_snap_hist_ = 0;

    KvCache        kv_;

    std::unique_ptr<DecodeState> state_;
    std::vector<LayerWeights>    weights_;
    // What one token reads out of the pinned set at each layer -- the
    // "resident" half of design §2.3's byte budget, summed from the manifest
    // rather than assumed, so the profiler's hot_bytes and the §9.8 NVMe
    // utilisation figures are the real ratio.
    std::vector<uint64_t>        layer_hot_bytes_;
    DeviceAddress                norm_w_ = kNoDeviceAddress;
    DeviceAddress                head_w_ = kNoDeviceAddress;
    const store::PinnedTensor*   embed_  = nullptr;
    uint64_t                     cache_budget_ = 0;
    // H1a: has any queue submission of this process ever succeeded? The first
    // one is the one that discovers an over-sized expert cache -- the driver
    // accepts every allocation and then refuses the submit that has to make
    // them resident -- so a failure here is a capacity report, not a raw code.
    bool                         any_submit_ok_ = false;

    // design §7.4's routing of caches between layers, which is model.py's
    // `shared_attn` written down. Fixed at bring-up except `pub_index_k_`.
    struct CedPlan {
        uint32_t cmp_src = 0;        // whose compressed-KV plane this layer reads
        uint32_t idx_src = 0;        // whose top-k list this layer reads
        uint32_t ratio   = 0;
        bool     is_kv_source    = false;
        bool     is_index_source = false;
    };
    std::vector<CedPlan> ced_;
    // Which layer's index-key cache was published last. NOT a static mapping:
    // a ratio-2 source whose group is incomplete publishes nothing, so at an
    // even position layers 2, 8 and 14 score against layer 20's keys and at an
    // odd one against their own (docs/p2_attention.md §9.3 item 5).
    bool       produce_ced_ = false;
    SessionConfig session_cfg_{};

    bool       ready_ = false;
    bool       gpu_ready_ = false;

    // docs/p4_hitrate.md §1: `DEEPMOE_ROUTE_DUMP=FILE` appends one fixed-size
    // record per decode step (prefill-by-decode steps included) -- the step's
    // LRU clock, its position, the forty layers' top-6 in gate order and each
    // layer's measured hit count -- so tools/hitrate_sim.py can replay exactly
    // the accesses this process made through cache_sim's LRU and compare it
    // step for step. Off (null) unless the variable is set.
    std::FILE*            route_dump_ = nullptr;
    MsSched               ms_sched_ = MsSched::Pipeline;
    void write_route_record(uint32_t position);

    // Whether the serve loop reheats the cache after every turn (§7).
    bool                  reheat_on_ = false;
    uint32_t              reheat_turn_ = 0;
    // The non-resident ranking the whole process uses: DEEPMOE_HEAT_FILE if one
    // was given, else store/static_heat.inc. Read once, by both the startup P3
    // backfill and every reheat pass (docs/p4_hitrate.md).
    mutable std::vector<ExpertKey> heat_order_;

    // Track R1 (docs/p4_hitrate.md §4). `overlap_`: dispatch A over the
    // resident experts goes out before a layer's NVMe wait. The eviction guard:
    // `guard_clock_` numbers the buffers that read expert slots; a slot read by
    // one is guarded with its number and the store's completed timeline is
    // advanced when that buffer's fence returns, so no fill -- a later layer's,
    // the P3 backfill's, the prefill handoff's -- can recycle a slot a
    // submitted buffer still reads.
    ResidentOnly       resident_only_ = ResidentOnly::Off;
    // What `Verify` does at the block's first position and at its four draft
    // positions (DEEPMOE_VERIFY_FIRST / DEEPMOE_VERIFY_DRAFT).
    ResidentOnly       verify_first_ = ResidentOnly::Off;
    ResidentOnly       verify_draft_ = ResidentOnly::All;
    ResidentRouteStats rr_{};
    // Ceiling on P3 reads this mode may have in flight at once, so a cold cache
    // cannot queue the whole model. docs/p4_resident_routing.md §3: the drive
    // can land about 19 experts per 80 ms step, so a couple of steps' worth.
    uint32_t      rr_inflight_cap_ = 48;
    void flush_resident_backfill(uint32_t layer);

    // --- Track Y step 3 (docs/p4_resident_routing.md section 8) -----------
    //
    // (B) The background miss queue is bounded to the most recent
    // `rr_queue_steps_` decode steps and drained newest-first, and no more
    // than `rr_outstanding_cap_` experts of it are allowed to be out at the
    // drive at once. Step 2 handed every miss straight to the IoEngine, whose
    // P3 queue is unbounded: the mean P3 latency was 5,594 ms against a 100 ms
    // step, so the drive was saturated with demand 56 steps out of date.
    // `DEEPMOE_RESIDENT_QUEUE_STEPS` sets the window (default 2).
    struct RrMiss { ExpertKey key; uint64_t step; };
    std::deque<RrMiss>    rr_queue_;             // oldest at the front
    uint32_t              rr_queue_steps_    = 2;
    uint32_t              rr_outstanding_cap_ = 24;   // ~ one step's drive budget
    std::atomic<uint32_t> rr_outstanding_{0};
    std::atomic<uint64_t> rr_fetch_ns_{0};       // P3 submit -> settle, summed
    std::atomic<uint64_t> rr_fetch_done_{0};
    //
    // (A) The LRU stamp of every REQUESTED top-k expert, resident or not. A
    // resident one is stamped by the planner's own lookup; a non-resident one
    // has nothing to stamp, so its request stamp is parked here and handed to
    // the P3 fetch as `stamp_in`, which is what `ExpertStore::settle_locked`
    // writes into `last_use_token` when it lands. Without it the arrival was
    // stamped at SUBMIT time, so a 5.6 s-late expert looked like the newest
    // thing in the cache; with it, eviction ranks against when it was wanted.
    std::unordered_map<uint64_t, uint64_t> rr_demand_;   // (layer<<16|expert) -> stamp
    static uint64_t rr_pack(ExpertKey k) {
        return (uint64_t(k.layer) << 16) | uint64_t(k.expert);
    }

    bool          overlap_ = true;
    bool          handoff_ = true;       // gpu_prefill's experts go to the decode cache (§3)
    TimelineValue guard_clock_ = 0;
    void guard_layer(uint32_t layer, std::span<const uint32_t> slots, const uint32_t* ids);
    // Track MS: the completed-guard value it is safe to publish to the store
    // when several streams have buffers in flight.
    void advance_store_guard(const Stream& me, TimelineValue mine);
};

}  // namespace deepmoe::runtime
