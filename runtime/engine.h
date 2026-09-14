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

#include <cstdint>
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
#include "runtime/sampler.h"
#include "storage/file.h"
#include "storage/io_engine.h"
#include "store/expert_store.h"
#include "store/pinned.h"
#include "store/planner.h"
#include "store/shard_set.h"

namespace deepmoe::runtime {

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
};

struct DecodeStepResult {
    uint32_t token    = 0;
    uint32_t position = 0;
    float    top1 = 0.0f, top2 = 0.0f;
    float    margin() const { return top1 - top2; }
    double   wall_ms = 0.0;
    StepBreakdown breakdown{};
    TokenRecord record{};
};

class Engine {
public:
    Engine() = default;
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

    const KvStore& kv() const { return kvs_; }
    const std::vector<uint32_t>& history() const { return history_; }

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
    TokenIndex              token_index() const { return token_; }
    bool                    gpu_ready()  const { return gpu_ready_; }

    // The per-layer breakdown of the last decode step (design §13.1).
    const std::vector<LayerTiming>& layer_timings() const { return timings_; }
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

    // Called after every layer's MoE, with the layer index and the DecodeLayer
    // holding that layer's `ffn_norm` output, gate ids and MoE output. Null on
    // a real run; the validators set it so a forty-layer disagreement can be
    // traced to the layer it starts at instead of being visible only as a
    // different token. Costs a host read of write-combining memory, so it is
    // not something to leave on.
    std::function<void(uint32_t, const DecodeLayer&)> layer_probe;

private:
    Result<void> open_model_files();
    Result<void> load_pinned();
    Result<void> build_expert_cache();
    Result<void> resolve_weights();
    Result<void> embed_token(uint32_t token);
    Result<void> run_layer(uint32_t layer, uint32_t position, bool& apply_post,
                           LayerTiming& t);
    Result<DecodeStepResult> collapse_and_sample(uint32_t position);

    // --- the token loop's command buffer (design §7.1) ----------------------
    // One buffer is open at a time. It is cut where the host HAS to look at
    // the GPU's output -- after the gate, for the ids -- so a layer's MoE and
    // the next layer's attention travel in one submit.
    Result<void> cmd_open();
    uint32_t     cmd_stamp();                           // ~0u when untimed
    Result<void> cmd_submit(TimelineValue wait_value);  // 0 = no gate
    Result<void> cmd_wait();
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
    gpu::Timeline        timeline_;
    // The GPU signals this one when a token-loop submit completes; the host
    // signals `timeline_` when a layer's experts are resident.
    gpu::Timeline        fence_;
    TimelineValue        fence_value_ = 0;
    gpu::CommandPool     tok_pool_;
    gpu::CommandBuffer   tok_cmd_{};
    bool                 tok_open_ = false;
    bool                 tok_first_ = true;       // the next open is a token's first
    gpu::QueryPool       tsq_;
    uint32_t             tsq_used_ = 0;
    uint32_t             submits_ = 0;
    double               rec_ms_ = 0.0, sub_ms_ = 0.0, wait_ms_ = 0.0, bind_ms_ = 0.0;
    double               mx_ms_ = 0.0, mq_ms_ = 0.0, mt_ms_ = 0.0;
    struct Stamp { uint32_t begin = ~0u, end = ~0u; };
    std::vector<Stamp>   ts_attn_, ts_moe_, ts_engram_;
    Stamp                ts_tail_{};
    std::vector<double>  engram_host_ms_;
    gpu::AttnRunner      attn_;
    gpu::DecodeRunner    dec_;
    gpu::GpuScratch      scratch_;
    gpu::GpuBuffer       logits_{};   // [vocab] fp32
    gpu::GpuBuffer       sample_{};   // SampleOut, host-coherent
    gpu::GpuBuffer       ffn_in_buf_{};   // the FFN input, in cached host pages (path B)
    gpu::MemoryAllocator* ffn_in_alloc_ = nullptr;

    KvCache        kv_;
    KvStore        kvs_;
    DecodeLayer    layer_;
    GpuMoeBridge   moe_;
    EngramRunner   engram_;

    std::unique_ptr<DecodeState> state_;
    std::vector<LayerWeights>    weights_;
    // What one token reads out of the pinned set at each layer -- the
    // "resident" half of design §2.3's byte budget, summed from the manifest
    // rather than assumed, so the profiler's hot_bytes and the §9.8 NVMe
    // utilisation figures are the real ratio.
    std::vector<uint64_t>        layer_hot_bytes_;
    std::vector<LayerTiming>     timings_;
    std::vector<uint32_t>        history_;
    DeviceAddress                norm_w_ = kNoDeviceAddress;
    DeviceAddress                head_w_ = kNoDeviceAddress;
    const store::PinnedTensor*   embed_  = nullptr;
    uint64_t                     cache_budget_ = 0;

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
    uint32_t   pub_index_k_ = 0;
    bool       produce_ced_ = false;
    // Whether the window ring came from the export or from `slow_prefill`.
    bool       prefill_loaded_ = false;

    TokenIndex token_ = 0;
    bool       ready_ = false;
    bool       gpu_ready_ = false;
};

}  // namespace deepmoe::runtime
