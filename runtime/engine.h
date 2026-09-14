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
//   engine/submit thread  records and submits the per-layer command buffers,
//                         reads the gate's ids out of host-coherent memory,
//                         host-signals the timeline, samples. Vulkan queue
//                         submission happens only here.
//   io dispatcher thread  owned by IoEngine: priority, chunking, completions.
//   iocp completion x2    owned by the backend: GetQueuedCompletionStatus.
//   planner thread        background prefetch and backfill between tokens
//                         (design §9.4, §9.6 P3) -- not started; §9.4 is a net
//                         loss at every (d, K) the simulator tried.
//
// What a decode step really does, and what it is handed
// ----------------------------------------------------
// Real: the embedding lookup, forty layers of design §7.14's dispatches 1-11
// with the §7.1 residency gate between 9 and 10, the engram at layers 1 and 14
// including its NVMe row fetch, the final collapse, the 1.32 GB head and the
// argmax. Handed to it: the window KV the prompt left and, per step, the
// compressed KV and the indexer's top-k list -- design §7.4's kernels are
// another track's and were not there when this was written. `status()` says so
// in one line, and docs/p2_decode.md says it at length.
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
struct LayerTiming {
    double   engram_ms = 0.0;
    double   attn_ms   = 0.0;   // dispatches 1-9
    double   gate_ms   = 0.0;   // the §7.1 residency gate: plan + wait for NVMe
    double   moe_ms    = 0.0;   // dispatches 10-11, routed + shared
    double   moe_gpu_ms = 0.0;  // of which the two kernels themselves
    double   moe_host_ms = 0.0; // of which the activation round trip and the sums
    uint32_t hits = 0, misses = 0;
    uint64_t miss_bytes = 0;
};

struct DecodeStepResult {
    uint32_t token    = 0;
    uint32_t position = 0;
    float    top1 = 0.0f, top2 = 0.0f;
    float    margin() const { return top1 - top2; }
    double   wall_ms = 0.0;
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

    // Loads the oracle's L3 export: the window KV every layer holds after the
    // prompt, the per-step compressed KV and top-k lists, the engram hash
    // tables, and the reference trajectory to check against. This stands in for
    // prefill, which is P5.
    Result<void> load_decode_state(const std::string& dir);
    const DecodeState* decode_state() const { return state_.get(); }

    // Encoder over the whole prompt, then decoder bounded replay over the last
    // 128 tokens (design §11.1, §11.2).
    // TODO(design §11, §9.7): implement in P5.
    Result<void> prefill(std::span<const uint32_t> prompt);

    // One decode step: forty layers, the head, the argmax. `state_step` selects
    // which of the L3 export's per-step compressed KV / top-k lists to seed
    // before running; -1 leaves whatever is in the KV store alone, which is
    // what a run that has its own compressor will pass.
    Result<DecodeStepResult> decode_step(uint32_t in_token, uint32_t position,
                                         int32_t state_step);

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
    gpu::AttnRunner      attn_;
    gpu::DecodeRunner    dec_;
    gpu::GpuScratch      scratch_;
    gpu::GpuBuffer       logits_{};   // [vocab] fp32
    gpu::GpuBuffer       sample_{};   // SampleOut, host-coherent

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

    TokenIndex token_ = 0;
    bool       ready_ = false;
    bool       gpu_ready_ = false;
};

}  // namespace deepmoe::runtime
