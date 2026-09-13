// Engine: the object that owns everything and runs the token loop
// (design §4, §15).
//
// Ownership graph (see docs/architecture.md):
//
//   Engine
//    +- V41Config, Manifest                    parsed once, immutable
//    +- storage::File x{hot, experts, mtp, engram.L1, engram.L14}
//    +- storage::IoEngine  -> Backend          owns 1 dispatcher thread,
//    |                                         the backend owns 2 IOCP threads
//    +- store::ExpertStore -> SlabPool -> SlabBacking
//    +- store::Planner     -> EvictionPolicy, Predictor
//    +- gpu::Device -> MemoryAllocator, CommandPool, Timeline, Pipelines
//    +- runtime::KvCache, Block[43], Dspark, Sampler
//    +- Profiler                               JSONL sink
//
// Threading model:
//   engine/submit thread  records and submits the per-token command buffer,
//                         spins on the gate readback counter, host-signals the
//                         timeline. Vulkan queue submission happens only here.
//   io dispatcher thread  owned by IoEngine: priority, chunking, completions.
//   iocp completion x2    owned by the backend: GetQueuedCompletionStatus.
//   planner thread        background prefetch and backfill between tokens
//                         (design §9.4, §9.6 P3) -- not started until the
//                         predictor exists.
//
// Everything the Engine exposes is called from one thread; the objects it owns
// are individually thread-safe where the design needs them to be.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/profiler.h"
#include "core/status.h"
#include "core/types.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/timeline.h"
#include "model/manifest.h"
#include "model/v41_config.h"
#include "runtime/kvcache.h"
#include "runtime/sampler.h"
#include "storage/file.h"
#include "storage/io_engine.h"
#include "store/expert_store.h"
#include "store/planner.h"

namespace deepmoe::runtime {

struct GenerateOptions {
    uint32_t max_tokens = 64;
    bool     greedy     = true;     // design §10.2 invariant
    bool     speculative = false;   // P4
};

struct GenerateResult {
    std::vector<uint32_t> tokens;
    RunSummary            summary;
};

class Engine {
public:
    Engine() = default;
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Brings up everything that does not need a GPU: config, manifest, files,
    // I/O engine, expert store, planner, profiler. Implemented -- this is what
    // `deepmoe bench nvme` and the storage tests exercise.
    Result<void> init(const RuntimeConfig& cfg);

    // Adds the Vulkan device, memory, pipelines and KV cache.
    // TODO(design §7, §15 P2): implement once the kernels exist.
    Result<void> init_gpu();

    void shutdown();

    // --- token loop (design §15 P2-P5) ------------------------------------

    // Encoder over the whole prompt, then decoder bounded replay over the last
    // 128 tokens (design §11.1, §11.2). Switches to the expert-major streaming
    // mode of §9.7 when the prompt is long enough that the activated expert set
    // exceeds the cache.
    // TODO(design §11, §9.7): implement in P5.
    Result<void> prefill(std::span<const uint32_t> prompt);

    // One decode step: record, submit, resolve routing per layer, sample.
    // TODO(design §7.14, §7.8): implement in P2/P3.
    Result<SampleResult> decode_step();

    // Full loop with optional DSpark speculation.
    // TODO(design §10): implement in P4.
    Result<GenerateResult> generate(std::span<const uint32_t> prompt,
                                    const GenerateOptions& opts);

    // --- accessors --------------------------------------------------------

    const RuntimeConfig&    config()   const { return cfg_; }
    const V41Config&        model()    const { return model_cfg_; }
    const Manifest&         manifest() const { return manifest_; }
    Profiler&               profiler()       { return profiler_; }
    store::ExpertStore&     store()          { return store_; }
    store::Planner&         planner()        { return planner_; }
    storage::IoEngine&      io()             { return io_; }
    gpu::Device&            device()         { return device_; }
    TokenIndex              token_index() const { return token_; }

    // One line per subsystem, for `deepmoe info` and the benchmark header.
    std::string status() const;

private:
    Result<void> open_model_files();

    RuntimeConfig cfg_{};
    V41Config     model_cfg_{};
    Manifest      manifest_{};
    Profiler      profiler_;

    storage::File hot_, experts_, mtp_, engram_l1_, engram_l14_;
    storage::IoEngine   io_;
    store::ExpertStore  store_;
    store::Planner      planner_;

    gpu::Device   device_;
    gpu::Timeline timeline_;
    KvCache       kv_;

    TokenIndex token_ = 0;
    bool       ready_ = false;
    bool       gpu_ready_ = false;
};

}  // namespace deepmoe::runtime
