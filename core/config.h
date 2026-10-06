// Runtime options. Everything the design leaves as "decided by measurement"
// (chunk size, queue depth, lookahead d/K, cache policy, path A vs B) is a
// field here with the design's starting guess as the default, so that P-1/P1
// results land in one place.
//
// Ownership/threading: a plain value struct. Engine copies it at init and
// resolves environment overrides before decode starts. Hot paths read the
// resolved fields, rather than inspecting the process environment.
#pragma once

#include "core/namespace.h"

#include "core/runtime_environment.h"
#include "core/runtime_facts.h"
#include <memory>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace cachedmoe {

// design §3.3: the two unified-memory paths. P-1 picks one by measured
// bandwidth x capacity; the ExpertStore exposes the same (host_ptr, dev_addr)
// pair either way.
enum class MemoryPath : uint8_t {
    Auto = 0,
    DeviceLocalHostVisible,  // path A: allocate from the DEVICE_LOCAL|HOST_VISIBLE type, CPU maps and writes
    ExternalMemoryHost,      // path B: allocate host memory, import with VK_EXT_external_memory_host
    HostOnly,                // no Vulkan at all: the host-memory backend used by tests and the CPU oracle
};

// design §9.3: cache replacement policy. Only Lru is implemented until
// tools/cache_sim.py has an answer (design §16: no policy code before P1).
enum class CachePolicy : uint8_t { Lru = 0, ScoreAwareLru, Lfu, StaticPinPlusLru };

// design §11.2 / §12: bounded replay vs the exact reference path.
enum class PrefillMode : uint8_t { BoundedReplay = 0, Oracle };

struct IoConfig {
    // design §9.2.1, measured on the dev machine's SN740: 4 MiB x QD 8 reaches
    // the drive's 4.5 GB/s plateau at 6.9 ms mean latency, and QD 64 gives the
    // same bandwidth at 37.7 ms. More bytes in flight buys only latency, which
    // is why this is 32 MB and not the 64 MB the design first guessed.
    uint32_t chunk_bytes        = 4u << 20;
    uint32_t max_inflight_bytes = 32u << 20;   // 8 x 4 MiB
    uint32_t max_inflight_ops   = 8;
    uint32_t completion_threads = 2;
    bool     unbuffered         = true;   // FILE_FLAG_NO_BUFFERING / O_DIRECT
    // When a P0 arrives, stop issuing new chunks from P1..P3 until it drains.
    // Reserved API field: the dispatcher currently always preempts P1-P3.
    bool     preempt_on_blocking = true;
    // P0 (blocking miss) only; 0 = chunk_bytes / max_inflight_ops. The
    // CACHEDMOE_IO_P0_CHUNK_MB / _QD environment knobs override these.
    uint32_t p0_chunk_bytes     = 0;
    uint32_t p0_qd              = 0;
    // P2 (engram rows, 4 KiB random reads) only; 0 = the background classes'
    // depth. CACHEDMOE_IO_ENGRAM_QD overrides it.
    uint32_t engram_qd          = 0;
    // Shared by pre-backend widening and post-capability derivation.
    std::shared_ptr<const configuration::RuntimeEnvironment> environment;
};

struct CacheConfig {
    // design §5.3: slab = 1.88 GB = 100 expert slots, bounded by the 2 GiB
    // maxMemoryAllocationSize.
    uint32_t slots_per_slab   = 100;
    uint64_t budget_bytes     = 2ull << 30;  // total slab-pool budget; the real run uses ~90 GB
    CachePolicy policy        = CachePolicy::Lru;
    std::shared_ptr<const configuration::RuntimeEnvironment> environment;
};

struct PrefetchConfig {
    // Reserved predictor API: depth/width are logged, but predictive prefetch
    // is not implemented. This does not control the live P3 backfill API.
    // design §9.4: lookahead depth d and width K. d >= 3-4 from the T_layer /
    // T_io estimate; K is adapted online by measured precision.
    uint32_t lookahead_depth  = 4;
    uint32_t lookahead_width  = 12;
    uint32_t min_width        = 6;
    uint32_t max_width        = 16;
    float    precision_floor  = 0.35f;   // shrink K below this
    bool     enabled          = true;
};

struct SpeculationConfig {
    bool     enabled   = false;   // real three-stage MTP draft
    uint32_t max_draft = configuration::facts::DRAFT_BLOCK_SIZE; // dspark_block_size
    uint32_t accept_topk =
        configuration::facts::ACCEPT_TOP_K; // accept the unchanged main-path token in target top-K
    // Experimental raw-score prefix policy; unset keeps fixed max_draft.
    // The draft has already run when this selects k, including k=0.
    std::optional<float> min_confidence;
};

// Resolved once by Engine::init. Environment overrides remain compatible with
// existing experiment commands; changing the process environment later cannot
// change a live engine's pipelines or verification policy.
struct GpuExecutionConfig {
    bool batch_gpu_route = false;
    bool batch_engram_early = true;
    bool spec_gpu_readout = true;
    bool draft_onecb = false;
    bool draft_mega = false;
    bool draft_profile = false;
    bool draft_diagnostics = false;
    bool draft_trim_tail = true;
    bool mgt_pair_dot = false;
    bool mgt_fold_scale = false;
    bool mgt_attn_cm = false;

    void apply_environment(const configuration::RuntimeEnvironment &environment) {
        const auto &overrides = environment.gpu;
        if (overrides.batch_gpu_route)
            batch_gpu_route = *overrides.batch_gpu_route;
        if (overrides.batch_engram_early)
            batch_engram_early = *overrides.batch_engram_early;
        if (overrides.spec_gpu_readout)
            spec_gpu_readout = *overrides.spec_gpu_readout;
        if (overrides.draft_onecb)
            draft_onecb = *overrides.draft_onecb;
        if (overrides.draft_mega)
            draft_mega = *overrides.draft_mega;
        if (overrides.draft_profile)
            draft_profile = *overrides.draft_profile;
        if (overrides.draft_diagnostics)
            draft_diagnostics = *overrides.draft_diagnostics;
        if (overrides.draft_trim_tail)
            draft_trim_tail = *overrides.draft_trim_tail;
        if (overrides.mgt_pair_dot)
            mgt_pair_dot = *overrides.mgt_pair_dot;
        if (overrides.mgt_fold_scale)
            mgt_fold_scale = *overrides.mgt_fold_scale;
        if (overrides.mgt_attn_cm)
            mgt_attn_cm = *overrides.mgt_attn_cm;
    }
    void apply_environment() { apply_environment(*configuration::RuntimeEnvironment::capture()); }
};

// Decode scheduling and diagnostics are fixed at engine startup. Platform
// defaults remain optional so RADV-specific choices can be resolved after the
// device is known, without polling process environment during a layer.
struct DecodeExecutionConfig {
    double gpu_wait_budget_seconds = configuration::kGpuWaitSeconds;
    double fence_spin_microseconds = configuration::kFenceSpinMicroseconds;
    std::optional<bool> shared_early;
    std::optional<bool> eager_moe;
    std::optional<bool> engram_deadline;
    bool shared_early_multistream = false;
    bool shared_early_check = false;
    bool dynamic_mask_lru = true;

    void apply_environment(const configuration::RuntimeEnvironment &environment) {
        const auto &overrides = environment.decode;
        if (overrides.gpu_wait_seconds)
            gpu_wait_budget_seconds = *overrides.gpu_wait_seconds;
        if (overrides.fence_spin_microseconds)
            fence_spin_microseconds = *overrides.fence_spin_microseconds;
        if (overrides.shared_early_present)
            shared_early = overrides.shared_early;
        if (overrides.eager_moe_present)
            eager_moe = overrides.eager_moe;
        if (overrides.shared_early_multistream)
            shared_early_multistream = *overrides.shared_early_multistream;
        if (overrides.shared_early_check)
            shared_early_check = *overrides.shared_early_check;
        if (overrides.dynamic_mask_lru)
            dynamic_mask_lru = *overrides.dynamic_mask_lru;
        if (overrides.engram_deadline)
            engram_deadline = *overrides.engram_deadline;
    }
    void apply_environment() { apply_environment(*configuration::RuntimeEnvironment::capture()); }
};

struct RuntimeConfig {
    // CLI captures before validation; library callers may omit this and init
    // captures once. Later mutations cannot affect this engine's resources.
    std::shared_ptr<const configuration::RuntimeEnvironment> environment;

    // The checkpoint directory: the 48 original safetensors shards, config.json
    // and the deepmoe_manifest.json that tools/manifest.py writes beside them
    // (design §5.1 v0.5 -- there is no repack).
    std::string model_dir;
    // Track D2 (docs/p4_dual_source.md): further directories holding
    // byte-identical copies of the same shards, on other physical drives.
    // Empty (the default) is the single-drive run, unchanged down to the byte.
    // Filled from --mirror DIR or CACHEDMOE_MODEL_MIRRORS (';'-separated).
    std::vector<std::string> model_mirrors;
    // Hold every engram layer's scale plane (8 B a row, 3.07 GB a layer) in
    // host memory, so a row is one 4 KiB read instead of two (STATUS §7 0as).
    // Off by default: the cache budget shrinks by what it takes.
    bool        engram_scales_resident = false;
    std::string profile_jsonl;    // empty = no JSONL sink
    std::string trace_file;       // ADDITIVE (Track W): per-dispatch GPU trace,
                                  // empty = off. runtime/trace.h has the format.

    // Reserved API fields; Engine currently selects dual-path GPU backing
    // and its implemented bounded prefill independently of these placeholders.
    MemoryPath  memory_path = MemoryPath::Auto;
    PrefillMode prefill     = PrefillMode::BoundedReplay;
    // The GPU prefill's expert transit: segments of 64 slots (1.2 GB of pinned
    // host pages each). More than 2 lets a long prompt's read-ahead take the
    // whole layer while the drives would idle under its attention (§7 0av).
    uint32_t    prefill_transit_segments = 2;

    IoConfig          io;
    CacheConfig       cache;
    PrefetchConfig    prefetch;
    SpeculationConfig speculation;
    GpuExecutionConfig gpu;
    DecodeExecutionConfig decode;

    uint32_t max_context = configuration::facts::RUNTIME_CONTEXT; // design §1.2 stage-one target
    // Reserved legacy sampler fields. GenerateRequest/GenerateOptions supply
    // the sampling values that the implemented token loop actually consumes.
    uint64_t seed        = 0;       // Philox counter seed, decode is reproducible
    float    temperature = 0.0f;    // 0 = greedy; the speculation invariant of §10.2

    // Pin the I/O and planner threads to a few physical cores so they do not
    // fight the GPU for LPDDR bandwidth (design §8).
    // Reserved; startup CPU_AFFINITY is the implemented affinity control.
    uint32_t io_core_mask = 0;      // 0 = let the OS decide
};

}  // namespace cachedmoe
