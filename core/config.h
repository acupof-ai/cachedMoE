// Runtime options. Everything the design leaves as "decided by measurement"
// (chunk size, queue depth, lookahead d/K, cache policy, path A vs B) is a
// field here with the design's starting guess as the default, so that P-1/P1
// results land in one place.
//
// Ownership/threading: a plain value struct. The Engine takes a copy at
// construction and treats it as immutable afterwards; nothing reads it on a hot
// path.
#pragma once

#include <cstdint>
#include <string>

#include "core/types.h"

namespace deepmoe {

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
    bool     preempt_on_blocking = true;
};

struct CacheConfig {
    // design §5.3: slab = 1.88 GB = 100 expert slots, bounded by the 2 GiB
    // maxMemoryAllocationSize.
    uint32_t slots_per_slab   = 100;
    uint64_t budget_bytes     = 2ull << 30;  // total slab-pool budget; the real run uses ~90 GB
    CachePolicy policy        = CachePolicy::Lru;
};

struct PrefetchConfig {
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
    bool     enabled   = false;   // design §15: DSpark lands in P4
    uint32_t max_draft = 5;       // dspark_block_size
    bool     greedy    = true;    // sampling-mode verification is P4b
};

struct RuntimeConfig {
    // The checkpoint directory: the 48 original safetensors shards, config.json
    // and the deepmoe_manifest.json that tools/manifest.py writes beside them
    // (design §5.1 v0.5 -- there is no repack).
    std::string model_dir;
    std::string profile_jsonl;    // empty = no JSONL sink
    std::string kvcache_dir;      // design §11.4 prefix KV persistence

    MemoryPath  memory_path = MemoryPath::Auto;
    PrefillMode prefill     = PrefillMode::BoundedReplay;

    IoConfig          io;
    CacheConfig       cache;
    PrefetchConfig    prefetch;
    SpeculationConfig speculation;

    uint32_t max_context = 65536;   // design §1.2 stage-one target
    uint64_t seed        = 0;       // Philox counter seed, decode is reproducible
    float    temperature = 0.0f;    // 0 = greedy; the speculation invariant of §10.2

    // Pin the I/O and planner threads to a few physical cores so they do not
    // fight the GPU for LPDDR bandwidth (design §8).
    uint32_t io_core_mask = 0;      // 0 = let the OS decide
};

}  // namespace deepmoe
