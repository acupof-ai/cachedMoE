// Engram sublayer (layers 1 and 14) -- design §2.4, §7.10.
//
// Each engram layer owns a 384M-row hash table of 256 fp8 values (98.3 GB of
// values + 3.07 GB of scales per layer, 203 GB total). Per token it reads 24
// rows: 3 n-gram orders x 8 heads. The 24 x 256 values concatenate into a
// 6144-wide vector that goes through `wkv [25600 x 6144]` fp8 (157 MB) to
// produce 4 key copies and 1 value; the gate is sigmoid of the signed square
// root of a normalised dot product, and the value is added into the residual
// stream weighted by it.
//
// Because the addresses depend only on token ids, prefetch can start the moment
// a token exists -- including an unverified DSpark draft token (design §9.5).
// This first implementation fetches them synchronously at P2 when the layer is
// reached; the interface is the one a prefetcher slots behind.
//
// Where the hash constants come from
// ----------------------------------
// `inference/engram.py` derives them from the TOKENIZER and config.json, not
// from the checkpoint: the compressed-vocabulary map is a normalisation
// (NFKC, strip accents, lowercase, whitespace-collapse) over all 129,280
// decoded tokens, the per-(layer, lookback) multipliers come from
// `numpy.random.default_rng(10007 * layer_id)`, and the 24 bucket moduli are
// the next 24 unused primes above `engram_vocab_size - 1`. Reproducing a
// PCG64 stream and a tokenizer normaliser in C++ to rederive constants would
// be a second implementation to keep correct, so `tools/oracle.py --level l3`
// exports them once (`EngramTables::load`) and this hashes with them. The
// addresses are still computed here, on whatever trajectory the runtime is
// actually on -- what is loaded is a constant table, not golden data.
//
// Ownership/threading: EngramRunner borrows the manifest, the shards, the
// IoEngine, the pinned set and the DecodeRunner; it owns its staging buffers.
// Called from the GPU submit thread.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/profiler.h"
#include "core/status.h"
#include "core/types.h"
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/decode_kernels.h"
#include "gpu/vulkan/memory.h"
#include "model/layout.h"
#include "model/manifest.h"
#include "model/v41_config.h"
#include "storage/io_engine.h"
#include "store/pinned.h"
#include "store/shard_set.h"

namespace deepmoe::runtime {

// The constant tables `NgramHashState` derives; see the header note.
struct EngramTables {
    struct LayerTable {
        uint32_t layer = 0;
        uint64_t num_embeddings = 0;
        std::array<int64_t, layout::kEngramMaxNgram> multipliers{};
        std::array<int64_t, layout::kEngramRowsPerToken> primes{};
        std::array<int64_t, layout::kEngramRowsPerToken> offsets{};
    };

    uint32_t compressed_vocab_size = 0;
    uint32_t max_ngram = layout::kEngramMaxNgram;
    uint32_t n_heads   = layout::kEngramHeads;
    uint32_t head_dim  = layout::kEngramHeadDim;
    int64_t  pad_id    = 0;                // ALREADY compressed
    std::vector<int32_t>    token_map;     // [vocab_size] -> compressed id
    std::vector<LayerTable> layers;

    bool valid() const { return !token_map.empty() && !layers.empty(); }
    const LayerTable* for_layer(uint32_t layer) const;

    // Reads `<dir>/index.json`'s "engram" object and the token map beside it.
    // The directory is the oracle's L3 export.
    static Result<EngramTables> load(const std::string& dir);

    // The 24 row ids for the token at `position`. `history` is the whole token
    // sequence so far, `history[i]` being the token at absolute position i, and
    // must contain `position`. Look-back stops at the start of the sequence,
    // exactly as `NgramHashState.forward` stops it (`positions < shift`).
    Result<void> hash_rows(uint32_t layer, std::span<const uint32_t> history,
                           uint64_t position, std::span<uint64_t> out) const;
};

// Everything one engram layer needs at decode time: the row fetch and the two
// dispatches of design §7.10.
class EngramRunner {
public:
    EngramRunner() = default;
    ~EngramRunner() { destroy(); }

    EngramRunner(const EngramRunner&) = delete;
    EngramRunner& operator=(const EngramRunner&) = delete;

    Result<void> create(gpu::Device& device, gpu::MemoryAllocator& alloc,
                        gpu::DecodeRunner& runner,
                        const Manifest& manifest, const store::ShardSet& shards,
                        storage::IoEngine& io, const store::PinnedStore& pinned,
                        const TextConfig& cfg, EngramTables tables);
    void destroy();

    bool has_layer(uint32_t layer) const { return tables_.for_layer(layer) != nullptr; }

    // Fetches the 24 rows and runs the GEMV and the gate, reading the residual
    // stream at `x_in` and writing it to `x_out` (both [hc][dim] fp32 device
    // addresses; the same address for both is safe -- every thread reads its
    // own element before writing it).
    Result<void> run(uint32_t layer, std::span<const uint32_t> history, uint64_t position,
                     DeviceAddress x_in, DeviceAddress x_out, Profiler* profiler = nullptr);

    // `run` in its two halves, for a caller that owns the command buffer: the
    // 48-read row fetch, which is host work and can overlap whatever the GPU
    // is doing, and the two dispatches recorded into `cmd` (barrier after
    // each). The rows staged by `fetch` are the ones the next `record` of the
    // same layer consumes.
    Result<void> fetch(uint32_t layer, std::span<const uint32_t> history, uint64_t position,
                       Profiler* profiler = nullptr);
    // Whether `layer`'s staged rows are the ones for `position`.
    bool fetched(uint32_t layer, uint64_t position) const;
    Result<void> record(gpu::CommandBuffer& cmd, uint32_t layer, DeviceAddress x_in,
                        DeviceAddress x_out);

    // What the last `run` fetched, for the profiler and the report.
    uint64_t rows_fetched() const { return rows_fetched_; }
    uint64_t bytes_read()   const { return bytes_read_; }
    const EngramTables& tables() const { return tables_; }

private:
    struct LayerBind { DeviceAddress w = 0, s = 0, qw = 0, kw = 0; };
    Result<LayerBind> bind_layer(uint32_t layer) const;

    gpu::MemoryAllocator* alloc_   = nullptr;
    gpu::DecodeRunner*    runner_  = nullptr;
    const Manifest*       manifest_ = nullptr;
    const store::ShardSet* shards_  = nullptr;
    storage::IoEngine*     io_      = nullptr;
    const store::PinnedStore* pinned_ = nullptr;
    const TextConfig*     cfg_     = nullptr;
    EngramTables          tables_{};

    gpu::GpuBuffer        buf_{};        // rowval | rowsc | kv, all device-addressable
    uint64_t              off_kv_ = 0;
    // One pair of row planes PER engram layer, so both layers' rows can be
    // fetched at the start of a token (design §9.5: the addresses are known
    // the instant the token is) and consumed whenever the layer is reached.
    struct Planes {
        uint32_t layer = 0xFFFFFFFFu;
        uint64_t off_val = 0, off_sc = 0;
        uint64_t fetched_position = ~0ull;   // what the staged rows are for
    };
    std::vector<Planes>   planes_;
    Planes*               planes_for(uint32_t layer);
    Result<void>          fetch_rows(uint32_t layer, const uint64_t* rows, const Planes& dst);
    gpu::HostAllocInfo    staging_{};    // 4 KiB-aligned landing zone for the I/O
    uint64_t              rows_fetched_ = 0, bytes_read_ = 0;
    // `run`'s own command buffer, acquired once: DecodeRunner::dispatch_now
    // allocates one per call and never frees it.
    gpu::CommandPool      pool_;
    gpu::CommandBuffer    cmd_{};
    gpu::Device*          device_ = nullptr;
};

}  // namespace deepmoe::runtime
