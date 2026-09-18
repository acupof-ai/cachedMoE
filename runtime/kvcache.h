// KV cache geometry and storage (design §11.3, §11.4).
//
// KV is not a memory problem here: 64K context costs ~90 MB total, all of it in
// the GPU heap. What matters is the *shape*, because DSpark rejection has to
// roll it back (design §10.2) and because the prefix-persistence path of §11.4
// has to serialise exactly this.
//
//   window KV      43 x 128 x (512 B + 16 B scale)          2.8 MB
//   compressed KV  sources 2/8/14 at ratio 2, 20 at ratio 1  84 MB fp8 / 48 MB fp4
//   indexer K      4 kv sources, 128 dims fp4 + scale        11 MB
//   candidate pool layer 20 -> layers 24..36, 2048 blocks    KB
//   engram hash    C x 8 B                                   0.5 MB
//
// Ownership/threading: KvCache owns its buffers. Written by the GPU only;
// the CPU touches it for rollback bookkeeping and for §11.4 persistence, both
// on the engine thread.
#pragma once

#include <cstdint>
#include <format>
#include <string>
#include <vector>

#include "core/status.h"
#include "core/types.h"
#include "model/v41_config.h"

namespace deepmoe::runtime {

struct KvGeometry {
    uint32_t max_context   = 65536;
    uint32_t window        = 128;     // sliding_window
    uint32_t latent_dim    = 512;     // head_dim, MQA over one KV head
    uint32_t layers        = 40;
    uint32_t mtp_blocks    = 3;
    bool     compressed_fp4 = false;  // design §6: fp8 in oracle mode, fp4 in production

    // 128 slots x (512 B latent + 16 B fp8 block scale) per layer, plus the
    // three mtp blocks (design §11.3).
    uint64_t window_bytes() const {
        return uint64_t(layers + mtp_blocks) * window * (latent_dim + 16);
    }
    // One compressed latent per `ratio` tokens for each kv_source layer.
    uint64_t compressed_bytes(const TextConfig& cfg) const {
        const uint64_t per_entry = compressed_fp4 ? (latent_dim / 2 + 32) : latent_dim;
        uint64_t n = 0;
        for (int64_t src : cfg.kv_source_layer_ids) {
            const uint32_t r = cfg.compress_ratio(static_cast<uint32_t>(src));
            if (r) n += (max_context / r) * per_entry;
        }
        return n;
    }
    // 128-dim FP4 index keys plus one E8M0 scale per 32, for each layer that
    // OWNS a key cache: model.py's `Indexer.owns_k` is `layer_id in
    // kv_source_layers`. Layers 24..36 are index sources that score layer 20's
    // keys with their own weights, so they add no cache (Track R2 correction of
    // design 搂11.3's v0.4 "8 sources, 28.97 MB"; docs/p4_kv_ux.md 搂1).
    uint64_t indexer_bytes(const TextConfig& cfg) const {
        uint64_t n = 0;
        for (int64_t src : cfg.kv_source_layer_ids) {
            const uint32_t r = cfg.compress_ratio(static_cast<uint32_t>(src));
            if (r) n += (max_context / r) * (cfg.index_head_dim / 2 + cfg.index_head_dim / 32);
        }
        return n;
    }
    uint64_t total_bytes(const TextConfig& cfg) const {
        return window_bytes() + compressed_bytes(cfg) + indexer_bytes(cfg)
             + uint64_t(max_context) * 8;   // engram hash cache
    }
    std::string to_string(const TextConfig& cfg) const {
        return std::format("kv @ {}: window {:.1f} MB, compressed {:.1f} MB ({}), "
                           "indexer {:.1f} MB, total {:.1f} MB",
                           max_context, window_bytes() / 1e6,
                           compressed_bytes(cfg) / 1e6, compressed_fp4 ? "fp4" : "fp8",
                           indexer_bytes(cfg) / 1e6, total_bytes(cfg) / 1e6);
    }
};

// A snapshot taken before a speculative block, so a rejection can be undone
// (design §10.2). Window KV is overwritten in place by position, so only the
// compressor's partial group state actually needs copying -- at ratio <= 2 that
// is a few hundred bytes.
struct KvSnapshot {
    TokenIndex position = 0;
    std::vector<std::byte> compressor_state;
};

class KvCache {
public:
    KvCache() = default;
    ~KvCache() = default;

    KvCache(const KvCache&) = delete;
    KvCache& operator=(const KvCache&) = delete;

    // TODO(design §11.3): allocate through gpu/vulkan/memory.h once that lands.
    // The geometry above is already exact, so the budget is knowable today.
    Result<void> init(const TextConfig&, const KvGeometry& geom) {
        geom_ = geom;
        return unimplemented("runtime::KvCache::init (design §11.3, needs gpu memory)");
    }
    void reset() { position_ = 0; ready_ = false; }

    const KvGeometry& geometry() const { return geom_; }
    TokenIndex        position()  const { return position_; }

    // TODO(design §10.2): snapshot/rollback around a DSpark verify block.
    Result<KvSnapshot> snapshot() const {
        return unimplemented("runtime::KvCache::snapshot (design §10.2)");
    }
    Result<void> rollback(const KvSnapshot&) {
        return unimplemented("runtime::KvCache::rollback (design §10.2)");
    }

    // TODO(design §11.4): persist the encoder output, compressed KV and
    // indexer K under a prompt-prefix hash so a second turn skips the ~270 GB
    // encoder prefill.
    Result<void> save_prefix(const std::string&, uint64_t) const {
        return unimplemented("runtime::KvCache::save_prefix (design §11.4)");
    }
    Result<void> load_prefix(const std::string&, uint64_t) {
        return unimplemented("runtime::KvCache::load_prefix (design §11.4)");
    }

private:
    KvGeometry geom_{};
    TokenIndex position_ = 0;
    bool       ready_ = false;
};

}  // namespace deepmoe::runtime
