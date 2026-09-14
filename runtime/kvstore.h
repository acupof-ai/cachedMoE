// The decode-time KV storage of design §11.3, allocated in GPU-addressable
// memory and shaped exactly as gpu/shaders/wkv.slang writes it and
// sparse_attn.slang reads it.
//
// runtime/kvcache.h next to this file is the *geometry*: what 64K of context
// costs, and the rollback and prefix-persistence interfaces of §10.2 and §11.4.
// This is the buffer that exists during a decode step.
//
// What is real, as of P2 step 3
// ----------------------------
//   window KV       REAL. A 128-slot ring per layer of E4M3 bytes plus one
//                   UE8M0 scale per 32 dims, written by the wkv kernel every
//                   decode step exactly as `Attention._window_kv` writes it.
//   compressed KV   REAL. `compressor.slang` writes one row per completed
//                   group into the source layer's plane (design §7.4).
//   indexer K       REAL. `indexer.slang` stage 2 writes the key cache of the
//                   layer that owns it.
//   compressor state REAL. The [ratio][latent_dim] tail of an incomplete group,
//                   carried across decode steps, which is why it lives here
//                   and not in the per-step scratch.
//   top-k list      REAL. The window half is written by the host
//                   (`set_decode_topk`, model.py's `get_window_topk_idxs`) and
//                   the compressed half by `indexer.slang` stage 5.
//
// The `seed_*` entry points remain, because a decode run still starts from
// what the prompt left behind -- either the L3 export's prefill record or our
// own slow prefill. What they no longer do is stand in for a kernel.
//
// Storage is per LAYER, not per source, even though only the four
// `kv_source_layers` ever write a compressed plane, an index-key cache or a
// compressor state. 40 planes instead of 4 costs 45 MB at this decode geometry
// and keeps `layer(l)` a pure function of l; the Engine points a reuse layer's
// view at its source's addresses (model.py's `shared_attn`), which is a
// decision about which cache, not about where caches live.
//
// Storage note: the compressed half is bf16 here, not the FP4 E2M1 + E4M3/16
// of design §11.3. The VALUES are already on the fp4 grid -- the compressor
// quantises them -- so this is a packing choice, not a precision one: 66 MB
// instead of 48 MB at 64K context. Packing it costs a nibble unpack in
// sparse_attn's inner loop and buys 18 MB; that is a P3 decision, once the
// compressor kernel exists to write the packed form.
//
// Ownership/threading: KvStore owns its allocation through the MemoryAllocator
// it was created with. Written by the GPU, seeded by the host before a step.
#pragma once

#include <cstdint>
#include <vector>

#include "core/status.h"
#include "core/types.h"
#include "gpu/vulkan/memory.h"

namespace deepmoe::runtime {

struct KvStoreConfig {
    uint32_t layers      = 40;
    uint32_t window      = 128;
    uint32_t latent_dim  = 512;    // head_dim; MQA over one KV head
    uint32_t index_dim   = 128;
    uint32_t max_context = 1024;   // decode geometry, not the 64K of §11.3
    uint32_t scale_block = 32;
    // Slots in a compressor's carried group state. compressor.slang's
    // `kMaxRatio`; the checkpoint's largest compress_ratio is 2.
    uint32_t max_ratio   = 8;

    uint32_t window_scales() const { return latent_dim / scale_block; }
    uint64_t window_bytes() const {
        return uint64_t(layers) * window * (latent_dim + window_scales());
    }
    // One compressed entry per source; ratio 1 is the worst case, so the
    // decode-time allocation is sized for it and the caller says how many
    // entries are live.
    uint64_t compressed_bytes() const {
        return uint64_t(layers) * max_context * latent_dim * 2;   // bf16
    }
    uint64_t topk_bytes() const {
        return uint64_t(layers) * (window + max_context) * sizeof(uint32_t);
    }
    // One index key per compressed position, bf16, on the layers that own one.
    uint64_t index_key_bytes() const {
        return uint64_t(layers) * max_context * index_dim * 2;
    }
    // kv_state and score_state, [max_ratio][latent_dim] fp32 each.
    uint64_t cmp_state_bytes() const {
        return uint64_t(layers) * max_ratio * latent_dim * 4 * 2;
    }
    uint64_t total_bytes() const {
        return window_bytes() + compressed_bytes() + topk_bytes() +
               index_key_bytes() + cmp_state_bytes();
    }
};

// Where one layer's KV lives, as the kernels address it.
struct KvLayerView {
    DeviceAddress win_val   = kNoDeviceAddress;   // [window][latent_dim] E4M3
    DeviceAddress win_scale = kNoDeviceAddress;   // [window][latent_dim/32] UE8M0
    DeviceAddress cmp_kv    = kNoDeviceAddress;   // [n_cmp][latent_dim] bf16
    DeviceAddress top_idx   = kNoDeviceAddress;   // [n_kv] int32, into the concatenation
    DeviceAddress idx_key   = kNoDeviceAddress;   // [n_cmp][index_dim] bf16
    DeviceAddress cmp_state_kv    = kNoDeviceAddress;  // [max_ratio][latent_dim] f32
    DeviceAddress cmp_state_score = kNoDeviceAddress;
    uint8_t*      win_val_host   = nullptr;
    uint8_t*      win_scale_host = nullptr;
    uint16_t*     cmp_kv_host    = nullptr;
    uint32_t*     top_idx_host   = nullptr;
    uint16_t*     idx_key_host   = nullptr;
    float*        cmp_state_kv_host    = nullptr;
    float*        cmp_state_score_host = nullptr;
    uint32_t      n_cmp = 0;
    uint32_t      n_kv  = 0;
};

class KvStore {
public:
    KvStore() = default;
    ~KvStore() { destroy(); }

    KvStore(const KvStore&) = delete;
    KvStore& operator=(const KvStore&) = delete;

    Result<void> create(gpu::MemoryAllocator& alloc, const KvStoreConfig& cfg);
    void         destroy();
    // Back to the state a fresh `create` leaves: everything zero, every
    // `score_state` slot -inf, every count 0. What a prefill from position 0
    // starts from.
    void         clear();

    const KvStoreConfig& config() const { return cfg_; }
    uint64_t bytes() const { return buf_.bytes; }

    // The addresses layer `l`'s dispatches need. `n_cmp` and `n_kv` come from
    // whatever seeded the layer.
    Result<KvLayerView> layer(uint32_t l) const;

    // --- seeding (see the header: prefill is not ours yet) -----------------

    // `values` is [rows][latent_dim] in fp32, the post-quantisation window KV
    // the oracle exported. Re-encoded to the E4M3 byte + UE8M0 scale the
    // kernels read; because the values are already on that grid the round trip
    // is exact, which `tests/test_gpu_attn.cpp` asserts.
    Result<void> seed_window(uint32_t layer, const float* values, uint32_t rows);
    // `values` is [n_cmp][latent_dim] fp32, stored as bf16.
    Result<void> seed_compressed(uint32_t layer, const float* values, uint32_t n_cmp);
    // The concatenated index list `sparse_attn` walks: window ring slots below
    // `window`, compressed rows offset by it, -1 for a slot holding nothing.
    Result<void> seed_topk(uint32_t layer, const int32_t* idx, uint32_t n_kv);
    // `values` is [rows][index_dim] fp32, stored as bf16: the indexer's key
    // cache, which only a kv_source_layer owns.
    Result<void> seed_index_k(uint32_t layer, const float* values, uint32_t rows);
    // The compressor's carried group state. `score` is -inf in the slots the
    // reference has never written, which is what makes them score zero.
    Result<void> seed_cmp_state(uint32_t layer, const float* kv, const float* score,
                               uint32_t ratio);

    // --- per-step bookkeeping (design §7.4, produced not loaded) ------------

    // model.py's `get_window_topk_idxs` for one decode query: the ring slots
    // oldest first, with a slot the sequence has not reached yet marked -1.
    // Writes the window half of the top-k list and sets n_cmp / n_kv, leaving
    // the compressed half for `indexer.slang` stage 5 to fill at offset
    // `window`. `n_sel` is how many compressed picks the list carries --
    // min(index_topk, n_cmp) -- so n_kv = window + n_sel.
    Result<void> set_decode_topk(uint32_t layer, uint32_t position, uint32_t n_cmp,
                                 uint32_t n_sel);
    // n_cmp alone, for a layer whose compressed plane belongs to a source.
    Result<void> set_counts(uint32_t layer, uint32_t n_cmp, uint32_t n_kv);

private:
    gpu::MemoryAllocator* alloc_ = nullptr;
    gpu::GpuBuffer        buf_{};
    KvStoreConfig         cfg_{};
    uint64_t              off_win_val_ = 0, off_win_scale_ = 0, off_cmp_ = 0, off_top_ = 0;
    uint64_t              off_idx_k_ = 0, off_state_ = 0;
    std::vector<uint32_t> n_cmp_, n_kv_;
};

}  // namespace deepmoe::runtime
