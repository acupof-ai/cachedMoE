// The decode-time KV storage of design §11.3, allocated in GPU-addressable
// memory and shaped exactly as gpu/shaders/wkv.slang writes it and
// sparse_attn.slang reads it.
//
// runtime/kvcache.h next to this file is the *geometry*: what 64K of context
// costs, and the rollback and prefix-persistence interfaces of §10.2 and §11.4.
// This is the buffer that exists during a decode step.
//
// What is real and what is not, as of P2 step 1
// --------------------------------------------
//   window KV     REAL. A 128-slot ring per layer of E4M3 bytes plus one UE8M0
//                 scale per 32 dims, written by the wkv kernel every decode
//                 step exactly as `Attention._window_kv` writes it.
//   compressed KV LOADED, not produced. The compressor and indexer kernels of
//                 design §7.4 are not written yet, so a decode step reads the
//                 compressed latents and the indexer's top-k list that the
//                 oracle's prefill produced (`seed_compressed`, `seed_topk`).
//                 Everything downstream of them -- sparse_attn, wo_a, wo_b --
//                 is real and is measured against the oracle.
//   indexer K     ALLOCATED, unused. Reserved so the geometry is right when
//                 the indexer lands.
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
    uint64_t total_bytes() const {
        return window_bytes() + compressed_bytes() + topk_bytes();
    }
};

// Where one layer's KV lives, as the kernels address it.
struct KvLayerView {
    DeviceAddress win_val   = kNoDeviceAddress;   // [window][latent_dim] E4M3
    DeviceAddress win_scale = kNoDeviceAddress;   // [window][latent_dim/32] UE8M0
    DeviceAddress cmp_kv    = kNoDeviceAddress;   // [n_cmp][latent_dim] bf16
    DeviceAddress top_idx   = kNoDeviceAddress;   // [n_kv] int32, into the concatenation
    uint8_t*      win_val_host   = nullptr;
    uint8_t*      win_scale_host = nullptr;
    uint16_t*     cmp_kv_host    = nullptr;
    uint32_t*     top_idx_host   = nullptr;
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

private:
    gpu::MemoryAllocator* alloc_ = nullptr;
    gpu::GpuBuffer        buf_{};
    KvStoreConfig         cfg_{};
    uint64_t              off_win_val_ = 0, off_win_scale_ = 0, off_cmp_ = 0, off_top_ = 0;
    std::vector<uint32_t> n_cmp_, n_kv_;
};

}  // namespace deepmoe::runtime
