// Attention sublayer interface (design §7.3-§7.6, §2.4).
//
// V4.1-Flash attention is MQA over a 512-wide latent with low-rank Q (1280),
// grouped low-rank O (8 x 1024), and a per-layer range of "sliding window 128 +
// indexer top-512 of the compressed KV". Layers 0, 1 and the mtp blocks are
// window only.
//
// The order of operations the oracle has to match exactly (design §2.4):
//   qr = wq_a(u); q = wq_b(rmsnorm(qr)); RoPE on the last 64 dims of each head
//   kv = rmsnorm(wkv(u)); RoPE on the last 64; quantise to fp8 into the window
//   o  = sparse_attention(q, window ++ top512, attn_sink)
//   o  = inverse_RoPE(o); a = wo_b(wo_a_grouped(o))
//
// Ownership/threading: Attention borrows the device, the pipelines and the KV
// cache. Recording happens on the GPU submit thread; nothing here allocates per
// token.
#pragma once

#include <cstdint>
#include <span>

#include "core/status.h"
#include "core/types.h"
#include "model/v41_config.h"
#include "runtime/kvcache.h"

namespace deepmoe::runtime {

struct AttentionLayerInfo {
    uint32_t layer = 0;
    uint32_t compress_ratio = 0;   // 0 = window only (layers 0, 1, mtp)
    bool     kv_source    = false; // 2, 8, 14, 20: runs the compressor
    bool     index_source = false; // 8 layers: runs the indexer
    bool     candidate_source = false;  // layer 20 builds the block pool for 24..36
};

class Attention {
public:
    virtual ~Attention() = default;

    // Records dispatches #2..#7 of design §7.14 for one layer into the
    // per-token command buffer.
    // TODO(design §7.3-§7.6): implement in P2.
    virtual Result<void> record_decode(const AttentionLayerInfo& info, uint32_t batch_m) = 0;

    // Prefill uses cooperative-matrix GEMM and band attention instead
    // (design §7.13).
    // TODO(design §7.13): implement in P5.
    virtual Result<void> record_prefill(const AttentionLayerInfo& info, uint32_t tokens) = 0;
};

// The indexer: scores compressed positions with 32 heads x 128 dims of FP4 and
// keeps the top 512. This is the only decode kernel whose cost grows with
// context length -- at 64K and ratio 2 it is 32K positions, 2 MB of K and
// 134 MFLOP, which is small, but the two-stage radix select is not trivial.
// TODO(design §7.4): implement in P2.
class Indexer {
public:
    virtual ~Indexer() = default;
    virtual Result<void> record(uint32_t layer, uint32_t context_len) = 0;
};

// The compressor: every `ratio` tokens it emits one compressed latent for a
// kv_source layer, with a softmax-gated accumulator carried between calls.
// That accumulator is the only thing a speculative rollback has to restore
// (design §10.2).
// TODO(design §7.4): implement in P2.
class Compressor {
public:
    virtual ~Compressor() = default;
    virtual Result<void> record(uint32_t layer, uint32_t position) = 0;
};

}  // namespace deepmoe::runtime
