// One transformer layer: the mHC residual machinery plus attention, optional
// engram, and MoE (design §2.4, §7.14).
//
// A V4.1-Flash layer is not the usual "norm -> attn -> add -> norm -> ffn ->
// add". The residual stream is four parallel copies (hc_mult = 4) and each
// sublayer is wrapped by the mHC mixer:
//
//   mega_mhc(attn): reads x[4][5120], computes this sublayer's input u and the
//                   pre/post/comb coefficients the *next* sublayer will use
//   attention:      dispatches 2..7 of design §7.14
//   mega_mhc(ffn):  applies hc_post for attention, then mixes for the FFN
//   moe:            gate, timeline wait, gate/up + SwiGLU, down + reduce
//
// The subtlety design §2.4 calls out: the `pre` computed by a sublayer belongs
// to the *next* one. Attention uses the previous layer's FFN pre; the FFN uses
// this layer's attention pre. Getting that wrong produces plausible-looking
// garbage, so it is an L2 oracle check.
//
// Ownership/threading: a Block borrows the sublayer implementations and records
// into the caller's command buffer on the GPU submit thread.
#pragma once

#include <cstdint>
#include <memory>

#include "core/status.h"
#include "core/types.h"
#include "model/layout.h"
#include "model/v41_config.h"
#include "runtime/attention.h"
#include "runtime/engram.h"
#include "runtime/moe.h"

namespace deepmoe::runtime {

// Which optional pieces this layer carries; derived from config.json.
struct BlockInfo {
    uint32_t layer = 0;
    bool     is_encoder = false;   // layers 0..19 (CED, design §11.1)
    bool     has_engram = false;   // layers 1 and 14
    bool     is_mtp     = false;   // logical layers 40..42
    AttentionLayerInfo attn{};

    static BlockInfo derive(const TextConfig& cfg, uint32_t layer) {
        BlockInfo b;
        b.layer      = layer;
        b.is_mtp     = layer >= cfg.num_hidden_layers;
        b.is_encoder = !b.is_mtp && layer < layout::kEncoderLayers;
        b.has_engram = !b.is_mtp && cfg.is_engram_layer(layer);
        b.attn.layer            = layer;
        // mtp blocks are window-only, like layers 0 and 1 (design §2.1).
        b.attn.compress_ratio   = b.is_mtp ? 0 : cfg.compress_ratio(layer);
        b.attn.kv_source        = !b.is_mtp && cfg.is_kv_source(layer);
        b.attn.index_source     = !b.is_mtp && cfg.is_index_source(layer);
        b.attn.candidate_source = !b.is_mtp && layer == cfg.candidate_source_layer_id;
        return b;
    }
};

class Block {
public:
    virtual ~Block() = default;
    virtual const BlockInfo& info() const = 0;

    // Records the ~11 dispatches of design §7.14 for this layer.
    // TODO(design §7.14): implement in P2.
    virtual Result<void> record_decode(TokenIndex token, uint32_t batch_m) = 0;

    // TODO(design §7.13, §11.2): prefill path, including the decoder's bounded
    // replay over only the last 128 prompt tokens.
    virtual Result<void> record_prefill(uint32_t tokens) = 0;
};

}  // namespace deepmoe::runtime
