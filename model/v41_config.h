// Parsed DeepSeek-V4.1-Flash config.json. Covers every field of design §2.1
// that the runtime needs; the vision tower is parsed but unused in stage one
// (design §1.2).
//
// Ownership/threading: an immutable value object built once at load and shared
// read-only by every module. `parse` takes the JSON text so tests can feed a
// stored copy without touching the filesystem.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/status.h"

namespace deepmoe {

// config.json -> quantization_config (design §6)
struct QuantConfig {
    std::string quant_method;       // "fp8"
    std::string activation_scheme;  // "dynamic"
    std::string scale_fmt;          // "ue8m0"
    std::string expert_dtype;       // "fp4"
    uint32_t    weight_block_m = 32;
    uint32_t    weight_block_k = 32;
};

// config.json -> text_config.rope_scaling
struct RopeScaling {
    std::string type;                        // "yarn"
    double      factor = 1.0;
    double      beta_fast = 32.0;
    double      beta_slow = 1.0;
    uint32_t    original_max_position = 0;
};

// config.json -> text_config. Field names follow the checkpoint, not our
// internal vocabulary, so a reader can diff this against config.json directly.
struct TextConfig {
    std::string model_type;
    uint32_t vocab_size            = 0;
    uint32_t hidden_size           = 0;
    uint32_t moe_intermediate_size = 0;
    uint32_t num_hidden_layers     = 0;
    uint32_t num_attention_heads   = 0;
    uint32_t num_key_value_heads   = 0;
    uint32_t head_dim              = 0;
    uint32_t qk_rope_head_dim      = 0;
    uint32_t q_lora_rank           = 0;
    uint32_t o_lora_rank           = 0;
    uint32_t o_groups              = 0;

    std::string hidden_act;                  // "silu"
    double   swiglu_limit          = 0.0;    // §2.4 expert clamp
    double   rms_norm_eps          = 0.0;    // 1e-20
    bool     attention_bias        = false;
    double   attention_dropout     = 0.0;
    double   initializer_range     = 0.0;
    bool     use_cache             = true;
    bool     tie_word_embeddings   = false;
    uint32_t max_position_embeddings = 0;
    double   rope_theta            = 0.0;
    RopeScaling rope_scaling;

    // MoE (design §2.1)
    uint32_t n_routed_experts      = 0;
    uint32_t n_shared_experts      = 0;
    uint32_t num_experts_per_tok   = 0;
    std::string scoring_func;                // "sqrtsoftplus"
    std::string topk_method;                 // "noaux_tc"
    bool     norm_topk_prob        = true;
    double   routed_scaling_factor = 0.0;    // 1.5

    // attention range / CED (design §2.1, §11.1)
    uint32_t sliding_window        = 0;      // 128
    std::vector<int64_t> compress_ratios;    // per layer, 0 = window only
    double   compress_rope_theta   = 0.0;
    std::vector<int64_t> kv_source_layer_ids;      // {2, 8, 14, 20}
    std::vector<int64_t> index_source_layer_ids;   // 8 layers
    uint32_t index_n_heads         = 0;
    uint32_t index_head_dim        = 0;
    uint32_t index_topk            = 0;
    uint32_t candidate_source_layer_id = 0;
    uint32_t candidate_topk_blocks = 0;
    uint32_t candidate_block_size  = 0;

    // mHC (design §2.4, §7.2)
    uint32_t hc_mult               = 0;      // 4
    uint32_t hc_sinkhorn_iters     = 0;      // 20
    double   hc_eps                = 0.0;

    // Engram (design §2.1, §7.10)
    std::vector<int64_t>  engram_layer_ids;      // {1, 14}
    std::vector<int64_t>  engram_num_embeddings; // per engram layer, ~384M rows
    uint32_t engram_max_ngram_size = 0;
    uint64_t engram_vocab_size     = 0;
    uint32_t engram_n_heads        = 0;
    uint32_t engram_head_dim       = 0;
    uint32_t engram_pad_token_id   = 0;
    uint32_t engram_compressed_vocab_size = 0;

    // DSpark (design §2.1, §7.12, §10)
    uint32_t num_nextn_predict_layers = 0;   // 3
    uint32_t dspark_block_size        = 0;   // 5
    uint32_t dspark_noise_token_id    = 0;
    std::vector<int64_t> dspark_target_layer_ids;  // {37, 38, 39}
    uint32_t dspark_markov_rank       = 0;
    uint32_t dspark_n_routed_experts  = 0;   // 128
    uint32_t dspark_num_experts_per_tok = 0; // 3

    // Number of rows the engram table for `engram_layer_ids[i]` holds.
    Result<uint64_t> engram_rows_for_layer(uint32_t layer) const;
    // Per-layer compress ratio; 0 means "window only" (layers 0, 1 and mtp).
    uint32_t compress_ratio(uint32_t layer) const;
    bool is_kv_source(uint32_t layer) const;
    bool is_index_source(uint32_t layer) const;
    bool is_engram_layer(uint32_t layer) const;
};

// Stage one ignores this, but the fields are read so an unexpected checkpoint
// is rejected loudly rather than silently (design §1.2).
struct VisionConfig {
    std::string model_type;
    uint32_t num_hidden_layers   = 0;
    uint32_t hidden_size         = 0;
    uint32_t num_attention_heads = 0;
    uint32_t intermediate_size   = 0;
    uint32_t patch_size          = 0;
    double   rope_theta          = 0.0;
    uint32_t downsample_ratio    = 0;
    uint32_t max_image_tokens    = 0;
    uint64_t min_pixels          = 0;
};

struct V41Config {
    std::vector<std::string> architectures;
    std::string model_type;        // "deepseek_v41"
    std::string dtype;             // "bfloat16"
    int64_t bos_token_id   = -1;
    int64_t eos_token_id   = -1;
    int64_t pad_token_id   = -1;
    int64_t image_token_id = -1;

    QuantConfig  quantization;
    TextConfig   text;
    VisionConfig vision;
    bool         has_vision = false;

    // Parses the whole document. Missing required keys are Corrupt, not
    // silently defaulted -- a wrong checkpoint must fail at load.
    static Result<V41Config> parse(std::string_view json_text);
    static Result<V41Config> load(const std::string& path);

    // Cross-checks the parsed config against model/layout.h, which the kernels
    // hard-code. Returns FailedPrecondition listing every mismatch.
    Result<void> validate_against_layout() const;

    std::string summary() const;
};

}  // namespace deepmoe
