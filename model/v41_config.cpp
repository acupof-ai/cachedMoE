#include "model/v41_config.h"

#include <algorithm>
#include <format>

#include "core/json.h"
#include "model/layout.h"

namespace deepmoe {
namespace {

// Required-key readers: any miss aborts the parse with the offending key name.
Result<uint32_t> req_u32(const JsonValue& o, std::string_view k) {
    auto v = o.uint_at(k);
    if (!v) return std::unexpected(v.error());
    if (*v > 0xFFFFFFFFull) return fail(Err::Corrupt, std::format("key '{}' exceeds uint32", k));
    return static_cast<uint32_t>(*v);
}
Result<double> req_f64(const JsonValue& o, std::string_view k) { return o.double_at(k); }
Result<std::string> req_str(const JsonValue& o, std::string_view k) { return o.string_at(k); }

std::vector<int64_t> opt_int_array(const JsonValue& o, std::string_view k) {
    auto r = o.int_array_at(k);
    return r ? *std::move(r) : std::vector<int64_t>{};
}

bool contains(const std::vector<int64_t>& v, int64_t x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

Result<RopeScaling> parse_rope(const JsonValue& text) {
    RopeScaling rs;
    const JsonValue* n = text.find("rope_scaling");
    if (!n || n->is_null()) return rs;
    rs.type   = n->string_or("rope_type", n->string_or("type", ""));
    rs.factor = n->double_or("factor", 1.0);
    rs.beta_fast = n->double_or("beta_fast", 32.0);
    rs.beta_slow = n->double_or("beta_slow", 1.0);
    rs.original_max_position = static_cast<uint32_t>(n->int_or("original_max_position_embeddings", 0));
    return rs;
}

Result<QuantConfig> parse_quant(const JsonValue& root) {
    QuantConfig q;
    const JsonValue* n = root.find("quantization_config");
    if (!n || n->is_null())
        return fail(Err::Corrupt, "config.json has no quantization_config (design §6 requires native FP4/FP8)");
    q.quant_method      = n->string_or("quant_method", "");
    q.activation_scheme = n->string_or("activation_scheme", "");
    q.scale_fmt         = n->string_or("scale_fmt", "");
    q.expert_dtype      = n->string_or("expert_dtype", "");
    auto blocks = n->int_array_at("weight_block_size");
    if (blocks && blocks->size() == 2) {
        q.weight_block_m = static_cast<uint32_t>((*blocks)[0]);
        q.weight_block_k = static_cast<uint32_t>((*blocks)[1]);
    }
    return q;
}

Result<TextConfig> parse_text(const JsonValue& root) {
    auto tp = root.at("text_config");
    if (!tp) return fail(Err::Corrupt, "config.json has no text_config");
    const JsonValue& t = **tp;

    TextConfig c;
    c.model_type = t.string_or("model_type", "");

#define REQ_U32(field) { auto r = req_u32(t, #field); if (!r) return std::unexpected(r.error()); c.field = *r; }
#define REQ_F64(field) { auto r = req_f64(t, #field); if (!r) return std::unexpected(r.error()); c.field = *r; }
#define REQ_STR(field) { auto r = req_str(t, #field); if (!r) return std::unexpected(r.error()); c.field = *r; }

    REQ_U32(vocab_size)
    REQ_U32(hidden_size)
    REQ_U32(moe_intermediate_size)
    REQ_U32(num_hidden_layers)
    REQ_U32(num_attention_heads)
    REQ_U32(num_key_value_heads)
    REQ_U32(head_dim)
    REQ_U32(qk_rope_head_dim)
    REQ_U32(q_lora_rank)
    REQ_U32(o_lora_rank)
    REQ_U32(o_groups)
    REQ_STR(hidden_act)
    REQ_F64(swiglu_limit)
    REQ_F64(rms_norm_eps)
    REQ_U32(max_position_embeddings)
    REQ_F64(rope_theta)

    c.attention_bias    = t.bool_or("attention_bias", false);
    c.attention_dropout = t.double_or("attention_dropout", 0.0);
    c.initializer_range = t.double_or("initializer_range", 0.0);
    c.use_cache         = t.bool_or("use_cache", true);
    c.tie_word_embeddings = t.bool_or("tie_word_embeddings", false);

    auto rs = parse_rope(t);
    if (!rs) return std::unexpected(rs.error());
    c.rope_scaling = *rs;

    REQ_U32(n_routed_experts)
    REQ_U32(n_shared_experts)
    REQ_U32(num_experts_per_tok)
    REQ_STR(scoring_func)
    REQ_STR(topk_method)
    REQ_F64(routed_scaling_factor)
    c.norm_topk_prob = t.bool_or("norm_topk_prob", true);

    REQ_U32(sliding_window)
    c.compress_ratios        = opt_int_array(t, "compress_ratios");
    c.compress_rope_theta    = t.double_or("compress_rope_theta", 0.0);
    c.kv_source_layer_ids    = opt_int_array(t, "kv_source_layer_ids");
    c.index_source_layer_ids = opt_int_array(t, "index_source_layer_ids");
    REQ_U32(index_n_heads)
    REQ_U32(index_head_dim)
    REQ_U32(index_topk)
    c.candidate_source_layer_id = static_cast<uint32_t>(t.int_or("candidate_source_layer_id", 0));
    c.candidate_topk_blocks     = static_cast<uint32_t>(t.int_or("candidate_topk_blocks", 0));
    c.candidate_block_size      = static_cast<uint32_t>(t.int_or("candidate_block_size", 0));

    REQ_U32(hc_mult)
    REQ_U32(hc_sinkhorn_iters)
    c.hc_eps = t.double_or("hc_eps", 0.0);

    c.engram_layer_ids      = opt_int_array(t, "engram_layer_ids");
    c.engram_num_embeddings = opt_int_array(t, "engram_num_embeddings");
    REQ_U32(engram_max_ngram_size)
    { auto r = t.uint_at("engram_vocab_size"); if (!r) return std::unexpected(r.error()); c.engram_vocab_size = *r; }
    REQ_U32(engram_n_heads)
    REQ_U32(engram_head_dim)
    c.engram_pad_token_id = static_cast<uint32_t>(t.int_or("engram_pad_token_id", 0));
    REQ_U32(engram_compressed_vocab_size)

    REQ_U32(num_nextn_predict_layers)
    REQ_U32(dspark_block_size)
    c.dspark_noise_token_id   = static_cast<uint32_t>(t.int_or("dspark_noise_token_id", 0));
    c.dspark_target_layer_ids = opt_int_array(t, "dspark_target_layer_ids");
    REQ_U32(dspark_markov_rank)
    REQ_U32(dspark_n_routed_experts)
    REQ_U32(dspark_num_experts_per_tok)

#undef REQ_U32
#undef REQ_F64
#undef REQ_STR
    return c;
}

VisionConfig parse_vision(const JsonValue& n) {
    VisionConfig v;
    v.model_type           = n.string_or("model_type", "");
    v.num_hidden_layers    = static_cast<uint32_t>(n.int_or("num_hidden_layers", 0));
    v.hidden_size          = static_cast<uint32_t>(n.int_or("hidden_size", 0));
    v.num_attention_heads  = static_cast<uint32_t>(n.int_or("num_attention_heads", 0));
    v.intermediate_size    = static_cast<uint32_t>(n.int_or("intermediate_size", 0));
    v.patch_size           = static_cast<uint32_t>(n.int_or("patch_size", 0));
    v.rope_theta           = n.double_or("rope_theta", 0.0);
    v.downsample_ratio     = static_cast<uint32_t>(n.int_or("downsample_ratio", 0));
    v.max_image_tokens     = static_cast<uint32_t>(n.int_or("max_image_tokens", 0));
    v.min_pixels           = static_cast<uint64_t>(n.int_or("min_pixels", 0));
    return v;
}

}  // namespace

Result<uint64_t> TextConfig::engram_rows_for_layer(uint32_t layer) const {
    for (size_t i = 0; i < engram_layer_ids.size(); ++i) {
        if (engram_layer_ids[i] == static_cast<int64_t>(layer)) {
            if (i >= engram_num_embeddings.size())
                return fail(Err::Corrupt, "engram_num_embeddings shorter than engram_layer_ids");
            return static_cast<uint64_t>(engram_num_embeddings[i]);
        }
    }
    return fail(Err::NotFound, std::format("layer {} is not an engram layer", layer));
}

uint32_t TextConfig::compress_ratio(uint32_t layer) const {
    if (layer >= compress_ratios.size()) return 0;
    int64_t r = compress_ratios[layer];
    return r > 0 ? static_cast<uint32_t>(r) : 0;
}
bool TextConfig::is_kv_source(uint32_t layer) const    { return contains(kv_source_layer_ids, layer); }
bool TextConfig::is_index_source(uint32_t layer) const { return contains(index_source_layer_ids, layer); }
bool TextConfig::is_engram_layer(uint32_t layer) const { return contains(engram_layer_ids, layer); }

Result<V41Config> V41Config::parse(std::string_view json_text) {
    auto doc = json_parse(json_text);
    if (!doc) return std::unexpected(doc.error());
    if (!doc->is_object()) return fail(Err::Corrupt, "config.json root is not an object");

    V41Config c;
    if (const JsonValue* a = doc->find("architectures")) {
        if (auto arr = a->as_array())
            for (const JsonValue& e : **arr)
                if (auto s = e.as_string()) c.architectures.emplace_back(*s);
    }
    c.model_type     = doc->string_or("model_type", "");
    c.dtype          = doc->string_or("dtype", doc->string_or("torch_dtype", ""));
    c.bos_token_id   = doc->int_or("bos_token_id", -1);
    c.eos_token_id   = doc->int_or("eos_token_id", -1);
    c.pad_token_id   = doc->int_or("pad_token_id", -1);
    c.image_token_id = doc->int_or("image_token_id", -1);

    auto q = parse_quant(*doc);
    if (!q) return std::unexpected(q.error());
    c.quantization = *q;

    auto t = parse_text(*doc);
    if (!t) return std::unexpected(t.error());
    c.text = *std::move(t);

    if (const JsonValue* v = doc->find("vision_config"); v && v->is_object()) {
        c.vision = parse_vision(*v);
        c.has_vision = true;
    }
    return c;
}

Result<V41Config> V41Config::load(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(Err::Io, std::format("cannot open '{}'", path));
    std::string buf;
    char chunk[65536];
    size_t n;
    while ((n = std::fread(chunk, 1, sizeof chunk, f)) > 0) buf.append(chunk, n);
    bool bad = std::ferror(f) != 0;
    std::fclose(f);
    if (bad) return fail(Err::Io, std::format("read error on '{}'", path));
    auto c = parse(buf);
    if (!c) return fail(c.error().code, std::format("{}: {}", path, c.error().message));
    return c;
}

Result<void> V41Config::validate_against_layout() const {
    namespace L = layout;
    std::string bad;
    auto check = [&bad](const char* name, uint64_t got, uint64_t want) {
        if (got != want) bad += std::format("  {}: config {} != layout.h {}\n", name, got, want);
    };
    check("hidden_size",           text.hidden_size,           L::kHiddenSize);
    check("num_hidden_layers",     text.num_hidden_layers,     L::kNumLayers);
    check("moe_intermediate_size", text.moe_intermediate_size, L::kMoeIntermediate);
    check("n_routed_experts",      text.n_routed_experts,      L::kRoutedExperts);
    check("n_shared_experts",      text.n_shared_experts,      L::kSharedExperts);
    check("num_experts_per_tok",   text.num_experts_per_tok,   L::kExpertsPerTok);
    check("num_attention_heads",   text.num_attention_heads,   L::kAttnHeads);
    check("num_key_value_heads",   text.num_key_value_heads,   L::kKvHeads);
    check("head_dim",              text.head_dim,              L::kHeadDim);
    check("qk_rope_head_dim",      text.qk_rope_head_dim,      L::kRopeHeadDim);
    check("q_lora_rank",           text.q_lora_rank,           L::kQLoraRank);
    check("o_lora_rank",           text.o_lora_rank,           L::kOLoraRank);
    check("o_groups",              text.o_groups,              L::kOGroups);
    check("hc_mult",               text.hc_mult,               L::kHcMult);
    check("hc_sinkhorn_iters",     text.hc_sinkhorn_iters,     L::kSinkhornIters);
    check("sliding_window",        text.sliding_window,        L::kSlidingWindow);
    check("vocab_size",            text.vocab_size,            L::kVocabSize);
    check("engram_n_heads",        text.engram_n_heads,        L::kEngramHeads);
    check("engram_head_dim",       text.engram_head_dim,       L::kEngramHeadDim);
    check("engram_max_ngram_size", text.engram_max_ngram_size, L::kEngramMaxNgram);
    check("index_n_heads",         text.index_n_heads,         L::kIndexHeads);
    check("index_head_dim",        text.index_head_dim,        L::kIndexHeadDim);
    check("index_topk",            text.index_topk,            L::kIndexTopK);
    check("num_nextn_predict_layers", text.num_nextn_predict_layers, L::kMtpBlocks);
    check("dspark_block_size",     text.dspark_block_size,     L::kDsparkBlockSize);
    check("dspark_n_routed_experts", text.dspark_n_routed_experts, L::kDsparkExperts);
    check("dspark_num_experts_per_tok", text.dspark_num_experts_per_tok, L::kDsparkTopK);
    check("dspark_markov_rank",    text.dspark_markov_rank,    L::kDsparkMarkovRank);
    check("quant weight_block_m",  quantization.weight_block_m, L::kFp8ScaleBlockM);
    check("quant weight_block_k",  quantization.weight_block_k, L::kFp8ScaleBlockK);

    if (text.scoring_func != "sqrtsoftplus")
        bad += std::format("  scoring_func: '{}' != 'sqrtsoftplus'\n", text.scoring_func);
    if (text.topk_method != "noaux_tc")
        bad += std::format("  topk_method: '{}' != 'noaux_tc'\n", text.topk_method);
    if (quantization.expert_dtype != "fp4")
        bad += std::format("  expert_dtype: '{}' != 'fp4'\n", quantization.expert_dtype);
    if (quantization.scale_fmt != "ue8m0")
        bad += std::format("  scale_fmt: '{}' != 'ue8m0'\n", quantization.scale_fmt);
    if (text.engram_layer_ids.size() != layout::kEngramLayers)
        bad += std::format("  engram_layer_ids: {} entries, expected {}\n",
                           text.engram_layer_ids.size(), layout::kEngramLayers);

    if (!bad.empty())
        return fail(Err::FailedPrecondition, "config.json disagrees with model/layout.h:\n" + bad);
    return {};
}

std::string V41Config::summary() const {
    return std::format(
        "{} ({}): {} layers, hidden {} x hc {}, {} routed experts top-{} ({}), "
        "{} heads x {} (kv {}), vocab {}, ctx {}, engram layers {}, dspark {}x{}",
        model_type, dtype, text.num_hidden_layers, text.hidden_size, text.hc_mult,
        text.n_routed_experts, text.num_experts_per_tok, quantization.expert_dtype,
        text.num_attention_heads, text.head_dim, text.num_key_value_heads,
        text.vocab_size, text.max_position_embeddings,
        text.engram_layer_ids.size(), text.num_nextn_predict_layers, text.dspark_block_size);
}

}  // namespace deepmoe
