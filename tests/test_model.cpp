// model/: parsing the real DeepSeek-V4.1-Flash config.json (a verbatim copy
// from ModelScope lives in tests/data/) and the manifest reader.
//
// This is the test that fails loudly if the checkpoint ever changes shape: every
// number in design §2.1 is asserted against the file, not against a constant
// copied into the test.
#include <string>
#include <vector>

#include "model/layout.h"
#include "model/manifest.h"
#include "model/v41_config.h"
#include "runtime/block.h"
#include "runtime/kvcache.h"
#include "tests/test_framework.h"

using namespace deepmoe;

// Set by tests/CMakeLists.txt so the test finds its data whatever the cwd is.
#ifndef DEEPMOE_TEST_DATA_DIR
#define DEEPMOE_TEST_DATA_DIR "tests/data"
#endif

namespace {
std::string data_path(const char* name) {
    return std::string(DEEPMOE_TEST_DATA_DIR) + "/" + name;
}
}  // namespace

DEEPMOE_TEST(v41_config, parses_the_real_config_json) {
    auto c = V41Config::load(data_path("v41_config.json"));
    REQUIRE_OK(c);

    CHECK_EQ(c->model_type, std::string("deepseek_v41"));
    CHECK_EQ(c->dtype, std::string("bfloat16"));
    REQUIRE_EQ(c->architectures.size(), 1u);
    CHECK_EQ(c->architectures[0], std::string("DeepseekV41ForCausalLM"));
    CHECK_EQ(c->bos_token_id, 0);
    CHECK_EQ(c->eos_token_id, 1);
    CHECK_EQ(c->pad_token_id, 2);
    CHECK_EQ(c->image_token_id, 129264);

    // design §6: native FP4 experts, FP8 everything else, UE8M0 scales, 32x32.
    CHECK_EQ(c->quantization.quant_method, std::string("fp8"));
    CHECK_EQ(c->quantization.expert_dtype, std::string("fp4"));
    CHECK_EQ(c->quantization.scale_fmt, std::string("ue8m0"));
    CHECK_EQ(c->quantization.activation_scheme, std::string("dynamic"));
    CHECK_EQ(c->quantization.weight_block_m, 32u);
    CHECK_EQ(c->quantization.weight_block_k, 32u);

    const TextConfig& t = c->text;
    // design §2.1 topology
    CHECK_EQ(t.vocab_size, 129280u);
    CHECK_EQ(t.hidden_size, 5120u);
    CHECK_EQ(t.num_hidden_layers, 40u);
    CHECK_EQ(t.moe_intermediate_size, 2304u);
    CHECK_EQ(t.hc_mult, 4u);
    CHECK_EQ(t.hc_sinkhorn_iters, 20u);
    CHECK_EQ(t.n_routed_experts, 384u);
    CHECK_EQ(t.n_shared_experts, 1u);
    CHECK_EQ(t.num_experts_per_tok, 6u);
    CHECK_EQ(t.scoring_func, std::string("sqrtsoftplus"));
    CHECK_EQ(t.topk_method, std::string("noaux_tc"));
    CHECK_CLOSE(t.routed_scaling_factor, 1.5, 1e-12);
    CHECK_CLOSE(t.swiglu_limit, 10.0, 1e-12);
    CHECK_CLOSE(t.rms_norm_eps, 1e-20, 1e-12);

    // attention: MQA over a 512 latent, low-rank Q and grouped low-rank O
    CHECK_EQ(t.num_attention_heads, 64u);
    CHECK_EQ(t.num_key_value_heads, 1u);
    CHECK_EQ(t.head_dim, 512u);
    CHECK_EQ(t.qk_rope_head_dim, 64u);
    CHECK_EQ(t.q_lora_rank, 1280u);
    CHECK_EQ(t.o_lora_rank, 1024u);
    CHECK_EQ(t.o_groups, 8u);
    CHECK_EQ(t.sliding_window, 128u);
    CHECK_EQ(t.max_position_embeddings, 1048576u);   // 1M context
    CHECK_CLOSE(t.rope_theta, 10000.0, 1e-9);
    CHECK_EQ(t.rope_scaling.type, std::string("yarn"));
    CHECK_CLOSE(t.rope_scaling.factor, 16.0, 1e-9);
    CHECK_EQ(t.rope_scaling.original_max_position, 65536u);

    // CED / compression (design §11.1)
    REQUIRE_EQ(t.kv_source_layer_ids.size(), 4u);
    CHECK_EQ(t.kv_source_layer_ids[0], 2);
    CHECK_EQ(t.kv_source_layer_ids[3], 20);
    REQUIRE_EQ(t.index_source_layer_ids.size(), 8u);
    CHECK_EQ(t.index_source_layer_ids[7], 36);
    CHECK_EQ(t.index_n_heads, 32u);
    CHECK_EQ(t.index_head_dim, 128u);
    CHECK_EQ(t.index_topk, 512u);
    CHECK_EQ(t.candidate_source_layer_id, 20u);
    CHECK_EQ(t.candidate_topk_blocks, 2048u);
    CHECK_EQ(t.candidate_block_size, 8u);

    // design §2.1: layers 0/1 window-only, 2..19 ratio 2, 20..39 ratio 1,
    // and the three mtp entries at the end are 0.
    REQUIRE_EQ(t.compress_ratios.size(), 43u);
    CHECK_EQ(t.compress_ratio(0), 0u);
    CHECK_EQ(t.compress_ratio(1), 0u);
    CHECK_EQ(t.compress_ratio(2), 2u);
    CHECK_EQ(t.compress_ratio(19), 2u);
    CHECK_EQ(t.compress_ratio(20), 1u);
    CHECK_EQ(t.compress_ratio(39), 1u);
    CHECK_EQ(t.compress_ratio(40), 0u);
    CHECK_CLOSE(t.compress_rope_theta, 160000.0, 1e-9);

    // engram (design §2.1, §7.10)
    REQUIRE_EQ(t.engram_layer_ids.size(), 2u);
    CHECK_EQ(t.engram_layer_ids[0], 1);
    CHECK_EQ(t.engram_layer_ids[1], 14);
    CHECK_EQ(t.engram_n_heads, 8u);
    CHECK_EQ(t.engram_head_dim, 256u);
    CHECK_EQ(t.engram_max_ngram_size, 4u);
    CHECK_EQ(t.engram_compressed_vocab_size, 99092u);
    CHECK_EQ(t.engram_vocab_size, 16000000ull);
    auto rows1 = t.engram_rows_for_layer(1);
    REQUIRE_OK(rows1);
    CHECK_EQ(*rows1, 384006168ull);
    auto rows14 = t.engram_rows_for_layer(14);
    REQUIRE_OK(rows14);
    CHECK_EQ(*rows14, 384016682ull);
    CHECK_ERR(t.engram_rows_for_layer(2), Err::NotFound);

    // DSpark (design §2.1, §10)
    CHECK_EQ(t.num_nextn_predict_layers, 3u);
    CHECK_EQ(t.dspark_block_size, 5u);
    CHECK_EQ(t.dspark_noise_token_id, 128799u);
    CHECK_EQ(t.dspark_markov_rank, 256u);
    CHECK_EQ(t.dspark_n_routed_experts, 128u);
    CHECK_EQ(t.dspark_num_experts_per_tok, 3u);
    REQUIRE_EQ(t.dspark_target_layer_ids.size(), 3u);
    CHECK_EQ(t.dspark_target_layer_ids[0], 37);
    CHECK_EQ(t.dspark_target_layer_ids[2], 39);

    // vision is parsed but stage one ignores it (design §1.2)
    CHECK(c->has_vision);
    CHECK_EQ(c->vision.num_hidden_layers, 32u);
    CHECK_EQ(c->vision.hidden_size, 1024u);
    CHECK_EQ(c->vision.patch_size, 14u);

    // predicates used all over the runtime
    CHECK(t.is_kv_source(20));
    CHECK(!t.is_kv_source(21));
    CHECK(t.is_index_source(36));
    CHECK(!t.is_index_source(37));
    CHECK(t.is_engram_layer(14));
    CHECK(!t.is_engram_layer(13));
}

DEEPMOE_TEST(v41_config, agrees_with_the_hard_coded_layout) {
    // model/layout.h is what the kernels are compiled against; if the
    // checkpoint ever disagrees the runtime must refuse to start.
    auto c = V41Config::load(data_path("v41_config.json"));
    REQUIRE_OK(c);
    CHECK_OK(c->validate_against_layout());
    CHECK(!c->summary().empty());

    // Corrupting one field must be caught, with the offending name reported.
    V41Config bad = *c;
    bad.text.n_routed_experts = 128;
    auto r = bad.validate_against_layout();
    CHECK(!r);
    CHECK_EQ(r.error().code, Err::FailedPrecondition);
    CHECK(r.error().message.find("n_routed_experts") != std::string::npos);

    V41Config bad2 = *c;
    bad2.quantization.expert_dtype = "int4";
    auto r2 = bad2.validate_against_layout();
    CHECK(!r2);
    CHECK(r2.error().message.find("expert_dtype") != std::string::npos);
}

DEEPMOE_TEST(v41_config, rejects_a_truncated_or_wrong_config) {
    CHECK_ERR(V41Config::parse("{}"), Err::Corrupt);                 // no quantization_config
    CHECK_ERR(V41Config::parse(R"({"quantization_config":{}})"), Err::Corrupt);  // no text_config
    // A text_config missing a required field names the field.
    auto r = V41Config::parse(R"({"quantization_config":{},"text_config":{"vocab_size":1}})");
    CHECK(!r);
    CHECK(r.error().message.find("hidden_size") != std::string::npos);
    CHECK_ERR(V41Config::load("tests/data/definitely_not_here.json"), Err::Io);
}

DEEPMOE_TEST(manifest, parses_and_validates) {
    // The schema tools/repack.py writes (design §5.1).
    const std::string json = R"({
      "version": 1,
      "model": "DeepSeek-V4.1-Flash",
      "files": {
        "hot":      {"path": "hot.bin",        "bytes": 11274289152, "sha256": "ab"},
        "experts":  {"path": "experts.bin",    "bytes": 288777830400},
        "engramL1": {"path": "engram.L1.bin",  "bytes": 101377628352}
      },
      "tensors": {
        "layers.5.attn.wq_a.weight": {
          "file": "hot", "offset": 4096, "bytes": 6553600, "dtype": "fp8_e4m3",
          "shape": [1280, 5120],
          "scale": {"offset": 6557696, "bytes": 6400, "dtype": "e8m0",
                    "shape": [40, 160], "block": [32, 32]}
        },
        "layers.5.ffn.gate.weight": {
          "file": "hot", "offset": 6565888, "bytes": 3932160, "dtype": "bf16",
          "shape": [384, 5120]
        }
      },
      "experts": {"file": "experts", "stride": 18800640, "layers": 40, "per_layer": 384, "base": 0},
      "engram":  [{"layer": 1, "file": "engramL1", "rows": 384006168, "row_bytes": 264, "base": 0}]
    })";

    auto m = Manifest::parse(json);
    REQUIRE_OK(m);
    CHECK_EQ(m->version(), 1u);
    CHECK_EQ(m->model(), std::string("DeepSeek-V4.1-Flash"));
    REQUIRE_EQ(m->files().size(), 3u);
    REQUIRE(m->file("experts") != nullptr);
    CHECK_EQ(m->file("experts")->path, std::string("experts.bin"));
    CHECK(m->file("nope") == nullptr);

    auto t = m->require_tensor("layers.5.attn.wq_a.weight");
    REQUIRE_OK(t);
    CHECK_EQ((*t)->dtype, QuantType::Fp8E4M3);
    CHECK_EQ((*t)->offset, 4096u);
    CHECK_EQ((*t)->bytes, 6553600u);
    REQUIRE_EQ((*t)->shape.size(), 2u);
    CHECK_EQ((*t)->shape[0], 1280u);
    CHECK_EQ((*t)->elements(), 1280ull * 5120ull);
    CHECK((*t)->scale.present());
    CHECK_EQ((*t)->scale.dtype, QuantType::E8M0);
    CHECK_EQ((*t)->scale.block_m, 32u);
    CHECK_EQ((*t)->scale.block_k, 32u);

    auto g = m->require_tensor("layers.5.ffn.gate.weight");
    REQUIRE_OK(g);
    CHECK_EQ((*g)->dtype, QuantType::Bf16);
    CHECK(!(*g)->scale.present());
    CHECK_ERR(m->require_tensor("nope"), Err::NotFound);

    // Arithmetic expert addressing (design §5.1).
    CHECK_EQ(m->experts().stride, layout::kExpertBytes);
    auto off = m->expert_offset(ExpertKey{0, 0});
    REQUIRE_OK(off);
    CHECK_EQ(*off, 0u);
    auto off2 = m->expert_offset(ExpertKey{1, 3});
    REQUIRE_OK(off2);
    CHECK_EQ(*off2, layout::expert_file_offset(1, 3));
    CHECK_ERR(m->expert_offset(ExpertKey{40, 0}), Err::OutOfRange);
    CHECK_ERR(m->expert_offset(ExpertKey{0, 384}), Err::OutOfRange);

    REQUIRE_EQ(m->engram().size(), 1u);
    REQUIRE(m->engram_for_layer(1) != nullptr);
    CHECK_EQ(m->engram_for_layer(1)->row_bytes, 264u);
    CHECK(m->engram_for_layer(14) == nullptr);

    CHECK_OK(m->validate());
    CHECK(m->total_bytes() > layout::kRoutedExpertTotalBytes);
}

DEEPMOE_TEST(manifest, catches_inconsistencies) {
    // A tensor pointing past the end of its file.
    auto over = Manifest::parse(R"({
      "version": 1, "files": {"hot": {"path": "hot.bin", "bytes": 4096}},
      "tensors": {"t": {"file": "hot", "offset": 0, "bytes": 8192, "dtype": "bf16", "shape": [1]}}
    })");
    REQUIRE_OK(over);
    auto r = over->validate();
    CHECK(!r);
    CHECK(r.error().message.find("runs past the end") != std::string::npos);

    // A tensor whose offset is not sector-aligned breaks unbuffered I/O.
    auto mis = Manifest::parse(R"({
      "version": 1, "files": {"hot": {"path": "hot.bin", "bytes": 1048576}},
      "tensors": {"t": {"file": "hot", "offset": 100, "bytes": 16, "dtype": "bf16", "shape": [8]}}
    })");
    REQUIRE_OK(mis);
    CHECK(mis->validate().error().message.find("4 KiB aligned") != std::string::npos);

    // A wrong expert stride would silently misaddress every expert.
    auto stride = Manifest::parse(R"({
      "version": 1, "files": {"e": {"path": "e.bin", "bytes": 1048576}}, "tensors": {},
      "experts": {"file": "e", "stride": 1024, "layers": 1, "per_layer": 1}
    })");
    REQUIRE_OK(stride);
    CHECK(stride->validate().error().message.find("kExpertBytes") != std::string::npos);

    // Unknown file reference.
    auto unknown = Manifest::parse(R"({
      "version": 1, "files": {"hot": {"path": "hot.bin", "bytes": 1048576}},
      "tensors": {"t": {"file": "cold", "offset": 0, "bytes": 16, "dtype": "bf16", "shape": [8]}}
    })");
    REQUIRE_OK(unknown);
    CHECK(unknown->validate().error().message.find("unknown file") != std::string::npos);

    // Version and structure.
    CHECK_ERR(Manifest::parse(R"({"version": 2})"), Err::Corrupt);
    CHECK_ERR(Manifest::parse(R"({"version": 1})"), Err::Corrupt);          // no files
    CHECK_ERR(Manifest::parse(R"({"version": 1, "files": {}})"), Err::Corrupt);  // no tensors
    CHECK_ERR(Manifest::load("tests/data/no_such_manifest.json"), Err::Io);
}

DEEPMOE_TEST(manifest, dtype_names_round_trip) {
    CHECK_EQ(quant_from_string("fp4_e2m1"), QuantType::Fp4E2M1);
    CHECK_EQ(quant_from_string("fp4"), QuantType::Fp4E2M1);
    CHECK_EQ(quant_from_string("fp8_e4m3"), QuantType::Fp8E4M3);
    CHECK_EQ(quant_from_string("ue8m0"), QuantType::E8M0);
    CHECK_EQ(quant_from_string("bfloat16"), QuantType::Bf16);
    CHECK_EQ(quant_from_string("nonsense"), QuantType::Unknown);
    CHECK_EQ(std::string(quant_to_string(QuantType::Fp4E2M1)), std::string("fp4_e2m1"));
    CHECK_EQ(quant_bits(QuantType::Fp4E2M1), 4u);
    CHECK_EQ(quant_bits(QuantType::Fp8E4M3), 8u);
    CHECK_EQ(quant_bits(QuantType::Bf16), 16u);
}

DEEPMOE_TEST(block_info, derives_layer_roles_from_the_config) {
    // Cross-check runtime/block.h against the real config.
    auto c = V41Config::load(data_path("v41_config.json"));
    REQUIRE_OK(c);
    const TextConfig& t = c->text;

    auto b0 = runtime::BlockInfo::derive(t, 0);
    CHECK(b0.is_encoder);
    CHECK(!b0.has_engram);
    CHECK_EQ(b0.attn.compress_ratio, 0u);

    auto b1 = runtime::BlockInfo::derive(t, 1);
    CHECK(b1.has_engram);

    auto b2 = runtime::BlockInfo::derive(t, 2);
    CHECK(b2.attn.kv_source);
    CHECK(b2.attn.index_source);
    CHECK_EQ(b2.attn.compress_ratio, 2u);

    auto b20 = runtime::BlockInfo::derive(t, 20);
    CHECK(!b20.is_encoder);
    CHECK(b20.attn.kv_source);
    CHECK(b20.attn.candidate_source);
    CHECK_EQ(b20.attn.compress_ratio, 1u);

    auto b40 = runtime::BlockInfo::derive(t, 40);
    CHECK(b40.is_mtp);
    CHECK(!b40.is_encoder);
    CHECK(!b40.has_engram);
    CHECK_EQ(b40.attn.compress_ratio, 0u);   // mtp blocks are window only
}

DEEPMOE_TEST(kvcache, geometry_matches_the_design_budget) {
    auto c = V41Config::load(data_path("v41_config.json"));
    REQUIRE_OK(c);
    runtime::KvGeometry g;
    g.max_context = 65536;
    // design §11.3: window 2.8 MB, compressed 84 MB fp8, indexer ~2 MB.
    CHECK_CLOSE(g.window_bytes() / 1e6, 2.9, 0.1);
    CHECK_CLOSE(g.compressed_bytes(c->text) / 1e6, 84.0, 0.05);
    // design 11.3 (corrected in v0.4): 8 index sources, five of them at
    // ratio 1, so 28.97 MB -- not the ~2 MB that table used to claim.
    CHECK_CLOSE(g.indexer_bytes(c->text) / 1e6, 28.97, 0.01);
    g.compressed_fp4 = true;
    CHECK_CLOSE(g.compressed_bytes(c->text) / 1e6, 48.2, 0.05);
    // "KV is not a memory problem": well under a GB at 64K.
    CHECK(g.total_bytes(c->text) < (1ull << 30));
    CHECK_CLOSE(g.total_bytes(c->text) / 1e6, 80.4, 0.02);   // fp4 compressed
    CHECK(!g.to_string(c->text).empty());
}
