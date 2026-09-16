// model/: parsing the real DeepSeek-V4.1-Flash config.json (a verbatim copy
// from ModelScope lives in tests/data/) and the manifest reader.
//
// This is the test that fails loudly if the checkpoint ever changes shape: every
// number in design §2.1 is asserted against the file, not against a constant
// copied into the test.
#include <algorithm>
#include <string>
#include <vector>

#include "core/align.h"
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

DEEPMOE_TEST(manifest, parses_schema_v2) {
    // The shape tools/manifest.py writes (design §5.1 v0.5): an address book
    // over the original shards, with sector-aligned runs and per-part skews.
    const std::string json = R"({
      "version": 2,
      "model": "DeepSeek-V4.1-Flash",
      "alignment": 4096,
      "expert_slot_bytes": 18808832,
      "expert_parts": ["w1.weight","w1.scale","w2.weight","w2.scale","w3.weight","w3.scale"],
      "files": [
        {"path": "model-00001-of-00048.safetensors", "bytes": 970533624, "data_start": 4184},
        {"path": "model-00003-of-00048.safetensors", "bytes": 7389759032, "data_start": 3800,
         "sha256": "ab"}
      ],
      "tensors": {
        "layers.0.attn.wq_a.weight": {
          "file": 1, "offset": 510842328, "bytes": 6553600, "dtype": "fp8_e4m3",
          "shape": [1280, 5120],
          "scale": {"file": 1, "offset": 7968216, "bytes": 6400, "dtype": "e8m0",
                    "shape": [40, 160], "block": [32, 32]}
        },
        "layers.0.ffn.gate.weight": {
          "file": 1, "offset": 3949528, "bytes": 3932160, "dtype": "bf16",
          "shape": [384, 5120]
        }
      },
      "experts": [
        {"layer": 0, "name": "layers.0", "experts": [
          [ {"file": 1, "aligned_off": 8269824, "aligned_bytes": 1110016, "slot_offset": 0,
             "parts": [
               {"tensor": "w1.scale", "skew": 3896,   "bytes": 368640, "slot_offset": 3896},
               {"tensor": "w2.scale", "skew": 372536, "bytes": 368640, "slot_offset": 372536},
               {"tensor": "w3.scale", "skew": 741176, "bytes": 368640, "slot_offset": 741176}]},
            {"file": 1, "aligned_off": 594984960, "aligned_bytes": 17698816,
             "slot_offset": 1110016,
             "parts": [
               {"tensor": "w1.weight", "skew": 1592,     "bytes": 5898240, "slot_offset": 1111608},
               {"tensor": "w2.weight", "skew": 5899832,  "bytes": 5898240, "slot_offset": 7009848},
               {"tensor": "w3.weight", "skew": 11798072, "bytes": 5898240, "slot_offset": 12908088}]}
          ]
        ]}
      ],
      "engram": [
        {"layer": 1, "rows": 1000,
         "value": {"file": 0, "offset": 664, "bytes": 256000, "row_bytes": 256,
                   "dtype": "fp8_e4m3"},
         "scale": {"file": 0, "offset": 300000, "bytes": 8000, "row_bytes": 8,
                   "dtype": "e8m0"}}
      ]
    })";

    auto m = Manifest::parse(json);
    REQUIRE_OK(m);
    CHECK_EQ(m->version(), 2u);
    CHECK_EQ(m->model(), std::string("DeepSeek-V4.1-Flash"));
    CHECK_EQ(m->alignment(), 4096u);
    CHECK_EQ(m->expert_slot_bytes(), layout::kExpertSlotBytes);

    // files are an ordered array; the index is the id every entry references.
    REQUIRE_EQ(m->files().size(), 2u);
    REQUIRE(m->file(1) != nullptr);
    CHECK_EQ(m->file(1)->path, std::string("model-00003-of-00048.safetensors"));
    CHECK_EQ(m->file(1)->data_start, 3800u);
    CHECK(m->file(2) == nullptr);
    CHECK_EQ(m->file_index("model-00001-of-00048.safetensors").value_or(99), 0u);
    CHECK(!m->file_index("nope").has_value());

    auto t = m->require_tensor("layers.0.attn.wq_a.weight");
    REQUIRE_OK(t);
    CHECK_EQ((*t)->dtype, QuantType::Fp8E4M3);
    CHECK_EQ((*t)->file, 1u);
    CHECK_EQ((*t)->offset, 510842328u);
    CHECK_EQ((*t)->elements(), 1280ull * 5120ull);
    CHECK((*t)->scale.present());
    CHECK_EQ((*t)->scale.block_m, 32u);
    CHECK_EQ((*t)->scale.block_k, 32u);
    CHECK(!(*m->require_tensor("layers.0.ffn.gate.weight"))->scale.present());
    CHECK_ERR(m->require_tensor("nope"), Err::NotFound);

    // Reading a tensor means one widened, sector-aligned read plus a skew. The
    // offsets in the shards are multiples of 8, never of 4096, which is the
    // whole reason this schema exists.
    CHECK(!is_aligned((*t)->offset));
    auto rd = m->tensor_read("layers.0.attn.wq_a.weight");
    REQUIRE_OK(rd);
    CHECK(is_aligned(rd->aligned_off));
    CHECK(is_aligned(rd->aligned_bytes));
    CHECK_EQ(rd->aligned_off + rd->skew, (*t)->offset);
    CHECK(rd->skew + rd->bytes <= rd->aligned_bytes);
    CHECK_OK(m->tensor_scale_read("layers.0.attn.wq_a.weight"));
    CHECK_ERR(m->tensor_scale_read("layers.0.ffn.gate.weight"), Err::NotFound);

    // --- the expert run table --------------------------------------------
    CHECK_EQ(m->experts_in_layer(0), 1u);
    auto e = m->require_expert(ExpertKey{0, 0});
    REQUIRE_OK(e);
    const ExpertEntry& x = **e;
    REQUIRE_EQ(x.runs.size(), 2u);
    CHECK_EQ(x.slot_bytes, 1110016ull + 17698816ull);
    CHECK_EQ(x.slot_bytes, layout::kExpertSlotBytes);

    // Every run is a legal unbuffered read and the runs tile the slot.
    uint64_t cursor = 0, payload = 0;
    for (const Run& r : x.runs) {
        CHECK(is_aligned(r.aligned_off));
        CHECK(is_aligned(r.aligned_bytes));
        CHECK_EQ(r.slot_offset, cursor);
        cursor += r.aligned_bytes;
        for (const RunPart& p : r.parts) {
            CHECK_EQ(p.slot_offset, r.slot_offset + p.skew);
            CHECK(p.skew + p.bytes <= r.aligned_bytes);
            payload += p.bytes;
        }
    }
    CHECK_EQ(payload, layout::kExpertBytes);

    // The six parts land where the pointer table will point.
    CHECK_EQ(x.offset_of(ExpertPart::W1Scale), 3896ull);
    CHECK_EQ(x.offset_of(ExpertPart::W3Scale), 741176ull);
    CHECK_EQ(x.offset_of(ExpertPart::W1Weight), 1111608ull);
    CHECK_EQ(x.offset_of(ExpertPart::W3Weight), 12908088ull);
    CHECK_EQ(x.bytes_of(ExpertPart::W1Weight), layout::kExpertWeightBytesPerMat);
    CHECK_EQ(x.bytes_of(ExpertPart::W2Scale), layout::kExpertScaleBytesPerMat);

    CHECK_ERR(m->require_expert(ExpertKey{0, 1}), Err::OutOfRange);
    CHECK_ERR(m->require_expert(ExpertKey{7, 0}), Err::OutOfRange);

    // --- engram rows ------------------------------------------------------
    REQUIRE_EQ(m->engram().size(), 1u);
    REQUIRE(m->engram_for_layer(1) != nullptr);
    CHECK_EQ(m->engram_for_layer(1)->value.row_bytes, layout::kEngramValueRowBytes);
    CHECK_EQ(m->engram_for_layer(1)->scale.row_bytes, layout::kEngramScaleRowBytes);
    CHECK(m->engram_for_layer(14) == nullptr);

    auto row = m->engram_row(1, 17);
    REQUIRE_OK(row);
    CHECK(is_aligned(row->value.aligned_off));
    CHECK(is_aligned(row->scale.aligned_off));
    CHECK_EQ(row->value.aligned_off + row->value.skew, 664ull + 17ull * 256);
    CHECK_EQ(row->scale.aligned_off + row->scale.skew, 300000ull + 17ull * 8);
    CHECK_EQ(row->value.bytes, 256ull);
    CHECK_ERR(m->engram_row(1, 1000), Err::OutOfRange);
    CHECK_ERR(m->engram_row(2, 0), Err::NotFound);

    CHECK_OK(m->validate());
    CHECK_EQ(m->total_bytes(), 6553600ull + 6400 + 3932160 + layout::kExpertBytes);
    CHECK_EQ(m->file_bytes(), 970533624ull + 7389759032ull);
}

DEEPMOE_TEST(manifest, parses_a_slice_of_the_real_checkpoint) {
    // tests/data/manifest_v2_slice.json is a verbatim cut of the manifest
    // tools/manifest.py wrote for the real 48-shard D:\models checkpoint: all 48
    // file entries, a handful of tensors, three experts each from layers 0, 39
    // and 40 (mtp.0), and both engram tables. The point is that the schema is
    // pinned to bytes that actually exist, not to a hand-written fixture.
    auto m = Manifest::load(data_path("manifest_v2_slice.json"));
    REQUIRE_OK(m);
    CHECK_EQ(m->version(), 2u);
    REQUIRE_EQ(m->files().size(), 48u);
    CHECK_EQ(m->files()[0].path, std::string("model-00001-of-00048.safetensors"));
    // design §2.2: 510.3 GB of shards on disk.
    CHECK_CLOSE(m->file_bytes() / 1e9, 510.3, 0.001);

    // Not one shard's data section is sector-aligned, which is the premise of
    // the whole runs-and-skews design.
    uint32_t aligned_starts = 0;
    for (const FileEntry& f : m->files())
        if (is_aligned(f.data_start)) ++aligned_starts;
    CHECK_EQ(aligned_starts, 0u);

    // Every expert in the slice is two runs -- one weights run, one scales run
    // -- and no expert needs more than one slot.
    uint64_t widest = 0;
    uint32_t experts = 0;
    for (uint32_t layer : {0u, 39u, 40u}) {
        REQUIRE_EQ(m->experts_in_layer(layer), 3u);
        for (uint16_t id = 0; id < 3; ++id) {
            auto e = m->require_expert(ExpertKey{static_cast<uint16_t>(layer), id});
            REQUIRE_OK(e);
            CHECK_EQ((*e)->runs.size(), 2u);
            CHECK_EQ((*e)->runs[0].aligned_bytes, 1110016ull);    // the three scales
            CHECK_EQ((*e)->runs[1].aligned_bytes, 17698816ull);   // the three weights
            CHECK_EQ((*e)->runs[0].file, (*e)->runs[1].file);     // and one shard holds both
            widest = std::max(widest, (*e)->slot_bytes);
            ++experts;
        }
    }
    CHECK_EQ(experts, 9u);
    // model/layout.h's compile-time slot size must be what the real manifest
    // needs; tools/manifest.py prints the maximum over all 15,744 experts.
    CHECK_EQ(widest, layout::kExpertSlotBytes);
    CHECK_EQ(m->expert_slot_bytes(), layout::kExpertSlotBytes);

    // The engram tables are two planes, not the interleaved 264 B rows the
    // pre-v0.5 repack produced.
    REQUIRE_EQ(m->engram().size(), 2u);
    for (uint32_t layer : {1u, 14u}) {
        const EngramEntry* e = m->engram_for_layer(layer);
        REQUIRE(e != nullptr);
        CHECK(e->rows > 384'000'000ull);
        CHECK_EQ(e->value.file, e->scale.file);
        auto row = m->engram_row(layer, e->rows - 1);
        REQUIRE_OK(row);
        // One 4 KiB read covers 16 value rows, another covers 512 scale rows.
        CHECK_EQ(row->value.aligned_bytes, 4096ull);
        CHECK_EQ(row->scale.aligned_bytes, 4096ull);
    }

    CHECK_OK(m->validate());
}

DEEPMOE_TEST(manifest, catches_inconsistencies) {
    auto base = [](std::string_view experts, std::string_view tensors = R"("t": {
            "file": 0, "offset": 0, "bytes": 16, "dtype": "bf16", "shape": [8]})") {
        return std::string(R"({"version": 2, "alignment": 4096,
          "expert_slot_bytes": 18808832,
          "files": [{"path": "a.safetensors", "bytes": 1048576, "data_start": 100}],
          "tensors": {)") + std::string(tensors) + "}" +
          (experts.empty() ? std::string() : ", \"experts\": " + std::string(experts)) + "}";
    };

    // A tensor pointing past the end of its shard.
    auto over = Manifest::parse(base("", R"("t": {
        "file": 0, "offset": 1048568, "bytes": 8192, "dtype": "bf16", "shape": [4096]})"));
    REQUIRE_OK(over);
    CHECK(over->validate().error().message.find("runs past the end") != std::string::npos);

    // A tensor in a shard that does not exist.
    auto unknown = Manifest::parse(base("", R"("t": {
        "file": 3, "offset": 0, "bytes": 16, "dtype": "bf16", "shape": [8]})"));
    REQUIRE_OK(unknown);
    CHECK(unknown->validate().error().message.find("only 1 are listed") != std::string::npos);

    // A run that is not sector-aligned would be illegal for unbuffered I/O --
    // this is the invariant that moved from the I/O layer into the manifest.
    auto misaligned = Manifest::parse(base(R"([{"layer":0,"experts":[[
        {"file":0,"aligned_off":100,"aligned_bytes":4096,"slot_offset":0,
         "parts":[{"tensor":"w1.weight","skew":0,"bytes":16,"slot_offset":0}]}]]}])"));
    REQUIRE_OK(misaligned);
    CHECK(misaligned->validate().error().message.find("not 4 KiB aligned") != std::string::npos);

    // Runs must tile the slot with no gap, or the parts' slot offsets lie.
    auto gapped = Manifest::parse(base(R"([{"layer":0,"experts":[[
        {"file":0,"aligned_off":0,"aligned_bytes":4096,"slot_offset":0,
         "parts":[{"tensor":"w1.weight","skew":0,"bytes":16,"slot_offset":0}]},
        {"file":0,"aligned_off":8192,"aligned_bytes":4096,"slot_offset":8192,
         "parts":[{"tensor":"w1.scale","skew":0,"bytes":16,"slot_offset":8192}]}]]}])"));
    REQUIRE_OK(gapped);
    CHECK(gapped->validate().error().message.find("running total") != std::string::npos);

    // An expert whose parts do not add up to 18,800,640 B would silently
    // misaddress every GEMV that reads it.
    auto shortfall = Manifest::parse(base(R"([{"layer":0,"experts":[[
        {"file":0,"aligned_off":0,"aligned_bytes":4096,"slot_offset":0,
         "parts":[{"tensor":"w1.weight","skew":0,"bytes":16,"slot_offset":0},
                  {"tensor":"w1.scale","skew":16,"bytes":16,"slot_offset":16},
                  {"tensor":"w2.weight","skew":32,"bytes":16,"slot_offset":32},
                  {"tensor":"w2.scale","skew":48,"bytes":16,"slot_offset":48},
                  {"tensor":"w3.weight","skew":64,"bytes":16,"slot_offset":64},
                  {"tensor":"w3.scale","skew":80,"bytes":16,"slot_offset":80}]}]]}])"));
    REQUIRE_OK(shortfall);
    CHECK(shortfall->validate().error().message.find("kExpertBytes") != std::string::npos);

    // A slot size that disagrees with model/layout.h means the C++ constant is
    // stale; tools/manifest.py prints the value to paste in.
    auto slot = Manifest::parse(R"({"version": 2, "expert_slot_bytes": 4096,
      "files": [{"path": "a", "bytes": 4096}], "tensors": {}})");
    REQUIRE_OK(slot);
    CHECK(slot->validate().error().message.find("kExpertSlotBytes") != std::string::npos);

    // An unknown part name is a schema mismatch, not something to guess at.
    CHECK_ERR(Manifest::parse(base(R"([{"layer":0,"experts":[[
        {"file":0,"aligned_off":0,"aligned_bytes":4096,"slot_offset":0,
         "parts":[{"tensor":"w4.weight","skew":0,"bytes":16,"slot_offset":0}]}]]}])")),
              Err::Corrupt);

    // Version and structure.
    CHECK_ERR(Manifest::parse(R"({"version": 1})"), Err::Corrupt);   // the repack schema
    CHECK_ERR(Manifest::parse(R"({"version": 2})"), Err::Corrupt);   // no files
    CHECK_ERR(Manifest::parse(R"({"version": 2, "files": []})"), Err::Corrupt);
    CHECK_ERR(Manifest::parse(R"({"version": 2, "files": [{"path":"a"}]})"), Err::Corrupt);
    CHECK_ERR(Manifest::load(data_path("no_such_manifest.json")), Err::Io);
}

DEEPMOE_TEST(manifest, expert_part_names_round_trip) {
    for (uint8_t i = 0; i < kExpertPartCount; ++i) {
        const auto p = static_cast<ExpertPart>(i);
        auto back = expert_part_from_string(expert_part_name(p));
        REQUIRE(back.has_value());
        CHECK_EQ(*back, p);
    }
    CHECK_EQ(std::string(expert_part_name(ExpertPart::W2Scale)), std::string("w2.scale"));
    CHECK(!expert_part_from_string("w1").has_value());
    CHECK(!expert_part_from_string("").has_value());
}

DEEPMOE_TEST(manifest, align_read_widens_to_sectors) {
    // The arithmetic every run in the manifest is built from.
    auto r = align_read(3, 4100, 8);
    CHECK_EQ(r.file, 3u);
    CHECK_EQ(r.aligned_off, 4096ull);
    CHECK_EQ(r.aligned_bytes, 4096ull);
    CHECK_EQ(r.skew, 4u);
    CHECK_EQ(r.bytes, 8ull);

    // A payload that straddles a sector boundary costs two sectors.
    auto s = align_read(0, 4094, 8);
    CHECK_EQ(s.aligned_off, 0ull);
    CHECK_EQ(s.aligned_bytes, 8192ull);
    CHECK_EQ(s.skew, 4094u);

    // An already-aligned payload costs nothing extra.
    auto t = align_read(0, 8192, 4096);
    CHECK_EQ(t.aligned_off, 8192ull);
    CHECK_EQ(t.aligned_bytes, 4096ull);
    CHECK_EQ(t.skew, 0u);
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
    // Track R2 (docs/p4_kv_ux.md 搂1): the key caches are the four kv sources'
    // (model.py Indexer.owns_k), 2.5 rows a token x 68 B -- not design 11.3
    // v0.4's eight index sources (28.97 MB).
    CHECK_CLOSE(g.indexer_bytes(c->text) / 1e6, 11.14, 0.01);
    g.compressed_fp4 = true;
    CHECK_CLOSE(g.compressed_bytes(c->text) / 1e6, 48.2, 0.05);
    // "KV is not a memory problem": well under a GB at 64K.
    CHECK(g.total_bytes(c->text) < (1ull << 30));
    CHECK_CLOSE(g.total_bytes(c->text) / 1e6, 61.76, 0.02);   // fp4 compressed
    CHECK(!g.to_string(c->text).empty());
}
