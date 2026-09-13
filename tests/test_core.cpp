// core/: alignment helpers, the JSON reader, the profiler record, and the
// layout constants the whole design is built on.
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "core/align.h"
#include "core/bytes.h"
#include "core/json.h"
#include "core/profiler.h"
#include "core/status.h"
#include "core/types.h"
#include "model/layout.h"
#include "tests/test_framework.h"

using namespace deepmoe;

DEEPMOE_TEST(align, round_up_and_down) {
    CHECK_EQ(align_up(0), 0u);
    CHECK_EQ(align_up(1), 4096u);
    CHECK_EQ(align_up(4096), 4096u);
    CHECK_EQ(align_up(4097), 8192u);
    CHECK_EQ(align_down(4097), 4096u);
    CHECK_EQ(align_down(4095), 0u);
    CHECK_EQ(align_up(100, 64), 128u);
    CHECK_EQ(align_down(100, 64), 64u);
    CHECK(is_aligned(8192));
    CHECK(!is_aligned(8193));
    CHECK(is_pow2(4096));
    CHECK(!is_pow2(4097));
}

DEEPMOE_TEST(align, page_span_counts_straddling_pages) {
    CHECK_EQ(page_span(0, 0), 0u);
    CHECK_EQ(page_span(0, 1), 1u);
    CHECK_EQ(page_span(0, 4096), 1u);
    CHECK_EQ(page_span(0, 4097), 2u);
    CHECK_EQ(page_span(4095, 2), 2u);        // straddles the boundary
    // design §5.1: an engram row is 264 B, so a 4 KiB read always covers it.
    CHECK_EQ(page_span(0, layout::kEngramRowBytes), 1u);
}

DEEPMOE_TEST(align, aligned_buffer_is_page_aligned_and_moves) {
    AlignedBuffer b(10000);
    REQUIRE(static_cast<bool>(b));
    CHECK(is_aligned(b.data()));
    CHECK_EQ(b.size(), 12288u);              // rounded up to 3 pages
    std::byte* p = b.data();
    AlignedBuffer c = std::move(b);
    CHECK_EQ(c.data(), p);
    CHECK(!static_cast<bool>(b));
    AlignedBuffer empty(0);
    CHECK(!static_cast<bool>(empty));
}

DEEPMOE_TEST(layout, constants_match_the_design_document) {
    namespace L = layout;
    // design §2.3: one routed expert is exactly 4590 x 4 KiB.
    CHECK_EQ(L::kExpertBytes, 18800640ull);
    CHECK_EQ(L::kExpertSectors, 4590ull);
    CHECK_EQ(L::kExpertBytes % 4096, 0ull);
    CHECK_EQ(L::kEngramRowBytes, 264ull);
    CHECK_EQ(L::kRoutedExpertCount, 15360ull);           // 40 x 384
    CHECK_EQ(L::kExpertsPerLayerBytes, 384ull * 18800640ull);
    CHECK_EQ(L::kHcFnCols, 20480u);                      // 4 x 5120
    CHECK_EQ(L::kTotalLogicalLayers, 43u);               // 40 + 3 mtp
    CHECK_EQ(L::kEngramWkvCols, L::kEngramRowsPerToken * L::kEngramHeadDim);

    // Arithmetic addressing (design §5.1).
    CHECK_EQ(L::expert_file_offset(0, 0), 0ull);
    CHECK_EQ(L::expert_file_offset(0, 1), L::kExpertBytes);
    CHECK_EQ(L::expert_file_offset(1, 0), L::kExpertsPerLayerBytes);
    CHECK_EQ(L::expert_file_offset(39, 383),
             39ull * L::kExpertsPerLayerBytes + 383ull * L::kExpertBytes);
    // 288.8 GB of routed experts, per design §2.2.
    CHECK_CLOSE(L::kRoutedExpertTotalBytes / 1e9, 288.8, 0.01);
}

DEEPMOE_TEST(types, expert_key_packs_and_orders) {
    ExpertKey a{5, 300}, b{5, 301}, c{6, 0};
    CHECK(a < b);
    CHECK(b < c);
    CHECK_EQ(ExpertKey::unpack(a.packed()), a);
    CHECK_EQ(a.packed(), (5u << 16) | 300u);
    ExpertKeyHash h;
    CHECK(h(a) != h(b));
    CHECK(h(a) != h(c));
}

DEEPMOE_TEST(json, parses_scalars_and_containers) {
    auto v = json_parse(R"({"a":1,"b":-2.5,"c":"hi","d":true,"e":null,"f":[1,2,3],"g":{"h":4}})");
    REQUIRE_OK(v);
    CHECK_EQ(v->int_or("a", -1), 1);
    CHECK_CLOSE(v->double_or("b", 0), -2.5, 1e-12);
    CHECK_EQ(v->string_or("c", ""), std::string("hi"));
    CHECK(v->bool_or("d", false));
    CHECK(v->find("e")->is_null());
    auto arr = v->int_array_at("f");
    REQUIRE_OK(arr);
    REQUIRE_EQ(arr->size(), 3u);
    CHECK_EQ((*arr)[2], 3);
    CHECK_EQ(v->find("g")->int_or("h", 0), 4);
    CHECK_EQ(v->size(), 7u);
}

DEEPMOE_TEST(json, keeps_large_integers_exact) {
    // Manifest byte offsets exceed 2^53; going through a double would lose them.
    auto v = json_parse(R"({"offset":9007199254740993,"bytes":18800640})");
    REQUIRE_OK(v);
    auto off = v->uint_at("offset");
    REQUIRE_OK(off);
    CHECK_EQ(*off, 9007199254740993ull);
    CHECK_EQ(v->uint_at("bytes").value_or(0), 18800640ull);
}

DEEPMOE_TEST(json, handles_escapes_and_unicode) {
    // Written with explicit backslashes (not a raw string) so the \u escapes
    // reach the parser as escapes rather than as literal UTF-8.
    const char* doc = "{\"s\":\"a\\\"b\\\\c\\nd\\u00e9\\ud83d\\ude00\"}";
    auto v = json_parse(doc);
    REQUIRE_OK(v);
    const std::string s = v->string_or("s", "");
    CHECK_EQ(s.substr(0, 6), std::string("a\"b\\c\n"));
    CHECK(s.find("\xc3\xa9") != std::string::npos);          // \u00e9 -> U+00E9
    CHECK(s.find("\xf0\x9f\x98\x80") != std::string::npos);  // surrogate pair -> U+1F600
    CHECK_EQ(s.size(), 7u + 2u + 4u);   // 7 ascii + a 2-byte and a 4-byte code point
    // A lone high surrogate is emitted as-is rather than dropped.
    auto lone = json_parse("{\"s\":\"\\ud83d\"}");
    CHECK_OK(lone);
}

DEEPMOE_TEST(json, rejects_malformed_input) {
    CHECK_ERR(json_parse("{"), Err::Corrupt);
    CHECK_ERR(json_parse("{\"a\":}"), Err::Corrupt);
    CHECK_ERR(json_parse("{\"a\":1,}"), Err::Corrupt);      // no trailing commas
    CHECK_ERR(json_parse("[1,2"), Err::Corrupt);
    CHECK_ERR(json_parse("{} garbage"), Err::Corrupt);
    CHECK_ERR(json_parse("tru"), Err::Corrupt);
    CHECK_ERR(json_parse(""), Err::Corrupt);
    // Deep nesting must be refused rather than blowing the stack.
    std::string deep(200, '[');
    CHECK_ERR(json_parse(deep), Err::Corrupt);
}

DEEPMOE_TEST(json, missing_and_mistyped_keys) {
    auto v = json_parse(R"({"a":1,"s":"x"})");
    REQUIRE_OK(v);
    CHECK_ERR(v->int_at("nope"), Err::NotFound);
    CHECK_ERR(v->int_at("s"), Err::Corrupt);
    CHECK_EQ(v->int_or("nope", 7), 7);
    CHECK_EQ(v->string_or("nope", "dflt"), std::string("dflt"));
}

DEEPMOE_TEST(json, tolerates_a_utf8_bom) {
    auto v = json_parse("\xEF\xBB\xBF{\"a\":1}");
    REQUIRE_OK(v);
    CHECK_EQ(v->int_or("a", 0), 1);
}

DEEPMOE_TEST(profiler, token_record_accumulates_and_serialises) {
    Profiler p;
    p.set_enabled(false);                    // no sink, in-memory summary only
    p.token_begin(7);
    p.add_phase(Phase::HotGemv, Nanos(2'000'000));
    p.add_phase(Phase::NvmeStall, Nanos(5'000'000));
    p.note_expert_lookup(true);
    p.note_expert_lookup(true);
    p.note_expert_lookup(false);
    p.note_hot_bytes(1000);
    p.note_miss_bytes(18800640);
    p.note_nvme_busy(Nanos(4'000'000));
    p.note_prefetch_issued(4);
    p.note_prefetch_used(1);
    const TokenRecord r = p.token_end();

    CHECK_EQ(r.token, 7u);
    CHECK_EQ(r.expert_requests, 3u);
    CHECK_EQ(r.expert_hits, 2u);
    CHECK_EQ(r.expert_misses, 1u);
    CHECK_CLOSE(r.hit_rate(), 2.0 / 3.0, 1e-9);
    CHECK_CLOSE(r.prefetch_precision(), 0.25, 1e-9);
    CHECK_EQ(r.phase_ns[static_cast<size_t>(Phase::HotGemv)], 2'000'000u);
    CHECK_EQ(r.phase_ns[static_cast<size_t>(Phase::NvmeStall)], 5'000'000u);
    // 18.8 MB over 4 ms is ~4.7 GB/s, the design §9.8 "effective NVMe GB/s".
    CHECK_CLOSE(r.nvme_gbps(), 18800640.0 / 4'000'000.0, 1e-9);

    const std::string line = r.to_jsonl();
    CHECK(line.find("\"token\":7") != std::string::npos);
    CHECK(line.find("\"nvme_stall_ms\"") != std::string::npos);
    CHECK(line.find("\"hit_rate\"") != std::string::npos);
    auto parsed = json_parse(line);
    REQUIRE_OK(parsed);                       // the JSONL must be real JSON
    CHECK_EQ(parsed->int_or("expert_hits", -1), 2);

    // A second token must start from zero but keep the running summary.
    p.token_begin(8);
    const TokenRecord r2 = p.token_end();
    CHECK_EQ(r2.expert_requests, 0u);
    CHECK_EQ(p.summary().tokens, 2u);
    CHECK_EQ(p.summary().expert_hits, 2u);
}

DEEPMOE_TEST(profiler, scoped_phase_records_time) {
    Profiler p;
    p.set_enabled(false);
    p.token_begin(0);
    {
        ScopedPhase s(p, Phase::Dispatch);
        // steady_clock is ~100 ns granular on Windows, so an empty scope can
        // legitimately measure zero; sleep past the tick.
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    { ScopedPhase s(p, Phase::CpuSync); s.dismiss(); }   // dismissed: records nothing
    const TokenRecord r = p.token_end();
    CHECK(r.phase_ns[static_cast<size_t>(Phase::Dispatch)] >= 1'000'000u);
    CHECK_EQ(r.phase_ns[static_cast<size_t>(Phase::CpuSync)], 0u);
}

DEEPMOE_TEST(status, error_formatting) {
    Status s{Err::Io, "ReadFile", 5};
    CHECK(s.str().find("io") != std::string::npos);
    CHECK(s.str().find("ReadFile") != std::string::npos);
    CHECK(s.str().find("os 5") != std::string::npos);
    Result<int> r = fail(Err::NotFound, "gone");
    CHECK(!r);
    CHECK_EQ(r.error().code, Err::NotFound);
    Result<int> ok = 42;
    REQUIRE_OK(ok);
    CHECK_EQ(*ok, 42);
}

DEEPMOE_TEST(bytes, human_and_readers) {
    CHECK_EQ(human_bytes(512), std::string("512 B"));
    CHECK_EQ(human_bytes(4096), std::string("4.00 KiB"));
    CHECK_EQ(human_bytes(18800640), std::string("17.93 MiB"));
    const uint8_t raw[8] = {1, 0, 0, 0, 2, 0, 0, 0};
    auto span = as_bytes(raw, 8);
    auto a = read_le<uint32_t>(span, 0);
    REQUIRE_OK(a);
    CHECK_EQ(*a, 1u);
    auto b = read_le<uint32_t>(span, 4);
    REQUIRE_OK(b);
    CHECK_EQ(*b, 2u);
    CHECK_ERR(read_le<uint32_t>(span, 6), Err::OutOfRange);
}
