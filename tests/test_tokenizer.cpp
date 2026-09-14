// The C++ tokenizer against HF `tokenizers` (docs/p3_chat.md §3).
//
// tests/data/tokenizer/golden.json is written by tools/tokenizer_golden.py from
// HF's own encode/decode: 909 (text, ids, decode, decode_skip) cases --
// hand-written adversarial strings, every rendered encoding/tests prompt, added
// tokens alone and glued to text, seeded fuzz over emoji / CJK / combining
// marks / every Unicode space, samples of the code point sweep, and the head of
// every corpus file. That script also runs the full sets (1.5 M characters of
// corpus, 2.5 M characters covering every scalar value, 20,000 fuzz strings)
// through `deepmoe tokenize` and requires 100% id equality; this test is the
// committed, model-dir-only subset of it.
//
// Needs tokenizer.json from DEEPMOE_MODEL_DIR (it is not copied into the repo).
#include <cstdio>
#include <cstdlib>
#include <format>
#include <string>
#include <vector>

#include "core/json.h"
#include "tests/test_framework.h"
#include "text/tokenizer.h"

#ifndef DEEPMOE_TEST_DATA_DIR
#define DEEPMOE_TEST_DATA_DIR "tests/data"
#endif

using namespace deepmoe;

namespace {

const text::Tokenizer* shared_tokenizer(std::string& why) {
    static std::string err;
    static text::Tokenizer* tok = [&]() -> text::Tokenizer* {
        const char* dir = std::getenv("DEEPMOE_MODEL_DIR");
        if (!dir) { err = "set DEEPMOE_MODEL_DIR"; return nullptr; }
        auto t = text::Tokenizer::load(std::string(dir) + "/tokenizer.json");
        if (!t) { err = t.error().str(); return nullptr; }
        return new text::Tokenizer(std::move(*t));
    }();
    why = err;
    return tok;
}

}  // namespace

DEEPMOE_TEST(tokenizer, golden_ids_and_decode_match_hf) {
    std::string why;
    const text::Tokenizer* tok = shared_tokenizer(why);
    if (!tok) { std::printf("      SKIP tokenizer: %s\n", why.c_str()); return; }
    auto doc = json_parse_file(std::string(DEEPMOE_TEST_DATA_DIR) + "/tokenizer/golden.json");
    REQUIRE(doc.has_value());
    const JsonValue* cases = doc->find("cases");
    REQUIRE(cases != nullptr && cases->is_array());
    uint32_t n = 0, bad_ids = 0, bad_dec = 0, bad_skip = 0, bad_stream = 0;
    uint64_t ids_total = 0;
    for (const JsonValue& c : **cases->as_array()) {
        const std::string text = c.string_or("text", "");
        std::vector<uint32_t> want;
        for (int64_t v : *c.int_array_at("ids")) want.push_back(static_cast<uint32_t>(v));
        const std::vector<uint32_t> got = tok->encode(text);
        ++n;
        ids_total += want.size();
        if (got != want) {
            if (bad_ids++ < 3)
                _ctx.fail(__FILE__, __LINE__,
                          std::format("ids differ for [{}] case of {} bytes: {} vs {} ids",
                                      c.string_or("set", ""), text.size(), got.size(), want.size()));
        }
        const std::string dec = tok->decode(want);
        if (dec != c.string_or("decode", "")) ++bad_dec;
        if (tok->decode(want, true) != c.string_or("decode_skip", "")) ++bad_skip;
        text::StreamDecoder sd(*tok);
        std::string stream;
        for (uint32_t id : want) stream += sd.push(id);
        stream += sd.flush();
        if (stream != dec) ++bad_stream;
    }
    std::printf("      %u cases, %llu ids: ids %u/%u, decode %u/%u, skip-special %u/%u, stream %u/%u\n",
                n, (unsigned long long)ids_total, n - bad_ids, n, n - bad_dec, n, n - bad_skip, n,
                n - bad_stream, n);
    CHECK(n >= 900);
    CHECK_EQ(bad_ids, 0u);
    CHECK_EQ(bad_dec, 0u);
    CHECK_EQ(bad_skip, 0u);
    CHECK_EQ(bad_stream, 0u);
}

DEEPMOE_TEST(tokenizer, special_tokens_and_vocabulary) {
    std::string why;
    const text::Tokenizer* tok = shared_tokenizer(why);
    if (!tok) { std::printf("      SKIP tokenizer: %s\n", why.c_str()); return; }
    CHECK_EQ(tok->vocab_size(), 129280u);
    const auto eos = tok->token_id("<\xEF\xBD\x9C" "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>");
    REQUIRE(eos.has_value());
    CHECK_EQ(*eos, 1u);
    CHECK(tok->is_special(1));
    const auto user = tok->token_id("<\xEF\xBD\x9C" "User\xEF\xBD\x9C>");
    REQUIRE(user.has_value());
    CHECK_EQ(*user, 128803u);
    CHECK(tok->is_added(128803) && !tok->is_special(128803));
    CHECK_EQ(*tok->token_id("</think>"), 128822u);
    // The chat header encodes to its special ids with nothing in between.
    const std::vector<uint32_t> ids =
        tok->encode("<\xEF\xBD\x9C" "begin\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C><\xEF\xBD\x9C"
                    "User\xEF\xBD\x9C>hi<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C></think>");
    REQUIRE(ids.size() == 5);
    CHECK_EQ(ids[0], 0u);
    CHECK_EQ(ids[1], 128803u);
    CHECK_EQ(ids[3], 128804u);
    CHECK_EQ(ids[4], 128822u);
    // No BOS is added by encode itself.
    CHECK(tok->encode("hello").front() != 0u);
}

DEEPMOE_TEST(tokenizer, streaming_decode_holds_partial_utf8) {
    std::string why;
    const text::Tokenizer* tok = shared_tokenizer(why);
    if (!tok) { std::printf("      SKIP tokenizer: %s\n", why.c_str()); return; }
    // "你好" is two 3-byte characters; byte-level BPE may split them across
    // tokens. Whatever the split, no push may emit a partial character.
    const std::string text = "\xE4\xBD\xA0\xE5\xA5\xBD \xF0\x9F\x98\x80 ok";
    const std::vector<uint32_t> ids = tok->encode(text);
    text::StreamDecoder sd(*tok);
    std::string all;
    for (uint32_t id : ids) {
        const std::string part = sd.push(id);
        CHECK_EQ(text::utf8_lossy(part), part);   // every emitted piece is valid UTF-8
        all += part;
    }
    all += sd.flush();
    CHECK_EQ(all, text);
    // A single byte token of a multi-byte character is held, then released.
    uint32_t e4 = 0;
    for (uint32_t id = 0; id < 128000; ++id)
        if (tok->token_bytes(id) == std::string("\xE4", 1)) { e4 = id; break; }
    REQUIRE(e4 != 0);
    text::StreamDecoder s2(*tok);
    CHECK_EQ(s2.push(e4), std::string());
    CHECK_EQ(s2.flush(), std::string("\xEF\xBF\xBD"));
}

DEEPMOE_TEST(tokenizer, utf8_lossy_matches_rust) {
    DM_UNUSED_CTX();
    using text::utf8_lossy;
    CHECK_EQ(utf8_lossy("abc"), std::string("abc"));
    // Maximal subparts: E0 80 is two errors (80 cannot follow E0), F0 9F 98 then
    // 'x' is one error for the three-byte valid prefix.
    CHECK_EQ(utf8_lossy(std::string("\xE0\x80", 2)), std::string("\xEF\xBF\xBD\xEF\xBF\xBD"));
    CHECK_EQ(utf8_lossy(std::string("\xF0\x9F\x98x", 4)), std::string("\xEF\xBF\xBDx"));
    CHECK_EQ(utf8_lossy(std::string("\xED\xA0\x80", 3)),
             std::string("\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD"));   // a surrogate
    size_t held = 0;
    CHECK_EQ(utf8_lossy(std::string("a\xE4\xBD", 3), true, &held), std::string("a"));
    CHECK_EQ(held, size_t(2));
}
