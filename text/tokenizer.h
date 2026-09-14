// The DeepSeek-V4.1 tokenizer, in C++: byte-level BPE read from the
// checkpoint's own tokenizer.json (docs/p3_chat.md §3).
//
// What tokenizer.json specifies, and therefore all this implements
// ---------------------------------------------------------------
//   normalizer      an empty Sequence: the identity.
//   added tokens    1,283 of them (ids 0-2 and 128000-129279). The text is
//                   first split on the ones with `normalized: false`, then the
//                   remaining pieces on the ones with `normalized: true`, each
//                   pass leftmost-longest -- HF's AddedVocabulary, two tries.
//                   None has lstrip / rstrip / single_word.
//   pre-tokenizer   three `Split(regex, Isolated)` passes and a ByteLevel with
//                   use_regex false:
//                     1. \p{N}{1,3}
//                     2. [一-龥぀-ゟ゠-ヿ]+
//                     3. the punctuation / letters / whitespace alternation
//                   Each pass runs its regex inside every piece the previous
//                   pass produced. The regexes are hand-compiled below (no
//                   regex engine: std::regex has no \p{..}), and the character
//                   classes come from HF tokenizers' own engine
//                   (tools/gen_unicode_tables.py), not from a Unicode database
//                   that might be a different version.
//   model           BPE, 128,000 tokens, 127,741 merges, no dropout, no unk,
//                   no byte fallback, no ignore_merges: HF's `Word::merge_all`
//                   priority queue, ties to the leftmost pair.
//   post-processor  ByteLevel(trim_offsets): adds nothing. No BOS is added
//                   (tokenizer_config.json add_bos_token false); the chat
//                   renderer writes `<｜begin▁of▁sentence｜>` itself.
//   decoder         ByteLevel: each token's characters mapped back to bytes,
//                   or the token's own UTF-8 when any character is outside the
//                   byte map (the added tokens), then lossy UTF-8.
//
// Validated against HF `tokenizers` for 100% id equality by
// tools/tokenizer_golden.py and tests/test_tokenizer.cpp.
//
// Ownership/threading: a loaded Tokenizer is immutable and safe to share
// between threads. StreamDecoder is a small per-stream value.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/status.h"

namespace deepmoe::text {

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    static Result<Tokenizer> load(const std::string& tokenizer_json);

    // `utf8` is encoded exactly as HF's `Tokenizer.encode(text).ids` would.
    // Invalid UTF-8 is read as U+FFFD per maximal bad subsequence (a Python
    // str cannot hold it, so there is no reference behaviour to match).
    std::vector<uint32_t> encode(std::string_view utf8) const;

    // HF `decode(ids, skip_special_tokens=skip_special)`. Unknown ids are
    // dropped, as HF drops them.
    std::string decode(std::span<const uint32_t> ids, bool skip_special = false) const;

    // The raw bytes one id stands for: the byte-level mapping reversed, or the
    // added token's UTF-8. Not necessarily valid UTF-8 on its own.
    std::string token_bytes(uint32_t id) const;
    // The token's string as tokenizer.json writes it (byte-level characters).
    std::string_view token_string(uint32_t id) const;
    // An added token's id, or a vocabulary token's, by its tokenizer.json string.
    std::optional<uint32_t> token_id(std::string_view content) const;
    bool is_special(uint32_t id) const;
    bool is_added(uint32_t id) const;
    // One past the largest id (129,280).
    uint32_t vocab_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Incremental decode: `push` returns the text that became complete with this
// token, holding back a trailing incomplete UTF-8 sequence until the bytes
// that finish it arrive. The concatenation of every `push` and the final
// `flush` equals `Tokenizer::decode` of the same ids.
class StreamDecoder {
public:
    explicit StreamDecoder(const Tokenizer& tok, bool skip_special = false)
        : tok_(&tok), skip_special_(skip_special) {}
    std::string push(uint32_t id);
    std::string flush();
    void reset() { pending_.clear(); }

private:
    const Tokenizer* tok_;
    bool skip_special_;
    std::string pending_;
};

// UTF-8 helpers the tokenizer and the server share.
// Appends `cp` as UTF-8.
void utf8_append(std::string& out, char32_t cp);
// Rust's String::from_utf8_lossy: every maximal invalid subsequence becomes
// one U+FFFD. With `hold_incomplete`, a trailing sequence that is a valid
// prefix of a code point is not decoded; its length is returned in *held.
std::string utf8_lossy(std::string_view bytes, bool hold_incomplete = false,
                       size_t* held = nullptr);

}  // namespace deepmoe::text
