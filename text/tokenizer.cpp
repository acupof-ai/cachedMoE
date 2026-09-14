#include "text/tokenizer.h"

#include <algorithm>
#include <format>
#include <mutex>
#include <queue>

#include "core/json.h"

namespace deepmoe::text {
namespace {

#include "text/unicode_tables.inc"

// --- character classes -------------------------------------------------------

enum : uint8_t {
    kN = 1, kL = 2, kM = 4, kP = 8, kS = 16, kWS = 32, kCJK = 64,
};

const std::vector<uint8_t>& class_table() {
    static const std::vector<uint8_t> t = [] {
        std::vector<uint8_t> v(0x110000, 0);
        auto fill = [&](const uint32_t (*r)[2], size_t n, uint8_t bit) {
            for (size_t i = 0; i < n; ++i)
                for (uint32_t c = r[i][0]; c <= r[i][1]; ++c) v[c] |= bit;
        };
        fill(kClassN, std::size(kClassN), kN);
        fill(kClassL, std::size(kClassL), kL);
        fill(kClassM, std::size(kClassM), kM);
        fill(kClassP, std::size(kClassP), kP);
        fill(kClassS, std::size(kClassS), kS);
        fill(kClassWS, std::size(kClassWS), kWS);
        // pre-tokenizer 2's literal class [一-龥぀-ゟ゠-ヿ]
        for (uint32_t c = 0x4E00; c <= 0x9FA5; ++c) v[c] |= kCJK;
        for (uint32_t c = 0x3040; c <= 0x309F; ++c) v[c] |= kCJK;
        for (uint32_t c = 0x30A0; c <= 0x30FF; ++c) v[c] |= kCJK;
        return v;
    }();
    return t;
}

// --- UTF-8 -------------------------------------------------------------------

enum class U8 { Ok, Invalid, Incomplete };

// One code point at s[i]. `len` is the bytes consumed: the whole sequence when
// Ok, the maximal valid subpart (>= 1) when Invalid or Incomplete.
U8 utf8_next(std::string_view s, size_t i, char32_t& cp, size_t& len) {
    const uint8_t b0 = static_cast<uint8_t>(s[i]);
    if (b0 < 0x80) { cp = b0; len = 1; return U8::Ok; }
    uint32_t n = 0, lo = 0x80, hi = 0xBF;
    if (b0 >= 0xC2 && b0 <= 0xDF)      { n = 2; cp = b0 & 0x1F; }
    else if (b0 == 0xE0)               { n = 3; cp = b0 & 0x0F; lo = 0xA0; }
    else if (b0 >= 0xE1 && b0 <= 0xEC) { n = 3; cp = b0 & 0x0F; }
    else if (b0 == 0xED)               { n = 3; cp = b0 & 0x0F; hi = 0x9F; }
    else if (b0 >= 0xEE && b0 <= 0xEF) { n = 3; cp = b0 & 0x0F; }
    else if (b0 == 0xF0)               { n = 4; cp = b0 & 0x07; lo = 0x90; }
    else if (b0 >= 0xF1 && b0 <= 0xF3) { n = 4; cp = b0 & 0x07; }
    else if (b0 == 0xF4)               { n = 4; cp = b0 & 0x07; hi = 0x8F; }
    else { len = 1; return U8::Invalid; }
    for (uint32_t k = 1; k < n; ++k) {
        if (i + k >= s.size()) { len = k; return U8::Incomplete; }
        const uint8_t b = static_cast<uint8_t>(s[i + k]);
        const uint32_t l = (k == 1) ? lo : 0x80, h = (k == 1) ? hi : 0xBF;
        if (b < l || b > h) { len = k; return U8::Invalid; }
        cp = (cp << 6) | (b & 0x3F);
    }
    len = n;
    return U8::Ok;
}

// --- byte-level (GPT-2 bytes_to_unicode) ------------------------------------

struct ByteMap {
    char32_t to_char[256];
    int16_t  to_byte[512];   // char -> byte, -1 if not a byte-level character
    ByteMap() {
        std::fill(std::begin(to_byte), std::end(to_byte), int16_t(-1));
        uint32_t extra = 0;
        for (uint32_t b = 0; b < 256; ++b) {
            const bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174);
            const char32_t c = keep ? char32_t(b) : char32_t(256 + extra++);
            to_char[b] = c;
            to_byte[c] = static_cast<int16_t>(b);
        }
    }
};
const ByteMap& byte_map() {
    static const ByteMap m;
    return m;
}

// The ByteLevel decoder's rule for one token string: every character mapped
// back to its byte, or -- if any character is not a byte-level one -- the
// string's own UTF-8.
std::string bytes_of_token_string(std::string_view s) {
    const ByteMap& bm = byte_map();
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        char32_t cp = 0;
        size_t len = 0;
        if (utf8_next(s, i, cp, len) != U8::Ok || cp >= 512 || bm.to_byte[cp] < 0)
            return std::string(s);
        out.push_back(static_cast<char>(bm.to_byte[cp]));
        i += len;
    }
    return out;
}

// --- added-token trie (leftmost-longest) -------------------------------------

struct Trie {
    struct Node { int32_t id = -1; };
    std::vector<Node> nodes{Node{}};
    std::unordered_map<uint64_t, uint32_t> edges;   // (node << 8 | byte) -> node

    void insert(std::string_view s, uint32_t id) {
        uint32_t n = 0;
        for (unsigned char b : s) {
            const uint64_t key = (uint64_t(n) << 8) | b;
            auto it = edges.find(key);
            if (it == edges.end()) {
                nodes.push_back(Node{});
                it = edges.emplace(key, uint32_t(nodes.size() - 1)).first;
            }
            n = it->second;
        }
        nodes[n].id = static_cast<int32_t>(id);
    }
    // The longest token starting at s[i]: its length, and the id in *id.
    size_t longest_at(std::string_view s, size_t i, uint32_t* id) const {
        uint32_t n = 0;
        size_t best = 0;
        for (size_t j = i; j < s.size(); ++j) {
            auto it = edges.find((uint64_t(n) << 8) | static_cast<unsigned char>(s[j]));
            if (it == edges.end()) break;
            n = it->second;
            if (nodes[n].id >= 0) { best = j + 1 - i; *id = uint32_t(nodes[n].id); }
        }
        return best;
    }
    bool empty() const { return nodes.size() == 1; }
};

struct Segment {
    size_t begin, end;
    int64_t id;   // -1 = ordinary text
};

// HF `split_with_indices` over one trie, applied to the text-only segments.
void split_on(const Trie& trie, std::string_view s, std::vector<Segment>& segs) {
    if (trie.empty()) return;
    std::vector<Segment> out;
    for (const Segment& g : segs) {
        if (g.id >= 0) { out.push_back(g); continue; }
        size_t last = g.begin, i = g.begin;
        std::string_view sub = s.substr(0, g.end);
        while (i < g.end) {
            uint32_t id = 0;
            const size_t n = trie.longest_at(sub, i, &id);
            if (n == 0) { ++i; continue; }
            if (i > last) out.push_back({last, i, -1});
            out.push_back({i, i + n, int64_t(id)});
            i += n;
            last = i;
        }
        if (g.end > last) out.push_back({last, g.end, -1});
    }
    segs.swap(out);
}

struct Piece { uint32_t b, e; };

// Split(regex, Isolated) inside each piece: matches and the gaps between them.
template <class Match>
void split_isolated(const std::vector<Piece>& in, std::vector<Piece>& out, Match&& match) {
    out.clear();
    for (const Piece& p : in) {
        uint32_t last = p.b, i = p.b;
        while (i < p.e) {
            const uint32_t n = match(i, p.e);
            if (n == 0) { ++i; continue; }
            if (i > last) out.push_back({last, i});
            out.push_back({i, i + n});
            i += n;
            last = i;
        }
        if (p.e > last) out.push_back({last, p.e});
    }
}

bool is_ascii_punct(char32_t c) {
    return (c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x40) ||
           (c >= 0x5B && c <= 0x60) || (c >= 0x7B && c <= 0x7E);
}
bool is_ascii_alpha(char32_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

}  // namespace

// --- Impl ------------------------------------------------------------------------

struct Tokenizer::Impl {
    uint32_t vocab_size = 0;
    std::vector<std::string> str;          // tokenizer.json string per id
    std::vector<std::string> bytes;        // decoded bytes per id
    std::vector<uint8_t>     flags;        // 1 exists, 2 added, 4 special
    std::unordered_map<std::string, uint32_t> vocab;   // model vocabulary
    std::unordered_map<std::string, uint32_t> added;   // added tokens by content
    std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> merges;  // (a,b) -> (rank, new)
    uint32_t byte_id[256]{};
    Trie raw_trie, norm_trie;

    void bpe(std::string_view word, std::vector<uint32_t>& out) const;
    void encode_text(std::string_view s, std::vector<uint32_t>& out) const;
};

// HF tokenizers' Word::merge_all, without dropout.
void Tokenizer::Impl::bpe(std::string_view word, std::vector<uint32_t>& out) const {
    const size_t n = word.size();
    if (n == 0) return;
    if (n == 1) { out.push_back(byte_id[static_cast<unsigned char>(word[0])]); return; }
    struct Sym { uint32_t c; int64_t prev, next; uint32_t len; };
    std::vector<Sym> sym(n);
    for (size_t i = 0; i < n; ++i)
        sym[i] = {byte_id[static_cast<unsigned char>(word[i])], int64_t(i) - 1,
                  i + 1 < n ? int64_t(i + 1) : -1, 1};
    struct M { uint32_t pos, rank, id; };
    auto cmp = [](const M& a, const M& b) {
        return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos;   // min-heap
    };
    std::priority_queue<M, std::vector<M>, decltype(cmp)> q(cmp);
    auto pair_of = [&](uint32_t a, uint32_t b) { return (uint64_t(a) << 32) | b; };
    for (size_t i = 0; i + 1 < n; ++i) {
        auto it = merges.find(pair_of(sym[i].c, sym[i + 1].c));
        if (it != merges.end()) q.push({uint32_t(i), it->second.first, it->second.second});
    }
    while (!q.empty()) {
        const M top = q.top();
        q.pop();
        Sym& cur = sym[top.pos];
        if (cur.len == 0 || cur.next == -1) continue;
        const size_t next_pos = size_t(cur.next);
        const Sym right = sym[next_pos];
        auto it = merges.find(pair_of(cur.c, right.c));
        if (it == merges.end() || it->second.second != top.id) continue;
        cur.c = top.id;
        cur.len += right.len;
        cur.next = right.next;
        sym[next_pos].len = 0;
        if (right.next > -1 && size_t(right.next) < n) sym[size_t(right.next)].prev = top.pos;
        if (cur.prev >= 0) {
            auto p = merges.find(pair_of(sym[size_t(cur.prev)].c, cur.c));
            if (p != merges.end()) q.push({uint32_t(cur.prev), p->second.first, p->second.second});
        }
        if (cur.next > -1 && size_t(cur.next) < n) {
            auto p = merges.find(pair_of(cur.c, sym[size_t(cur.next)].c));
            if (p != merges.end()) q.push({top.pos, p->second.first, p->second.second});
        }
    }
    for (const Sym& s : sym)
        if (s.len) out.push_back(s.c);
}

void Tokenizer::Impl::encode_text(std::string_view s, std::vector<uint32_t>& out) const {
    const std::vector<uint8_t>& cls = class_table();
    // Code points (invalid UTF-8 -> U+FFFD).
    std::vector<char32_t> cp;
    cp.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        char32_t c = 0;
        size_t len = 0;
        cp.push_back(utf8_next(s, i, c, len) == U8::Ok ? c : char32_t(0xFFFD));
        i += len;
    }
    if (cp.empty()) return;
    auto has = [&](uint32_t i, uint8_t bits) { return (cls[cp[i]] & bits) != 0; };

    std::vector<Piece> a{{0, uint32_t(cp.size())}}, b;
    // 1. \p{N}{1,3}
    split_isolated(a, b, [&](uint32_t i, uint32_t e) -> uint32_t {
        uint32_t n = 0;
        while (n < 3 && i + n < e && has(i + n, kN)) ++n;
        return n;
    });
    // 2. [一-龥぀-ゟ゠-ヿ]+
    split_isolated(b, a, [&](uint32_t i, uint32_t e) -> uint32_t {
        uint32_t n = 0;
        while (i + n < e && has(i + n, kCJK)) ++n;
        return n;
    });
    // 3. [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
    //  | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
    //  | ?[\p{P}\p{S}]+[\r\n]*
    //  | \s*[\r\n]+
    //  | \s+(?!\S)
    //  | \s+
    // leftmost-first alternation, each alternative with its backtracking
    // resolved by hand (docs/p3_chat.md §3.2).
    auto crlf = [&](uint32_t i) { return cp[i] == U'\r' || cp[i] == U'\n'; };
    split_isolated(a, b, [&](uint32_t i, uint32_t e) -> uint32_t {
        const char32_t c = cp[i];
        // A1
        if (is_ascii_punct(c) && i + 1 < e && is_ascii_alpha(cp[i + 1])) {
            uint32_t j = i + 1;
            while (j < e && is_ascii_alpha(cp[j])) ++j;
            return j - i;
        }
        // A2
        {
            const bool opt = !crlf(i) && !has(i, kL | kP | kS);
            uint32_t j = i;
            if (opt && i + 1 < e && has(i + 1, kL | kM)) j = i + 1;
            if (has(j, kL | kM)) {
                while (j < e && has(j, kL | kM)) ++j;
                return j - i;
            }
        }
        // A3
        {
            uint32_t j = i;
            if (c == U' ' && i + 1 < e && has(i + 1, kP | kS)) j = i + 1;
            if (has(j, kP | kS)) {
                while (j < e && has(j, kP | kS)) ++j;
                while (j < e && crlf(j)) ++j;
                return j - i;
            }
        }
        if (!has(i, kWS)) return 0;
        uint32_t j = i;
        while (j < e && has(j, kWS)) ++j;
        // A4: through the last \r or \n of the whitespace run
        for (uint32_t k = j; k > i; --k)
            if (crlf(k - 1)) return k - i;
        // A5
        if (j == e) return j - i;
        if (j - i >= 2) return j - 1 - i;
        // A6
        return j - i;
    });

    std::string word;
    for (const Piece& p : b) {
        word.clear();
        for (uint32_t i = p.b; i < p.e; ++i) utf8_append(word, cp[i]);
        bpe(word, out);
    }
}

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

Result<Tokenizer> Tokenizer::load(const std::string& path) {
    auto doc = json_parse_file(path);
    if (!doc) return std::unexpected(doc.error());
    auto im = std::make_unique<Impl>();

    const JsonValue* model = doc->find("model");
    if (!model || model->string_or("type", "") != "BPE")
        return fail(Err::Corrupt, std::format("{}: model.type is not BPE", path));
    if (model->bool_or("byte_fallback", false) || model->bool_or("ignore_merges", false) ||
        !model->find("dropout")->is_null())
        return fail(Err::Unimplemented,
                    std::format("{}: byte_fallback / ignore_merges / dropout are not implemented", path));
    // The pre-tokenizer is hand-compiled; refuse a file whose regexes differ.
    {
        const JsonValue* pt = doc->find("pre_tokenizer");
        const JsonValue* seq = pt ? pt->find("pretokenizers") : nullptr;
        static const char* kWant[] = {
            "\\p{N}{1,3}",
            "[\xE4\xB8\x80-\xE9\xBE\xA5\xE3\x81\x80-\xE3\x82\x9F\xE3\x82\xA0-\xE3\x83\xBF]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
        };
        auto arr = seq ? seq->as_array() : Result<const JsonArray*>(fail(Err::Corrupt));
        if (!arr || (*arr)->size() != 4)
            return fail(Err::Corrupt, std::format("{}: unexpected pre_tokenizer", path));
        for (size_t i = 0; i < 3; ++i) {
            const JsonValue& s = (**arr)[i];
            const JsonValue* pat = s.find("pattern");
            const std::string rx = pat ? pat->string_or("Regex", "") : "";
            if (s.string_or("type", "") != "Split" || s.string_or("behavior", "") != "Isolated" ||
                s.bool_or("invert", false) || rx != kWant[i])
                return fail(Err::Unimplemented,
                            std::format("{}: pre_tokenizer {} is not the one this tokenizer "
                                        "hand-compiles", path, i));
        }
        if ((**arr)[3].string_or("type", "") != "ByteLevel" || (**arr)[3].bool_or("use_regex", true) ||
            (**arr)[3].bool_or("add_prefix_space", true))
            return fail(Err::Unimplemented, std::format("{}: unexpected ByteLevel step", path));
        const JsonValue* norm = doc->find("normalizer");
        if (norm && !norm->is_null()) {
            const JsonValue* ns = norm->find("normalizers");
            if (norm->string_or("type", "") != "Sequence" || !ns || ns->size() != 0)
                return fail(Err::Unimplemented, std::format("{}: a normalizer is set", path));
        }
    }

    auto vocab = model->at("vocab");
    if (!vocab) return std::unexpected(vocab.error());
    auto vobj = (*vocab)->as_object();
    if (!vobj) return std::unexpected(vobj.error());
    uint32_t max_id = 0;
    im->vocab.reserve((*vobj)->size() * 2);
    for (const auto& [k, v] : **vobj) {
        auto id = v.as_uint();
        if (!id) return std::unexpected(id.error());
        im->vocab.emplace(k, uint32_t(*id));
        max_id = std::max<uint32_t>(max_id, uint32_t(*id));
    }
    auto added = doc->at("added_tokens");
    if (!added) return std::unexpected(added.error());
    auto aarr = (*added)->as_array();
    if (!aarr) return std::unexpected(aarr.error());
    for (const JsonValue& a : **aarr)
        max_id = std::max<uint32_t>(max_id, uint32_t(a.int_or("id", 0)));
    im->vocab_size = max_id + 1;
    im->str.assign(im->vocab_size, {});
    im->bytes.assign(im->vocab_size, {});
    im->flags.assign(im->vocab_size, 0);
    for (const auto& [k, id] : im->vocab) {
        im->str[id] = k;
        im->flags[id] |= 1;
    }
    for (const JsonValue& a : **aarr) {
        const uint32_t id = uint32_t(a.int_or("id", 0));
        const std::string content = a.string_or("content", "");
        if (content.empty()) return fail(Err::Corrupt, "an added token has no content");
        if (a.bool_or("lstrip", false) || a.bool_or("rstrip", false) || a.bool_or("single_word", false))
            return fail(Err::Unimplemented,
                        std::format("added token '{}' uses lstrip/rstrip/single_word", content));
        im->str[id] = content;
        im->flags[id] |= 1 | 2 | (a.bool_or("special", false) ? 4 : 0);
        im->added[content] = id;
        (a.bool_or("normalized", false) ? im->norm_trie : im->raw_trie).insert(content, id);
    }
    for (uint32_t id = 0; id < im->vocab_size; ++id)
        if (im->flags[id] & 1) im->bytes[id] = bytes_of_token_string(im->str[id]);

    const ByteMap& bm = byte_map();
    for (uint32_t b = 0; b < 256; ++b) {
        std::string s;
        utf8_append(s, bm.to_char[b]);
        auto it = im->vocab.find(s);
        if (it == im->vocab.end())
            return fail(Err::Corrupt, std::format("byte {} has no vocabulary token", b));
        im->byte_id[b] = it->second;
    }

    auto merges = model->at("merges");
    if (!merges) return std::unexpected(merges.error());
    auto marr = (*merges)->as_array();
    if (!marr) return std::unexpected(marr.error());
    im->merges.reserve((*marr)->size() * 2);
    uint32_t rank = 0;
    for (const JsonValue& m : **marr) {
        std::string a, b;
        if (m.is_string()) {
            const std::string_view s = *m.as_string();
            const size_t sp = s.find(' ');
            if (sp == std::string_view::npos || s.find(' ', sp + 1) != std::string_view::npos)
                return fail(Err::Corrupt, std::format("merge {} is not 'a b'", rank));
            a = std::string(s.substr(0, sp));
            b = std::string(s.substr(sp + 1));
        } else if (m.is_array() && m.size() == 2) {
            a = std::string(*(**m.as_array())[0].as_string());
            b = std::string(*(**m.as_array())[1].as_string());
        } else {
            return fail(Err::Corrupt, std::format("merge {} has an unknown form", rank));
        }
        auto ia = im->vocab.find(a), ib = im->vocab.find(b), in = im->vocab.find(a + b);
        if (ia == im->vocab.end() || ib == im->vocab.end() || in == im->vocab.end())
            return fail(Err::Corrupt, std::format("merge {} refers to a token outside the vocabulary", rank));
        im->merges.emplace((uint64_t(ia->second) << 32) | ib->second,
                           std::make_pair(rank, in->second));
        ++rank;
    }

    Tokenizer t;
    t.impl_ = std::move(im);
    return t;
}

std::vector<uint32_t> Tokenizer::encode(std::string_view utf8) const {
    std::vector<uint32_t> out;
    if (!impl_) return out;
    std::vector<Segment> segs{{0, utf8.size(), -1}};
    split_on(impl_->raw_trie, utf8, segs);
    split_on(impl_->norm_trie, utf8, segs);
    for (const Segment& g : segs) {
        if (g.id >= 0) { out.push_back(uint32_t(g.id)); continue; }
        impl_->encode_text(utf8.substr(g.begin, g.end - g.begin), out);
    }
    return out;
}

std::string Tokenizer::decode(std::span<const uint32_t> ids, bool skip_special) const {
    std::string raw;
    if (!impl_) return raw;
    for (uint32_t id : ids) {
        if (id >= impl_->vocab_size || !(impl_->flags[id] & 1)) continue;
        if (skip_special && (impl_->flags[id] & 4)) continue;
        raw += impl_->bytes[id];
    }
    return utf8_lossy(raw);
}

std::string Tokenizer::token_bytes(uint32_t id) const {
    if (!impl_ || id >= impl_->vocab_size) return {};
    return impl_->bytes[id];
}
std::string_view Tokenizer::token_string(uint32_t id) const {
    if (!impl_ || id >= impl_->vocab_size) return {};
    return impl_->str[id];
}
std::optional<uint32_t> Tokenizer::token_id(std::string_view content) const {
    if (!impl_) return std::nullopt;
    const std::string k(content);
    if (auto it = impl_->added.find(k); it != impl_->added.end()) return it->second;
    if (auto it = impl_->vocab.find(k); it != impl_->vocab.end()) return it->second;
    return std::nullopt;
}
bool Tokenizer::is_special(uint32_t id) const {
    return impl_ && id < impl_->vocab_size && (impl_->flags[id] & 4);
}
bool Tokenizer::is_added(uint32_t id) const {
    return impl_ && id < impl_->vocab_size && (impl_->flags[id] & 2);
}
uint32_t Tokenizer::vocab_size() const { return impl_ ? impl_->vocab_size : 0; }

// --- streaming ------------------------------------------------------------------

std::string StreamDecoder::push(uint32_t id) {
    if (id >= tok_->vocab_size()) return {};
    if (skip_special_ && tok_->is_special(id)) return {};
    pending_ += tok_->token_bytes(id);
    size_t held = 0;
    std::string out = utf8_lossy(pending_, /*hold_incomplete=*/true, &held);
    pending_.erase(0, pending_.size() - held);
    return out;
}

std::string StreamDecoder::flush() {
    std::string out = utf8_lossy(pending_);
    pending_.clear();
    return out;
}

// --- UTF-8 helpers --------------------------------------------------------------

void utf8_append(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::string utf8_lossy(std::string_view s, bool hold_incomplete, size_t* held) {
    std::string out;
    out.reserve(s.size());
    if (held) *held = 0;
    size_t i = 0;
    while (i < s.size()) {
        char32_t cp = 0;
        size_t len = 0;
        const U8 r = utf8_next(s, i, cp, len);
        if (r == U8::Ok) {
            out.append(s.data() + i, len);
        } else if (r == U8::Incomplete && hold_incomplete) {
            if (held) *held = s.size() - i;
            break;
        } else {
            out += "\xEF\xBF\xBD";
        }
        i += len;
    }
    return out;
}

}  // namespace deepmoe::text
