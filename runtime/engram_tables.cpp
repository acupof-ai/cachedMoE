#include "runtime/engram_tables.h"

#include <algorithm>
#include <format>
#include <limits>
#include <set>
#include <unordered_map>

#include "model/layout.h"

namespace deepmoe::runtime {
namespace {

#include "runtime/engram_norm.inc"

// --- the per-code-point normaliser ---------------------------------------------

struct NormMap {
    // cp -> [offset, len) into `out`; len 0 = deleted.
    std::unordered_map<char32_t, std::pair<uint32_t, uint32_t>> map;
    std::vector<char32_t> out;

    NormMap() {
        for (const auto& r : kNormDelete)
            for (uint32_t c = r[0]; c <= r[1]; ++c) map[c] = {0, 0};
        for (const auto& r : kNormRuns) {
            for (int64_t c = r[0]; c <= r[1]; c += r[2]) {
                map[static_cast<char32_t>(c)] = {static_cast<uint32_t>(out.size()), 1};
                out.push_back(static_cast<char32_t>(c + r[3]));
            }
        }
        for (const auto& r : kNormMulti) {
            map[r[0]] = {static_cast<uint32_t>(out.size()), r[2]};
            for (uint32_t i = 0; i < r[2]; ++i) out.push_back(kNormMultiOut[r[1] + i]);
        }
    }
};

const NormMap& norm_map() {
    static const NormMap m;
    return m;
}

std::u32string utf8_to_u32(std::string_view s) {
    std::u32string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned char b = static_cast<unsigned char>(s[i]);
        char32_t cp = 0xFFFD;
        size_t n = 1;
        if (b < 0x80) { cp = b; }
        else if ((b >> 5) == 0x6 && i + 1 < s.size()) { cp = ((b & 0x1F) << 6) | (s[i + 1] & 0x3F); n = 2; }
        else if ((b >> 4) == 0xE && i + 2 < s.size()) {
            cp = ((b & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F); n = 3;
        } else if ((b >> 3) == 0x1E && i + 3 < s.size()) {
            cp = ((b & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) | ((s[i + 2] & 0x3F) << 6) |
                 (s[i + 3] & 0x3F);
            n = 4;
        }
        out.push_back(cp);
        i += n;
    }
    return out;
}

// Rust `char::is_whitespace` (the White_Space property), which HF's Strip uses.
bool rust_whitespace(char32_t c) {
    switch (c) {
    case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20: case 0x85: case 0xA0:
    case 0x1680: case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000:
        return true;
    default:
        return c >= 0x2000 && c <= 0x200A;
    }
}

// --- numpy's SeedSequence, PCG64 and bounded draw ------------------------------

std::vector<uint64_t> seed_sequence_state(uint64_t entropy, size_t n_words64) {
    constexpr uint32_t kInitA = 0x43b0d7e5, kMultA = 0x931e8875;
    constexpr uint32_t kInitB = 0x8b51f9dd, kMultB = 0x58f38ded;
    constexpr uint32_t kMixL = 0xca01f9dd, kMixR = 0x4973f715;
    std::vector<uint32_t> ent;
    do { ent.push_back(static_cast<uint32_t>(entropy)); entropy >>= 32; } while (entropy);
    uint32_t hc = kInitA;
    auto hashmix = [&](uint32_t v) {
        v ^= hc;
        hc *= kMultA;
        v *= hc;
        v ^= v >> 16;
        return v;
    };
    auto mix = [](uint32_t x, uint32_t y) {
        uint32_t r = kMixL * x - kMixR * y;
        r ^= r >> 16;
        return r;
    };
    uint32_t pool[4];
    for (size_t i = 0; i < 4; ++i) pool[i] = hashmix(i < ent.size() ? ent[i] : 0u);
    for (size_t s = 0; s < 4; ++s)
        for (size_t d = 0; d < 4; ++d)
            if (s != d) pool[d] = mix(pool[d], hashmix(pool[s]));
    for (size_t s = 4; s < ent.size(); ++s)
        for (size_t d = 0; d < 4; ++d) pool[d] = mix(pool[d], hashmix(ent[s]));
    std::vector<uint64_t> out(n_words64);
    uint32_t h = kInitB;
    for (size_t i = 0; i < n_words64 * 2; ++i) {
        uint32_t v = pool[i % 4];
        v ^= h;
        h *= kMultB;
        v *= h;
        v ^= v >> 16;
        if (i % 2 == 0) out[i / 2] = v;
        else            out[i / 2] |= uint64_t(v) << 32;
    }
    return out;
}

using u128 = unsigned __int128;

struct Pcg64 {
    u128 state = 0, inc = 0;
    static constexpr u128 kMult = (u128(2549297995355413924ull) << 64) | u128(4865540595714422341ull);

    explicit Pcg64(uint64_t seed) {
        const std::vector<uint64_t> v = seed_sequence_state(seed, 4);
        const u128 initstate = (u128(v[0]) << 64) | v[1];
        const u128 initseq   = (u128(v[2]) << 64) | v[3];
        inc = (initseq << 1) | 1u;
        step();
        state += initstate;
        step();
    }
    void step() { state = state * kMult + inc; }
    uint64_t next64() {
        step();
        const uint64_t x = static_cast<uint64_t>(state >> 64) ^ static_cast<uint64_t>(state);
        const unsigned rot = static_cast<unsigned>(state >> 122);
        return (x >> rot) | (x << ((64 - rot) & 63));
    }
};

bool is_prime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
        if (n % p == 0) return n == p;
    }
    auto mulmod = [](uint64_t a, uint64_t b, uint64_t m) { return static_cast<uint64_t>(u128(a) * b % m); };
    auto powmod = [&](uint64_t a, uint64_t e, uint64_t m) {
        uint64_t r = 1;
        a %= m;
        while (e) {
            if (e & 1) r = mulmod(r, a, m);
            a = mulmod(a, a, m);
            e >>= 1;
        }
        return r;
    };
    uint64_t d = n - 1;
    unsigned s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }
    for (uint64_t a : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
        uint64_t x = powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (unsigned r = 1; r < s && comp; ++r) {
            x = mulmod(x, x, n);
            if (x == n - 1) comp = false;
        }
        if (comp) return false;
    }
    return true;
}

}  // namespace

std::string engram_normalize(std::string_view utf8) {
    const NormMap& nm = norm_map();
    std::u32string s;
    s.reserve(utf8.size());
    for (char32_t c : utf8_to_u32(utf8)) {
        if (c >= 0xAC00 && c <= 0xD7A3) {   // Hangul syllable: L V (T)
            const uint32_t k = c - 0xAC00;
            s.push_back(0x1100 + k / 588);
            s.push_back(0x1161 + (k % 588) / 28);
            if (k % 28) s.push_back(0x11A7 + k % 28);
            continue;
        }
        auto it = nm.map.find(c);
        if (it == nm.map.end()) { s.push_back(c); continue; }
        for (uint32_t i = 0; i < it->second.second; ++i) s.push_back(nm.out[it->second.first + i]);
    }
    // Replace(Regex("[ \t\r\n]+"), " ")
    std::u32string w;
    w.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        auto ws = [](char32_t c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
        if (ws(s[i])) {
            w.push_back(' ');
            while (i < s.size() && ws(s[i])) ++i;
        } else {
            w.push_back(s[i++]);
        }
    }
    // Replace(Regex("^ $"), sentinel), Strip(), Replace(sentinel, " ")
    constexpr char32_t kSentinel = 0xE000;
    if (w.size() == 1 && w[0] == ' ') w[0] = kSentinel;
    size_t a = 0, b = w.size();
    while (a < b && rust_whitespace(w[a])) ++a;
    while (b > a && rust_whitespace(w[b - 1])) --b;
    std::string out;
    out.reserve(b - a);
    for (size_t i = a; i < b; ++i) text::utf8_append(out, w[i] == kSentinel ? U' ' : w[i]);
    return out;
}

std::vector<int64_t> numpy_rng_integers(uint64_t seed, int64_t high, size_t n) {
    Pcg64 g(seed);
    std::vector<int64_t> out(n, 0);
    if (high <= 1) return out;
    const uint64_t rng = static_cast<uint64_t>(high) - 1;   // closed interval [0, rng]
    for (size_t i = 0; i < n; ++i) {
        if (rng <= 0xFFFFFFFFull) {
            // numpy draws 32-bit values for a range this small; engram's bound
            // is ~4.6e13, so this path is never taken and not reimplemented.
            out[i] = -1;
            continue;
        }
        const uint64_t ex = rng + 1;
        u128 m = u128(g.next64()) * ex;
        uint64_t lo = static_cast<uint64_t>(m);
        if (lo < ex) {
            const uint64_t threshold = (~0ull - rng) % ex;
            while (lo < threshold) {
                m = u128(g.next64()) * ex;
                lo = static_cast<uint64_t>(m);
            }
        }
        out[i] = static_cast<int64_t>(m >> 64);
    }
    return out;
}

Result<EngramTables> derive_engram_tables(const text::Tokenizer& tok, const TextConfig& cfg) {
    EngramTables t;
    t.max_ngram = cfg.engram_max_ngram_size;
    t.n_heads   = cfg.engram_n_heads;
    t.head_dim  = cfg.engram_head_dim;
    if (t.max_ngram != layout::kEngramMaxNgram || t.n_heads != layout::kEngramHeads ||
        t.head_dim != layout::kEngramHeadDim)
        return fail(Err::Corrupt, std::format("engram geometry {}x{}x{} does not match model/layout.h",
                                              t.max_ngram, t.n_heads, t.head_dim));
    if (cfg.engram_layer_ids.size() != cfg.engram_num_embeddings.size())
        return fail(Err::Corrupt, "engram_layer_ids and engram_num_embeddings differ in length");

    // build_compressed_token_map
    const uint32_t n = tok.vocab_size();
    t.token_map.resize(n);
    std::unordered_map<std::string, int32_t> keys;
    keys.reserve(n);
    for (uint32_t id = 0; id < n; ++id) {
        const uint32_t one[1] = {id};
        const std::string text = tok.decode(one, /*skip_special=*/false);
        std::string key;
        if (text.find("\xEF\xBF\xBD") != std::string::npos) {
            key = std::string(tok.token_string(id));
        } else {
            key = engram_normalize(text);
            if (key.empty()) key = text;
        }
        auto [it, fresh] = keys.try_emplace(std::move(key), static_cast<int32_t>(keys.size()));
        t.token_map[id] = it->second;
    }
    t.compressed_vocab_size = static_cast<uint32_t>(keys.size());
    if (cfg.engram_compressed_vocab_size && t.compressed_vocab_size != cfg.engram_compressed_vocab_size)
        return fail(Err::Corrupt,
                    std::format("the tokenizer compresses to {} ids but config.json says {}; every "
                                "engram hash would differ", t.compressed_vocab_size,
                                cfg.engram_compressed_vocab_size));
    if (cfg.engram_pad_token_id >= n) return fail(Err::Corrupt, "engram_pad_token_id outside the vocabulary");
    t.pad_id = t.token_map[cfg.engram_pad_token_id];

    // compute_hash_multipliers
    const int64_t bound =
        std::max<int64_t>(1, (std::numeric_limits<int64_t>::max() / int64_t(t.compressed_vocab_size)) / 2);

    // EngramLayout.from_args: primes drawn in order, never reused
    std::set<uint64_t> seen;
    for (size_t li = 0; li < cfg.engram_layer_ids.size(); ++li) {
        EngramTables::LayerTable lt;
        lt.layer = static_cast<uint32_t>(cfg.engram_layer_ids[li]);
        lt.num_embeddings = static_cast<uint64_t>(cfg.engram_num_embeddings[li]);
        const std::vector<int64_t> m = numpy_rng_integers(10007ull * lt.layer, bound, t.max_ngram);
        for (uint32_t i = 0; i < t.max_ngram; ++i) {
            if (m[i] < 0) return fail(Err::Internal, "engram multiplier bound below 2^32");
            lt.multipliers[i] = m[i] * 2 + 1;
        }
        uint32_t col = 0;
        int64_t off = 0;
        for (uint32_t g = 0; g + 1 < t.max_ngram; ++g) {
            uint64_t current = cfg.engram_vocab_size - 1;
            for (uint32_t h = 0; h < t.n_heads; ++h) {
                uint64_t c = current + 1;
                while (!is_prime(c) || seen.count(c)) ++c;
                seen.insert(c);
                current = c;
                lt.primes[col] = static_cast<int64_t>(c);
                lt.offsets[col] = off;
                off += static_cast<int64_t>(c);
                ++col;
            }
        }
        if (static_cast<uint64_t>(off) != lt.num_embeddings)
            return fail(Err::Corrupt,
                        std::format("engram layer {}: the primes sum to {} rows, config.json says {}",
                                    lt.layer, off, lt.num_embeddings));
        t.layers.push_back(lt);
    }
    return t;
}

Result<EngramTables> derive_engram_tables(const std::string& model_dir, const TextConfig& cfg) {
    auto tok = text::Tokenizer::load(model_dir + "/tokenizer.json");
    if (!tok) return std::unexpected(tok.error());
    return derive_engram_tables(*tok, cfg);
}

}  // namespace deepmoe::runtime
