// tests/data/l2/ -- the oracle L2 per-stage golden tensors of design §12,
// written by `tools/oracle.py --level l2`.
//
// One `index.json` describes every (layer, step) file and, inside it, every
// named tensor: dtype, shape, byte offset. The binary is a plain concatenation
// after a 16-byte header, so reading it needs nothing but core/json.h.
//
// Everything comes back as fp32. A "bf16" tensor was bf16 in the reference and
// is widened on load, which is lossless and is also what the GPU side produces
// (design §6 keeps our residual stream in fp32 where the reference keeps it in
// bf16 -- see docs/p2_attention.md).
//
// Ownership/threading: plain value types, loaded once per test.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <map>
#include <string>
#include <vector>

#include "core/json.h"
#include "core/status.h"

namespace deepmoe::testing {

struct L2Tensor {
    std::string        dtype;
    std::vector<uint64_t> shape;
    std::vector<float> f;        // always fp32
    std::vector<uint8_t> raw;    // the bytes as stored, for the u8 planes

    uint64_t elements() const {
        uint64_t n = 1;
        for (uint64_t d : shape) n *= d;
        return n;
    }
};

// One (layer, step) export.
struct L2Step {
    uint32_t layer = 0;
    std::string step;
    std::map<std::string, L2Tensor, std::less<>> t;

    const L2Tensor* find(std::string_view name) const {
        auto it = t.find(name);
        return it == t.end() ? nullptr : &it->second;
    }
    // A missing tensor is a test bug, not a data condition, so this aborts
    // loudly rather than returning something empty that silently passes.
    const std::vector<float>& f(std::string_view name) const {
        const L2Tensor* v = find(name);
        if (!v) {
            std::printf("FATAL: L2 layer %u has no tensor '%.*s'\n", layer,
                        static_cast<int>(name.size()), name.data());
            std::abort();
        }
        return v->f;
    }
};

struct L2Set {
    uint32_t prefill_len = 0, decode_pos = 0;
    std::map<std::string, int64_t> config;
    std::vector<L2Step> steps;

    const L2Step* layer(uint32_t l) const {
        for (const L2Step& s : steps) if (s.layer == l) return &s;
        return nullptr;
    }
    int64_t cfg(std::string_view k, int64_t dflt = 0) const {
        auto it = config.find(std::string(k));
        return it == config.end() ? dflt : it->second;
    }
};

inline float bf16_to_f32(uint16_t h) {
    const uint32_t b = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

inline Result<L2Set> load_l2(const std::string& dir) {
    auto doc = json_parse_file(dir + "/index.json");
    if (!doc) return std::unexpected(doc.error());
    L2Set out;
    out.prefill_len = static_cast<uint32_t>(doc->int_or("prefill_len", 0));
    out.decode_pos  = static_cast<uint32_t>(doc->int_or("decode_pos", 0));
    if (const JsonValue* c = doc->find("config"))
        if (auto o = c->as_object())
            for (const auto& [k, v] : **o)
                if (v.is_number()) out.config[k] = static_cast<int64_t>(v.as_int().value_or(0));

    auto steps = doc->at("steps");
    if (!steps) return std::unexpected(steps.error());
    auto arr = (*steps)->as_array();
    if (!arr) return std::unexpected(arr.error());

    for (const JsonValue& s : **arr) {
        L2Step st;
        st.layer = static_cast<uint32_t>(s.int_or("layer", 0));
        st.step  = s.string_or("step", "");
        const std::string path = dir + "/" + s.string_or("file", "");
        const uint64_t base = static_cast<uint64_t>(s.int_or("data_offset", 16));

        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return fail(Err::NotFound, std::format("cannot open '{}'", path));
        std::fseek(f, 0, SEEK_END);
        const long fsize = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        std::vector<uint8_t> blob(static_cast<size_t>(fsize));
        const size_t got = std::fread(blob.data(), 1, blob.size(), f);
        std::fclose(f);
        if (got != blob.size()) return fail(Err::Io, std::format("short read of '{}'", path));

        auto tens = s.at("tensors");
        if (!tens) return std::unexpected(tens.error());
        auto ta = (*tens)->as_array();
        if (!ta) return std::unexpected(ta.error());
        for (const JsonValue& e : **ta) {
            L2Tensor t;
            t.dtype = e.string_or("dtype", "");
            if (const JsonValue* sh = e.find("shape"))
                if (auto a = sh->as_array())
                    for (const JsonValue& d : **a) t.shape.push_back(d.as_uint().value_or(0));
            const uint64_t off = base + static_cast<uint64_t>(e.int_or("offset", 0));
            const uint64_t n   = static_cast<uint64_t>(e.int_or("bytes", 0));
            if (off + n > blob.size())
                return fail(Err::Corrupt, std::format("'{}' runs past the file", path));
            t.raw.assign(blob.begin() + static_cast<long>(off),
                         blob.begin() + static_cast<long>(off + n));
            const uint64_t count = t.elements();
            t.f.resize(static_cast<size_t>(count));
            if (t.dtype == "f32") {
                std::memcpy(t.f.data(), t.raw.data(), static_cast<size_t>(count) * 4);
            } else if (t.dtype == "bf16") {
                for (uint64_t i = 0; i < count; ++i) {
                    uint16_t h;
                    std::memcpy(&h, t.raw.data() + i * 2, 2);
                    t.f[static_cast<size_t>(i)] = bf16_to_f32(h);
                }
            } else if (t.dtype == "i32") {
                for (uint64_t i = 0; i < count; ++i) {
                    int32_t v;
                    std::memcpy(&v, t.raw.data() + i * 4, 4);
                    t.f[static_cast<size_t>(i)] = static_cast<float>(v);
                }
            } else {   // u8: the fp8 / fp4 byte planes, compared as bytes
                for (uint64_t i = 0; i < count; ++i)
                    t.f[static_cast<size_t>(i)] = static_cast<float>(t.raw[static_cast<size_t>(i)]);
            }
            st.t.emplace(e.string_or("name", ""), std::move(t));
        }
        out.steps.push_back(std::move(st));
    }
    return out;
}

// --- comparison -------------------------------------------------------------

struct Agreement {
    double cos = 0.0;         // cosine similarity
    double max_abs = 0.0;     // max |a - b|
    double rel = 0.0;         // max_abs / max |b|, the design §12 L1 criterion
    double rel_l2 = 0.0;      // ||a - b|| / ||b||
    size_t n = 0;

    std::string str() const {
        return std::format("n={} cos={:.9f} max|d|={:.3e} rel={:.3e} relL2={:.3e}",
                           n, cos, max_abs, rel, rel_l2);
    }
};

inline Agreement agree(const float* a, const float* b, size_t n) {
    Agreement g;
    g.n = n;
    double dot = 0, na = 0, nb = 0, maxd = 0, maxb = 0, sd = 0;
    for (size_t i = 0; i < n; ++i) {
        const double x = a[i], y = b[i];
        dot += x * y; na += x * x; nb += y * y;
        const double d = std::fabs(x - y);
        if (d > maxd) maxd = d;
        if (std::fabs(y) > maxb) maxb = std::fabs(y);
        sd += d * d;
    }
    g.cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 1.0;
    g.max_abs = maxd;
    g.rel = maxb > 0 ? maxd / maxb : maxd;
    g.rel_l2 = nb > 0 ? std::sqrt(sd) / std::sqrt(nb) : std::sqrt(sd);
    return g;
}

inline Agreement agree(const std::vector<float>& a, const std::vector<float>& b) {
    return agree(a.data(), b.data(), a.size() < b.size() ? a.size() : b.size());
}

}  // namespace deepmoe::testing
