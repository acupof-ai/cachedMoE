#include "model/manifest.h"

#include <algorithm>
#include <cstdio>
#include <format>

#include "core/align.h"
#include "core/json.h"
#include "model/layout.h"

namespace deepmoe {

QuantType quant_from_string(std::string_view s) {
    if (s == "fp4_e2m1" || s == "fp4" || s == "e2m1")   return QuantType::Fp4E2M1;
    if (s == "fp8_e4m3" || s == "fp8" || s == "e4m3")   return QuantType::Fp8E4M3;
    if (s == "e8m0"     || s == "ue8m0")                return QuantType::E8M0;
    if (s == "bf16"     || s == "bfloat16")             return QuantType::Bf16;
    if (s == "f32"      || s == "float32" || s == "fp32") return QuantType::Fp32;
    if (s == "f16"      || s == "float16" || s == "fp16") return QuantType::Fp16;
    return QuantType::Unknown;
}

const char* quant_to_string(QuantType q) {
    switch (q) {
        case QuantType::Fp4E2M1: return "fp4_e2m1";
        case QuantType::Fp8E4M3: return "fp8_e4m3";
        case QuantType::E8M0:    return "e8m0";
        case QuantType::Bf16:    return "bf16";
        case QuantType::Fp32:    return "f32";
        case QuantType::Fp16:    return "f16";
        case QuantType::Unknown: return "unknown";
    }
    return "unknown";
}

uint64_t TensorEntry::elements() const {
    uint64_t n = shape.empty() ? 0 : 1;
    for (uint64_t d : shape) n *= d;
    return n;
}

namespace {

Result<std::vector<uint64_t>> shape_of(const JsonValue& n, std::string_view key) {
    std::vector<uint64_t> out;
    const JsonValue* v = n.find(key);
    if (!v) return out;
    auto a = v->as_array();
    if (!a) return fail(Err::Corrupt, std::format("'{}' is not an array", key));
    for (const JsonValue& e : **a) {
        auto d = e.as_uint();
        if (!d) return fail(Err::Corrupt, std::format("'{}' has a non-integer dimension", key));
        out.push_back(*d);
    }
    return out;
}

Result<ScaleEntry> parse_scale(const JsonValue& t) {
    ScaleEntry s;
    const JsonValue* n = t.find("scale");
    if (!n || n->is_null()) return s;
    if (!n->is_object()) return fail(Err::Corrupt, "'scale' is not an object");
    auto off = n->uint_at("offset");
    if (!off) return std::unexpected(off.error());
    auto by = n->uint_at("bytes");
    if (!by) return std::unexpected(by.error());
    s.offset = *off;
    s.bytes  = *by;
    s.dtype  = quant_from_string(n->string_or("dtype", "e8m0"));
    auto sh = shape_of(*n, "shape");
    if (!sh) return std::unexpected(sh.error());
    s.shape = *std::move(sh);
    auto blk = n->int_array_at("block");
    if (blk && blk->size() == 2) {
        s.block_m = static_cast<uint32_t>((*blk)[0]);
        s.block_k = static_cast<uint32_t>((*blk)[1]);
    }
    return s;
}

}  // namespace

const FileEntry* Manifest::file(std::string_view id) const {
    auto it = std::find_if(files_.begin(), files_.end(),
                           [&](const FileEntry& f) { return f.id == id; });
    return it == files_.end() ? nullptr : &*it;
}

const TensorEntry* Manifest::tensor(std::string_view name) const {
    auto it = tensors_.find(std::string(name));
    return it == tensors_.end() ? nullptr : &it->second;
}

Result<const TensorEntry*> Manifest::require_tensor(std::string_view name) const {
    if (const TensorEntry* t = tensor(name)) return t;
    return fail(Err::NotFound, std::format("manifest has no tensor '{}'", name));
}

const EngramEntry* Manifest::engram_for_layer(uint32_t layer) const {
    auto it = std::find_if(engram_.begin(), engram_.end(),
                           [&](const EngramEntry& e) { return e.layer == layer; });
    return it == engram_.end() ? nullptr : &*it;
}

Result<uint64_t> Manifest::expert_offset(ExpertKey key) const {
    if (experts_.stride == 0) return fail(Err::FailedPrecondition, "manifest has no 'experts' section");
    if (key.layer >= experts_.layers)
        return fail(Err::OutOfRange, std::format("layer {} >= {}", key.layer, experts_.layers));
    if (key.expert >= experts_.per_layer)
        return fail(Err::OutOfRange, std::format("expert {} >= {}", key.expert, experts_.per_layer));
    return experts_.base
         + (uint64_t(key.layer) * experts_.per_layer + key.expert) * experts_.stride;
}

Result<Manifest> Manifest::parse(std::string_view json_text) {
    auto doc = json_parse(json_text);
    if (!doc) return std::unexpected(doc.error());
    if (!doc->is_object()) return fail(Err::Corrupt, "manifest root is not an object");

    Manifest m;
    m.version_ = static_cast<uint32_t>(doc->int_or("version", 0));
    if (m.version_ != 1)
        return fail(Err::Corrupt, std::format("manifest version {} is not supported (expected 1)", m.version_));
    m.model_ = doc->string_or("model", "");

    // files
    if (auto fp = doc->at("files"); fp) {
        auto obj = (*fp)->as_object();
        if (!obj) return fail(Err::Corrupt, "'files' is not an object");
        for (const auto& [id, v] : **obj) {
            FileEntry f;
            f.id   = id;
            f.path = v.string_or("path", id);
            f.bytes = static_cast<uint64_t>(v.int_or("bytes", 0));
            f.sha256 = v.string_or("sha256", "");
            m.files_.push_back(std::move(f));
        }
    } else {
        return fail(Err::Corrupt, "manifest has no 'files' section");
    }

    // tensors
    if (auto tp = doc->at("tensors"); tp) {
        auto obj = (*tp)->as_object();
        if (!obj) return fail(Err::Corrupt, "'tensors' is not an object");
        m.tensors_.reserve((*obj)->size());
        for (const auto& [name, v] : **obj) {
            TensorEntry t;
            t.name = name;
            auto fileid = v.string_at("file");
            if (!fileid) return fail(Err::Corrupt, std::format("tensor '{}': {}", name, fileid.error().message));
            t.file = *std::move(fileid);
            auto off = v.uint_at("offset");
            if (!off) return fail(Err::Corrupt, std::format("tensor '{}': {}", name, off.error().message));
            t.offset = *off;
            auto by = v.uint_at("bytes");
            if (!by) return fail(Err::Corrupt, std::format("tensor '{}': {}", name, by.error().message));
            t.bytes = *by;
            t.dtype = quant_from_string(v.string_or("dtype", "unknown"));
            auto sh = shape_of(v, "shape");
            if (!sh) return fail(Err::Corrupt, std::format("tensor '{}': {}", name, sh.error().message));
            t.shape = *std::move(sh);
            auto sc = parse_scale(v);
            if (!sc) return fail(Err::Corrupt, std::format("tensor '{}': {}", name, sc.error().message));
            t.scale = *std::move(sc);
            m.tensors_.emplace(name, std::move(t));
        }
    } else {
        return fail(Err::Corrupt, "manifest has no 'tensors' section");
    }

    // experts (arithmetic addressing, design §5.1)
    if (const JsonValue* e = doc->find("experts")) {
        m.experts_.file      = e->string_or("file", "experts");
        m.experts_.stride    = static_cast<uint64_t>(e->int_or("stride", 0));
        m.experts_.layers    = static_cast<uint32_t>(e->int_or("layers", 0));
        m.experts_.per_layer = static_cast<uint32_t>(e->int_or("per_layer", 0));
        m.experts_.base      = static_cast<uint64_t>(e->int_or("base", 0));
    }

    // engram tables
    if (const JsonValue* g = doc->find("engram")) {
        auto arr = g->as_array();
        if (!arr) return fail(Err::Corrupt, "'engram' is not an array");
        for (const JsonValue& v : **arr) {
            EngramEntry e;
            e.layer     = static_cast<uint32_t>(v.int_or("layer", 0));
            e.file      = v.string_or("file", "");
            e.rows      = static_cast<uint64_t>(v.int_or("rows", 0));
            e.row_bytes = static_cast<uint32_t>(v.int_or("row_bytes", layout::kEngramRowBytes));
            e.base      = static_cast<uint64_t>(v.int_or("base", 0));
            m.engram_.push_back(std::move(e));
        }
    }
    return m;
}

Result<Manifest> Manifest::load(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(Err::Io, std::format("cannot open '{}'", path));
    std::string buf;
    char chunk[65536];
    size_t n;
    while ((n = std::fread(chunk, 1, sizeof chunk, f)) > 0) buf.append(chunk, n);
    bool bad = std::ferror(f) != 0;
    std::fclose(f);
    if (bad) return fail(Err::Io, std::format("read error on '{}'", path));
    auto m = parse(buf);
    if (!m) return fail(m.error().code, std::format("{}: {}", path, m.error().message));
    return m;
}

Result<void> Manifest::validate() const {
    std::string bad;
    auto file_bytes = [&](std::string_view id) -> uint64_t {
        const FileEntry* f = file(id);
        return f ? f->bytes : 0;
    };
    for (const auto& [name, t] : tensors_) {
        const FileEntry* f = file(t.file);
        if (!f) { bad += std::format("  tensor '{}' references unknown file '{}'\n", name, t.file); continue; }
        if (f->bytes && t.offset + t.bytes > f->bytes)
            bad += std::format("  tensor '{}' runs past the end of '{}'\n", name, t.file);
        if (t.scale.present() && f->bytes && t.scale.offset + t.scale.bytes > f->bytes)
            bad += std::format("  tensor '{}' scale runs past the end of '{}'\n", name, t.file);
        if (t.dtype == QuantType::Unknown)
            bad += std::format("  tensor '{}' has an unknown dtype\n", name);
        if (!is_aligned(t.offset))
            bad += std::format("  tensor '{}' offset {} is not 4 KiB aligned\n", name, t.offset);
    }
    if (experts_.stride) {
        if (experts_.stride != layout::kExpertBytes)
            bad += std::format("  experts.stride {} != layout::kExpertBytes {}\n",
                               experts_.stride, layout::kExpertBytes);
        if (!file(experts_.file))
            bad += std::format("  experts.file '{}' is not in 'files'\n", experts_.file);
        else {
            uint64_t need = experts_.base
                          + uint64_t(experts_.layers) * experts_.per_layer * experts_.stride;
            uint64_t have = file_bytes(experts_.file);
            if (have && need > have)
                bad += std::format("  experts region needs {} B but '{}' is {} B\n",
                                   need, experts_.file, have);
        }
    }
    for (const EngramEntry& e : engram_) {
        if (!file(e.file)) { bad += std::format("  engram layer {} references unknown file '{}'\n", e.layer, e.file); continue; }
        if (e.row_bytes != layout::kEngramRowBytes)
            bad += std::format("  engram layer {} row_bytes {} != {}\n", e.layer, e.row_bytes, layout::kEngramRowBytes);
        uint64_t need = e.base + e.rows * e.row_bytes;
        uint64_t have = file_bytes(e.file);
        if (have && need > have)
            bad += std::format("  engram layer {} needs {} B but '{}' is {} B\n", e.layer, need, e.file, have);
    }
    if (!bad.empty()) return fail(Err::Corrupt, "manifest is inconsistent:\n" + bad);
    return {};
}

uint64_t Manifest::total_bytes() const {
    uint64_t n = 0;
    for (const auto& [name, t] : tensors_) n += t.bytes + t.scale.bytes;
    n += uint64_t(experts_.layers) * experts_.per_layer * experts_.stride;
    for (const EngramEntry& e : engram_) n += e.rows * e.row_bytes;
    return n;
}

}  // namespace deepmoe
