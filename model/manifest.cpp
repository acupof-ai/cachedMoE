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

const char* expert_part_name(ExpertPart p) {
    switch (p) {
        case ExpertPart::W1Weight: return "w1.weight";
        case ExpertPart::W1Scale:  return "w1.scale";
        case ExpertPart::W2Weight: return "w2.weight";
        case ExpertPart::W2Scale:  return "w2.scale";
        case ExpertPart::W3Weight: return "w3.weight";
        case ExpertPart::W3Scale:  return "w3.scale";
    }
    return "?";
}

std::optional<ExpertPart> expert_part_from_string(std::string_view s) {
    for (uint8_t i = 0; i < kExpertPartCount; ++i) {
        const auto p = static_cast<ExpertPart>(i);
        if (s == expert_part_name(p)) return p;
    }
    return std::nullopt;
}

AlignedRead align_read(uint32_t file, uint64_t off, uint64_t bytes) {
    AlignedRead r;
    r.file          = file;
    r.aligned_off   = align_down(off);
    r.aligned_bytes = align_up(off + bytes) - r.aligned_off;
    r.skew          = static_cast<uint32_t>(off - r.aligned_off);
    r.bytes         = bytes;
    return r;
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
    s.file   = static_cast<uint32_t>(n->int_or("file", 0));
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

Result<Run> parse_run(const JsonValue& v) {
    if (!v.is_object()) return fail(Err::Corrupt, "a run is not an object");
    Run r;
    r.file = static_cast<uint32_t>(v.int_or("file", 0));
    auto off = v.uint_at("aligned_off");
    if (!off) return std::unexpected(off.error());
    auto by = v.uint_at("aligned_bytes");
    if (!by) return std::unexpected(by.error());
    r.aligned_off   = *off;
    r.aligned_bytes = *by;
    r.slot_offset   = static_cast<uint64_t>(v.int_or("slot_offset", 0));

    const JsonValue* parts = v.find("parts");
    if (!parts) return fail(Err::Corrupt, "a run has no 'parts'");
    auto arr = parts->as_array();
    if (!arr) return fail(Err::Corrupt, "'parts' is not an array");
    r.parts.reserve((*arr)->size());
    for (const JsonValue& pv : **arr) {
        auto name = pv.string_at("tensor");
        if (!name) return std::unexpected(name.error());
        auto part = expert_part_from_string(*name);
        if (!part) return fail(Err::Corrupt, std::format("unknown expert part '{}'", *name));
        RunPart p;
        p.part        = *part;
        p.skew        = static_cast<uint32_t>(pv.int_or("skew", 0));
        p.bytes       = static_cast<uint64_t>(pv.int_or("bytes", 0));
        p.slot_offset = static_cast<uint64_t>(pv.int_or("slot_offset",
                            static_cast<int64_t>(r.slot_offset + p.skew)));
        r.parts.push_back(p);
    }
    return r;
}

}  // namespace

const FileEntry* Manifest::file(uint32_t index) const {
    return index < files_.size() ? &files_[index] : nullptr;
}

std::optional<uint32_t> Manifest::file_index(std::string_view path) const {
    for (size_t i = 0; i < files_.size(); ++i)
        if (files_[i].path == path) return static_cast<uint32_t>(i);
    return std::nullopt;
}

const TensorEntry* Manifest::tensor(std::string_view name) const {
    auto it = tensors_.find(std::string(name));
    return it == tensors_.end() ? nullptr : &it->second;
}

Result<const TensorEntry*> Manifest::require_tensor(std::string_view name) const {
    if (const TensorEntry* t = tensor(name)) return t;
    return fail(Err::NotFound, std::format("manifest has no tensor '{}'", name));
}

Result<AlignedRead> Manifest::tensor_read(std::string_view name) const {
    auto t = require_tensor(name);
    if (!t) return std::unexpected(t.error());
    return align_read((*t)->file, (*t)->offset, (*t)->bytes);
}

Result<AlignedRead> Manifest::tensor_scale_read(std::string_view name) const {
    auto t = require_tensor(name);
    if (!t) return std::unexpected(t.error());
    if (!(*t)->scale.present())
        return fail(Err::NotFound, std::format("tensor '{}' has no scale plane", name));
    const ScaleEntry& s = (*t)->scale;
    return align_read(s.file, s.offset, s.bytes);
}

uint32_t Manifest::experts_in_layer(uint32_t layer) const {
    return layer < experts_.size() ? static_cast<uint32_t>(experts_[layer].size()) : 0;
}

const ExpertEntry* Manifest::expert(ExpertKey key) const {
    if (key.layer >= experts_.size()) return nullptr;
    const auto& row = experts_[key.layer];
    return key.expert < row.size() ? &row[key.expert] : nullptr;
}

Result<const ExpertEntry*> Manifest::require_expert(ExpertKey key) const {
    if (const ExpertEntry* e = expert(key)) return e;
    return fail(Err::OutOfRange,
                std::format("manifest has no expert ({}, {}); layer holds {} of {} layers",
                            key.layer, key.expert, experts_in_layer(key.layer), experts_.size()));
}

const EngramEntry* Manifest::engram_for_layer(uint32_t layer) const {
    auto it = std::find_if(engram_.begin(), engram_.end(),
                           [&](const EngramEntry& e) { return e.layer == layer; });
    return it == engram_.end() ? nullptr : &*it;
}

Result<EngramRowPlan> Manifest::engram_row(uint32_t layer, uint64_t row) const {
    const EngramEntry* e = engram_for_layer(layer);
    if (!e) return fail(Err::NotFound, std::format("no engram table for layer {}", layer));
    if (row >= e->rows)
        return fail(Err::OutOfRange, std::format("engram row {} >= {}", row, e->rows));
    EngramRowPlan plan;
    plan.value = align_read(e->value.file, e->value.offset + row * e->value.row_bytes,
                            e->value.row_bytes);
    plan.scale = align_read(e->scale.file, e->scale.offset + row * e->scale.row_bytes,
                            e->scale.row_bytes);
    return plan;
}

Result<Manifest> Manifest::parse(std::string_view json_text) {
    auto doc = json_parse(json_text);
    if (!doc) return std::unexpected(doc.error());
    if (!doc->is_object()) return fail(Err::Corrupt, "manifest root is not an object");

    Manifest m;
    m.version_ = static_cast<uint32_t>(doc->int_or("version", 0));
    if (m.version_ != 2)
        return fail(Err::Corrupt,
                    std::format("manifest version {} is not supported (expected 2; "
                                "re-run tools/manifest.py)", m.version_));
    m.model_             = doc->string_or("model", "");
    m.alignment_         = static_cast<uint32_t>(doc->int_or("alignment", kPageSize));
    m.expert_slot_bytes_ = static_cast<uint64_t>(doc->int_or("expert_slot_bytes", 0));

    // --- files: an ordered array; the index is the file id used everywhere ---
    if (const JsonValue* fp = doc->find("files")) {
        auto arr = fp->as_array();
        if (!arr) return fail(Err::Corrupt, "'files' is not an array");
        m.files_.reserve((*arr)->size());
        for (const JsonValue& v : **arr) {
            FileEntry f;
            f.path       = v.string_or("path", "");
            f.bytes      = static_cast<uint64_t>(v.int_or("bytes", 0));
            f.data_start = static_cast<uint64_t>(v.int_or("data_start", 0));
            f.sha256     = v.string_or("sha256", "");
            if (f.path.empty()) return fail(Err::Corrupt, "a 'files' entry has no path");
            m.files_.push_back(std::move(f));
        }
    }
    if (m.files_.empty()) return fail(Err::Corrupt, "manifest has no 'files' section");

    // --- tensors ----------------------------------------------------------
    if (const JsonValue* tp = doc->find("tensors")) {
        auto obj = tp->as_object();
        if (!obj) return fail(Err::Corrupt, "'tensors' is not an object");
        m.tensors_.reserve((*obj)->size() * 2);
        for (const auto& [name, v] : **obj) {
            TensorEntry t;
            t.name = name;
            t.file = static_cast<uint32_t>(v.int_or("file", 0));
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

    // --- experts: runs with skews (design §5.1) ---------------------------
    if (const JsonValue* ep = doc->find("experts")) {
        auto arr = ep->as_array();
        if (!arr) return fail(Err::Corrupt, "'experts' is not an array");
        for (const JsonValue& lv : **arr) {
            const auto layer = static_cast<uint32_t>(lv.int_or("layer", 0));
            const JsonValue* ev = lv.find("experts");
            if (!ev) return fail(Err::Corrupt, std::format("expert layer {} has no 'experts'", layer));
            auto earr = ev->as_array();
            if (!earr) return fail(Err::Corrupt, std::format("expert layer {}: not an array", layer));
            if (m.experts_.size() <= layer) m.experts_.resize(layer + 1);
            auto& row = m.experts_[layer];
            row.clear();
            row.reserve((*earr)->size());
            for (const JsonValue& xv : **earr) {
                auto runs = xv.as_array();
                if (!runs)
                    return fail(Err::Corrupt,
                                std::format("expert layer {}: an expert is not a list of runs", layer));
                ExpertEntry e;
                e.runs.reserve((*runs)->size());
                for (const JsonValue& rv : **runs) {
                    auto r = parse_run(rv);
                    if (!r) return fail(Err::Corrupt,
                                        std::format("expert layer {}, expert {}: {}",
                                                    layer, row.size(), r.error().message));
                    for (const RunPart& p : r->parts) {
                        const auto i = static_cast<uint8_t>(p.part);
                        e.part_offset[i] = p.slot_offset;
                        e.part_bytes[i]  = p.bytes;
                    }
                    e.slot_bytes += r->aligned_bytes;
                    e.runs.push_back(*std::move(r));
                }
                row.push_back(std::move(e));
            }
        }
    }

    // --- engram tables ----------------------------------------------------
    if (const JsonValue* g = doc->find("engram")) {
        auto arr = g->as_array();
        if (!arr) return fail(Err::Corrupt, "'engram' is not an array");
        auto plane = [](const JsonValue* v) {
            EngramPlane p;
            if (!v) return p;
            p.file      = static_cast<uint32_t>(v->int_or("file", 0));
            p.offset    = static_cast<uint64_t>(v->int_or("offset", 0));
            p.bytes     = static_cast<uint64_t>(v->int_or("bytes", 0));
            p.row_bytes = static_cast<uint32_t>(v->int_or("row_bytes", 0));
            p.dtype     = quant_from_string(v->string_or("dtype", "unknown"));
            return p;
        };
        for (const JsonValue& v : **arr) {
            EngramEntry e;
            e.layer = static_cast<uint32_t>(v.int_or("layer", 0));
            e.rows  = static_cast<uint64_t>(v.int_or("rows", 0));
            e.value = plane(v.find("value"));
            e.scale = plane(v.find("scale"));
            m.engram_.push_back(std::move(e));
        }
    }
    return m;
}

Result<Manifest> Manifest::load(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(Err::Io, std::format("cannot open '{}'", path));
    std::string buf;
    char chunk[1 << 20];
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
    const uint64_t align = alignment_ ? alignment_ : kPageSize;
    if (align != kPageSize)
        bad += std::format("  alignment {} != the 4 KiB the I/O layer requires\n", align);

    // A payload must lie wholly inside its shard.
    auto check_range = [&](const char* what, uint32_t fi, uint64_t off, uint64_t bytes) {
        const FileEntry* f = file(fi);
        if (!f) {
            bad += std::format("  {} references file index {} but only {} are listed\n",
                               what, fi, files_.size());
            return;
        }
        if (f->bytes && off + bytes > f->bytes)
            bad += std::format("  {} runs past the end of '{}' ({} + {} > {})\n",
                               what, f->path, off, bytes, f->bytes);
    };

    // A sector-widened *read*, by contrast, is allowed to overshoot EOF by less
    // than one sector: the last tensor of a shard ends at the file's byte
    // length, which is never a multiple of 4096. Windows serves the valid bytes
    // and reports a short read; storage/backend.h ChunkRequest::min_bytes is how
    // the I/O layer is told that is legal. What must not happen is a read that
    // starts past EOF, or one that overshoots by a whole sector or more.
    auto check_read = [&](const std::string& what, uint32_t fi,
                          uint64_t aligned_off, uint64_t aligned_bytes) {
        const FileEntry* f = file(fi);
        if (!f) {
            bad += std::format("  {} references file index {} but only {} are listed\n",
                               what, fi, files_.size());
            return;
        }
        if (!f->bytes) return;
        if (aligned_off >= f->bytes) {
            bad += std::format("  {} starts at {}, past the end of '{}' ({} B)\n",
                               what, aligned_off, f->path, f->bytes);
        } else if (aligned_off + aligned_bytes > align_up(f->bytes)) {
            bad += std::format("  {} overshoots '{}' by a whole sector ({} + {} vs {})\n",
                               what, f->path, aligned_off, aligned_bytes, f->bytes);
        }
    };

    for (const auto& [name, t] : tensors_) {
        check_range(name.c_str(), t.file, t.offset, t.bytes);
        if (t.dtype == QuantType::Unknown)
            bad += std::format("  tensor '{}' has an unknown dtype\n", name);
        if (t.scale.present())
            check_range(name.c_str(), t.scale.file, t.scale.offset, t.scale.bytes);
        // Offsets are deliberately NOT 4 KiB aligned here: the shards are
        // whatever safetensors wrote. What must hold is that widening the read
        // to sector boundaries still lands inside the file.
        const AlignedRead r = align_read(t.file, t.offset, t.bytes);
        const FileEntry* f = file(t.file);
        if (f && f->bytes && r.aligned_off + r.aligned_bytes > align_up(f->bytes))
            bad += std::format("  tensor '{}' cannot be read sector-aligned inside '{}'\n",
                               name, f->path);
    }

    if (expert_slot_bytes_ == 0 && !experts_.empty())
        bad += "  'expert_slot_bytes' is missing\n";
    if (expert_slot_bytes_ && expert_slot_bytes_ % kPageSize != 0)
        bad += std::format("  expert_slot_bytes {} is not a 4 KiB multiple\n", expert_slot_bytes_);
    if (expert_slot_bytes_ && expert_slot_bytes_ != layout::kExpertSlotBytes)
        bad += std::format("  expert_slot_bytes {} != layout::kExpertSlotBytes {}; "
                           "update model/layout.h from tools/manifest.py's summary\n",
                           expert_slot_bytes_, layout::kExpertSlotBytes);

    for (uint32_t layer = 0; layer < experts_.size(); ++layer) {
        for (uint32_t id = 0; id < experts_[layer].size(); ++id) {
            const ExpertEntry& e = experts_[layer][id];
            const std::string what = std::format("expert ({}, {})", layer, id);
            if (e.runs.empty()) { bad += std::format("  {} has no runs\n", what); continue; }
            if (e.runs.size() > layout::kMaxExpertRuns)
                bad += std::format("  {} has {} runs, more than kMaxExpertRuns {}\n",
                                   what, e.runs.size(), layout::kMaxExpertRuns);
            if (e.slot_bytes > expert_slot_bytes_)
                bad += std::format("  {} needs {} B but a slot is {} B\n",
                                   what, e.slot_bytes, expert_slot_bytes_);
            uint64_t cursor = 0, payload = 0;
            for (const Run& r : e.runs) {
                if (!is_aligned(r.aligned_off) || !is_aligned(r.aligned_bytes))
                    bad += std::format("  {} has a run that is not 4 KiB aligned ({} + {})\n",
                                       what, r.aligned_off, r.aligned_bytes);
                if (r.slot_offset != cursor)
                    bad += std::format("  {} run slot_offset {} != the running total {}\n",
                                       what, r.slot_offset, cursor);
                if (!is_aligned(r.slot_offset))
                    bad += std::format("  {} run slot_offset {} is not 4 KiB aligned\n",
                                       what, r.slot_offset);
                check_read(what, r.file, r.aligned_off, r.aligned_bytes);
                for (const RunPart& p : r.parts) {
                    // The payload itself always lies inside the shard; only the
                    // sector padding at the tail may spill past EOF.
                    check_range(what.c_str(), r.file, r.aligned_off + p.skew, p.bytes);
                    if (p.skew + p.bytes > r.aligned_bytes)
                        bad += std::format("  {} part '{}' spills out of its run\n",
                                           what, expert_part_name(p.part));
                    if (p.slot_offset != r.slot_offset + p.skew)
                        bad += std::format("  {} part '{}' slot_offset {} != run {} + skew {}\n",
                                           what, expert_part_name(p.part), p.slot_offset,
                                           r.slot_offset, p.skew);
                    payload += p.bytes;
                }
                cursor += r.aligned_bytes;
            }
            if (cursor != e.slot_bytes)
                bad += std::format("  {} slot_bytes {} != the sum of its runs {}\n",
                                   what, e.slot_bytes, cursor);
            if (payload != layout::kExpertBytes)
                bad += std::format("  {} payload {} B != layout::kExpertBytes {}\n",
                                   what, payload, layout::kExpertBytes);
            for (uint8_t p = 0; p < kExpertPartCount; ++p)
                if (e.part_bytes[p] == 0)
                    bad += std::format("  {} is missing part '{}'\n",
                                       what, expert_part_name(static_cast<ExpertPart>(p)));
        }
    }

    for (const EngramEntry& e : engram_) {
        if (e.value.row_bytes != layout::kEngramValueRowBytes)
            bad += std::format("  engram layer {} value row is {} B, expected {}\n",
                               e.layer, e.value.row_bytes, layout::kEngramValueRowBytes);
        if (e.scale.row_bytes != layout::kEngramScaleRowBytes)
            bad += std::format("  engram layer {} scale row is {} B, expected {}\n",
                               e.layer, e.scale.row_bytes, layout::kEngramScaleRowBytes);
        check_range("engram values", e.value.file, e.value.offset, e.rows * e.value.row_bytes);
        check_range("engram scales", e.scale.file, e.scale.offset, e.rows * e.scale.row_bytes);
    }

    if (!bad.empty()) return fail(Err::Corrupt, "manifest is inconsistent:\n" + bad);
    return {};
}

uint64_t Manifest::total_bytes() const {
    uint64_t n = 0;
    for (const auto& [name, t] : tensors_) n += t.bytes + t.scale.bytes;
    for (const auto& row : experts_)
        for (const ExpertEntry& e : row)
            for (uint8_t p = 0; p < kExpertPartCount; ++p) n += e.part_bytes[p];
    return n;
}

uint64_t Manifest::file_bytes() const {
    uint64_t n = 0;
    for (const FileEntry& f : files_) n += f.bytes;
    return n;
}

}  // namespace deepmoe
