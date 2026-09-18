#include "runtime/decode_state.h"

#include <cstdio>
#include <cstring>
#include <format>

#include "core/json.h"
#include "core/log.h"

namespace deepmoe::runtime {

namespace {

float bf16_to_f32(uint16_t h) {
    const uint32_t b = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

Result<std::vector<uint8_t>> read_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(Err::NotFound, std::format("cannot open '{}'", path));
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> blob(static_cast<size_t>(n < 0 ? 0 : n));
    const size_t got = std::fread(blob.data(), 1, blob.size(), f);
    std::fclose(f);
    if (got != blob.size()) return fail(Err::Io, std::format("short read of '{}'", path));
    return blob;
}

std::vector<uint32_t> uint_array(const JsonValue& doc, std::string_view key) {
    std::vector<uint32_t> out;
    if (const JsonValue* a = doc.find(key))
        if (auto arr = a->as_array())
            for (const JsonValue& e : **arr)
                out.push_back(static_cast<uint32_t>(e.as_int().value_or(0)));
    return out;
}

}  // namespace

// Widens `count` elements of `dtype` at `p` into `t`.
static Result<void> decode_tensor(StateTensor& t, const uint8_t* p) {
    const uint64_t count = t.elements();
    if (t.dtype == "f32") {
        t.f.resize(static_cast<size_t>(count));
        std::memcpy(t.f.data(), p, static_cast<size_t>(count) * 4);
    } else if (t.dtype == "bf16") {
        t.f.resize(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            uint16_t h;
            std::memcpy(&h, p + i * 2, 2);
            t.f[static_cast<size_t>(i)] = bf16_to_f32(h);
        }
    } else if (t.dtype == "i32") {
        t.i.resize(static_cast<size_t>(count));
        std::memcpy(t.i.data(), p, static_cast<size_t>(count) * 4);
        t.f.resize(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
            t.f[static_cast<size_t>(i)] = static_cast<float>(t.i[static_cast<size_t>(i)]);
    } else {
        return fail(Err::Corrupt, std::format("unhandled L3 dtype '{}'", t.dtype));
    }
    return {};
}

Result<void> DecodeState::read_record(const std::string& path, const JsonValue& rec,
                                      Record& out, bool defer_cmp_kv) {
    auto blob = read_file(path);
    if (!blob) return std::unexpected(blob.error());
    const uint64_t base = static_cast<uint64_t>(rec.int_or("data_offset", 12));

    auto tens = rec.at("tensors");
    if (!tens) return std::unexpected(tens.error());
    auto ta = (*tens)->as_array();
    if (!ta) return std::unexpected(ta.error());
    for (const JsonValue& e : **ta) {
        StateTensor t;
        t.dtype = e.string_or("dtype", "");
        if (const JsonValue* sh = e.find("shape"))
            if (auto a = sh->as_array())
                for (const JsonValue& d : **a) t.shape.push_back(d.as_uint().value_or(0));
        const uint64_t off = base + static_cast<uint64_t>(e.int_or("offset", 0));
        const uint64_t n   = static_cast<uint64_t>(e.int_or("bytes", 0));
        if (off + n > blob->size())
            return fail(Err::Corrupt, std::format("'{}' runs past the file", path));
        const std::string name = e.string_or("name", "");
        if (defer_cmp_kv && name.size() > 7 && name.compare(name.size() - 7, 7, ".cmp_kv") == 0) {
            out.deferred.emplace(name, Deferred{path, off, n});
            out.t.emplace(name, std::move(t));     // shape and dtype only
            continue;
        }
        if (auto r = decode_tensor(t, blob->data() + off); !r) return r;
        out.t.emplace(name, std::move(t));
    }
    return {};
}

Result<DecodeState> DecodeState::load(const std::string& dir) {
    auto doc = json_parse_file(dir + "/index.json");
    if (!doc) return std::unexpected(doc.error());

    DecodeState s;
    s.dir_         = dir;
    s.prefill_len_ = static_cast<uint32_t>(doc->int_or("prefill_len", 0));
    s.decode_pos_  = static_cast<uint32_t>(doc->int_or("decode_pos", s.prefill_len_));
    s.steps_       = static_cast<uint32_t>(doc->int_or("steps_exported", 0));
    s.prompt_ids_  = uint_array(*doc, "prompt_ids");
    s.greedy_      = uint_array(*doc, "greedy_tokens");
    if (const JsonValue* c = doc->find("config")) {
        s.layers_   = static_cast<uint32_t>(c->int_or("n_layers", 40));
        s.window_   = static_cast<uint32_t>(c->int_or("window_size", 128));
        s.head_dim_ = static_cast<uint32_t>(c->int_or("head_dim", 512));
    }
    if (s.prompt_ids_.empty() || s.steps_ == 0 || s.layers_ == 0)
        return fail(Err::Corrupt, "L3 index.json is missing prompt_ids / steps / layers");

    auto steps = doc->at("steps");
    if (!steps) return std::unexpected(steps.error());
    auto arr = (*steps)->as_array();
    if (!arr) return std::unexpected(arr.error());

    for (const JsonValue& r : **arr) {
        Record rec;
        rec.name = r.string_or("step", "");
        const std::string path = dir + "/" + r.string_or("file", "");
        if (auto ok = s.read_record(path, r, rec, !s.records_.empty()); !ok)
            return std::unexpected(ok.error());

        RefLogits lg;
        if (const StateTensor* ids = [&] {
                auto it = rec.t.find("top_ids");
                return it == rec.t.end() ? nullptr : &it->second;
            }()) {
            lg.top_ids.reserve(ids->i.size());
            for (int32_t v : ids->i) lg.top_ids.push_back(static_cast<uint32_t>(v));
        }
        auto get = [&](const char* n) -> const StateTensor* {
            auto it = rec.t.find(n);
            return it == rec.t.end() ? nullptr : &it->second;
        };
        if (const StateTensor* v = get("top_logits")) lg.top_logits = v->f;
        if (const StateTensor* v = get("logit_stats"); v && v->f.size() >= 3) {
            lg.max_logit = v->f[0];
            lg.logsumexp = v->f[1];
            lg.min_logit = v->f[2];
        }
        if (const StateTensor* v = get("argmax"); v && !v->i.empty())
            lg.argmax = static_cast<uint32_t>(v->i[0]);
        if (const StateTensor* v = get("in_token"); v && !v->i.empty()) {
            lg.in_token = static_cast<uint32_t>(v->i[0]);
            lg.has_in_token = true;
        }

        // The geometry the KV store must be sized for.
        for (const auto& [name, t] : rec.t) {
            if (name.size() > 7 && name.compare(name.size() - 7, 7, ".cmp_kv") == 0)
                s.max_cmp_ = std::max<uint32_t>(
                    s.max_cmp_, static_cast<uint32_t>(t.elements() / s.head_dim_));
            else if (name.size() > 10 && name.compare(name.size() - 10, 10, ".topk_idxs") == 0)
                s.max_topk_ = std::max<uint32_t>(s.max_topk_,
                                                 static_cast<uint32_t>(t.elements()));
            else if (s.records_.empty() && !t.shape.empty() &&
                     ((name.size() > 10 && name.compare(name.size() - 10, 10, ".cmp_cache") == 0) ||
                      (name.size() > 8 && name.compare(name.size() - 8, 8, ".index_k") == 0)))
                s.max_prefill_rows_ = std::max<uint32_t>(s.max_prefill_rows_,
                                                         static_cast<uint32_t>(t.shape[0]));
        }
        s.records_.push_back(std::move(rec));
        s.logits_.push_back(std::move(lg));
    }
    if (s.records_.size() != size_t(s.steps_) + 1)
        return fail(Err::Corrupt,
                    std::format("L3 export has {} records for {} steps + prefill",
                                s.records_.size(), s.steps_));
    log_info("l3 state: {} prompt tokens, {} steps, {} layers, cmp<={} topk<={}",
             s.prompt_ids_.size(), s.steps_, s.layers_, s.max_cmp_, s.max_topk_);
    return s;
}

const StateTensor* DecodeState::tensor(uint32_t step, const std::string& name) const {
    if (step >= records_.size()) return nullptr;
    const Record& rec = records_[step];
    auto it = rec.t.find(name);
    if (it == rec.t.end()) return nullptr;
    if (auto d = rec.deferred.find(name); d != rec.deferred.end()) {
        std::FILE* f = std::fopen(d->second.path.c_str(), "rb");
        std::vector<uint8_t> bytes(static_cast<size_t>(d->second.bytes));
        bool ok = f != nullptr;
        if (ok) ok = _fseeki64(f, static_cast<int64_t>(d->second.offset), SEEK_SET) == 0 &&
                     std::fread(bytes.data(), 1, bytes.size(), f) == bytes.size();
        if (f) std::fclose(f);
        if (!ok || !decode_tensor(it->second, bytes.data())) {
            log_warn("l3 state: cannot read '{}' from '{}'", name, d->second.path);
            return nullptr;
        }
        rec.deferred.erase(d);
    }
    return &it->second;
}

Result<void> DecodeState::seed_prefill(KvStore& kv) const {
    for (uint32_t L = 0; L < layers_; ++L) {
        const StateTensor* w = tensor(0, std::format("L{:02d}.win_kv", L));
        if (!w)
            return fail(Err::NotFound,
                        std::format("L3 prefill record has no window KV for layer {}", L));
        const uint32_t rows = static_cast<uint32_t>(w->elements() / head_dim_);
        if (auto r = kv.seed_window(L, w->f.data(), rows); !r) return r;

        // The rest of the state the prompt leaves behind, on the four
        // kv_source_layers. An export that predates them has none of these and
        // the caller falls back to seeding the compressed KV per step; see
        // `has_prefill_ced`.
        if (const StateTensor* c = tensor(0, std::format("L{:02d}.cmp_cache", L))) {
            const uint32_t n = static_cast<uint32_t>(c->elements() / head_dim_);
            if (auto r = kv.seed_compressed(L, c->f.data(), n); !r) return r;
        }
        if (const StateTensor* k = tensor(0, std::format("L{:02d}.index_k", L))) {
            const uint32_t dim = kv.config().index_dim;
            const uint32_t n = static_cast<uint32_t>(k->elements() / dim);
            if (auto r = kv.seed_index_k(L, k->f.data(), n); !r) return r;
        }
        const StateTensor* sk = tensor(0, std::format("L{:02d}.cmp_state_kv", L));
        const StateTensor* ss = tensor(0, std::format("L{:02d}.cmp_state_score", L));
        if (sk && ss) {
            const uint32_t ratio = static_cast<uint32_t>(sk->elements() / head_dim_);
            if (auto r = kv.seed_cmp_state(L, sk->f.data(), ss->f.data(), ratio); !r)
                return r;
        }
    }
    return {};
}

bool DecodeState::has_prefill_ced() const {
    for (uint32_t L = 0; L < layers_; ++L)
        if (tensor(0, std::format("L{:02d}.index_k", L)) != nullptr) return true;
    return false;
}

Result<void> DecodeState::seed_step(KvStore& kv, uint32_t s) const {
    if (s >= steps_)
        return fail(Err::OutOfRange, std::format("step {} of {}", s, steps_));
    const uint32_t rec = s + 1;
    for (uint32_t L = 0; L < layers_; ++L) {
        const std::string cname = std::format("L{:02d}.cmp_kv", L);
        const uint32_t owner = kv.config().owner(L);
        if (owner != L && owner != KvStoreConfig::kNoPlane) {
            // A reuse layer's per-step plane is its source's (model.py
            // `shared_attn`), which the store already holds: only the count is
            // this layer's, and it comes from the shape without decoding a
            // [T][512] tensor (Track R2).
            auto it = records_[rec].t.find(cname);
            if (it != records_[rec].t.end())
                if (auto r = kv.set_counts(L, static_cast<uint32_t>(it->second.elements() / head_dim_),
                                           kv.layer(L)->n_kv); !r)
                    return r;
        } else if (const StateTensor* c = tensor(rec, cname)) {
            const uint32_t n = static_cast<uint32_t>(c->elements() / head_dim_);
            if (auto r = kv.seed_compressed(L, c->f.data(), n); !r) return r;
        }
        const StateTensor* t = tensor(rec, std::format("L{:02d}.topk_idxs", L));
        if (!t)
            return fail(Err::NotFound,
                        std::format("L3 step {} has no top-k list for layer {}", s, L));
        if (auto r = kv.seed_topk(L, t->i.data(), static_cast<uint32_t>(t->i.size())); !r)
            return r;
    }
    return {};
}

}  // namespace deepmoe::runtime
