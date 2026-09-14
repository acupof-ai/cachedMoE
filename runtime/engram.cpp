#include "runtime/engram.h"

#include <cstdio>
#include <cstring>
#include <format>

#include "core/align.h"
#include "core/json.h"
#include "core/log.h"

namespace deepmoe::runtime {

namespace {

// One row is two aligned reads (value plane + scale plane), and a 256 B or 8 B
// payload widened to sector boundaries is at most two sectors.
constexpr uint64_t kStagingPerRead = 2 * kPageSize;
constexpr uint32_t kReadsPerToken  = 2 * layout::kEngramRowsPerToken;   // 48

Result<std::vector<int64_t>> int_array(const JsonValue& v, std::string_view key,
                                       size_t want) {
    auto a = v.at(key);
    if (!a) return std::unexpected(a.error());
    auto arr = (*a)->as_array();
    if (!arr) return std::unexpected(arr.error());
    if ((*arr)->size() != want)
        return fail(Err::Corrupt, std::format("engram '{}' has {} entries, expected {}",
                                              key, (*arr)->size(), want));
    std::vector<int64_t> out;
    out.reserve(want);
    for (const JsonValue& e : **arr) {
        auto n = e.as_int();
        if (!n) return std::unexpected(n.error());
        out.push_back(*n);
    }
    return out;
}

}  // namespace

const EngramTables::LayerTable* EngramTables::for_layer(uint32_t layer) const {
    for (const LayerTable& t : layers)
        if (t.layer == layer) return &t;
    return nullptr;
}

Result<EngramTables> EngramTables::load(const std::string& dir) {
    auto doc = json_parse_file(dir + "/index.json");
    if (!doc) return std::unexpected(doc.error());
    auto eg = doc->at("engram");
    if (!eg) return std::unexpected(eg.error());
    const JsonValue& e = **eg;

    EngramTables t;
    t.compressed_vocab_size = static_cast<uint32_t>(e.int_or("compressed_vocab_size", 0));
    t.max_ngram = static_cast<uint32_t>(e.int_or("max_ngram_size", layout::kEngramMaxNgram));
    t.n_heads   = static_cast<uint32_t>(e.int_or("n_heads", layout::kEngramHeads));
    t.head_dim  = static_cast<uint32_t>(e.int_or("head_dim", layout::kEngramHeadDim));
    t.pad_id    = e.int_or("pad_id", 0);
    if (t.max_ngram != layout::kEngramMaxNgram || t.n_heads != layout::kEngramHeads ||
        t.head_dim != layout::kEngramHeadDim)
        return fail(Err::Corrupt,
                    std::format("engram geometry {}x{}x{} does not match model/layout.h",
                                t.max_ngram, t.n_heads, t.head_dim));

    const uint64_t entries = static_cast<uint64_t>(e.int_or("token_map_entries", 0));
    const std::string mapfile = e.string_or("token_map_file", "engram_token_map.bin");
    const std::string path = dir + "/" + mapfile;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(Err::NotFound, std::format("cannot open '{}'", path));
    t.token_map.resize(static_cast<size_t>(entries));
    const size_t got = std::fread(t.token_map.data(), sizeof(int32_t),
                                  t.token_map.size(), f);
    std::fclose(f);
    if (got != t.token_map.size())
        return fail(Err::Io, std::format("'{}': {} of {} entries", path, got,
                                         t.token_map.size()));

    auto ls = e.at("layers");
    if (!ls) return std::unexpected(ls.error());
    auto la = (*ls)->as_array();
    if (!la) return std::unexpected(la.error());
    for (const JsonValue& lv : **la) {
        LayerTable lt;
        lt.layer = static_cast<uint32_t>(lv.int_or("layer", 0));
        lt.num_embeddings = static_cast<uint64_t>(lv.int_or("num_embeddings", 0));
        auto m = int_array(lv, "multipliers", t.max_ngram);
        if (!m) return std::unexpected(m.error());
        auto p = int_array(lv, "primes", layout::kEngramRowsPerToken);
        if (!p) return std::unexpected(p.error());
        auto o = int_array(lv, "offsets", layout::kEngramRowsPerToken);
        if (!o) return std::unexpected(o.error());
        for (size_t i = 0; i < m->size(); ++i) lt.multipliers[i] = (*m)[i];
        for (size_t i = 0; i < p->size(); ++i) lt.primes[i] = (*p)[i];
        for (size_t i = 0; i < o->size(); ++i) lt.offsets[i] = (*o)[i];
        t.layers.push_back(lt);
    }
    return t;
}

Result<void> EngramTables::hash_rows(uint32_t layer, std::span<const uint32_t> history,
                                     uint64_t position, std::span<uint64_t> out) const {
    const LayerTable* t = for_layer(layer);
    if (!t) return fail(Err::NotFound, std::format("no engram table for layer {}", layer));
    if (out.size() != layout::kEngramRowsPerToken)
        return fail(Err::InvalidArgument, "engram needs exactly 24 row slots");
    if (position >= history.size())
        return fail(Err::OutOfRange,
                    std::format("position {} but only {} tokens of history", position,
                                history.size()));

    // `NgramHashState.forward`, for one position. `blocked` is cumulative: once
    // the look-back runs off the start of the sequence every longer n-gram is
    // padded too, so a 4-gram at position 1 is (t1, t0, pad, pad).
    int64_t prod[layout::kEngramMaxNgram]{};
    bool blocked = false;
    for (uint32_t shift = 0; shift < max_ngram; ++shift) {
        if (position < shift) blocked = true;
        int64_t compressed = pad_id;
        if (!blocked) {
            const uint32_t tok = history[static_cast<size_t>(position - shift)];
            if (tok >= token_map.size())
                return fail(Err::OutOfRange,
                            std::format("token {} is outside the {}-entry engram map", tok,
                                        token_map.size()));
            compressed = token_map[tok];
        }
        prod[shift] = compressed * t->multipliers[shift];
    }

    // XOR one lookback at a time: the running value after step i is the hash of
    // the (i+1)-gram, and each n-gram order owns `n_heads` disjoint prime-sized
    // bucket ranges whose bases are `offsets`.
    int64_t rolling = prod[0];
    for (uint32_t i = 1; i < max_ngram; ++i) {
        rolling ^= prod[i];
        for (uint32_t h = 0; h < n_heads; ++h) {
            const uint32_t col = (i - 1) * n_heads + h;
            const int64_t row = rolling % t->primes[col] + t->offsets[col];
            if (row < 0 || static_cast<uint64_t>(row) >= t->num_embeddings)
                return fail(Err::Internal,
                            std::format("engram row {} outside layer {}'s {} rows", row,
                                        layer, t->num_embeddings));
            out[col] = static_cast<uint64_t>(row);
        }
    }
    return {};
}

// --- runner -----------------------------------------------------------------

Result<void> EngramRunner::create(gpu::MemoryAllocator& alloc, gpu::DecodeRunner& runner,
                                  const Manifest& manifest, const store::ShardSet& shards,
                                  storage::IoEngine& io, const store::PinnedStore& pinned,
                                  const TextConfig& cfg, EngramTables tables) {
    destroy();
    if (!tables.valid()) return fail(Err::InvalidArgument, "engram tables are empty");
    alloc_ = &alloc;
    runner_ = &runner;
    manifest_ = &manifest;
    shards_ = &shards;
    io_ = &io;
    pinned_ = &pinned;
    cfg_ = &cfg;
    tables_ = std::move(tables);

    const uint64_t val_bytes = uint64_t(layout::kEngramRowsPerToken) * layout::kEngramValueRowBytes;
    const uint64_t sc_bytes  = uint64_t(layout::kEngramRowsPerToken) * layout::kEngramScaleRowBytes;
    const uint64_t kv_bytes  = uint64_t(cfg.hidden_size) * (cfg.hc_mult + 1) * sizeof(float);
    uint64_t off = 0;
    auto place = [&](uint64_t n) { const uint64_t at = align_up(off, 256); off = at + n; return at; };
    off_val_ = place(val_bytes);
    off_sc_  = place(sc_bytes);
    off_kv_  = place(kv_bytes);

    auto b = alloc.allocate(off, /*host_visible=*/true, /*device_address=*/true);
    if (!b) return std::unexpected(b.error());
    if (!b->host_ptr || b->dev_addr == kNoDeviceAddress) {
        alloc.free(*b);
        return fail(Err::Internal, "engram staging must be host-writable and device-addressable");
    }
    buf_ = *b;
    std::memset(buf_.host_ptr, 0, static_cast<size_t>(off));

    // The I/O lands in ordinary host pages, not in the path-A mapping: the 264 B
    // payload has to be gathered out of the sector-aligned reads, and gathering
    // it means READING the destination, which on a write-combining device
    // mapping is the one thing design §3.3 says never to do.
    auto st = gpu::alloc_host_pages(uint64_t(kReadsPerToken) * kStagingPerRead, false);
    if (!st) { destroy(); return std::unexpected(st.error()); }
    staging_ = *st;
    return {};
}

void EngramRunner::destroy() {
    if (staging_.ptr) gpu::free_host_pages(staging_);
    staging_ = gpu::HostAllocInfo{};
    if (alloc_ && buf_.valid()) alloc_->free(buf_);
    buf_ = gpu::GpuBuffer{};
    alloc_ = nullptr;
    runner_ = nullptr;
}

Result<EngramRunner::LayerBind> EngramRunner::bind_layer(uint32_t layer) const {
    const std::string pre = std::format("layers.{}.engram", layer);
    auto w = pinned_->require(pre + ".wkv.weight");
    if (!w) return std::unexpected(w.error());
    if ((*w)->scale == kNoDeviceAddress)
        return fail(Err::FailedPrecondition, pre + ".wkv has no fp8 scale plane");
    auto q = pinned_->require(pre + ".q_weight");
    if (!q) return std::unexpected(q.error());
    auto k = pinned_->require(pre + ".k_weight");
    if (!k) return std::unexpected(k.error());
    return LayerBind{(*w)->data, (*w)->scale, (*q)->data, (*k)->data};
}

Result<void> EngramRunner::fetch_rows(uint32_t layer, const uint64_t* rows) {
    auto* base = static_cast<std::byte*>(staging_.ptr);
    std::vector<std::future<storage::IoResult>> pending;
    struct Landing { uint32_t slot; uint32_t skew; bool is_scale; uint32_t row; };
    std::vector<Landing> landings;
    pending.reserve(kReadsPerToken);
    landings.reserve(kReadsPerToken);

    uint64_t submitted = 0;
    for (uint32_t r = 0; r < layout::kEngramRowsPerToken; ++r) {
        auto plan = manifest_->engram_row(layer, rows[r]);
        if (!plan) return std::unexpected(plan.error());
        const AlignedRead reads[2] = {plan->value, plan->scale};
        for (uint32_t j = 0; j < 2; ++j) {
            const AlignedRead& a = reads[j];
            if (a.aligned_bytes > kStagingPerRead)
                return fail(Err::Internal,
                            std::format("engram row {} needs {} B of staging, have {}",
                                        rows[r], a.aligned_bytes, kStagingPerRead));
            auto file = shards_->require(a.file);
            if (!file) return std::unexpected(file.error());
            const uint32_t slot = r * 2 + j;
            storage::IoRequest req;
            req.priority = IoPriority::Engram;       // design §9.6 P2
            req.file     = *file;
            req.file_off = a.aligned_off;
            req.bytes    = a.aligned_bytes;
            req.dst      = base + uint64_t(slot) * kStagingPerRead;
            auto f = io_->submit_future(req);
            if (!f) return std::unexpected(f.error());
            pending.push_back(std::move(*f));
            landings.push_back({slot, a.skew, j == 1, r});
            submitted += a.aligned_bytes;
        }
    }
    for (size_t i = 0; i < pending.size(); ++i) {
        const storage::IoResult res = pending[i].get();
        if (!res.ok())
            return fail(res.status.code,
                        std::format("engram layer {} row {}{}: {}", layer,
                                    rows[landings[i].row],
                                    landings[i].is_scale ? " (scale)" : "",
                                    res.status.message));
    }

    auto* vals = static_cast<std::byte*>(buf_.host_ptr) + off_val_;
    auto* scs  = static_cast<std::byte*>(buf_.host_ptr) + off_sc_;
    for (const Landing& l : landings) {
        const std::byte* src = base + uint64_t(l.slot) * kStagingPerRead + l.skew;
        if (l.is_scale)
            std::memcpy(scs + uint64_t(l.row) * layout::kEngramScaleRowBytes, src,
                        layout::kEngramScaleRowBytes);
        else
            std::memcpy(vals + uint64_t(l.row) * layout::kEngramValueRowBytes, src,
                        layout::kEngramValueRowBytes);
    }
    rows_fetched_ += layout::kEngramRowsPerToken;
    bytes_read_   += submitted;
    return {};
}

Result<void> EngramRunner::run(uint32_t layer, std::span<const uint32_t> history,
                               uint64_t position, DeviceAddress x_in, DeviceAddress x_out,
                               Profiler* profiler) {
    if (!runner_) return fail(Err::FailedPrecondition, "engram runner is not created");
    auto bound = bind_layer(layer);
    if (!bound) return std::unexpected(bound.error());

    uint64_t rows[layout::kEngramRowsPerToken];
    if (auto r = tables_.hash_rows(layer, history, position, rows); !r) return r;

    const uint64_t before = bytes_read_;
    {
        // The fetch is 48 four-KiB reads and the GPU has nothing else to do, so
        // it is charged to the NVMe stall bucket of design §13.1 rather than to
        // AttnMisc -- the dispatches below are the AttnMisc half.
        ScopedPhaseIf phase(profiler, Phase::NvmeStall);
        if (auto r = fetch_rows(layer, rows); !r) return r;
    }
    if (profiler) profiler->note_miss_bytes(bytes_read_ - before);

    const uint32_t dim = cfg_->hidden_size;
    const uint32_t hc  = cfg_->hc_mult;
    const uint32_t k   = layout::kEngramWkvCols;
    const uint32_t rowsn = dim * (hc + 1);

    uint64_t* g = runner_->slots(gpu::DecodeStage::EngramGemv);
    g[gpu::dslot::kRowVal] = buf_.dev_addr + off_val_;
    g[gpu::dslot::kRowSc]  = buf_.dev_addr + off_sc_;
    g[gpu::dslot::kW]      = bound->w;
    g[gpu::dslot::kS]      = bound->s;
    g[gpu::dslot::kKv]     = buf_.dev_addr + off_kv_;

    uint64_t* a = runner_->slots(gpu::DecodeStage::EngramGate);
    a[gpu::dslot::kKv]   = buf_.dev_addr + off_kv_;
    a[gpu::dslot::kX]    = x_in;
    a[gpu::dslot::kQW]   = bound->qw;
    a[gpu::dslot::kKW]   = bound->kw;
    a[gpu::dslot::kXout] = x_out;

    gpu::EngramPush push{rowsn, k, k / 32, dim, hc,
                         static_cast<float>(cfg_->rms_norm_eps)};
    ScopedPhaseIf phase(profiler, Phase::AttnMisc);
    if (auto r = runner_->dispatch_now(gpu::DecodeStage::EngramGemv, &push, sizeof push,
                                       runner_->gemv_groups(rowsn)); !r)
        return r;
    return runner_->dispatch_now(gpu::DecodeStage::EngramGate, &push, sizeof push, hc);
}

}  // namespace deepmoe::runtime
