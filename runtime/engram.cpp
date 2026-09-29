#include "runtime/engram.h"
#include <future>

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

Result<std::shared_ptr<const EngramTables::ScalePlanes>> EngramTables::load_scales(
    const Manifest& m, const store::ShardSet& shards, std::span<const int64_t> layers) {
    auto planes = std::make_shared<ScalePlanes>();
    for (const int64_t L : layers) {
        const EngramEntry* e = m.engram_for_layer(static_cast<uint32_t>(L));
        if (!e) return fail(Err::NotFound, std::format("no engram table for layer {}", L));
        auto f = shards.require(e->scale.file);
        if (!f) return std::unexpected(f.error());
        const uint64_t bytes = e->rows * layout::kEngramScaleRowBytes;
        std::vector<std::byte> plane(bytes);
        // sector-aligned pieces (the shards are open unbuffered)
        constexpr uint64_t kPiece = 8ull << 20;
        AlignedBuffer buf(kPiece + 2 * kPageSize, kPageSize);
        for (uint64_t done = 0; done < bytes;) {
            const uint64_t off = e->scale.offset + done;
            const uint64_t a0 = off & ~(kPageSize - 1);
            const uint64_t want = std::min(kPiece, bytes - done);
            const uint64_t a1 = (off + want + kPageSize - 1) & ~(kPageSize - 1);
            auto n = (*f)->read_at(a0, MutBytes(buf.data(), a1 - a0));
            if (!n) return fail(n.error().code, std::format("engram layer {} scale plane: {}", L, n.error().message));
            if (*n < (off - a0) + want) return fail(Err::Io, std::format("engram layer {} scale plane: short read", L));
            std::memcpy(plane.data() + done, buf.data() + (off - a0), want);
            done += want;
        }
        planes->by_layer.emplace(static_cast<uint32_t>(L), std::move(plane));
    }
    return std::shared_ptr<const ScalePlanes>(std::move(planes));
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

Result<void> EngramRunner::create(gpu::Device& device, gpu::MemoryAllocator& alloc,
                                  gpu::DecodeRunner& runner,
                                  const Manifest& manifest, const store::ShardSet& shards,
                                  storage::IoEngine& io, const store::PinnedStore& pinned,
                                  const TextConfig& cfg, EngramTables tables) {
    destroy();
    if (!tables.valid()) return fail(Err::InvalidArgument, "engram tables are empty");
    device_ = &device;
    if (auto r = pool_.create(device); !r) return r;
    {
        auto cb = pool_.acquire();
        if (!cb) return std::unexpected(cb.error());
        cmd_ = *cb;
    }
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
    planes_.clear();
    for (const EngramTables::LayerTable& lt : tables_.layers) {
        Planes pl;
        pl.layer   = lt.layer;
        pl.off_val = place(val_bytes);
        pl.off_sc  = place(sc_bytes);
        pl.staging = planes_.size() * kReadsPerToken * kStagingPerRead;
        planes_.push_back(std::move(pl));
    }
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
    auto st = gpu::alloc_host_pages(planes_.size() * kReadsPerToken * kStagingPerRead, false);
    if (!st) { destroy(); return std::unexpected(st.error()); }
    staging_ = *st;
    return {};
}

void EngramRunner::destroy() {
    pool_.destroy();
    cmd_ = gpu::CommandBuffer{};
    device_ = nullptr;
    for (Planes& p : planes_)
        for (auto& [f, skew] : p.pending) if (f.valid()) f.wait();   // reads still landing in `staging_`
    planes_.clear();
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

EngramRunner::Planes* EngramRunner::planes_for(uint32_t layer) {
    for (Planes& p : planes_)
        if (p.layer == layer) return &p;
    return nullptr;
}

bool EngramRunner::fetched(uint32_t layer, uint64_t position) const {
    for (const Planes& p : planes_)
        if (p.layer == layer) return p.fetched_position == position;
    return false;
}

Result<void> EngramRunner::fetch(uint32_t layer, std::span<const uint32_t> history,
                                 uint64_t position) {
    if (!runner_) return fail(Err::FailedPrecondition, "engram runner is not created");
    Planes* pl = planes_for(layer);
    if (!pl) return fail(Err::InvalidArgument, std::format("layer {} has no engram", layer));
    if (auto r = land(*pl); !r) return r;   // the landing zone must be free
    uint64_t rows[layout::kEngramRowsPerToken];
    if (auto r = tables_.hash_rows(layer, history, position, rows); !r) return r;
    auto* base = static_cast<std::byte*>(staging_.ptr) + pl->staging;
    pl->fetched_position = ~0ull;
    for (uint32_t r = 0; r < layout::kEngramRowsPerToken; ++r) {
        auto plan = manifest_->engram_row(layer, rows[r]);
        if (!plan) return std::unexpected(plan.error());
        for (const AlignedRead& a : {plan->value, plan->scale}) {
            if (const std::byte* sc = &a == &plan->scale ? tables_.scale_row(layer, rows[r]) : nullptr) {
                // resident: the 8 bytes go straight into the staging slot,
                // behind an empty future (no shared state to allocate)
                std::memcpy(base + pl->pending.size() * kStagingPerRead, sc, layout::kEngramScaleRowBytes);
                pl->pending.emplace_back(std::future<storage::IoResult>{}, 0u);
                continue;
            }
            if (a.aligned_bytes > kStagingPerRead)
                return fail(Err::Internal,
                            std::format("engram row {} needs {} B of staging, have {}",
                                        rows[r], a.aligned_bytes, kStagingPerRead));
            auto file = shards_->require(a.file);
            if (!file) return std::unexpected(file.error());
            storage::IoRequest req;
            req.priority = IoPriority::Engram;       // design §9.6 P2
            req.file     = *file;
            req.file_off = a.aligned_off;
            req.bytes    = a.aligned_bytes;
            req.dst      = base + pl->pending.size() * kStagingPerRead;
            auto f = io_->submit_future(req);
            if (!f) return std::unexpected(f.error());
            pl->pending.emplace_back(std::move(*f), a.skew);
            bytes_read_ += a.aligned_bytes;
        }
    }
    rows_fetched_ += layout::kEngramRowsPerToken;
    pl->fetched_position = position;
    return {};
}

Result<void> EngramRunner::land(Planes& pl) {
    if (pl.pending.empty()) return {};
    const auto* base = static_cast<const std::byte*>(staging_.ptr) + pl.staging;
    // Gathered into ordinary host memory first and written to the mapping in
    // one memcpy per plane: 24 scattered writes of 256 and 8 bytes each would
    // be the write-combining pattern docs/p2_decode.md §3.3 warns about.
    std::vector<std::byte> vals(size_t(layout::kEngramRowsPerToken) * layout::kEngramValueRowBytes);
    std::vector<std::byte> scs(size_t(layout::kEngramRowsPerToken) * layout::kEngramScaleRowBytes);
    Status failed{Err::Ok};   // the first failed read; the rest still have to land
    for (size_t i = 0; i < pl.pending.size(); ++i) {
        auto& [f, skew] = pl.pending[i];
        if (f.valid()) {   // a resident scale row has no read behind it
            const storage::IoResult res = f.get();
            if (!res.ok() && failed.code == Err::Ok) failed = res.status;
        }
        const std::byte* src = base + i * kStagingPerRead + skew;
        const size_t row = i / 2;
        if (i % 2)
            std::memcpy(scs.data() + row * layout::kEngramScaleRowBytes, src, layout::kEngramScaleRowBytes);
        else
            std::memcpy(vals.data() + row * layout::kEngramValueRowBytes, src, layout::kEngramValueRowBytes);
    }
    pl.pending.clear();
    if (failed.code != Err::Ok) {
        pl.fetched_position = ~0ull;
        return fail(failed.code, std::format("engram layer {}: {}", pl.layer, failed.message));
    }
    std::memcpy(static_cast<std::byte*>(buf_.host_ptr) + pl.off_val, vals.data(), vals.size());
    std::memcpy(static_cast<std::byte*>(buf_.host_ptr) + pl.off_sc, scs.data(), scs.size());
    return {};
}

Result<void> EngramRunner::record(gpu::CommandBuffer& cmd, uint32_t layer, DeviceAddress x_in,
                                  DeviceAddress x_out) {
    if (!runner_) return fail(Err::FailedPrecondition, "engram runner is not created");
    Planes* pl = planes_for(layer);
    if (!pl || pl->fetched_position == ~0ull)
        return fail(Err::FailedPrecondition,
                    std::format("engram layer {} recorded before its rows were fetched", layer));
    if (auto r = land(*pl); !r) return r;
    auto bound = bind_layer(layer);
    if (!bound) return std::unexpected(bound.error());

    const uint32_t dim = cfg_->hidden_size;
    const uint32_t hc  = cfg_->hc_mult;
    const uint32_t k   = layout::kEngramWkvCols;
    const uint32_t rowsn = dim * (hc + 1);

    uint64_t* g = runner_->slots(gpu::DecodeStage::EngramGemv);
    g[gpu::dslot::kRowVal] = buf_.dev_addr + pl->off_val;
    g[gpu::dslot::kRowSc]  = buf_.dev_addr + pl->off_sc;
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
    if (auto r = runner_->record(cmd, gpu::DecodeStage::EngramGemv, &push, sizeof push,
                                 runner_->gemv_groups(rowsn)); !r)
        return r;
    if (auto r = cmd.barrier(); !r) return r;
    if (auto r = runner_->record(cmd, gpu::DecodeStage::EngramGate, &push, sizeof push, hc); !r)
        return r;
    return cmd.barrier();
}

Result<void> EngramRunner::run(uint32_t layer, std::span<const uint32_t> history,
                               uint64_t position, DeviceAddress x_in, DeviceAddress x_out,
                               Profiler* profiler) {
    {
        ScopedPhaseIf phase(profiler, Phase::NvmeStall);   // the reads, landed here, not in `record`
        if (auto r = fetch(layer, history, position); !r) return r;
        if (auto r = land(*planes_for(layer)); !r) return r;
    }
    ScopedPhaseIf phase(profiler, Phase::AttnMisc);
    if (auto r = cmd_.begin(); !r) return r;
    if (auto r = record(cmd_, layer, x_in, x_out); !r) return r;
    if (auto r = cmd_.end(); !r) return r;
    return gpu::submit_and_wait(*device_, cmd_);
}

}  // namespace deepmoe::runtime
