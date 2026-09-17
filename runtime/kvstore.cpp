#include "runtime/kvstore.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>

#include "core/align.h"
#include "cpu/dequant.h"

namespace deepmoe::runtime {

// --- geometry ---------------------------------------------------------------------

KvStoreConfig KvStoreConfig::for_model(const TextConfig& c, uint32_t max_context,
                                       uint32_t initial_context) {
    KvStoreConfig k;
    k.layers          = c.num_hidden_layers;
    k.window          = c.sliding_window;
    k.latent_dim      = c.head_dim;
    k.index_dim       = c.index_head_dim;
    k.index_topk      = c.index_topk;
    k.max_context     = max_context;
    k.initial_context = initial_context;
    k.plane_of.assign(k.layers, kNoPlane);
    k.ratio.assign(k.layers, 0);
    uint32_t cur = kNoPlane;
    for (uint32_t L = 0; L < k.layers; ++L) {
        k.ratio[L] = c.compress_ratio(L);
        if (c.is_kv_source(L) && k.ratio[L]) cur = L;
        k.plane_of[L] = k.ratio[L] ? cur : kNoPlane;
    }
    return k;
}

uint32_t KvStoreConfig::owner(uint32_t l) const {
    if (l >= layers) return kNoPlane;
    return per_source() ? plane_of[l] : l;
}

uint32_t KvStoreConfig::plane_ratio(uint32_t o) const {
    if (!per_source() || o >= ratio.size()) return 1;
    return std::max<uint32_t>(1, ratio[o]);
}

std::vector<uint32_t> KvStoreConfig::owners() const {
    std::vector<uint32_t> out;
    for (uint32_t l = 0; l < layers; ++l)
        if (owner(l) == l) out.push_back(l);
    return out;
}

uint32_t KvStoreConfig::topk_rows() const {
    return window + std::min(index_topk, max_context);
}

uint64_t KvStoreConfig::compressed_bytes(uint32_t positions) const {
    uint64_t n = 0;
    for (uint32_t o : owners()) n += uint64_t(rows_for(o, positions)) * latent_dim * 2;
    return n;
}

uint64_t KvStoreConfig::index_key_bytes(uint32_t positions) const {
    uint64_t n = 0;
    for (uint32_t o : owners()) n += uint64_t(rows_for(o, positions)) * index_dim * 2;
    return n;
}

uint64_t KvStoreConfig::cmp_state_bytes() const {
    return uint64_t(owners().size()) * max_ratio * latent_dim * 4 * 2;
}

uint64_t KvStoreConfig::packed_bytes(uint32_t positions) const {
    uint64_t n = 0;
    for (uint32_t o : owners()) {
        const uint32_t r = plane_ratio(o);
        n += uint64_t(positions / r) *
             (latent_dim / 2 + latent_dim / 16 + index_dim / 2 + index_dim / 32);
        if (r > 1) n += uint64_t(r) * latent_dim * 4 * 2;
    }
    return n;
}

// --- packing ----------------------------------------------------------------------

namespace {

const float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float scale_of(uint8_t code, bool e8m0) {
    return e8m0 ? cpu::e8m0_to_float(code) : cpu::fp8_e4m3_to_float(code);
}

// One block at a given scale: nibbles, or false if some value is off the grid.
bool pack_block(const uint16_t* b, uint32_t n, float sc, uint8_t* codes) {
    for (uint32_t i = 0; i < n; ++i) {
        const float v = cpu::bf16_to_float(b[i]);
        const float a = std::fabs(v);
        const uint8_t sign = (b[i] & 0x8000u) ? 8u : 0u;
        uint8_t j = 0xFF;
        for (uint8_t m = 0; m < 8; ++m)
            if (kMag[m] * sc == a) { j = m; break; }
        if (j == 0xFF) return false;
        const float back = sign ? -(kMag[j] * sc) : kMag[j] * sc;
        if (cpu::float_to_bf16(back) != b[i]) return false;
        codes[i] = static_cast<uint8_t>(sign | j);
    }
    return true;
}

}  // namespace

bool pack_fp4_row(const uint16_t* bf16, uint32_t n, uint32_t block, bool e8m0,
                  uint8_t* nibbles, uint8_t* scales) {
    if (block == 0 || n % block || n % 2) return false;
    uint8_t codes[64];
    if (block > 64) return false;
    std::memset(nibbles, 0, n / 2);
    for (uint32_t b0 = 0, bi = 0; b0 < n; b0 += block, ++bi) {
        const uint16_t* blk = bf16 + b0;
        float amax = 0.0f;
        for (uint32_t i = 0; i < block; ++i) {
            const float v = cpu::bf16_to_float(blk[i]);
            if (!std::isfinite(v)) return false;
            amax = std::max(amax, std::fabs(v));
        }
        bool ok = false;
        uint8_t code = 0;
        // The quantiser's own choice first: E4M3 amax / 6 (the largest value
        // lands on +-6), E8M0 the smallest power of two >= amax / 6. Then every
        // scale, for the blocks whose largest value is not on 6 (a floored
        // all-small block, or E8M0's amax/sc in (3, 5)).
        auto try_code = [&](uint32_t c) {
            if (e8m0 ? c > 254 : (c > 127 || !std::isfinite(cpu::fp8_e4m3_to_float(uint8_t(c)))))
                return false;
            if (amax > 0.0f && scale_of(uint8_t(c), e8m0) <= 0.0f) return false;
            if (!pack_block(blk, block, scale_of(uint8_t(c), e8m0), codes)) return false;
            code = uint8_t(c);
            return true;
        };
        if (amax == 0.0f) {
            ok = try_code(0);
        } else if (e8m0) {
            int e = 0;
            (void)std::frexp(amax / 6.0f, &e);   // amax/6 in [2^(e-1), 2^e)
            for (int d : {-1, 0, 1, 2}) {
                const int c = e - 1 + d + 127;
                if (c >= 0 && c <= 254 && try_code(uint32_t(c))) { ok = true; break; }
            }
        } else {
            const float want = amax / 6.0f;
            for (uint32_t c = 1; c < 128 && !ok; ++c)
                if (cpu::fp8_e4m3_to_float(uint8_t(c)) == want) ok = try_code(c);
        }
        for (uint32_t c = 0; !ok && c < (e8m0 ? 255u : 128u); ++c) ok = try_code(c);
        if (!ok) return false;
        scales[bi] = code;
        for (uint32_t i = 0; i < block; ++i) {
            const uint32_t k = b0 + i;
            if (k & 1) nibbles[k >> 1] |= static_cast<uint8_t>(codes[i] << 4);
            else       nibbles[k >> 1] |= codes[i];
        }
    }
    return true;
}

void unpack_fp4_row(const uint8_t* nibbles, const uint8_t* scales, uint32_t n, uint32_t block,
                    bool e8m0, uint16_t* bf16) {
    for (uint32_t k = 0; k < n; ++k) {
        const uint8_t c = (k & 1) ? uint8_t(nibbles[k >> 1] >> 4) : uint8_t(nibbles[k >> 1] & 0x0F);
        const float v = kMag[c & 7] * scale_of(scales[k / block], e8m0);
        bf16[k] = cpu::float_to_bf16((c & 8) ? -v : v);
    }
}

uint64_t KvPacked::bytes() const {
    uint64_t n = sizeof(*this);
    for (const KvPackedPlane& p : planes)
        n += p.cmp_fp4.size() + p.cmp_scale.size() + p.key_fp4.size() + p.key_scale.size() +
             p.raw_rows.size() * 4 + (p.raw_cmp.size() + p.raw_key.size()) * 2 +
             (p.carry_kv.size() + p.carry_score.size()) * 4;
    return n;
}

uint32_t KvPacked::raw_rows() const {
    uint32_t n = 0;
    for (const KvPackedPlane& p : planes) n += static_cast<uint32_t>(p.raw_rows.size());
    return n;
}

uint64_t KvRowBackup::bytes() const {
    uint64_t n = 0;
    for (const Plane& p : planes)
        n += (p.cmp.size() + p.key.size()) * 2 + (p.carry_kv.size() + p.carry_score.size()) * 4;
    return n;
}

// --- the store --------------------------------------------------------------------

uint32_t KvStore::Layout::slab_of(uint64_t off) const {
    for (size_t i = slab_at.size(); i-- > 0;)
        if (off >= slab_at[i]) return static_cast<uint32_t>(i);
    return 0;
}

// The regions in a flat space cut into slabs of at most `cfg_.slab_bytes`: a
// region that would cross the cap starts the next slab instead, so every
// region -- and so every row address derived from one -- lies in a single
// allocation (design 5.3; the 2 GiB maxMemoryAllocationSize of 1.1 is what
// 11.3 called "max_context <~ 41.7K").
KvStore::Layout KvStore::layout_for(uint32_t cap) const {
    Layout l;
    l.cap = cap;
    const uint64_t limit = cfg_.slab_bytes ? cfg_.slab_bytes : ~0ull;
    uint64_t off = 0;
    l.slab_at.push_back(0);
    auto place = [&](uint64_t bytes) {
        uint64_t at = align_up(off, 256);
        if (bytes && at + bytes - l.slab_at.back() > limit && at > l.slab_at.back()) {
            // Close this slab and open the next. `bytes` may still be over the
            // cap on its own, which allocate_slabs() reports as it is.
            l.slab_size.push_back(align_up(at, 4096) - l.slab_at.back());
            at = l.slab_at.back() + l.slab_size.back();
            l.slab_at.push_back(at);
        }
        off = at + bytes;
        return at;
    };
    const KvStoreConfig& c = cfg_;
    l.off_win_val   = place(uint64_t(c.layers) * c.window * c.latent_dim);
    l.off_win_scale = place(uint64_t(c.layers) * c.window * c.window_scales());
    l.off_top       = place(c.topk_bytes());
    // kv_state for every owner, then score_state for every owner, so the -inf
    // fill is one contiguous range.
    l.off_state     = place(uint64_t(owners_.size()) * c.max_ratio * c.latent_dim * 4 * 2);
    for (uint32_t o : owners_) l.off_cmp.push_back(place(uint64_t(c.rows_for(o, cap)) * c.latent_dim * 2));
    for (uint32_t o : owners_) l.off_idx.push_back(place(uint64_t(c.rows_for(o, cap)) * c.index_dim * 2));
    l.total = align_up(off, 4096);
    l.slab_size.push_back(l.total - l.slab_at.back());
    return l;
}

std::byte* KvStore::at(uint64_t off) const {
    const uint32_t s = lay_.slab_of(off);
    return static_cast<std::byte*>(bufs_[s].host_ptr) + (off - lay_.slab_at[s]);
}

DeviceAddress KvStore::dev(uint64_t off) const {
    const uint32_t s = lay_.slab_of(off);
    return bufs_[s].dev_addr + (off - lay_.slab_at[s]);
}

uint64_t KvStore::bytes() const {
    uint64_t n = 0;
    for (const gpu::GpuBuffer& b : bufs_) n += b.bytes;
    return n;
}

uint64_t KvStore::largest_slab() const {
    uint64_t n = 0;
    for (const gpu::GpuBuffer& b : bufs_) n = std::max(n, b.bytes);
    return n;
}

void KvStore::free_slabs(gpu::MemoryAllocator* a, std::vector<gpu::GpuBuffer>& v) {
    if (a)
        for (gpu::GpuBuffer& b : v)
            if (b.valid()) a->free(b);
    v.clear();
}

Result<std::vector<gpu::GpuBuffer>> KvStore::allocate_slabs(const Layout& l) const {
    std::vector<gpu::GpuBuffer> out;
    for (size_t i = 0; i < l.slab_size.size(); ++i) {
        auto b = alloc_->allocate(l.slab_size[i], /*host_visible=*/true, /*device_address=*/true);
        if (!b) {
            free_slabs(alloc_, out);
            return fail(b.error().code,
                        std::format("KV slab {} of {} ({} B): {}", i + 1, l.slab_size.size(),
                                    l.slab_size[i], b.error().message));
        }
        if (!b->host_ptr || b->dev_addr == kNoDeviceAddress) {
            alloc_->free(*b);
            free_slabs(alloc_, out);
            return fail(Err::Internal, "the KV store must be host-writable and device-addressable");
        }
        out.push_back(*b);
    }
    return out;
}

uint32_t KvStore::owner_index(uint32_t l) const {
    return l < owner_of_.size() ? owner_of_[l] : KvStoreConfig::kNoPlane;
}

Result<void> KvStore::create(gpu::MemoryAllocator& alloc, const KvStoreConfig& cfg) {
    destroy();
    if (cfg.latent_dim % cfg.scale_block || cfg.latent_dim % 16 || cfg.index_dim % 32)
        return fail(Err::InvalidArgument, "latent_dim / index_dim do not divide into scale blocks");
    if (cfg.per_source() && (cfg.plane_of.size() != cfg.layers || cfg.ratio.size() != cfg.layers))
        return fail(Err::InvalidArgument, "plane_of / ratio must have one entry per layer");
    if (cfg.max_context == 0) return fail(Err::InvalidArgument, "max_context is 0");
    cfg_ = cfg;
    owners_ = cfg.owners();
    owner_of_.assign(cfg.layers, KvStoreConfig::kNoPlane);
    for (uint32_t l = 0; l < cfg.layers; ++l) {
        const uint32_t o = cfg.owner(l);
        if (o == KvStoreConfig::kNoPlane) continue;
        const auto it = std::find(owners_.begin(), owners_.end(), o);
        if (it == owners_.end())
            return fail(Err::InvalidArgument, std::format("layer {} reads layer {}'s planes, which "
                                                          "owns none", l, o));
        owner_of_[l] = static_cast<uint32_t>(it - owners_.begin());
    }
    const uint32_t cap = cfg.initial_context ? std::min(cfg.initial_context, cfg.max_context)
                                             : cfg.max_context;
    alloc_ = &alloc;
    // A driver whose cap is below `slab_bytes` refuses the SIZE (InvalidArgument)
    // rather than failing to find memory; halve and lay out again, so a device
    // with a smaller maxMemoryAllocationSize needs no configuration.
    Status last{};
    for (int tries = 0; tries < 8 && bufs_.empty(); ++tries) {
        Layout lay = layout_for(cap);
        auto b = allocate_slabs(lay);
        if (b) {
            bufs_ = std::move(*b);
            lay_  = std::move(lay);
            break;
        }
        last = b.error();
        if (b.error().code != Err::InvalidArgument || cfg_.slab_bytes <= (64ull << 20)) break;
        cfg_.slab_bytes /= 2;
    }
    if (bufs_.empty()) {
        alloc_ = nullptr;
        return std::unexpected(last);
    }
    cap_   = cap;
    rows_hw_.assign(owners_.size(), 0);
    n_cmp_.assign(cfg.layers, 0);
    n_kv_.assign(cfg.layers, 0);
    for (gpu::GpuBuffer& b : bufs_) std::memset(b.host_ptr, 0, static_cast<size_t>(b.bytes));
    clear();
    return {};
}

void KvStore::clear() {
    if (!valid()) return;
    const KvStoreConfig& c = cfg_;
    std::memset(at(lay_.off_win_val), 0, size_t(c.layers) * c.window * c.latent_dim);
    std::memset(at(lay_.off_win_scale), 0, size_t(c.layers) * c.window * c.window_scales());
    std::memset(at(lay_.off_top), 0, static_cast<size_t>(c.topk_bytes()));
    // `Compressor.score_state` starts at -inf, not at zero: a slot that has
    // never held a token must contribute nothing to the pooling softmax, and
    // exp(-inf - max) is the only value that does that. Zero would make an
    // unwritten slot an equal partner (model.py, `torch.full(..., -torch.inf)`).
    //
    // -inf is not a byte pattern, so the fill is built ONCE in ordinary host
    // memory and memcpy'd in: design 7.1 rule 10 forbids a scalar loop over
    // GPU-visible memory, and written that way this one is 16,384 uncached
    // stores (~230 ns each = 3.8 ms) on every clear() -- every context reset,
    // every restore, every rollback that resets.
    const uint64_t n = uint64_t(owners_.size()) * c.max_ratio * c.latent_dim;
    std::memset(at(lay_.off_state), 0, size_t(n) * 4);
    if (ninf_.size() < n) ninf_.assign(size_t(n), -std::numeric_limits<float>::infinity());
    std::memcpy(at(lay_.off_state + n * 4), ninf_.data(), size_t(n) * 4);
    for (size_t oi = 0; oi < owners_.size(); ++oi) {
        std::memset(at(lay_.off_cmp[oi]), 0, size_t(rows_hw_[oi]) * c.latent_dim * 2);
        std::memset(at(lay_.off_idx[oi]), 0, size_t(rows_hw_[oi]) * c.index_dim * 2);
        rows_hw_[oi] = 0;
    }
    std::fill(n_cmp_.begin(), n_cmp_.end(), 0u);
    std::fill(n_kv_.begin(), n_kv_.end(), 0u);
    win_lo_ = 0;
    slot_pos_.assign(c.window, kSlotEmpty);
}

void KvStore::resolve_ring(uint32_t n) {
    const uint32_t w = cfg_.window;
    for (uint32_t s = 0; s < slot_pos_.size(); ++s) {
        if (slot_pos_[s] != kSlotUnknown) continue;
        if (n <= s) { slot_pos_[s] = kSlotEmpty; continue; }
        slot_pos_[s] = int64_t(s) + int64_t((n - 1 - s) / w) * w;
    }
}

bool KvStore::ring_holds(uint32_t position) const {
    if (slot_pos_.empty() || position < win_lo_) return false;
    return slot_pos_[position % cfg_.window] == int64_t(position);
}

void KvStore::destroy() {
    free_slabs(alloc_, bufs_);
    alloc_ = nullptr;
    lay_ = Layout{};
    cap_ = 0;
    owners_.clear();
    owner_of_.clear();
    rows_hw_.clear();
    n_cmp_.clear();
    n_kv_.clear();
    win_lo_ = 0;
    slot_pos_.clear();
}

Result<void> KvStore::reserve(uint32_t positions) {
    if (!valid()) return fail(Err::FailedPrecondition, "KV store is not created");
    if (positions <= cap_) return {};
    if (positions > cfg_.max_context)
        return fail(Err::ResourceExhausted,
                    std::format("{} positions against a KV store limited to {}", positions,
                                cfg_.max_context));
    const uint32_t doubled = cap_ > cfg_.max_context / 2 ? cfg_.max_context : cap_ * 2;
    const uint32_t nc = std::min(cfg_.max_context, std::max(positions, doubled));
    Layout nl = layout_for(nc);
    auto nb = allocate_slabs(nl);
    if (!nb) return std::unexpected(nb.error());
    for (gpu::GpuBuffer& b : *nb) std::memset(b.host_ptr, 0, static_cast<size_t>(b.bytes));
    const KvStoreConfig& c = cfg_;
    // Region by region: the two layouts may cut their slabs differently, so
    // "the fixed head is one memcpy" no longer holds. Each of these is still
    // one bulk copy (7.1 rule 10) between two mappings of GPU-visible memory.
    auto dst = [&](uint64_t off) {
        const uint32_t si = nl.slab_of(off);
        return static_cast<std::byte*>((*nb)[si].host_ptr) + (off - nl.slab_at[si]);
    };
    std::memcpy(dst(nl.off_win_val), at(lay_.off_win_val),
                size_t(c.layers) * c.window * c.latent_dim);
    std::memcpy(dst(nl.off_win_scale), at(lay_.off_win_scale),
                size_t(c.layers) * c.window * c.window_scales());
    std::memcpy(dst(nl.off_top), at(lay_.off_top), static_cast<size_t>(c.topk_bytes()));
    std::memcpy(dst(nl.off_state), at(lay_.off_state),
                size_t(owners_.size()) * c.max_ratio * c.latent_dim * 4 * 2);
    for (size_t oi = 0; oi < owners_.size(); ++oi) {
        std::memcpy(dst(nl.off_cmp[oi]), at(lay_.off_cmp[oi]),
                    size_t(rows_hw_[oi]) * c.latent_dim * 2);
        std::memcpy(dst(nl.off_idx[oi]), at(lay_.off_idx[oi]),
                    size_t(rows_hw_[oi]) * c.index_dim * 2);
    }
    free_slabs(alloc_, bufs_);
    bufs_ = std::move(*nb);
    lay_ = std::move(nl);
    cap_ = nc;
    return {};
}

Result<KvLayerView> KvStore::layer(uint32_t l) const {
    if (!valid()) return fail(Err::FailedPrecondition, "KV store is not created");
    if (l >= cfg_.layers)
        return fail(Err::OutOfRange, std::format("layer {} of {}", l, cfg_.layers));
    const KvStoreConfig& c = cfg_;
    const uint64_t win  = uint64_t(c.window) * c.latent_dim;
    const uint64_t wins = uint64_t(c.window) * c.window_scales();
    const uint64_t top  = uint64_t(c.topk_rows()) * sizeof(uint32_t);
    const uint64_t stp  = uint64_t(c.max_ratio) * c.latent_dim * 4;
    const uint64_t score_base = lay_.off_state + uint64_t(owners_.size()) * stp;
    // A window-only layer never reads the compressed half; it is handed the
    // first plane so every address it binds is a real one.
    uint32_t oi = owner_index(l);
    const bool own = oi != KvStoreConfig::kNoPlane;
    if (!own) oi = 0;

    KvLayerView v;
    v.win_val   = dev(lay_.off_win_val + l * win);
    v.win_scale = dev(lay_.off_win_scale + l * wins);
    v.top_idx   = dev(lay_.off_top + l * top);
    v.win_val_host   = reinterpret_cast<uint8_t*>(at(lay_.off_win_val + l * win));
    v.win_scale_host = reinterpret_cast<uint8_t*>(at(lay_.off_win_scale + l * wins));
    v.top_idx_host   = reinterpret_cast<uint32_t*>(at(lay_.off_top + l * top));
    if (!owners_.empty()) {
        v.cmp_kv  = dev(lay_.off_cmp[oi]);
        v.idx_key = dev(lay_.off_idx[oi]);
        v.cmp_state_kv    = dev(lay_.off_state + oi * stp);
        v.cmp_state_score = dev(score_base + oi * stp);
        v.cmp_kv_host  = reinterpret_cast<uint16_t*>(at(lay_.off_cmp[oi]));
        v.idx_key_host = reinterpret_cast<uint16_t*>(at(lay_.off_idx[oi]));
        v.cmp_state_kv_host    = reinterpret_cast<float*>(at(lay_.off_state + oi * stp));
        v.cmp_state_score_host = reinterpret_cast<float*>(at(score_base + oi * stp));
        if (own) v.plane_owner = owners_[oi];
    }
    v.n_cmp = n_cmp_[l];
    v.n_kv  = n_kv_[l];
    return v;
}

Result<void> KvStore::seed_window(uint32_t layer, const float* values, uint32_t rows) {
    auto v = this->layer(layer);
    if (!v) return std::unexpected(v.error());
    if (rows > cfg_.window)
        return fail(Err::InvalidArgument,
                    std::format("{} window rows but the ring holds {}", rows, cfg_.window));
    const uint32_t blocks = cfg_.window_scales();
    std::vector<float> back(cfg_.scale_block);
    for (uint32_t r = 0; r < rows && r < slot_pos_.size(); ++r) slot_pos_[r] = kSlotUnknown;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t b = 0; b < blocks; ++b) {
            const float s = cpu::act_quant_block(values + size_t(r) * cfg_.latent_dim
                                                        + b * cfg_.scale_block,
                                                 cfg_.scale_block,
                                                 v->win_val_host + size_t(r) * cfg_.latent_dim
                                                                 + b * cfg_.scale_block,
                                                 back.data());
            v->win_scale_host[size_t(r) * blocks + b] = cpu::e8m0_encode(s);
        }
    }
    return {};
}

Result<void> KvStore::seed_compressed(uint32_t layer, const float* values, uint32_t n_cmp) {
    if (layer >= cfg_.layers)
        return fail(Err::OutOfRange, std::format("layer {} of {}", layer, cfg_.layers));
    const uint32_t oi = owner_index(layer);
    if (oi == KvStoreConfig::kNoPlane || owners_[oi] != layer) {
        // The rows are the owner's; what this layer may read of them is a count.
        n_cmp_[layer] = n_cmp;
        return {};
    }
    const uint32_t ratio = cfg_.plane_ratio(layer);
    if (uint64_t(n_cmp) * ratio > cfg_.max_context)
        return fail(Err::InvalidArgument,
                    std::format("{} compressed entries at ratio {} but the store holds {} positions",
                                n_cmp, ratio, cfg_.max_context));
    if (auto r = reserve(n_cmp * ratio); !r) return r;
    auto v = this->layer(layer);
    if (!v) return std::unexpected(v.error());
    for (size_t i = 0; i < size_t(n_cmp) * cfg_.latent_dim; ++i)
        v->cmp_kv_host[i] = cpu::float_to_bf16(values[i]);
    rows_hw_[oi] = std::max(rows_hw_[oi], n_cmp);
    n_cmp_[layer] = n_cmp;
    return {};
}

Result<void> KvStore::seed_topk(uint32_t layer, const int32_t* idx, uint32_t n_kv) {
    auto v = this->layer(layer);
    if (!v) return std::unexpected(v.error());
    if (n_kv > cfg_.topk_rows())
        return fail(Err::InvalidArgument,
                    std::format("a {}-entry top-k list; the store holds {}", n_kv, cfg_.topk_rows()));
    std::memcpy(v->top_idx_host, idx, size_t(n_kv) * sizeof(int32_t));
    n_kv_[layer] = n_kv;
    return {};
}

Result<void> KvStore::seed_index_k(uint32_t layer, const float* values, uint32_t rows) {
    if (layer >= cfg_.layers)
        return fail(Err::OutOfRange, std::format("layer {} of {}", layer, cfg_.layers));
    const uint32_t oi = owner_index(layer);
    if (oi == KvStoreConfig::kNoPlane || owners_[oi] != layer) return {};
    const uint32_t ratio = cfg_.plane_ratio(layer);
    if (uint64_t(rows) * ratio > cfg_.max_context)
        return fail(Err::InvalidArgument,
                    std::format("{} index keys at ratio {} but the store holds {} positions", rows,
                                ratio, cfg_.max_context));
    if (auto r = reserve(rows * ratio); !r) return r;
    auto v = this->layer(layer);
    if (!v) return std::unexpected(v.error());
    for (size_t i = 0; i < size_t(rows) * cfg_.index_dim; ++i)
        v->idx_key_host[i] = cpu::float_to_bf16(values[i]);
    rows_hw_[oi] = std::max(rows_hw_[oi], rows);
    return {};
}

Result<void> KvStore::seed_cmp_state(uint32_t layer, const float* kv, const float* score,
                                     uint32_t ratio) {
    if (layer >= cfg_.layers)
        return fail(Err::OutOfRange, std::format("layer {} of {}", layer, cfg_.layers));
    const uint32_t oi = owner_index(layer);
    if (oi == KvStoreConfig::kNoPlane || owners_[oi] != layer) return {};
    if (ratio > cfg_.max_ratio)
        return fail(Err::InvalidArgument,
                    std::format("compress ratio {} above the {} slots the state holds",
                                ratio, cfg_.max_ratio));
    auto v = this->layer(layer);
    if (!v) return std::unexpected(v.error());
    const size_t n = size_t(ratio) * cfg_.latent_dim;
    std::memcpy(v->cmp_state_kv_host, kv, n * sizeof(float));
    std::memcpy(v->cmp_state_score_host, score, n * sizeof(float));
    return {};
}

Result<void> KvStore::set_counts(uint32_t layer, uint32_t n_cmp, uint32_t n_kv) {
    if (layer >= cfg_.layers)
        return fail(Err::OutOfRange, std::format("layer {} of {}", layer, cfg_.layers));
    const uint32_t oi = owner_index(layer);
    const uint32_t ratio = oi == KvStoreConfig::kNoPlane ? 1 : cfg_.plane_ratio(owners_[oi]);
    if (uint64_t(n_cmp) * ratio > cfg_.max_context || n_kv > cfg_.topk_rows())
        return fail(Err::InvalidArgument,
                    std::format("layer {} wants {} compressed / {} total KV entries, "
                                "above the {} positions / {} list entries the store was sized for",
                                layer, n_cmp, n_kv, cfg_.max_context, cfg_.topk_rows()));
    if (oi != KvStoreConfig::kNoPlane) {
        if (auto r = reserve(n_cmp * ratio); !r) return r;
        rows_hw_[oi] = std::max(rows_hw_[oi], n_cmp);
    }
    n_cmp_[layer] = n_cmp;
    n_kv_[layer]  = n_kv;
    return {};
}

Result<void> KvStore::set_decode_topk(uint32_t layer, uint32_t position, uint32_t n_cmp,
                                      uint32_t n_sel) {
    if (n_sel > n_cmp)
        return fail(Err::InvalidArgument, std::format("{} picks of {} compressed positions", n_sel, n_cmp));
    if (auto r = reserve(position + 1); !r) return r;
    if (auto r = set_counts(layer, n_cmp, cfg_.window + n_sel); !r) return r;
    auto v = this->layer(layer);
    if (!v) return std::unexpected(v.error());
    // model.py's `get_window_topk_idxs(win, 1, 1, start_pos)`: the ring listed
    // oldest first from `start_pos % win + 1`, with any slot the sequence has
    // not reached yet marked -1. Order inside the list does not matter to
    // sparse_attn, which handles every slot independently -- but the -1s do.
    // The i-th entry holds position `position - (win - 1 - i)`; below 0 is the
    // reference's "ring still filling", below the window floor is a slot a
    // replay has not rebuilt (and whose bytes belong to some other position).
    const uint32_t win = cfg_.window;
    // This step writes the slot; the ring bookkeeping records it now.
    slot_pos_[position % win] = position;
    const uint32_t oldest = position % win + 1;
    auto* out = reinterpret_cast<int32_t*>(v->top_idx_host);
    for (uint32_t i = 0; i < win; ++i) {
        const uint32_t slot = (oldest + i) % win;
        const int64_t  pos  = int64_t(position) - int64_t(win - 1 - i);
        out[i] = (pos < 0 || pos < int64_t(win_lo_)) ? -1 : static_cast<int32_t>(slot);
    }
    return {};
}

// --- packing whole contexts -------------------------------------------------------

Result<KvPacked> KvStore::pack(uint32_t positions) const {
    if (!valid()) return fail(Err::FailedPrecondition, "KV store is not created");
    const KvStoreConfig& c = cfg_;
    KvPacked out;
    out.positions = positions;
    for (size_t oi = 0; oi < owners_.size(); ++oi) {
        const uint32_t L = owners_[oi];
        KvPackedPlane p;
        p.layer = L;
        p.ratio = c.plane_ratio(L);
        p.rows  = positions / p.ratio;
        if (p.rows > rows_hw_[oi] || p.rows > c.rows_for(L, cap_))
            return fail(Err::FailedPrecondition,
                        std::format("packing {} positions needs {} rows of layer {}'s plane; {} "
                                    "were ever written", positions, p.rows, L, rows_hw_[oi]));
        const uint32_t cd = c.latent_dim, kd = c.index_dim;
        p.cmp_fp4.resize(size_t(p.rows) * cd / 2);
        p.cmp_scale.resize(size_t(p.rows) * cd / 16);
        p.key_fp4.resize(size_t(p.rows) * kd / 2);
        p.key_scale.resize(size_t(p.rows) * kd / 32);
        // One copy out of the mapping first: reading GPU-visible memory element
        // by element is the slow way (a write-combining read, design §3.3).
        std::vector<uint16_t> cmp(size_t(p.rows) * cd), key(size_t(p.rows) * kd);
        std::memcpy(cmp.data(), at(lay_.off_cmp[oi]), cmp.size() * 2);
        std::memcpy(key.data(), at(lay_.off_idx[oi]), key.size() * 2);
        for (uint32_t r = 0; r < p.rows; ++r) {
            const bool okc = pack_fp4_row(cmp.data() + size_t(r) * cd, cd, 16, false,
                                          p.cmp_fp4.data() + size_t(r) * cd / 2,
                                          p.cmp_scale.data() + size_t(r) * cd / 16);
            const bool okk = pack_fp4_row(key.data() + size_t(r) * kd, kd, 32, true,
                                          p.key_fp4.data() + size_t(r) * kd / 2,
                                          p.key_scale.data() + size_t(r) * kd / 32);
            if (okc && okk) continue;
            p.raw_rows.push_back(r);
            p.raw_cmp.insert(p.raw_cmp.end(), cmp.begin() + size_t(r) * cd, cmp.begin() + size_t(r + 1) * cd);
            p.raw_key.insert(p.raw_key.end(), key.begin() + size_t(r) * kd, key.begin() + size_t(r + 1) * kd);
        }
        if (p.ratio > 1) {
            const uint64_t stp = uint64_t(c.max_ratio) * cd * 4;
            const uint64_t score_base = lay_.off_state + uint64_t(owners_.size()) * stp;
            const size_t n = size_t(p.ratio) * cd;
            p.carry_kv.resize(n);
            p.carry_score.resize(n);
            std::memcpy(p.carry_kv.data(), at(lay_.off_state + oi * stp), n * 4);
            std::memcpy(p.carry_score.data(), at(score_base + oi * stp), n * 4);
        }
        out.planes.push_back(std::move(p));
    }
    return out;
}

Result<void> KvStore::unpack(const KvPacked& pk) {
    if (!valid()) return fail(Err::FailedPrecondition, "KV store is not created");
    if (auto r = reserve(std::max<uint32_t>(pk.positions, 1)); !r) return r;
    const KvStoreConfig& c = cfg_;
    const uint32_t cd = c.latent_dim, kd = c.index_dim;
    for (const KvPackedPlane& p : pk.planes) {
        const uint32_t oi = owner_index(p.layer);
        if (oi == KvStoreConfig::kNoPlane || owners_[oi] != p.layer || c.plane_ratio(p.layer) != p.ratio)
            return fail(Err::InvalidArgument,
                        std::format("packed plane of layer {} at ratio {} does not match this store",
                                    p.layer, p.ratio));
        std::vector<uint16_t> cmp(size_t(p.rows) * cd), key(size_t(p.rows) * kd);
        for (uint32_t r = 0; r < p.rows; ++r) {
            unpack_fp4_row(p.cmp_fp4.data() + size_t(r) * cd / 2, p.cmp_scale.data() + size_t(r) * cd / 16,
                           cd, 16, false, cmp.data() + size_t(r) * cd);
            unpack_fp4_row(p.key_fp4.data() + size_t(r) * kd / 2, p.key_scale.data() + size_t(r) * kd / 32,
                           kd, 32, true, key.data() + size_t(r) * kd);
        }
        for (size_t i = 0; i < p.raw_rows.size(); ++i) {
            const uint32_t r = p.raw_rows[i];
            std::memcpy(cmp.data() + size_t(r) * cd, p.raw_cmp.data() + i * cd, size_t(cd) * 2);
            std::memcpy(key.data() + size_t(r) * kd, p.raw_key.data() + i * kd, size_t(kd) * 2);
        }
        std::memcpy(at(lay_.off_cmp[oi]), cmp.data(), cmp.size() * 2);
        std::memcpy(at(lay_.off_idx[oi]), key.data(), key.size() * 2);
        rows_hw_[oi] = std::max(rows_hw_[oi], p.rows);
        if (p.ratio > 1 && p.carry_kv.size() == size_t(p.ratio) * cd) {
            const uint64_t stp = uint64_t(c.max_ratio) * cd * 4;
            const uint64_t score_base = lay_.off_state + uint64_t(owners_.size()) * stp;
            std::memcpy(at(lay_.off_state + oi * stp), p.carry_kv.data(), p.carry_kv.size() * 4);
            std::memcpy(at(score_base + oi * stp), p.carry_score.data(), p.carry_score.size() * 4);
        }
    }
    for (uint32_t l = 0; l < c.layers; ++l) {
        const uint32_t oi = owner_index(l);
        n_cmp_[l] = oi == KvStoreConfig::kNoPlane ? 0 : pk.positions / c.plane_ratio(owners_[oi]);
    }
    return {};
}

Result<KvRowBackup> KvStore::backup_rows(uint32_t first_pos, uint32_t end_pos) const {
    if (!valid()) return fail(Err::FailedPrecondition, "KV store is not created");
    const KvStoreConfig& c = cfg_;
    KvRowBackup b;
    b.first_pos = first_pos;
    b.end_pos = end_pos;
    const uint64_t stp = uint64_t(c.max_ratio) * c.latent_dim * 4;
    const uint64_t score_base = lay_.off_state + uint64_t(owners_.size()) * stp;
    for (size_t oi = 0; oi < owners_.size(); ++oi) {
        KvRowBackup::Plane p;
        p.layer = owners_[oi];
        p.ratio = c.plane_ratio(p.layer);
        p.first_row = first_pos / p.ratio;
        const uint32_t end_row = end_pos / p.ratio;
        p.rows = end_row > p.first_row ? end_row - p.first_row : 0;
        if (p.first_row + p.rows > rows_hw_[oi])
            return fail(Err::FailedPrecondition,
                        std::format("backing up rows {}..{} of layer {}; {} were ever written",
                                    p.first_row, p.first_row + p.rows, p.layer, rows_hw_[oi]));
        p.cmp.resize(size_t(p.rows) * c.latent_dim);
        p.key.resize(size_t(p.rows) * c.index_dim);
        std::memcpy(p.cmp.data(), at(lay_.off_cmp[oi] + uint64_t(p.first_row) * c.latent_dim * 2),
                    p.cmp.size() * 2);
        std::memcpy(p.key.data(), at(lay_.off_idx[oi] + uint64_t(p.first_row) * c.index_dim * 2),
                    p.key.size() * 2);
        const size_t n = size_t(c.max_ratio) * c.latent_dim;
        p.carry_kv.resize(n);
        p.carry_score.resize(n);
        std::memcpy(p.carry_kv.data(), at(lay_.off_state + oi * stp), n * 4);
        std::memcpy(p.carry_score.data(), at(score_base + oi * stp), n * 4);
        b.planes.push_back(std::move(p));
    }
    return b;
}

Result<void> KvStore::restore_rows(const KvRowBackup& b, uint32_t position) {
    if (!valid()) return fail(Err::FailedPrecondition, "KV store is not created");
    const KvStoreConfig& c = cfg_;
    for (const KvRowBackup::Plane& p : b.planes) {
        if ((position + 1) % p.ratio) continue;
        const uint32_t r = (position + 1) / p.ratio - 1;
        if (r < p.first_row || r >= p.first_row + p.rows) continue;
        const uint32_t oi = owner_index(p.layer);
        if (oi == KvStoreConfig::kNoPlane) return fail(Err::InvalidArgument, "backup of another store");
        const size_t k = r - p.first_row;
        std::memcpy(at(lay_.off_cmp[oi] + uint64_t(r) * c.latent_dim * 2),
                    p.cmp.data() + k * c.latent_dim, size_t(c.latent_dim) * 2);
        std::memcpy(at(lay_.off_idx[oi] + uint64_t(r) * c.index_dim * 2),
                    p.key.data() + k * c.index_dim, size_t(c.index_dim) * 2);
    }
    return {};
}

Result<void> KvStore::restore_carry(const KvRowBackup& b) {
    if (!valid()) return fail(Err::FailedPrecondition, "KV store is not created");
    const KvStoreConfig& c = cfg_;
    const uint64_t stp = uint64_t(c.max_ratio) * c.latent_dim * 4;
    const uint64_t score_base = lay_.off_state + uint64_t(owners_.size()) * stp;
    for (const KvRowBackup::Plane& p : b.planes) {
        const uint32_t oi = owner_index(p.layer);
        if (oi == KvStoreConfig::kNoPlane) return fail(Err::InvalidArgument, "backup of another store");
        std::memcpy(at(lay_.off_state + oi * stp), p.carry_kv.data(), p.carry_kv.size() * 4);
        std::memcpy(at(score_base + oi * stp), p.carry_score.data(), p.carry_score.size() * 4);
    }
    return {};
}

Result<KvStore::RingSnapshot> KvStore::snapshot_ring(std::span<const uint32_t> slots,
                                                     std::span<const uint32_t> layers) const {
    if (!valid()) return fail(Err::FailedPrecondition, "KV store is not created");
    const KvStoreConfig& c = cfg_;
    if (slots.empty()) return fail(Err::InvalidArgument, "snapshot_ring: no slots");
    if (layers.empty()) return fail(Err::InvalidArgument, "snapshot_ring: no layers");
    RingSnapshot s;
    s.latent_dim = c.latent_dim;
    s.slot.assign(slots.begin(), slots.end());
    s.layer.assign(layers.begin(), layers.end());
    for (uint32_t l : layers)
        if (l >= c.layers) return fail(Err::OutOfRange, std::format("layer {} of {}", l, c.layers));
    for (uint32_t sl : slots)
        if (sl >= c.window)
            return fail(Err::OutOfRange, std::format("ring slot {} of {}", sl, c.window));
    const size_t n = s.layer.size() * s.slot.size();
    s.val.resize(n * c.latent_dim);
    s.scale.resize(n * c.window_scales());
    const uint64_t row = c.latent_dim, srow = c.window_scales();
    for (size_t li = 0; li < s.layer.size(); ++li) {
        const uint32_t l = s.layer[li];
        for (size_t si = 0; si < s.slot.size(); ++si) {
            const uint32_t sl = s.slot[si];
            const size_t o = li * s.slot.size() + si;
            std::memcpy(s.val.data() + o * row,
                        at(lay_.off_win_val + (uint64_t(l) * c.window + sl) * row), row);
            std::memcpy(s.scale.data() + o * srow,
                        at(lay_.off_win_scale + (uint64_t(l) * c.window + sl) * srow), srow);
        }
    }
    return s;
}

Result<void> KvStore::restore_ring(const RingSnapshot& s) {
    if (!valid()) return fail(Err::FailedPrecondition, "KV store is not created");
    const KvStoreConfig& c = cfg_;
    if (s.latent_dim != c.latent_dim)
        return fail(Err::InvalidArgument, "ring snapshot of another store");
    for (uint32_t l : s.layer)
        if (l >= c.layers) return fail(Err::InvalidArgument, "ring snapshot of another store");
    for (uint32_t sl : s.slot)
        if (sl >= c.window) return fail(Err::InvalidArgument, "ring snapshot of another store");
    const uint64_t row = c.latent_dim, srow = c.window_scales();
    for (size_t li = 0; li < s.layer.size(); ++li) {
        const uint32_t l = s.layer[li];
        for (size_t si = 0; si < s.slot.size(); ++si) {
            const uint32_t sl = s.slot[si];
            const size_t o = li * s.slot.size() + si;
            std::memcpy(at(lay_.off_win_val + (uint64_t(l) * c.window + sl) * row),
                        s.val.data() + o * row, row);
            std::memcpy(at(lay_.off_win_scale + (uint64_t(l) * c.window + sl) * srow),
                        s.scale.data() + o * srow, srow);
        }
    }
    return {};
}

}  // namespace deepmoe::runtime
