#include "runtime/kvstore.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <limits>

#include "core/align.h"
#include "cpu/dequant.h"

namespace deepmoe::runtime {

Result<void> KvStore::create(gpu::MemoryAllocator& alloc, const KvStoreConfig& cfg) {
    destroy();
    if (cfg.latent_dim % cfg.scale_block)
        return fail(Err::InvalidArgument, "latent_dim must be a multiple of the scale block");
    const uint64_t win_val   = uint64_t(cfg.layers) * cfg.window * cfg.latent_dim;
    const uint64_t win_scale = uint64_t(cfg.layers) * cfg.window * cfg.window_scales();
    const uint64_t cmp       = cfg.compressed_bytes();
    const uint64_t top       = cfg.topk_bytes();
    const uint64_t idxk      = cfg.index_key_bytes();
    const uint64_t state     = cfg.cmp_state_bytes();

    uint64_t off = 0;
    auto place = [&](uint64_t bytes) {
        const uint64_t at = align_up(off, 256);
        off = at + bytes;
        return at;
    };
    off_win_val_   = place(win_val);
    off_win_scale_ = place(win_scale);
    off_cmp_       = place(cmp);
    off_top_       = place(top);
    off_idx_k_     = place(idxk);
    off_state_     = place(state);

    auto b = alloc.allocate(off, /*host_visible=*/true, /*device_address=*/true);
    if (!b) return std::unexpected(b.error());
    if (!b->host_ptr || b->dev_addr == kNoDeviceAddress) {
        alloc.free(*b);
        return fail(Err::Internal, "the KV store must be host-writable and device-addressable");
    }
    alloc_ = &alloc;
    buf_   = *b;
    cfg_   = cfg;
    n_cmp_.assign(cfg.layers, 0);
    n_kv_.assign(cfg.layers, 0);
    clear();
    return {};
}

void KvStore::clear() {
    if (!buf_.valid()) return;
    std::memset(buf_.host_ptr, 0, static_cast<size_t>(buf_.bytes));
    // `Compressor.score_state` starts at -inf, not at zero: a slot that has
    // never held a token must contribute nothing to the pooling softmax, and
    // exp(-inf - max) is the only value that does that. Zero would make an
    // unwritten slot an equal partner (model.py, `torch.full(..., -torch.inf)`).
    const uint64_t n = uint64_t(cfg_.layers) * cfg_.max_ratio * cfg_.latent_dim;
    auto* score = reinterpret_cast<float*>(static_cast<std::byte*>(buf_.host_ptr) +
                                           off_state_ + n * 4);
    const float ninf = -std::numeric_limits<float>::infinity();
    for (uint64_t i = 0; i < n; ++i) score[i] = ninf;
    std::fill(n_cmp_.begin(), n_cmp_.end(), 0u);
    std::fill(n_kv_.begin(), n_kv_.end(), 0u);
}

void KvStore::destroy() {
    if (alloc_ && buf_.valid()) alloc_->free(buf_);
    buf_ = gpu::GpuBuffer{};
    alloc_ = nullptr;
    n_cmp_.clear();
    n_kv_.clear();
}

Result<KvLayerView> KvStore::layer(uint32_t l) const {
    if (!buf_.valid()) return fail(Err::FailedPrecondition, "KV store is not created");
    if (l >= cfg_.layers)
        return fail(Err::OutOfRange, std::format("layer {} of {}", l, cfg_.layers));
    auto* host = static_cast<std::byte*>(buf_.host_ptr);
    const uint64_t win   = uint64_t(cfg_.window) * cfg_.latent_dim;
    const uint64_t wins  = uint64_t(cfg_.window) * cfg_.window_scales();
    const uint64_t cmp   = uint64_t(cfg_.max_context) * cfg_.latent_dim * 2;
    const uint64_t top   = uint64_t(cfg_.window + cfg_.max_context) * sizeof(uint32_t);
    const uint64_t idxk  = uint64_t(cfg_.max_context) * cfg_.index_dim * 2;
    const uint64_t stp   = uint64_t(cfg_.max_ratio) * cfg_.latent_dim * 4;
    // The two state planes are laid out one after the other so the -inf fill
    // of `create` is one contiguous range.
    const uint64_t score_base = off_state_ + uint64_t(cfg_.layers) * stp;

    KvLayerView v;
    v.win_val   = buf_.dev_addr + off_win_val_ + l * win;
    v.win_scale = buf_.dev_addr + off_win_scale_ + l * wins;
    v.cmp_kv    = buf_.dev_addr + off_cmp_ + l * cmp;
    v.top_idx   = buf_.dev_addr + off_top_ + l * top;
    v.idx_key   = buf_.dev_addr + off_idx_k_ + l * idxk;
    v.cmp_state_kv    = buf_.dev_addr + off_state_ + l * stp;
    v.cmp_state_score = buf_.dev_addr + score_base + l * stp;
    v.win_val_host   = reinterpret_cast<uint8_t*>(host + off_win_val_ + l * win);
    v.win_scale_host = reinterpret_cast<uint8_t*>(host + off_win_scale_ + l * wins);
    v.cmp_kv_host    = reinterpret_cast<uint16_t*>(host + off_cmp_ + l * cmp);
    v.top_idx_host   = reinterpret_cast<uint32_t*>(host + off_top_ + l * top);
    v.idx_key_host   = reinterpret_cast<uint16_t*>(host + off_idx_k_ + l * idxk);
    v.cmp_state_kv_host    = reinterpret_cast<float*>(host + off_state_ + l * stp);
    v.cmp_state_score_host = reinterpret_cast<float*>(host + score_base + l * stp);
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
    auto v = this->layer(layer);
    if (!v) return std::unexpected(v.error());
    if (n_cmp > cfg_.max_context)
        return fail(Err::InvalidArgument,
                    std::format("{} compressed entries but the store holds {}",
                                n_cmp, cfg_.max_context));
    for (size_t i = 0; i < size_t(n_cmp) * cfg_.latent_dim; ++i)
        v->cmp_kv_host[i] = cpu::float_to_bf16(values[i]);
    n_cmp_[layer] = n_cmp;
    return {};
}

Result<void> KvStore::seed_topk(uint32_t layer, const int32_t* idx, uint32_t n_kv) {
    auto v = this->layer(layer);
    if (!v) return std::unexpected(v.error());
    if (n_kv > cfg_.window + cfg_.max_context)
        return fail(Err::InvalidArgument, "top-k list is longer than the store allows");
    std::memcpy(v->top_idx_host, idx, size_t(n_kv) * sizeof(int32_t));
    n_kv_[layer] = n_kv;
    return {};
}

Result<void> KvStore::seed_index_k(uint32_t layer, const float* values, uint32_t rows) {
    auto v = this->layer(layer);
    if (!v) return std::unexpected(v.error());
    if (rows > cfg_.max_context)
        return fail(Err::InvalidArgument,
                    std::format("{} index keys but the store holds {}", rows,
                                cfg_.max_context));
    for (size_t i = 0; i < size_t(rows) * cfg_.index_dim; ++i)
        v->idx_key_host[i] = cpu::float_to_bf16(values[i]);
    return {};
}

Result<void> KvStore::seed_cmp_state(uint32_t layer, const float* kv, const float* score,
                                     uint32_t ratio) {
    auto v = this->layer(layer);
    if (!v) return std::unexpected(v.error());
    if (ratio > cfg_.max_ratio)
        return fail(Err::InvalidArgument,
                    std::format("compress ratio {} above the {} slots the state holds",
                                ratio, cfg_.max_ratio));
    const size_t n = size_t(ratio) * cfg_.latent_dim;
    std::memcpy(v->cmp_state_kv_host, kv, n * sizeof(float));
    std::memcpy(v->cmp_state_score_host, score, n * sizeof(float));
    return {};
}

Result<void> KvStore::set_counts(uint32_t layer, uint32_t n_cmp, uint32_t n_kv) {
    if (layer >= cfg_.layers)
        return fail(Err::OutOfRange, std::format("layer {} of {}", layer, cfg_.layers));
    if (n_cmp > cfg_.max_context || n_kv > cfg_.window + cfg_.max_context)
        return fail(Err::InvalidArgument,
                    std::format("layer {} wants {} compressed / {} total KV entries, "
                                "above the {} / {} the store was sized for", layer, n_cmp,
                                n_kv, cfg_.max_context, cfg_.window + cfg_.max_context));
    n_cmp_[layer] = n_cmp;
    n_kv_[layer]  = n_kv;
    return {};
}

Result<void> KvStore::set_decode_topk(uint32_t layer, uint32_t position, uint32_t n_cmp) {
    auto v = this->layer(layer);
    if (!v) return std::unexpected(v.error());
    if (auto r = set_counts(layer, n_cmp, cfg_.window + n_cmp); !r) return r;
    // model.py's `get_window_topk_idxs(win, 1, 1, start_pos)`: the ring listed
    // oldest first from `start_pos % win + 1`, with any slot the sequence has
    // not reached yet marked -1. Order inside the list does not matter to
    // sparse_attn, which handles every slot independently -- but the -1s do.
    const uint32_t win = cfg_.window;
    const uint32_t oldest = position % win + 1;
    auto* out = reinterpret_cast<int32_t*>(v->top_idx_host);
    for (uint32_t i = 0; i < win; ++i) {
        const uint32_t slot = (oldest + i) % win;
        out[i] = (slot > position) ? -1 : static_cast<int32_t>(slot);
    }
    return {};
}

}  // namespace deepmoe::runtime
