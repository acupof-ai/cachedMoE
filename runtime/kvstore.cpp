#include "runtime/kvstore.h"

#include <cstring>
#include <format>

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

    auto b = alloc.allocate(off, /*host_visible=*/true, /*device_address=*/true);
    if (!b) return std::unexpected(b.error());
    if (!b->host_ptr || b->dev_addr == kNoDeviceAddress) {
        alloc.free(*b);
        return fail(Err::Internal, "the KV store must be host-writable and device-addressable");
    }
    alloc_ = &alloc;
    buf_   = *b;
    cfg_   = cfg;
    std::memset(buf_.host_ptr, 0, static_cast<size_t>(off));
    n_cmp_.assign(cfg.layers, 0);
    n_kv_.assign(cfg.layers, 0);
    return {};
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

    KvLayerView v;
    v.win_val   = buf_.dev_addr + off_win_val_ + l * win;
    v.win_scale = buf_.dev_addr + off_win_scale_ + l * wins;
    v.cmp_kv    = buf_.dev_addr + off_cmp_ + l * cmp;
    v.top_idx   = buf_.dev_addr + off_top_ + l * top;
    v.win_val_host   = reinterpret_cast<uint8_t*>(host + off_win_val_ + l * win);
    v.win_scale_host = reinterpret_cast<uint8_t*>(host + off_win_scale_ + l * wins);
    v.cmp_kv_host    = reinterpret_cast<uint16_t*>(host + off_cmp_ + l * cmp);
    v.top_idx_host   = reinterpret_cast<uint32_t*>(host + off_top_ + l * top);
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

}  // namespace deepmoe::runtime
