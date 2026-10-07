#include "gpu/vulkan/draft_head.h"
#include "core/wc_read.h"
#include "cpu/dequant.h"
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

namespace cachedmoe::gpu {
DraftHeadFp8::~DraftHeadFp8() {
    if (!allocator_) return;
    if (weights_.valid()) allocator_->free(weights_);
    if (scales_.valid()) allocator_->free(scales_);
}

Result<void> DraftHeadFp8::create(MemoryAllocator &allocator, const store::PinnedTensor &source,
                                  uint32_t vocab, uint32_t dim) {
    if (allocator_)
        return fail(Err::FailedPrecondition, "draft FP8 copy was already initialized");
    if (!vocab || !dim || source.dtype != QuantType::Bf16 || !source.data_host ||
        source.shape != std::vector<uint64_t>{vocab, dim} ||
        source.data_bytes != uint64_t(vocab) * dim * sizeof(uint16_t))
        return fail(Err::InvalidArgument, "draft FP8 source must be a complete BF16 head");
    allocator_ = &allocator;
    // Scalar reads of write-combining GPU memory are slow. The source is idle
    // at startup, so stage it into cached RAM once before the independent rows.
    std::vector<uint16_t> cached(uint64_t(vocab) * dim);
    wc_readback(cached.data(), source.data_host, source.data_bytes);
    std::vector<uint8_t> encoded(cached.size());
    std::vector<float> row_scales(vocab);
    std::atomic<bool> bad{false};
    constexpr uint32_t kWorkers = 8;
    std::vector<std::jthread> threads;
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        threads.emplace_back([&, worker] {
            for (uint32_t row = worker; row < vocab; row += kWorkers) {
                auto scale = cpu::quantize_bf16_row_fp8(
                    std::span(cached).subspan(size_t(row) * dim, dim),
                    std::span(encoded).subspan(size_t(row) * dim, dim));
                if (!scale) { bad = true; return; }
                row_scales[row] = *scale;
            }
        });
    }
    threads.clear(); // Join before publishing any GPU-visible data.
    if (bad) return fail(Err::InvalidArgument, "draft FP8 row conversion failed");
    auto weights = allocator.allocate(encoded.size(), true, true);
    if (!weights) return std::unexpected(weights.error());
    weights_ = *weights;
    auto scales = allocator.allocate(row_scales.size() * sizeof(float), true, true);
    if (!scales) return std::unexpected(scales.error());
    scales_ = *scales;
    std::memcpy(weights_.host_ptr, encoded.data(), encoded.size());
    std::memcpy(scales_.host_ptr, row_scales.data(), row_scales.size() * sizeof(float));
    return {};
}
} // namespace cachedmoe::gpu
