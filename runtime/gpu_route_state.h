#pragma once

#include <cstdint>
#include <vector>

#include "core/status.h"
#include "gpu/vulkan/attn_kernels.h"
#include "gpu/vulkan/memory.h"
#include "model/v41_config.h"
#include "runtime/decode_layer.h"

namespace deepmoe::runtime {

class Engine;

// GPU-written readback belongs to its stream and remains valid through the
// batch fence. Views point into scratch; allocation order matches the original
// batch path and is independent of the cache residency snapshot.
struct GpuRouteBuffers {
    static constexpr uint64_t kRouteScratchBytes = 8ull << 20;
    gpu::GpuScratch scratch;
    std::vector<gpu::GpuScratch::View> routes, carry_k, carry_g;
    std::vector<gpu::GpuScratch::View> rope, rope_lat, hidden;

    Result<void> initialize(gpu::MemoryAllocator& allocator, const TextConfig& config);
};

// One immutable residency snapshot and the batch steps recorded against it.
// Engine owns the scoped guard and submits; this component reconciles the
// completed readback with counters, draft carry, and asynchronous LRU demand.
struct GpuRouteState {
    bool enabled = false;
    bool logged = false;
    std::vector<uint64_t> snapshot;
    std::vector<BatchStep> steps;

    Result<void> finish(Engine& engine, uint32_t first_position, uint32_t rows);
};

} // namespace deepmoe::runtime
