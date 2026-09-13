// The raw-read upper bound of design §7.1 rule 2.
//
// "All GEMVs are scored in effective GB/s, and every kernel benchmark prints
// the raw-read shader over the same byte count as the ceiling." This is that
// shader's host side: point it at a VkBuffer, tell it how many workgroups to
// use, and it reports GPU seconds and GB/s for one full pass.
//
// bench/bw_matrix uses it to compare path A, path B and DEVICE_LOCAL-only
// memory (design §3.3); bench/kernel_bench uses it as the denominator of the
// MoE kernels' efficiency.
//
// Ownership/threading: owns its pipeline, descriptor pool, sink buffer, command
// pool and query pool. One instance is driven from one thread.
#pragma once

#include <cstdint>
#include <string>

#include "core/status.h"
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/descriptor.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/pipeline.h"

namespace deepmoe::gpu {

struct RawReadResult {
    uint64_t bytes    = 0;
    double   seconds  = 0.0;    // GPU timestamps when available, else wall clock
    double   gbps     = 0.0;
    bool     gpu_timed = false;
    uint32_t groups   = 0;
    uint32_t passes   = 0;
};

class RawReadKernel {
public:
    RawReadKernel() = default;
    ~RawReadKernel() { destroy(); }

    RawReadKernel(const RawReadKernel&) = delete;
    RawReadKernel& operator=(const RawReadKernel&) = delete;

    // `groups` workgroups of 256 threads each; each owns one contiguous slice.
    Result<void> create(Device& device, MemoryAllocator& alloc,
                        const std::string& shader_dir, uint32_t groups = 320);
    void destroy();

    // Streams `bytes` of `src` `passes` times. `bytes` must be a multiple of
    // groups * 16 KiB so every workgroup gets whole 1024-uint4 iterations.
    Result<RawReadResult> run(const GpuBuffer& src, uint64_t bytes, uint32_t passes = 1);

    // `dispatches` near-empty dispatches separated by the same global barrier
    // the decode chain uses. design §3.4 assumed 5-20 us per dispatch+barrier;
    // this is the measurement. `seconds` is the total, so divide by `passes`.
    Result<RawReadResult> run_empty(const GpuBuffer& src, uint32_t dispatches);

    uint32_t groups() const { return groups_; }
    // The largest multiple of the per-dispatch granularity not exceeding `bytes`.
    uint64_t round_bytes(uint64_t bytes) const;

private:
    Device*         device_ = nullptr;
    MemoryAllocator* alloc_ = nullptr;
    uint32_t        groups_ = 0;
    Pipeline        pipeline_;
    DescriptorPool  descriptors_;
    GpuBuffer       sink_{};
    CommandPool     pool_;
    CommandBuffer   cmd_{};
    QueryPool       queries_;
};

}  // namespace deepmoe::gpu
