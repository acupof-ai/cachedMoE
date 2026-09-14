// Host side of the batched prefill kernels (docs/p3_prefill.md, design §7.13):
// the `gpu/shaders/prefill_*.slang` family and the runner that drives it.
//
// Why a fourth runner
// -------------------
// AttnRunner (Track J), DecodeRunner / MoeRunner (Track I / H) and DsparkRunner
// (Track K) each own a fixed enum of stages, one pipeline per enumerator, and a
// slot table sliced per stage. The prefill kernels are different in one way
// that matters: most of their knobs are specialisation constants that the
// bench sweeps and the driver picks per call site -- the weight format, the
// activation format, the token tile, the cooperative-matrix shape -- so the set
// of pipelines is not a closed enum. `PrefillRunner` therefore creates a
// pipeline the first time a `PfKernel` key is asked for and hands back a small
// integer; everything after that (slot table slice, descriptor set, record,
// dispatch_now) is exactly the idiom the other runners use.
//
// Ownership/threading: created, recorded and submitted from the single GPU
// submit thread. Owns its pipelines, descriptor pool, slot table and command
// pool; the weights belong to store::PinnedStore / the expert transit buffers,
// and the activations to whoever allocated them.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "core/status.h"
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/descriptor.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/pipeline.h"
#include "model/manifest.h"
#include "storage/io_engine.h"
#include "store/shard_set.h"

namespace deepmoe::gpu {

// --- weight / activation formats, mirroring prefill_common.slang -------------
enum PfWeightFmt : uint32_t { kPfFp8 = 0, kPfBf16 = 1, kPfFp32 = 2, kPfFp4 = 3 };
enum PfActFmt    : uint32_t { kPfActF32 = 0, kPfActQ = 1 };

inline constexpr uint32_t kPfFlagAccumulate = 1u;
inline constexpr uint32_t kPfFlagRound      = 2u;
inline constexpr uint32_t kPfFlagGather     = 4u;
inline constexpr uint32_t kPfFlagScatter    = 8u;
inline constexpr uint32_t kPfFlagRowScale   = 16u;
inline constexpr uint32_t kPfFlagInverse    = 32u;

// One pipeline: which .spv, and its specialisation constants 4.. (Stage, WFmt,
// XFmt, TileM, then two kernel-specific ones).
struct PfKernel {
    std::string spv;
    uint32_t stage = 0;
    uint32_t wfmt  = 0;
    uint32_t xfmt  = 0;
    uint32_t tile  = 8;
    uint32_t extra0 = 0;    // prefill_coopmat: CmRows
    uint32_t extra1 = 0;    // prefill_coopmat: CmCols
    auto key() const { return std::tie(spv, stage, wfmt, xfmt, tile, extra0, extra1); }
    bool operator<(const PfKernel& o) const { return key() < o.key(); }
};

// --- push constants ----------------------------------------------------------

// prefill_gemm.slang, all four stages.
struct PfGemmPush {
    uint32_t rows = 0, k = 0, scale_cols = 0, n = 0;
    uint32_t x_stride = 0, y_stride = 0, rows_per_group = 0, flags = 0;
    uint32_t idx_off = 0, job = 0, row_base = 0;
    float    out_scale = 1.0f, swiglu_limit = 10.0f;
    uint32_t tile_base = 0;
};
// slots
enum : uint32_t { kPgW = 0, kPgS = 1, kPgX = 2, kPgXS = 3, kPgY = 4, kPgIdx = 5, kPgRW = 6,
                  kPgJob = 7, kPgRowScale = 8 };

// The job table entry stages 1 and 2 read: design §5.3's pointer table one
// level up. 64 B, laid out exactly as prefill_gemm.slang's `load_job`.
struct PfJob {
    uint64_t w1 = 0, s1 = 0, w3 = 0, s3 = 0, w2 = 0, s2 = 0;
    uint32_t rows_off = 0, n = 0, h_off = 0, fmt = kPfFp4;
};

// prefill_coopmat.slang
struct PfCoopPush { uint32_t n = 0, k = 0, idx_off = 0, flags = 0; };
enum : uint32_t { kPcW = 0, kPcX = 1, kPcY = 2, kPcQ = 3, kPcQS = 4, kPcIdx = 5 };

// --- host helpers shared by the driver, the test and the bench ---------------

// `act_quant(x, 32, "ue8m0")` for `n` rows of `k` (a multiple of 32): the E4M3
// VALUE of every element as fp16 -- every E4M3 number is exact in fp16 -- and
// one power-of-two fp32 scale per block. The layout `XFmt = 1` reads.
void pf_act_quant_host(const float* x, uint32_t n, uint32_t k, uint16_t* q16, float* scales);

// One routed expert read off NVMe into a GPU-visible, device-addressable
// buffer, exactly as store::ExpertStore lays a slot out. `addr[part]` follows
// model/manifest.h's ExpertPart order: w1.weight, w1.scale, w2.weight,
// w2.scale, w3.weight, w3.scale.
struct PfExpert {
    GpuBuffer buf{};
    uint64_t  addr[6] = {};
    uint64_t  part_off[6] = {};
    const void* host(uint32_t part) const {
        return static_cast<const std::byte*>(buf.host_ptr) + part_off[part];
    }
    void release(MemoryAllocator& a) { if (buf.valid()) a.free(buf); buf = GpuBuffer{}; }
};
Result<PfExpert> pf_load_expert(MemoryAllocator& alloc, const Manifest& manifest,
                                const store::ShardSet& shards, storage::IoEngine& io,
                                ExpertKey key);

inline constexpr uint32_t kPfSlotsPerKernel = 32;
inline constexpr uint32_t kPfKernelStride   = kPfSlotsPerKernel * sizeof(uint64_t);
inline constexpr uint32_t kPfPushBytes      = 64;

class PrefillRunner {
public:
    PrefillRunner() = default;
    ~PrefillRunner() { destroy(); }

    PrefillRunner(const PrefillRunner&) = delete;
    PrefillRunner& operator=(const PrefillRunner&) = delete;

    // `max_kernels` bounds the slot table; every distinct PfKernel costs one
    // pipeline, one descriptor set and 256 B of table.
    Result<void> create(Device& device, MemoryAllocator& alloc, const std::string& shader_dir,
                        uint32_t max_kernels = 96);
    void destroy();

    // The pipeline for `k`, created on first use.
    Result<uint32_t> kernel(const PfKernel& k);
    // Its 32 slots, host-writable.
    uint64_t* slots(uint32_t handle);
    uint32_t  kernels() const { return static_cast<uint32_t>(pipes_.size()); }

    Result<void> record(CommandBuffer& cmd, uint32_t handle, const void* push, uint32_t bytes,
                        uint32_t gx, uint32_t gy = 1);
    // Tests and the bench: one dispatch, submitted and waited on.
    Result<void> dispatch_now(uint32_t handle, const void* push, uint32_t bytes,
                              uint32_t gx, uint32_t gy = 1);

    CommandPool& pool() { return pool_; }
    Device*      device() { return device_; }

    // --- workgroup counts --------------------------------------------------
    // prefill_gemm stages 0-2: (row blocks of 8, token tiles).
    static uint32_t gemm_gx(uint32_t rows) { return (rows + 7) / 8; }
    static uint32_t gemm_gy(uint32_t n, uint32_t tile) { return (n + tile - 1) / tile; }
    // prefill_gemm stage 3 and every one-thread-per-(row, block) kernel.
    static uint32_t per_block_groups(uint64_t rows, uint32_t k) {
        const uint64_t t = rows * (k / 32);
        return static_cast<uint32_t>((t + 255) / 256);
    }

private:
    Device*          device_ = nullptr;
    MemoryAllocator* alloc_  = nullptr;
    std::string      shader_dir_;
    uint32_t         max_kernels_ = 0;
    std::map<PfKernel, uint32_t> index_;
    std::vector<std::unique_ptr<Pipeline>> pipes_;
    DescriptorPool   descriptors_;
    CommandPool      pool_;
    CommandBuffer    own_cmd_{};
    bool             own_cmd_valid_ = false;
    GpuBuffer        table_{};
#if defined(DEEPMOE_ENABLE_VULKAN)
    std::vector<VkDescriptorSet> sets_;
#endif
};

}  // namespace deepmoe::gpu
