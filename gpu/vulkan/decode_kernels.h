// Host side of the two decode-path kernels gpu/vulkan/attn_kernels.h does not
// carry: the engram of design §7.10 and the greedy sampler of §7.11.
//
// Why a second runner instead of two more AttnStages
// -------------------------------------------------
// `AttnRunner` owns one address-table slice per stage and is the attention
// track's file. These two stages belong to different parts of the model -- the
// engram runs on exactly two of the forty layers, the sampler once a token --
// and they are developed alongside runtime/engine.cpp rather than alongside the
// attention chain. Keeping them in their own runner means the two tracks never
// write the same table, the same `kStages` array or the same enum.
//
// It is otherwise the same shape as AttnRunner, deliberately: one descriptor
// per pipeline, holding a 32-slot slice of a shared `uint64_t` address table
// that the host writes and the shader dereferences (design §5.3's argument,
// applied to weights). Write addresses, push dimensions, dispatch.
//
// Ownership/threading: created, recorded and submitted from the single GPU
// submit thread. Owns its pipelines, descriptor pool, address table and command
// pool; the weights belong to store::PinnedStore and the activations to
// whoever allocated them.
#pragma once

#include <cstdint>
#include <string>

#include "core/status.h"
#include "gpu/vulkan/attn_kernels.h"
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/descriptor.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/pipeline.h"

namespace deepmoe::gpu {

enum class DecodeStage : uint32_t {
    EngramGemv = 0,   // §7.10: the 6144 -> 25600 fp8 GEMV over 24 hashed rows
    EngramGate,       // §7.10: the gate and the residual update
    Argmax,           // §7.11: 129,280 logits -> (id, top1, top2)
    SampleTopK,       // §7.11 sampled: logits -> top set + tail mass (sample_topk.slang)
    Count,
};

const char* decode_stage_name(DecodeStage s);

// Mirrors gpu/shaders/engram.slang's EngramPush.
struct EngramPush {
    uint32_t rows;        // 25600 = dim * (hc + 1)
    uint32_t k;           // 6144  = 24 rows x 256
    uint32_t scale_cols;  // k / 32
    uint32_t dim;         // 5120
    uint32_t hc;          // 4
    float    norm_eps;
};

namespace dslot {
// engram
enum : uint32_t { kRowVal = 0, kRowSc = 1, kW = 2, kS = 3, kKv = 4,
                  kX = 5, kQW = 6, kKW = 7, kXout = 8 };
// head stage 1 -- the same slice layout as gpu/shaders/head.slang's GEMV, plus
// the four-word sample result at slot 3.
enum : uint32_t { kHeadW = 0, kHeadX = 1, kHeadLogits = 2, kHeadSample = 3 };
// sample_topk: the logits, the output (header + candidate segments), the
// per-thread histograms.
enum : uint32_t { kTopKLogits = 0, kTopKOut = 1, kTopKHist = 2 };
}  // namespace dslot

// Mirrors gpu/shaders/sample_topk.slang's TopKPush.
struct TopKPush {
    uint32_t rows = 0;
    uint32_t k = 0;
    float    inv_t = 1.0f;
    float    bins_per_logit = 16.0f;
};

// What gpu/shaders/head.slang's argmax stage writes: the token, its logit, the
// runner-up's logit (design §12 L3 wants the margin at a divergence) and the
// row count it scanned, as a self-check that the dispatch saw the whole vocab.
struct SampleOut {
    uint32_t token = 0;
    float    top1  = 0.0f;
    float    top2  = 0.0f;
    uint32_t rows  = 0;
    float    margin() const { return top1 - top2; }
};

class DecodeRunner {
public:
    DecodeRunner() = default;
    ~DecodeRunner() { destroy(); }

    DecodeRunner(const DecodeRunner&) = delete;
    DecodeRunner& operator=(const DecodeRunner&) = delete;

    Result<void> create(Device& device, MemoryAllocator& alloc,
                        const std::string& shader_dir, const AttnSpec& spec = {});
    void destroy();

    uint64_t* slots(DecodeStage s);

    Result<void> record(CommandBuffer& cmd, DecodeStage s, const void* push,
                        uint32_t push_bytes, uint32_t groups);
    Result<void> dispatch_now(DecodeStage s, const void* push, uint32_t push_bytes,
                              uint32_t groups);

    // Workgroups for the engram GEMV: one row group per (256 / lanes) threads,
    // RowsPerLane fixed at 1 (engram.slang does not row-block -- 25600 rows over
    // 8 rows a workgroup is 3,200 workgroups, which already saturates 40 CUs).
    uint32_t gemv_groups(uint32_t rows) const {
        const uint32_t per = 256 / spec_.lanes_per_row;
        return (rows + per - 1) / per;
    }

private:
    Result<void> make(DecodeStage s, const std::string& spv, uint32_t stage_const);

    Device*          device_ = nullptr;
    MemoryAllocator* alloc_  = nullptr;
    AttnSpec         spec_{};
    Pipeline         pipes_[static_cast<uint32_t>(DecodeStage::Count)];
    DescriptorPool   descriptors_;
    CommandPool      pool_;
    GpuBuffer        table_{};
#if defined(DEEPMOE_ENABLE_VULKAN)
    VkDescriptorSet  sets_[static_cast<uint32_t>(DecodeStage::Count)]{};
#endif
};

}  // namespace deepmoe::gpu
