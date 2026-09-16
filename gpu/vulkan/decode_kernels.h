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
#include <vector>

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

// ===========================================================================
// Track T: the non-MoE decode stages for a verify batch of M = k + 1 <= 6
// tokens (docs/p4_mgt1.md). Eight shaders, gpu/shaders/mgt1_*.slang, each
// specialised to M: a stage owns one 32-slot slice of a shared address table
// (as in AttnRunner) and one pipeline per M, built the first time that M is
// asked for (`ensure`).
// ===========================================================================

enum class MgtStage : uint32_t {
    // mgt1_gemv: the fp8 GEMVs and their tails
    WqASplit = 0, WqACombine,
    QNorm,                      // q_norm, per token
    WqBSplit, WqBFinish,        // wq_b + RoPE + bf16 q
    WkvSplit, WkvFinish,        // wkv + kv_norm + RoPE + overflow copy + ring write
    WoASplit, WoACombine,       // ActQuant 0
    WoBSplit, WoBCombine,
    IdxQSplit, IdxQCombine,     // the indexer's wq_b
    // mgt1_mhc, one slice each for the three uses (mega_mhc's argument)
    MhcPost, MhcMix, MhcFinal, MhcPostB, MhcMixB, MhcFinalB, MhcClose,
    // mgt1_attn
    AttnScore, AttnPv,
    // mgt1_gate
    GateScore, GateTopK,
    // mgt1_cmp
    CmpKvGemv, CmpGateGemv, CmpPool, CmpState, CmpStore,
    // mgt1_idx
    IdxQFinish, IdxKey, IdxWeights, IdxScore, IdxTopK, IdxBlockKeys, IdxBlockSelect,
    IdxApplyCand,
    // mgt1_head
    Head, HeadArgmax, HeadTopK,
    // mgt1_engram
    EngramGemv, EngramGate,
    Count,
};

const char* mgt_stage_name(MgtStage s);

inline constexpr uint32_t kMgtMaxM = 6;

// The knobs. A K-split factor must be a power of two dividing K / 32 with
// K / factor <= 1024 (mgt1_gemv.slang's staged slice).
struct MgtSpec {
    uint32_t lanes_per_row = 32;
    uint32_t subgroup_size = 32;
    uint32_t ksplit_wq_a = 8;       // 5120 / 8 = 640
    uint32_t ksplit_wq_b = 2;       // 1280 / 2 = 640
    uint32_t ksplit_wkv  = 8;
    uint32_t ksplit_wo_a = 4;       // 4096 / 4 = 1024
    uint32_t ksplit_wo_b = 8;       // 8192 / 8 = 1024
    uint32_t ksplit_idx  = 2;
    uint32_t tile_heads_per_wg = 8;
    uint32_t head_slices   = 5;     // 5120 / 5 = 1024
    uint32_t engram_slices = 8;     // 6144 / 8 = 768
};

// mgt1_gemv.slang
struct MgtGemvPush {
    uint32_t rows = 0, k = 0, scale_cols = 0, rows_per_group = 0, part_stride = 0;
    uint32_t x_stride = 0, y_stride = 0, head_dim = 0, rope_dim = 0, p0 = 0, window = 0;
    float    eps = 0.0f;
};
// mgt1_attn.slang
struct MgtAttnPush {
    uint32_t n_kv = 0, n_win = 0, n_ovf = 0, head_dim = 0, rope_dim = 0, score_stride = 0;
    float    softmax_scale = 0.0f;
    uint32_t n_heads = 0, n_tiles = 0, tile_len = 0, list_stride = 0;
};
// mgt1_cmp.slang
struct MgtCmpPush {
    uint32_t rows = 0, k = 0, ratio = 1, p0 = 0, rope_dim = 0;
    float    norm_eps = 0.0f;
};
// mgt1_idx.slang
struct MgtIdxPush {
    uint32_t n_heads = 0, head_dim = 0, rope_dim = 0, k = 0, p0 = 0, ratio = 1, topk = 0;
    uint32_t offset = 0, score_stride = 0, list_stride = 0, key_sel = 0, blk_stride = 0;
    float    norm_eps = 0.0f, wscale = 0.0f;
};
// mgt1_head.slang
struct MgtHeadPush {
    uint32_t rows = 0, k = 0, slice = 0, slices = 1, x_stride = 0, topk_k = 0;
    float    inv_t = 1.0f, bins_per_logit = 16.0f;
};
// mgt1_engram.slang
struct MgtEngramPush {
    uint32_t rows = 0, k = 0, scale_cols = 0, dim = 0, hc = 0;
    float    norm_eps = 0.0f;
    uint32_t slice = 0, slices = 1;
};

namespace mslot {
// mgt1_gemv
enum : uint32_t { kGW = 0, kGS = 1, kGX = 2, kGY = 3, kGP = 4, kGNormW = 5, kGRope = 6,
                  kGRingVal = 7, kGRingScale = 8, kGOvfVal = 9, kGOvfScale = 10, kGKvOut = 11 };
// mgt1_mhc: gpu::slot's mega_mhc indices (kPostIn / kCombIn unused: offsets of kPreMix)
// mgt1_attn: gpu::slot's nine + tile max, then the overflow planes
enum : uint32_t { kAQ = 0, kAWinVal = 1, kAWinScale = 2, kACmpKv = 3, kATopIdx = 4,
                  kASink = 5, kARope = 6, kAScore = 7, kAO = 8, kATileMax = 9,
                  kAOvfVal = 10, kAOvfScale = 11 };
// mgt1_gate: gpu::slot's gate indices.  mgt1_cmp: gpu::slot's compressor indices.
// mgt1_idx
enum : uint32_t { kIQRaw = 0, kIQ = 1, kIQFp4 = 2, kIQScale = 3, kIRope = 4, kIWk = 5,
                  kIKNormW = 6, kILatent = 7, kIKRaw = 8, kIKCache = 9, kIKFp4 = 10,
                  kIKScale = 11, kIWProjW = 12, kIX = 13, kIWeights = 14, kIScore = 15,
                  kIOut = 16, kIKCachePub = 17, kIBlkKey = 18, kICand = 19 };
// mgt1_head
enum : uint32_t { kHW = 0, kHX = 1, kHLogits = 2, kHSample = 3, kHTopOut = 4, kHHist = 5 };
// mgt1_engram: dslot's engram indices
}  // namespace mslot

// Words of one row's record in mgt1_head stage 2's output.
inline constexpr uint32_t kMgtTopKRecordWords = 8 + 256 + 256 * 32 * 2;

class MgtRunner {
public:
    MgtRunner() = default;
    ~MgtRunner() { destroy(); }
    MgtRunner(const MgtRunner&) = delete;
    MgtRunner& operator=(const MgtRunner&) = delete;

    Result<void> create(Device& device, MemoryAllocator& alloc, const std::string& shader_dir,
                        const MgtSpec& spec = {});
    void destroy();
    const MgtSpec& spec() const { return spec_; }

    // Builds every pipeline for batch size `m` (1..6; the bench's M = 1 column
    // runs the same kernels specialised to one token).
    Result<void> ensure(uint32_t m);

    uint64_t* slots(MgtStage s);
    Result<void> record(CommandBuffer& cmd, uint32_t m, MgtStage s, const void* push,
                        uint32_t push_bytes, uint32_t gx, uint32_t gy = 1);
    Result<void> dispatch_now(uint32_t m, MgtStage s, const void* push, uint32_t push_bytes,
                              uint32_t gx, uint32_t gy = 1);

    uint32_t ksplit(MgtStage s) const;
    uint32_t row_groups(uint32_t rows) const {
        const uint32_t per = 256 / spec_.lanes_per_row;
        return (rows + per - 1) / per;
    }
    // Workgroups of a split stage: row groups x slices.
    uint32_t split_groups(MgtStage s, uint32_t rows) const { return row_groups(rows) * ksplit(s); }
    static uint32_t combine_groups(uint32_t rows) { return (rows + 255) / 256; }

private:
    Result<void> make(uint32_t m, MgtStage s);

    Device*          device_ = nullptr;
    MemoryAllocator* alloc_  = nullptr;
    MgtSpec          spec_{};
    std::string      dir_;
    struct PerM {
        bool ready = false;
        std::vector<Pipeline> pipes;
#if defined(DEEPMOE_ENABLE_VULKAN)
        std::vector<VkDescriptorSet> sets;
#endif
    };
    PerM             per_m_[kMgtMaxM + 1];
    DescriptorPool   descriptors_;
    CommandPool      pool_;
    GpuBuffer        table_{};
};

}  // namespace deepmoe::gpu
