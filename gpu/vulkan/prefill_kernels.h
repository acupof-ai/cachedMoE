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

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <span>
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
inline constexpr uint32_t kPfFlagRoundPre   = 64u;
inline constexpr uint32_t kPfFlagFromJob    = 128u;
// prefill_common.slang kWgRowX: a split 1-D dispatch's workgroups per gid.y row.
inline constexpr uint32_t kPfWgRowX = 16384u;

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
    // The spec a handle was created from.
    const PfKernel* spec(uint32_t handle) const;
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

// =============================================================================
// The prefill itself: docs/p3_prefill.md §2's schedule over the kernels above.
// =============================================================================

// prefill_elem.slang / prefill_attn.slang push constants.
struct PfElemPush {
    uint32_t n = 0, d = 0, a0 = 0, a1 = 0, a2 = 0, a3 = 0, flags = 0;
    float    eps = 0.0f, f1 = 0.0f;
    uint32_t pos0 = 0, pos_step = 1, row_off = 0;
};
struct PfAttnPush {
    uint32_t b = 0, n_idx = 0, n_heads = 0, head_dim = 0, n_win = 0, g = 0;
    float    scale = 1.0f;
    uint32_t ratio = 1, pos0 = 0, flags = 0;
};

// A weight as a GEMM sees it.
struct PfWeight {
    uint64_t data = 0, scale = 0;
    uint32_t fmt = kPfFp8, rows = 0, k = 0;
    uint64_t bytes = 0;         // data + scale, for the bandwidth accounting
};

struct PrefillConfig {
    // Decoder rows (design §11.2's bounded replay). >= the prompt length is the
    // oracle mode, which is exactly `inference/model.py`.
    uint32_t replay = 128;
    uint32_t query_block = 512;
    uint32_t tile = 8;
    // Routed-expert transit slots per half of the read-ahead (two halves:
    // one computing, one filling).
    uint32_t transit_slots = 32;
    // Round onto bf16 wherever the reference holds a bf16 tensor.
    bool     round = true;
    // The largest prompt the activation buffers are sized for.
    uint32_t max_tokens = 1024;
    // An expert (routed or shared) with at least this many tokens runs on
    // cooperative-matrix GEMM (docs/p3_prefill.md §5 option (a)); fewer tokens
    // run on the tiled GEMV job table (b), whose cost has no fixed per-matrix
    // decode. 0 = every expert on (a); UINT32_MAX = every expert on (b).
    uint32_t coopmat_min_rows = 16;
    // A dense linear over at least this many rows runs on cooperative-matrix
    // GEMM when its shape allows (rows and cols multiples of 16, no grouping,
    // row scale or output scale); fewer on the tiled GEMV. UINT32_MAX = never.
    uint32_t coopmat_dense_min_rows = 64;
    // Validation only: hand every layer's per-stage buffers to `probe`.
    bool     probe_layers = false;
};

// What a decode engine inherits (docs/p3_prefill.md §8.1).
struct PrefillHandoff {
    std::vector<uint32_t> prompt;
    uint32_t first_token = 0;
    float    top1 = 0.0f, top2 = 0.0f;
    std::vector<float> logits;              // the last position's, [vocab]
    struct Layer {
        std::vector<float> win_kv;          // [128][512], slot p % 128 holds position p
        std::vector<float> cmp_cache;       // [n_cmp][512], kv sources only
        std::vector<float> index_k;         // [n_cmp][128], kv sources only
        std::vector<float> cmp_state_kv;    // [ratio][512], ratio > 1 sources only
        std::vector<float> cmp_state_score; // [ratio][512]
        uint32_t n_cmp = 0, ratio = 0;
        // Validation: the last prompt position's index row as `sparse_attn` saw
        // it (window part = positions, compressed part = row + N, -1 unused; the
        // reference's `Lnn.topk_idxs_last` in oracle mode), and its top-6.
        std::vector<int32_t>  topk_last;
        std::vector<uint32_t> gate_ids_last;
    };
    std::vector<Layer> layers;
};

// The §10 breakdown. Wall milliseconds on the host clock, each GPU bucket
// including its submit and wait.
struct PrefillTimes {
    double embed = 0, engram_io = 0, engram = 0, mhc = 0, attention = 0, gate = 0;
    double shared_expert = 0, expert_io = 0, expert_gpu = 0, host = 0, head = 0, total = 0;
    uint64_t expert_bytes = 0, engram_reads = 0;
    uint32_t experts_read = 0, dispatches = 0, submits = 0;
    // Wall time of every single-dispatch op (submit + wait), by kernel shape:
    // "gemm RxK w<fmt> x<fmt>" or "<shader> s<stage>". ms and calls.
    std::map<std::string, std::pair<double, uint32_t>> per_op;
};

// Host views of one layer's buffers, valid only inside the probe callback.
// Rows are this layer's rows; `row0` is the absolute position of row 0.
struct PrefillProbe {
    uint32_t layer = 0, rows = 0, row0 = 0, attn_rows = 0, attn_row0 = 0;
    const float* block_in = nullptr;        // [rows][hc][dim], copied before the layer ran
    const float* engram_out = nullptr;      // engram layers only
    const float* attn_norm_out = nullptr;   // [front rows][dim] (N at layer 20)
    uint32_t     front_rows = 0;
    const float* mix_attn = nullptr;        // [front rows][24]
    const float* kv = nullptr;              // [front rows][512]
    const float* cmp_latent = nullptr;      // [G][512] pre-RoPE, kv sources
    const float* cmp_cache = nullptr;       // [G][512]
    const float* index_k = nullptr;         // [G][128]
    uint32_t     n_cmp = 0;
    const int32_t* topk_first = nullptr;    // [first block rows][n_idx]
    uint32_t     n_idx = 0;
    const float* attn_out = nullptr;        // [attn rows][dim] (wo_b)
    const float* attn_block_out = nullptr;  // [attn rows][hc][dim]
    const float* mix_ffn = nullptr;         // [attn rows][24]
    const float* ffn_norm_out = nullptr;    // [attn rows][dim]
    const float* moe_out = nullptr;         // [attn rows][dim]
    const float* block_out = nullptr;       // [attn rows][hc][dim]
    const uint32_t* gate_ids = nullptr;     // [attn rows][6]
    const float* gate_weights = nullptr;
};

}  // namespace deepmoe::gpu

namespace deepmoe::store { class PinnedStore; }
namespace deepmoe { struct TextConfig; }
namespace deepmoe::runtime { struct EngramTables; }

namespace deepmoe::gpu {

class Prefill {
public:
    Prefill() = default;
    ~Prefill() { destroy(); }
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;

    Result<void> create(Device& device, MemoryAllocator& alloc, PrefillRunner& runner,
                        const Manifest& manifest, const store::ShardSet& shards,
                        storage::IoEngine& io, const store::PinnedStore& pinned,
                        const TextConfig& cfg, const runtime::EngramTables* engram,
                        const PrefillConfig& pcfg);
    void destroy();

    // The whole prompt through the forty layers and the head.
    Result<PrefillHandoff> run(std::span<const uint32_t> prompt);

    std::function<void(const PrefillProbe&)> probe;
    const PrefillTimes& times() const { return times_; }
    const PrefillConfig& config() const { return pcfg_; }

    // Writes `h` as an L3-format directory `Engine::load_decode_state` reads
    // (docs/p3_prefill.md §8.2): our prefill record, plus the reference's step
    // records and engram tables copied from `reference_l3` so the loader's
    // `steps_exported > 0` holds and the steps are there to compare against.
    static Result<void> write_l3_dir(const PrefillHandoff& h, const TextConfig& cfg,
                                     const std::string& reference_l3, const std::string& dir);

    // --- the ops, immediate (record, submit, wait): what the per-stage test
    // --- drives with golden inputs, and what `run` is built from -----------
    Result<PfWeight> weight(const std::string& name, uint32_t fmt) const;
    Result<void> op_act_quant(uint64_t x, uint32_t n, uint32_t d, uint64_t q16, uint64_t sc);
    Result<void> op_gemm(const PfWeight& w, uint32_t xfmt, uint64_t x, uint64_t xs, uint32_t n,
                         uint32_t x_stride, uint64_t y, uint32_t flags, float out_scale = 1.0f,
                         uint32_t rows_per_group = 0, uint64_t row_scale = 0);
    // op_gemm's option (a) branch: decode W to fp16 (skipped when the transit
    // already holds it), stage x as fp16, cooperative-matrix GEMM into a padded
    // plane, copy (and round) into y. One submit. False = not applicable.
    Result<bool> op_gemm_coop(const PfWeight& w, uint32_t xfmt, uint64_t x, uint64_t xs,
                              uint32_t n, uint32_t x_stride, uint64_t y, uint32_t flags);
    Result<void> op_rmsnorm(uint64_t x, uint64_t y, uint32_t n, uint32_t d, uint64_t w);
    Result<void> op_mhc_pre_norm(uint64_t h, uint32_t n, uint64_t coeff, uint32_t coeff_stride,
                                 uint64_t norm_w, uint64_t out, uint64_t rs);
    Result<void> op_sinkhorn(uint64_t raw, uint64_t out, uint32_t n, uint64_t base, uint64_t scale);
    Result<void> op_mhc_post(uint64_t h, uint64_t a, uint64_t coeff, uint64_t out, uint32_t n);
    // mode: 0 RoPE only, 1 fp8/UE8M0-32, 2 FP4/UE8M0-32, 3 FP4/E4M3-16
    Result<void> op_rope(uint64_t x, uint64_t y, uint32_t n, uint32_t d, uint32_t head_dim,
                         uint32_t mode, uint32_t block, bool compressed_theta, uint32_t pos0,
                         uint32_t pos_step, bool inverse);
    Result<void> op_cmp_pool(uint64_t kv, uint64_t score, uint64_t out, uint32_t groups,
                             uint32_t ratio);
    Result<void> op_engram_gate(uint64_t h, uint64_t kv, uint64_t qw, uint64_t kw, uint64_t out,
                                uint32_t n);
    Result<void> op_attention(uint64_t q, uint64_t kv, uint32_t n_win, uint64_t cmp,
                              uint64_t idx, uint32_t n_idx, uint64_t sink, uint64_t o, uint32_t b);
    Result<void> op_index_score(uint64_t q, uint64_t keys, uint32_t g, uint64_t w, uint64_t score,
                                uint32_t b, uint32_t ratio, uint32_t pos0);
    // design §2.1's first level (`select_candidate_blocks`) for queries at
    // pos0..pos0+b-1 over [b][g] scores: per query, a keep flag per block of
    // `block` positions, or an empty row when every reachable block is kept.
    static std::vector<std::vector<uint8_t>> candidate_blocks(uint32_t b, uint32_t pos0, uint32_t ratio,
                                                              uint32_t g, const float* scores,
                                                              uint32_t topk_blocks, uint32_t block);
    // The window band plus the compressed picks for queries at positions
    // pos0..pos0+b-1 (docs/p3_prefill.md §1), host side. `scores` is
    // [b][g] or null (every visible block kept).
    static void topk_rows(uint32_t b, uint32_t pos0, uint32_t kv_pos0, uint32_t n_kv_rows,
                          uint32_t window, uint32_t ratio, uint32_t g, uint32_t index_topk,
                          const float* scores, int32_t* out, uint32_t n_idx);

    // Scratch the test can borrow: a host-visible, device-addressable buffer.
    Result<GpuBuffer> scratch(uint64_t bytes);

    // The engram rows of every prompt position of layer L, dequantised onto
    // bf16, into `engram_x()` ([n][6144] f32). Reads 48 x n rows off NVMe.
    Result<void> op_engram_rows(uint32_t L, std::span<const uint32_t> prompt);
    uint64_t engram_x() const { return b_.eng_x.dev_addr; }
    // The MoE of layer L over `rows` rows of x (f32 [rows][dim], the FFN
    // input): the activation round trip, the shared expert and/or the routed
    // experts of `ids`/`wts` ([rows][6]) streamed expert-major, accumulated into
    // `y` (which the caller zeroes).
    Result<void> op_moe(uint32_t L, uint32_t rows, uint64_t x, std::vector<uint32_t>& ids,
                        std::vector<float>& wts, uint64_t y, bool shared = true, bool routed = true);

private:
    Result<void> flush_one(uint32_t kernel, const void* push, uint32_t bytes, uint32_t gx,
                           uint32_t gy = 1);
    Result<void> build_rope(uint32_t positions);
    Result<void> run_layer(uint32_t L, std::span<const uint32_t> prompt, PrefillHandoff& out);
    Result<void> run_moe(uint32_t L, uint32_t rows, uint64_t x, uint64_t xq, uint64_t xs,
                         uint64_t y, std::vector<uint32_t>& ids, std::vector<float>& wts,
                         bool shared = true, bool routed = true);
    Result<void> engram_rows(uint32_t L, std::span<const uint32_t> prompt, uint64_t out);

    Device*                   device_ = nullptr;
    MemoryAllocator*          alloc_  = nullptr;
    PrefillRunner*            runner_ = nullptr;
    const Manifest*           manifest_ = nullptr;
    const store::ShardSet*    shards_ = nullptr;
    storage::IoEngine*        io_ = nullptr;
    const store::PinnedStore* pinned_ = nullptr;
    const TextConfig*         cfg_ = nullptr;
    const runtime::EngramTables* engram_ = nullptr;
    PrefillConfig             pcfg_{};
    PrefillTimes              times_{};
    CommandBuffer             cmd_{};
    bool                      cmd_valid_ = false;
    uint64_t                  w16_src_ = 0;   // weight whose fp16 decode b_.w16 holds
    // A host-side step's wall time into times_.per_op.
    void host_op(const char* name, std::chrono::steady_clock::time_point t0) {
        auto& slot = times_.per_op[name];
        slot.first += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++slot.second;
    }
    std::string               tag_;   // per_op key of the next flush_one, "" = the kernel spec

    std::vector<GpuBuffer> owned_;
    struct Bufs {
        GpuBuffer h_a, h_b, h_in_copy, x, rs, mix_raw, mix_a, mix_f, mix_prev, xq, xs;
        GpuBuffer kv_raw, kv_norm, kv;
        GpuBuffer ckv, cscore, latent_pre, latent, key_raw, key_norm;
        GpuBuffer qr_raw, qr, qrq, qrs, q, iq, iw, iscore, idx, score, o, woa, woaq, woas, attn;
        GpuBuffer fx, fxq, fxs, gate, y, hplane, hq, hs, csr_idx, csr_rw, jobs;
        GpuBuffer x16, h16, gu, dout, w16;          // the cooperative-matrix MoE
        GpuBuffer eng_x, eng_xq, eng_xs, eng_kv;
        GpuBuffer transit, rope_win, rope_cmp, logits, nrm;
    } b_{};
    struct SourceState { GpuBuffer cache, keys; uint32_t n = 0; bool valid = false; };
    std::vector<SourceState> sources_;      // per layer; only kv sources fill theirs
    uint32_t cmp_src_ = 0, key_src_ = 0, idx_src_ = 0;
    std::vector<int32_t> topk_shared_;      // the last index source's compressed picks, [rows][k]
    uint32_t topk_k_ = 0;
    // layer 20's candidate blocks per attention row; empty = identity (§1)
    std::vector<std::vector<uint8_t>> cand_;
    uint32_t rope_positions_ = 0;
    std::vector<int32_t>  probe_idx_;       // the first query block's index rows
    std::vector<uint32_t> probe_ids_;       // this layer's top-6, [rows][6]
    std::vector<float>    probe_wts_;
};

}  // namespace deepmoe::gpu
