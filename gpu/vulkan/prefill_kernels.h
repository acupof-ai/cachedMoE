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

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <map>
#include <span>
#include <memory>
#include <optional>
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
inline constexpr uint32_t kPfFlagBatched    = 256u;   // prefill_gemm_lds: gid.x is a batch index
inline constexpr uint32_t kPfFlagOutF16     = 512u;   // rope: write fp16
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
    uint32_t extra2 = 0;    // prefill_coopmat: CmTokTiles (0 = 2)
    uint32_t extra3 = 0;    // prefill_coopmat: CmRowTilesPerWg (0 = 1)
    uint32_t extra4 = 0;    // prefill_gemm_lds: LdsWt (W stored [K][R])
    uint32_t extra5 = 0;    // prefill_gemm_lds: LdsGroupRows (grouped weight: rows a group, x [groups][n][K])
    auto key() const { return std::tie(spv, stage, wfmt, xfmt, tile, extra0, extra1, extra2, extra3, extra4, extra5); }
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
struct PfCoopPush {
    uint32_t n = 0, k = 0, idx_off = 0, flags = 0, row0 = 0, x_off = 0;
    uint32_t x_stride = 0, x_col0 = 0, y_stride = 0, y_row0 = 0;   // stage 0: 0 = K, 0, R, 0
};
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

// Rows of slack on every plane a cooperative-matrix GEMM stages into: one
// workgroup covers 16 * CmTokTiles (<= 128) tokens, so the last workgroup of a
// dispatch touches up to that many rows past the real row count.
inline constexpr uint64_t kPfRowSlack = 160;

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
    // prefill_coopmat stages 1 and 2: a 32-thread workgroup, 8 elements a thread.
    static uint32_t stage_groups(uint64_t rows, uint32_t k) {
        return static_cast<uint32_t>((rows * (k / 8) + 31) / 32);
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
struct PfTopkPush {
    uint32_t b = 0, g = 0, n_idx = 0, window = 0, pos0 = 0, kv_pos0 = 0, n_kv_rows = 0, ratio = 1;
    uint32_t block = 0, nblocks = 0, flags = 0, pad = 0;   // flags & 1: a block mask
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
    // one computing, one filling), 2.4 GB at 64. With the P0 queue at the
    // prefill's depth (p0_qd) the read-ahead of 128 experts covers a 4K
    // prompt's serial phase and both drives run at their rate the whole
    // layer: 4,133 tokens 31.7 -> 29.8 s, 17,010 54.0 -> 52.6; 112 (4.2 GB)
    // is the same at 4K and about 0.7 s better at 17K (STATUS §7 0aj).
    uint32_t transit_slots = 64;   // <= 228: a half is one allocation of at most 4 GiB
    // The IO engine's P0 queue while the prefill runs (IoEngine::set_p0_depth):
    // 24 chunks / 96 MiB against the decode's 8 -- a layer's read-ahead is
    // hundreds of chunks, a decode miss is 18.
    uint32_t p0_qd = 24;
    uint64_t p0_inflight_bytes = 96u << 20;
    // A layer whose FFN runs over at least this many rows starts reading its
    // routed experts before its gate has picked any (docs/STATUS.md §7 0aa): at
    // 4,133 rows a layer uses 329-381 of its 384, and the drive is otherwise
    // idle until the gate. The first 2 x transit_slots in shard order go into
    // the transit; the gate then decides as before, and a picked expert that
    // was read ahead is computed from where it already is. 0 = off.
    uint32_t read_ahead_min_rows = 1024;
    // Round onto bf16 wherever the reference holds a bf16 tensor.
    bool     round = true;
    // The largest prompt the activation buffers are sized for.
    uint32_t max_tokens = 1024;
    // Rows a submit carries in a pass over every prompt row (dense linears, the
    // shared expert, the engram gate): Linux amdgpu resets a compute queue after
    // 2 s, which 17,010 tokens hit in layer 0. 0 = one submit.
    uint32_t max_rows_per_submit = 2048;
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
    // Band attention on cooperative-matrix tiles (docs/p4_prefill_speed.md §3):
    // gather, tile scores, softmax, tile P.V, finish, one submit. false = the
    // per-(head, query) kernel of docs/p3_prefill.md §6.
    bool     attn_coop = true;
    // 16-head tiles per workgroup of the two tile stages (1, 2 or 4).
    uint32_t attn_head_tiles = 1;
    // 16-dim output tiles held at once by the P.V stage (1, 2, 4 or 8). The
    // contraction there is over the entries and the output index is the dim, so
    // one dim tile at a time walks the gathered KV plane with a 16 KiB stride
    // and re-reads it head_dim/16 times. > 1 forces attn_head_tiles = 1 for
    // that stage (accumulators) and is the geometry docs/p4_prefill_speed.md
    // §3.1 measures.
    uint32_t attn_pv_dim_tiles = 4;
    // sqrtsoftplus + noaux_tc top-6 on the GPU (prefill_elem stage 11) instead
    // of reading all n x 384 gate scores back and sorting them on the host.
    // false = the host path, which is the correctness reference.
    bool     gate_topk_gpu = true;
    // The indexer's top-k rows on the GPU (prefill_topk.slang) instead of a
    // partial_sort per query on the host after reading the scores back. Same
    // rows bit for bit; false = the host path, which is the reference.
    bool     index_topk_gpu = true;
    // 16-token tiles per workgroup of the cooperative-matrix GEMM
    // (prefill_coopmat stage 0's CmTokTiles, 1..8). The weight tile is loaded
    // once per workgroup and reused by every token tile it holds, so the WEIGHT
    // traffic of a GEMM is ceil(n / (16 * this)) passes over the matrix: at
    // n = 512 and the old value of 2 that was 16 passes over wq_b's 84 MB.
    uint32_t coop_tok_tiles = 8;
    // The cooperative-matrix GEMM on prefill_gemm_lds.slang -- four waves
    // sharing LDS tiles, 128 rows x 64 tokens a workgroup (64 x 64 when the
    // rows only divide by 64) -- instead of prefill_coopmat stage 0, whenever
    // the shape tiles. Bit for bit the same result. DEEPMOE_PF_LDS=0 = stage 0.
    bool     lds_gemm = true;
    // wo_a (and any grouped linear) on cooperative matrix instead of the
    // tiled GEMV. false = the pre-F3 path, which is the reference.
    bool     coop_grouped_dense = true;
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
    // One per_op entry: milliseconds, calls, and the analytic work of those
    // calls from the shapes at dispatch -- FLOPs and the bytes the op must
    // move at least (each input read once, each output written once;
    // intermediates a fused kernel would keep on chip count nothing).
    // tools/prefill_model.py scores them against model_probe's ceilings.
    struct Op { double ms = 0, flop = 0, bytes = 0, reads = 0; uint32_t calls = 0; };
    // Keys: "gemm RxK w<fmt> x<fmt>" / "coop ..." / "<shader> s<stage>" are the
    // wall time of one submit (submit + wait); "gpu: " entries are GPU
    // timestamps between the steps of a multi-dispatch submit; "moe ..." the
    // MoE's GPU batches; "io: " disk reads (bytes and reads are the disk's,
    // ms the exposed wait); "host: " CPU steps.
    std::map<std::string, Op> per_op;
    // One JSON line: the run, its buckets, every op by wall time -- what
    // prefill_bench --ops-json writes and tools/prefill_model.py reads.
    std::string json(uint32_t n, std::string_view mode, std::string_view load) const;
};

// The analytic work of one op call (PrefillTimes::Op): disk requests too.
struct PfCost { double flop = 0, bytes = 0, reads = 0; };

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

// ADDITIVE (Track R1, docs/p4_hitrate.md §3; docs/p3_prefill.md §3.4 / §8.3
// item 3): where the routed experts this prefill streams come from and go to.
// Null = the prefill's own transit, every expert read and dropped (as before).
struct PfExpertSink {
    enum class Kind : uint8_t { Drop = 0, Fill, Resident };
    struct Dest {
        Kind     kind = Kind::Drop;
        void*    host = nullptr;     // the slot base, Fill: the runs go at run.slot_offset
        uint64_t dev  = 0;           // the slot base (Fill and Resident)
        uint64_t cookie = 0;         // the sink's own
    };
    // Before an expert's read is issued. `last_pos` is the absolute prompt
    // position of the LAST row routed to it and `last_slot` its rank in that
    // row's top-6. Resident = no read; Fill = read into the slot; Drop = transit.
    std::function<Dest(uint32_t layer, uint32_t expert, uint32_t last_pos, uint32_t last_slot)> reserve;
    // After the batch that computed from `d` has run (`ok`), or its read failed.
    std::function<void(uint32_t layer, uint32_t expert, const Dest& d, bool ok)> release;
    // Whether the cache holds the expert already: the read-ahead skips it.
    // Null = never.
    std::function<bool(uint32_t layer, uint32_t expert)> cached;
};

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
    PfExpertSink* expert_sink = nullptr;   // Track R1; borrowed
    const PrefillTimes& times() const { return times_; }
    // bytes of workspace `create` would allocate for pcfg.max_tokens tokens
    uint64_t workspace_bytes(const TextConfig& cfg, const PrefillConfig& pcfg);
    const PrefillConfig& config() const { return pcfg_; }
    PrefillConfig&       config() { return pcfg_; }   // tests flip kernel switches between ops

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
    // `rows_per_group` != 0 is the grouped form (wo_a: 8 groups of [1024][4096]
    // over one [32768] row): the weight is decoded once and each group is one
    // staging pass over its own column slice plus one GEMM into its own output
    // rows. Only the fp32-activation staging supports it.
    Result<bool> op_gemm_coop(const PfWeight& w, uint32_t xfmt, uint64_t x, uint64_t xs,
                              uint32_t n, uint32_t x_stride, uint64_t y, uint32_t flags,
                              uint32_t rows_per_group = 0);
    Result<void> op_rmsnorm(uint64_t x, uint64_t y, uint32_t n, uint32_t d, uint64_t w);
    Result<void> op_mhc_pre_norm(uint64_t h, uint32_t n, uint64_t coeff, uint32_t coeff_stride,
                                 uint64_t norm_w, uint64_t out, uint64_t rs);
    Result<void> op_sinkhorn(uint64_t raw, uint64_t out, uint32_t n, uint64_t base, uint64_t scale);
    // cpu::gate_topk for `n` rows on the GPU: raw gate scores [n][E] ->
    // sqrtsoftplus, top-6 of (score + bias), normalised weights. Only
    // n x 6 ids and weights come back, not n x 384 scores.
    Result<void> op_gate_topk(uint64_t scores, uint64_t bias, uint32_t n, uint64_t ids,
                              uint64_t wts);
    Result<void> op_mhc_post(uint64_t h, uint64_t a, uint64_t coeff, uint64_t out, uint32_t n);
    // mode: 0 RoPE only, 1 fp8/UE8M0-32, 2 FP4/UE8M0-32, 3 FP4/E4M3-16
    // out_f16 (mode 0 only): y is fp16 [n][d] -- the plane the attention's
    // q16 stage would have staged the f32 result to, so that pass and half
    // the rope's write go away (STATUS §7 0am). Not for o and wo_a's x16
    // stage: that pass leaves x in the MALL for the tiles (§3 77).
    Result<void> op_rope(uint64_t x, uint64_t y, uint32_t n, uint32_t d, uint32_t head_dim,
                         uint32_t mode, uint32_t block, bool compressed_theta, uint32_t pos0,
                         uint32_t pos_step, bool inverse, bool out_f16 = false);
    Result<void> op_cmp_pool(uint64_t kv, uint64_t score, uint64_t out, uint32_t groups,
                             uint32_t ratio);
    Result<void> op_engram_gate(uint64_t h, uint64_t kv, uint64_t qw, uint64_t kw, uint64_t out,
                                uint32_t n);
    // q: fp32 [b][H*D], or with q16 the fp16 plane b_.q16 already holds
    // (op_rope out_f16) and the q16 stage is skipped. attn_coop_ok says
    // whether the coopmat path -- the only one that takes q16 -- will run.
    bool attn_coop_ok(uint32_t b, uint32_t n_idx) const;
    Result<void> op_attention(uint64_t q, uint64_t kv, uint32_t n_win, uint64_t cmp,
                              uint64_t idx, uint32_t n_idx, uint64_t sink, uint64_t o, uint32_t b,
                              bool q16 = false);
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
    const float* engram_x_host() const { return static_cast<const float*>(b_.eng_x.host_ptr); }
    // The MoE of layer L over `rows` rows of x (f32 [rows][dim], the FFN
    // input): the activation round trip, the shared expert and/or the routed
    // experts of `ids`/`wts` ([rows][6]) streamed expert-major, accumulated into
    // `y` (which the caller zeroes).
    Result<void> op_moe(uint32_t L, uint32_t rows, uint64_t x, std::vector<uint32_t>& ids,
                        std::vector<float>& wts, uint64_t y, bool shared = true, bool routed = true);

private:
    Result<void> flush_one(uint32_t kernel, const void* push, uint32_t bytes, uint32_t gx,
                           uint32_t gy = 1);
    // A multi-dispatch submit with GPU timestamps between its steps:
    // cmd_open() begins cmd_ and stamps; rec() records one dispatch and a
    // barrier; mark(name) closes the step since the last mark; cmd_close()
    // submits, waits, and adds every step to times_.per_op as "gpu: <name>".
    Result<void> cmd_open();
    Result<void> rec(uint32_t kernel, const void* push, uint32_t bytes, uint32_t gx, uint32_t gy = 1);
    void mark(std::string name, PfCost cost = {});
    Result<void> cmd_close();
    void add_op(const std::string& key, double ms, PfCost cost = {}) {
        auto& op = times_.per_op[key];
        op.ms += ms; op.flop += cost.flop; op.bytes += cost.bytes; op.reads += cost.reads; ++op.calls;
    }
    Result<void> op_attention_legacy(uint64_t q, uint64_t kv, uint32_t n_win, uint64_t cmp,
                                     uint64_t idx, uint32_t n_idx, uint64_t sink, uint64_t o, uint32_t b);
    Result<void> build_rope(uint32_t positions);
    Result<void> run_layer(uint32_t L, std::span<const uint32_t> prompt, PrefillHandoff& out);
    Result<void> run_moe(uint32_t L, uint32_t rows, uint64_t x, uint64_t xq, uint64_t xs,
                         uint64_t y, std::vector<uint32_t>& ids, std::vector<float>& wts,
                         bool shared = true, bool routed = true);
    // An engram layer's row reads: issued by engram_issue, waited for and
    // converted by engram_rows. Kept as a member so that an error path (a
    // short read, STATUS §7 0ao) can never free the staging pages under
    // reads still in flight -- that was the segfault of 2026-09-30 02:10.
    // Issuing them ahead of their layer (the rows depend on the prompt
    // alone) was measured and reverted: STATUS §3 78.
    struct EngramAhead {
        uint32_t L = ~0u;
        std::vector<uint64_t> rows, uniq;
        std::vector<uint32_t> skew;
        std::vector<std::future<storage::IoResult>> futs;
        HostAllocInfo stage;
        double bytes = 0;
        uint64_t reads = 0;   // issued to the drive (resident scale rows are not)
        void drop() {   // nothing lands in freed pages
            for (auto& f : futs) if (f.valid()) f.wait();
            if (stage.ptr) free_host_pages(stage);
            L = ~0u; rows.clear(); uniq.clear(); skew.clear(); futs.clear(); stage = {}; bytes = 0; reads = 0;
        }
        ~EngramAhead() { drop(); }
    };
    EngramAhead engram_ahead_;
    Result<void> engram_issue(uint32_t L, std::span<const uint32_t> prompt);
    Result<void> engram_rows(uint32_t L, std::span<const uint32_t> prompt, uint64_t out);
    Result<void> read_ahead(uint32_t L);
    std::byte* transit_host(uint32_t slot) const {
        return static_cast<std::byte*>(b_.transit[slot / pcfg_.transit_slots].host_ptr) +
               uint64_t(slot % pcfg_.transit_slots) * layout::kExpertSlotBytes;
    }
    uint64_t transit_dev(uint32_t slot) const {
        return b_.transit[slot / pcfg_.transit_slots].dev_addr +
               uint64_t(slot % pcfg_.transit_slots) * layout::kExpertSlotBytes;
    }
    Result<void> op_topk_rows(uint32_t b, uint32_t pos0, uint32_t kv_pos0, uint32_t n_kv_rows,
                              uint32_t window, uint32_t ratio, uint32_t g, uint32_t n_idx,
                              uint64_t scores, uint64_t out, uint64_t mask, uint32_t nblocks);
    Result<void> op_copy(uint64_t src, uint64_t dst, uint64_t bytes);
    Result<void> read_expert(uint32_t L, uint32_t e, const ExpertEntry& ent, std::byte* dst,
                             IoPriority pri, std::vector<std::future<storage::IoResult>>& futs,
                             double& bytes);

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
    QueryPool                 qp_;
    bool                      qp_ok_ = false;
    std::vector<std::pair<std::string, PfCost>> marks_;
    struct Want { GpuBuffer* b; uint64_t bytes; };
    std::vector<Want> plan(const TextConfig& cfg, const PrefillConfig& pcfg);
    // b_.w16 is kW16Slots equal slots of fp16 weight decodes: a dense weight
    // decoded once a layer instead of once a query block (STATUS §7 0an), the
    // slot chosen least recently used. Slot 0 is also the MoE's transit (the
    // shared expert's decode, the routed experts' jobs) and the sliced
    // engram GEMM's, which invalidate it.
    static constexpr uint32_t kW16Slots = 4;
    std::array<uint64_t, kW16Slots> w16_src_{};    // weight each slot holds (0 = nothing)
    std::array<uint64_t, kW16Slots> w16_used_{};   // when it was last used
    uint64_t                  w16_clock_ = 0;
    uint64_t                  w16_slot_bytes_ = 0;
    // The slot for w's decode and whether it still has to be decoded.
    std::pair<uint64_t, bool> w16_slot(uint64_t src);
    uint32_t                  moe_pos0_ = 0;  // absolute position of run_moe's row 0 (Track R1)
    // read_ahead's reads for the layer it was called for: transit slot of each
    // expert (~0u = not read ahead) and, per slot, its futures' range in
    // `futs`; run_moe waits for a slot's reads only when its batch computes,
    // so the later ones keep landing while the GPU works (STATUS §7 0ai).
    struct Ahead {
        uint32_t layer = ~0u;
        std::vector<uint32_t> slot;
        std::vector<std::pair<uint32_t, uint32_t>> range;   // per transit slot: [first, last) in futs
        std::vector<std::future<storage::IoResult>> futs;
        double bytes = 0;
    } ahead_;
    // A host-side step's wall time into times_.per_op.
    void host_op(const char* name, std::chrono::steady_clock::time_point t0) {
        add_op(name, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::string               tag_;   // per_op key of the next flush_one, "" = the kernel spec
    PfCost                    cost_;  // the next flush_one's work

    std::vector<GpuBuffer> owned_;
    struct Bufs {
        GpuBuffer h_a, h_b, h_in_copy, x, rs, mix_raw, mix_a, mix_f, mix_prev, xq, xs;
        GpuBuffer kv_raw, kv_norm, kv;
        GpuBuffer ckv, cscore, latent_pre, latent, key_raw, key_norm;
        GpuBuffer qr_raw, qr, qrq, qrs, q, iq, iw, iscore, idx, cmask, score, o, woa, woaq, woas, attn;
        GpuBuffer fx, fxq, fxs, gate, gids, gwts, y, hplane, hq, hs, csr_idx, csr_rw, jobs;
        GpuBuffer x16, h16, gu, dout, w16;          // the cooperative-matrix MoE
        GpuBuffer eng_x, eng_xq, eng_xs, eng_kv;
        GpuBuffer transit[2], rope_win, rope_cmp, logits, nrm;   // one allocation per transit half
        GpuBuffer q16, g16, p16, inv;               // the cooperative-matrix attention
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
