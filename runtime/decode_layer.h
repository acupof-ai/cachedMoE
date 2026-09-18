// One decoder layer's decode-time execution: dispatches 1-9 of design §7.14
// recorded into one command buffer, the §7.1 timeline gate, and the MoE.
//
// What this is for
// ----------------
// tests/test_gpu_attn.cpp proves each kernel against the oracle's own input.
// That is deliberately not the same as proving the layer: a per-stage pass
// only says nothing is individually wrong, not that the stages compose -- a
// buffer wired to the wrong slot, a coefficient handed to the wrong sublayer
// (design §2.4's pre/post handoff), or a missing barrier all survive it. This
// runs the chain the way decode will, feeding each kernel the previous one's
// output, and compares the block output against the oracle's.
//
// The command buffer
// ------------------
// design §7.1 wants one pre-recorded command buffer per token with a timeline
// wait before each MoE dispatch. This records dispatches 1-9 of one layer into
// the caller's command buffer with a global shader-write -> shader-read
// barrier between each -- a decode layer is a strict chain, so nothing finer
// is needed -- and then stops, because the gate's output has to reach the CPU
// before the MoE can be dispatched. `submit_attention` signals a timeline
// value the planner answers.
//
// The MoE boundary
// ----------------
// `MoeBridge` is the single point of contact with the MoE kernels, which
// another track owns and is actively changing. Everything that knows about
// MoeRunner, MoeSpec, the pointer table and the fp8 shared-expert flag lives
// behind that one interface; a change to their push constants is a change to
// one file.
//
// Ownership/threading: a DecodeLayer borrows the runner, the pinned weights
// and the KV store, and records from the single GPU submit thread.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/status.h"
#include "core/types.h"
#include "gpu/vulkan/attn_kernels.h"
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/decode_kernels.h"
#include "gpu/vulkan/timeline.h"
#include "model/v41_config.h"
#include "runtime/kvstore.h"
#include "runtime/rope.h"
#include "store/pinned.h"

namespace deepmoe::runtime {

// The device addresses of one layer's pinned weights. Resolved once per layer,
// not per token: the whole point of the pinned set is that these never move.
struct LayerWeights {
    uint64_t hc_attn_fn = 0, hc_attn_base = 0, hc_attn_scale = 0;
    uint64_t hc_ffn_fn = 0, hc_ffn_base = 0, hc_ffn_scale = 0;
    uint64_t attn_norm = 0, ffn_norm = 0;
    uint64_t wq_a = 0, wq_a_scale = 0, q_norm = 0;
    uint64_t wq_b = 0, wq_b_scale = 0;
    uint64_t wkv = 0, wkv_scale = 0, kv_norm = 0;
    uint64_t attn_sink = 0;
    uint64_t wo_a = 0, wo_a_scale = 0, wo_b = 0, wo_b_scale = 0;
    uint64_t gate_w = 0, gate_bias = 0;
    // design §7.4, present only on a source layer -- 0 everywhere else, which
    // is what `LayerStep::run_compressor` / `run_indexer` already say.
    // `wgate` exists only at compress_ratio > 1.
    uint64_t cmp_wkv = 0, cmp_wgate = 0, cmp_norm = 0;
    uint64_t idx_wq_b = 0, idx_wq_b_scale = 0, idx_wk = 0, idx_k_norm = 0, idx_wproj = 0;

    static Result<LayerWeights> from_pinned(const store::PinnedStore& p, uint32_t layer);
};

// The activation buffers one token needs. All of them together are under 2 MB,
// so they come out of one GpuScratch (design §7.14's dispatch list touches 133
// MB of weights against this).
struct DecodeScratch {
    gpu::GpuScratch::View x, xout, utmp, partials, mix_raw, mix_a, mix_b, u;
    gpu::GpuScratch::View qr, q, rope;
    gpu::GpuScratch::View kv_raw, kv_partials, kv;
    gpu::GpuScratch::View score, o, woa, wob;
    gpu::GpuScratch::View gate_scores, gate_ids, gate_weights, layer_done;
    gpu::GpuScratch::View moe_x, moe_y;
    // design §7.4's compressor and indexer. Per-step working buffers only: the
    // state that outlives a step -- the compressed-KV plane, the index-key
    // cache and the compressor's carried group -- lives in runtime/kvstore.h.
    gpu::GpuScratch::View rope_lat;                    // the latent's own position
    gpu::GpuScratch::View cmp_y, cmp_g, latent, latent_q, cmp_fp4, cmp_scale;
    gpu::GpuScratch::View idx_q_raw, idx_q, idx_q_fp4, idx_q_scale;
    gpu::GpuScratch::View idx_k_raw, idx_k_fp4, idx_k_scale;
    gpu::GpuScratch::View idx_w, idx_score;
    // design §2.1's candidate blocks: the source's per-block radix keys and the
    // keep flags every consumer layer of the same step reads.
    gpu::GpuScratch::View idx_blk_key, idx_cand;

    Result<void> create(gpu::GpuScratch& s, const TextConfig& cfg);
};

// Compressed positions one `indexer.score` dispatch can be asked for. This is
// the KERNEL's limit, not a buffer size anyone chose: stage 4 covers
// kIdxScoreTile positions a workgroup and a dispatch has at most 65,535
// workgroups, so 524,280 positions -- eight times V4.1's original 65,536-token
// context. The score plane is sized for it once (2 MB of the 32 MB scratch)
// and so are the candidate-block planes (0.5 MB). What really bounds a decode's
// context is the KV store's memory (KvStoreConfig::total_bytes), which the
// caller sizes from the context it wants.
//
// History: this was 4,096 (then 16,384) and doubled as the context cap, which
// hid that sparse_attn was being handed window + n_cmp entries instead of
// window + min(index_topk, n_cmp) -- Track P's fix in Engine::prepare_ced.
inline constexpr uint32_t kMaxIndexPositions = 65535u * gpu::kIdxScoreTile;

// Row stride of sparse_attn's [heads][stride] score plane. A top-k list is
// window + min(index_topk, n_cmp) = 640 entries in V4.1; `record_attention`
// refuses any list longer than the stride, which is exactly the overrun that
// produced NaN logits past 1K tokens of context.
inline constexpr uint32_t kAttnScoreStride = 1024;

// What the MoE track is asked to do for one layer. Deliberately in terms of
// the model, not of their kernels: expert ids, routing weights, an input and
// an output.
struct MoeCall {
    uint32_t        layer = 0;
    const uint32_t* ids = nullptr;       // [topk] routed expert ids
    const float*    weights = nullptr;   // [topk] routing weights, already x route_scale
    uint32_t        topk = 0;
    const float*    x = nullptr;         // [hidden] ffn_norm output
    float*          y = nullptr;         // [hidden] routed + shared, fp32 accumulated
    uint32_t        hidden = 0;
};

class MoeBridge {
public:
    virtual ~MoeBridge() = default;
    virtual const char* name() const = 0;
    virtual Result<void> run(const MoeCall& call) = 0;
};

// A bridge that returns Unavailable. Lets the attention chain be validated
// against the oracle's `ffn_norm_out` and gate ids on a machine or a branch
// where the MoE kernels are not usable, and says so rather than silently
// producing a wrong block output.
std::unique_ptr<MoeBridge> make_null_moe_bridge();

// Everything one decode step of one layer needs beyond the weights.
struct LayerStep {
    uint32_t layer    = 0;
    uint32_t position = 0;       // absolute token position; the ring slot is position % window
    uint32_t compress_ratio = 2; // picks the RoPE base and whether YaRN is on (§2.1)
    bool     apply_hc_post = true;   // false only for the very first sublayer of a sequence
    // Whether `bind` writes the MhcClose slice. See `bind`.
    bool     bind_close = true;
    KvLayerView kv{};

    // --- design §7.4, decided by the Engine and not by this layer -----------
    //
    // `kv` above already carries the addresses of the caches this layer READS:
    // model.py's `shared_attn` makes a reuse layer read the cache its source
    // published, so `kv.cmp_kv`, `kv.top_idx` and `kv.idx_key` may belong to
    // another layer entirely. What a source layer WRITES is here.

    // Compressed positions visible to this layer's query: `(position + 1) /
    // compress_ratio`, which is both `compress_len` and the reference's
    // `end_pos // ratio` slice of the index-key cache.
    uint32_t n_cmp = 0;
    bool     run_compressor = false;   // this layer is a kv_source_layer
    bool     run_indexer    = false;   // this layer is an index_source_layer
    // `(position + 1) % compress_ratio == 0`: the group just filled, so the
    // compressor pools and publishes and the indexer derives a key from it. At
    // ratio 1 it is every step; at ratio 2 every other one.
    bool     cmp_complete = false;
    // design §2.1's two-level top-k, which only matters once n_cmp exceeds
    // candidate_topk_blocks * candidate_block_size. Derived by DecodeLayer from
    // the config and n_cmp -- the Engine does not have to set anything.
    //
    // This layer's OWN index-key cache, which only a kv_source_layer has. The
    // one being scored against is `kv.idx_key`, and the two differ whenever a
    // ratio-2 source's group is incomplete: it publishes nothing and scores
    // against whatever cache was published last (docs/p2_attention.md §9.3).
    DeviceAddress idx_key_write = kNoDeviceAddress;
};

// --- Track T: a verify batch of M = k + 1 <= 6 tokens (docs/p4_mgt1.md) -----
//
// The per-token activations of design §7.14 with a batch dimension, for the
// gpu/shaders/mgt1_*.slang kernels. Created once, for the largest M and the
// longest compressed run a batch will score.
struct BatchScratch {
    uint32_t m_cap = 0, list_stride = 0, score_stride = 0, blk_stride = 0;
    gpu::GpuScratch::View x, xout, utmp, u, partials, mix_raw, mix_a, mix_b;
    gpu::GpuScratch::View qr_raw, qr, q, rope, rope_lat, kv_out, ovf_val, ovf_scale;
    gpu::GpuScratch::View score, tile_max, o, woa, wob, part;
    gpu::GpuScratch::View gate_scores, gate_ids, gate_weights, layer_done, moe_y;
    gpu::GpuScratch::View cmp_y, cmp_g, latent, latent_q, cmp_fp4, cmp_scale;
    gpu::GpuScratch::View idx_q_raw, idx_q, idx_q_fp4, idx_q_scale;
    gpu::GpuScratch::View idx_k_raw, idx_k_fp4, idx_k_scale, idx_w, idx_score;
    gpu::GpuScratch::View idx_blk_key, idx_cand;
    // [n_lists][M][list_stride] int32: list 0 serves the window-only layers,
    // list 1 + i the i-th index source and every layer that reads it.
    gpu::GpuScratch::View lists;
    uint32_t n_lists = 0;

    Result<void> create(gpu::GpuScratch& s, const TextConfig& cfg, uint32_t m_cap,
                        uint32_t max_compressed);
    int32_t* list_host(uint32_t i) const {
        return static_cast<int32_t*>(lists.host) + uint64_t(i) * m_cap * list_stride;
    }
    uint64_t list_addr(uint32_t i) const { return lists.addr + uint64_t(i) * m_cap * list_stride * 4; }
};

// One layer of one verify batch. `kv` carries this layer's ring and carried
// compressor state and its source's compressed plane, as LayerStep's does.
struct BatchStep {
    uint32_t layer = 0;
    uint32_t p0 = 0;              // position of token 0
    uint32_t m = 1;               // tokens in the batch
    uint32_t compress_ratio = 2;
    bool     apply_hc_post = true;
    bool     bind_close = true;
    KvLayerView kv{};
    uint32_t list = 0;            // which BatchScratch list this layer reads
    bool     run_compressor = false;
    bool     run_indexer = false;
    // The index-key caches: this layer's own (a kv source) and the one a query
    // whose group did not complete here scores against (docs/p3_dspark.md §4.8).
    DeviceAddress idx_key_own = kNoDeviceAddress;
    DeviceAddress idx_key_pub = kNoDeviceAddress;
    uint32_t key_sel = 0;         // bit m: query m scores `idx_key_own`

    uint32_t n_cmp(uint32_t mm) const {
        return compress_ratio ? (p0 + mm + 1) / compress_ratio : 0u;
    }
    uint32_t n_win_ext() const;   // window + m - 1
};

// The window half of every batch list, and -1 over the compressed half: query
// m sees ring slot s iff the position it holds after the batch's writes is in
// [0, p0 + m], and overflow row j - 1 iff j > m and that row held a position.
void write_batch_window_lists(BatchScratch& b, uint32_t window, uint32_t p0, uint32_t m);

class DecodeLayer {
public:
    DecodeLayer() = default;
    ~DecodeLayer() { destroy(); }

    DecodeLayer(const DecodeLayer&) = delete;
    DecodeLayer& operator=(const DecodeLayer&) = delete;

    // Releases the command pool. The scratch and the weights are borrowed, so
    // there is nothing else to give back; teardown order matters only because
    // the pool belongs to the Device.
    void destroy() { pool_.destroy(); device_ = nullptr; runner_ = nullptr; }

    Result<void> create(gpu::Device& device, gpu::AttnRunner& runner,
                        gpu::GpuScratch& scratch, const TextConfig& cfg);

    DecodeScratch& scratch() { return buf_; }
    const DecodeScratch& scratch() const { return buf_; }

    // Points the runner's address tables at this layer's weights and this
    // step's KV, and writes the RoPE table for `position`. Call before
    // recording; it touches host memory only.
    Result<void> bind(const LayerWeights& w, const LayerStep& s);

    // Records dispatches 1-9 of design §7.14 into `cmd`, barrier-separated.
    // On a source layer that includes design §7.4's compressor and indexer,
    // recorded between the window-KV write and sparse_attn, which is where
    // `Attention.forward` runs them.
    Result<void> record_attention(gpu::CommandBuffer& cmd, const LayerStep& s);

    // After the command buffer holding `record_attention(s)` has run: proves
    // the indexer wrote the whole compressed half of this step's top-k list --
    // `record_attention` poisons it with -1 first -- and that what it wrote is
    // a strictly increasing run of in-range compressed rows. A layer that is
    // not an index source has nothing to check. `run_attention` calls it; the
    // token loop should call it after its wait. ~512 host reads a source.
    Result<void> verify_after_attention(const LayerStep& s) const;
    // Whether this step, on this layer, runs design §2.1's level one (the
    // candidate source) or level two (a consumer).
    bool candidate_source(const LayerStep& s) const;
    bool candidate_consumer(const LayerStep& s) const;

    // Records + submits + waits. The decode loop will not wait here -- §7.1 is
    // explicit about that -- but a layer-at-a-time validator does.
    Result<void> run_attention(const LayerStep& s);

    // The §7.1 gate: the CPU reads the gate's ids out of host-coherent memory,
    // `on_ready` makes the experts resident, the timeline is signalled, and the
    // MoE runs. With every expert pinned (design §15, P2) `on_ready` is a
    // formality and the wait is satisfied immediately -- but the shape is the
    // one P3 needs.
    Result<void> run_moe(MoeBridge& moe, const LayerStep& s, gpu::Timeline* timeline,
                         const std::function<Result<void>(const uint32_t*, uint32_t)>& on_ready);

    // hc_post of the MoE output into the stream, which design §7.7 fuses into
    // the NEXT layer's first mega_mhc. Recorded here so a layer-at-a-time
    // validator can close the block; the token loop lets the next layer do it.
    Result<void> record_close(gpu::CommandBuffer& cmd, const LayerStep& s);
    Result<void> run_close(const LayerStep& s);

    // Where the MoE output lives when it is not this layer's own scratch.
    // The token loop leaves it on the GPU in the MoE bridge's `y`, which the
    // next layer's hc_post reads by address; a layer-at-a-time validator keeps
    // the default, `scratch().moe_y`, which `run_moe` copies into. Takes
    // effect at the next `bind`.
    void set_moe_output(uint64_t addr, const float* host) {
        moe_out_addr_ = addr;
        moe_out_host_ = host;
    }
    uint64_t     moe_out_addr() const { return moe_out_addr_ ? moe_out_addr_ : buf_.moe_y.addr; }

    // Where the FFN half writes its normed input -- what the gate scores and
    // what the host copies into the MoE's fp16 `x` every layer. Default is the
    // scratch `u`, which lives in path A: DEVICE_LOCAL|HOST_VISIBLE, i.e.
    // write-combining, and a 20 KB read of it measured 155 us a layer, 6.2 ms
    // a token. The token loop points it at ordinary host pages imported for
    // the GPU (path B), where the same read is a cached memcpy.
    void set_ffn_input(uint64_t addr, float* host) { ffn_in_addr_ = addr; ffn_in_host_ = host; }
    uint64_t ffn_in_addr() const { return ffn_in_addr_ ? ffn_in_addr_ : buf_.u.addr; }
    const float* moe_out() const {
        return moe_out_host_ ? moe_out_host_ : static_cast<const float*>(buf_.moe_y.host);
    }

    // Host views of the two things the layer produces.
    const float* block_out() const;     // [hc][hidden] fp32 residual stream
    const float* gate_scores() const;
    const uint32_t* gate_ids() const;
    const float* gate_weights() const;
    const float* ffn_norm_out() const;

    const TextConfig& config() const { return *cfg_; }

    // --- Track T: the verify batch ---------------------------------------
    // Borrows the M > 1 runner and allocates the batch activations. Separate
    // from `create` so the M = 1 path is untouched by it.
    Result<void> create_batch(gpu::MgtRunner& mgt, gpu::GpuScratch& scratch, uint32_t m_cap,
                              uint32_t max_compressed);
    BatchScratch& batch() { return bb_; }
    const BatchScratch& batch() const { return bb_; }
    Result<void> bind_batch(const LayerWeights& w, const BatchStep& s);
    // Dispatches 1-9 of design §7.14 for all M tokens, and §7.4's compressor
    // and indexer on a source layer. The caller has written the lists' window
    // halves (`write_batch_window_lists`) once for the batch.
    Result<void> record_attention_batch(gpu::CommandBuffer& cmd, const BatchStep& s);
    Result<void> run_attention_batch(const BatchStep& s);
    // Every query's compressed half: min(index_topk, n_cmp(m)) strictly
    // increasing in-range entries, and -1 after them.
    Result<void> verify_after_attention_batch(const BatchStep& s) const;
    Result<void> record_close_batch(gpu::CommandBuffer& cmd, const BatchStep& s);
    Result<void> run_close_batch(const BatchStep& s);
    // `h = hc_pre(h, pre_mix); logits = head(norm(h))` for all M rows, then the
    // per-row argmax (and, when `topk_out` is set, the per-row top set). The
    // stream is `batch().x`, i.e. after `run_close_batch` of the last layer
    // has been folded in by this call's MhcClose.
    struct BatchTail {
        uint64_t norm_w = 0, head_w = 0;
        uint64_t logits = 0;          // [M][vocab] fp32
        uint64_t sample = 0;          // [M][4] words
        uint64_t topk_out = 0, topk_hist = 0;
        uint32_t topk_k = 1024;
        float    inv_t = 1.0f;
    };
    Result<void> record_tail_batch(gpu::CommandBuffer& cmd, const BatchStep& s, const BatchTail& t);
    // Where token m's MoE output lives (`batch().moe_y` + m * hidden).
    const float* batch_ffn_norm_out(uint32_t m) const {
        return static_cast<const float*>(bb_.u.host) + uint64_t(m) * cfg_->hidden_size;
    }

private:
    gpu::MgtRunner*   mgt_ = nullptr;
    BatchScratch      bb_{};
    std::vector<float> batch_rope_;
    Result<void> record_ced_batch(gpu::CommandBuffer& cmd, const BatchStep& s);
    Result<void> submit(gpu::CommandBuffer& cmd);
    Result<void> bind_ced(const LayerWeights& w, const LayerStep& s, const RopeConfig& rc);
    const std::vector<float>& rope_cached(const RopeConfig& rc, uint32_t position);
    struct RopeEntry {
        bool     valid = false;
        uint32_t position = 0, original_seq_len = 0, dim = 0;
        double   base = 0.0, factor = 0.0;
        std::vector<float> table;
    };
    std::array<RopeEntry, 4> rope_cache_{};
    uint32_t                 rope_next_ = 0;
    // `scratch().moe_y` with its address replaced by wherever the MoE output
    // actually is -- the only field of it hc_post reads.
    gpu::GpuScratch::View moe_view() const {
        gpu::GpuScratch::View v = buf_.moe_y;
        v.addr = moe_out_addr();
        return v;
    }
    Result<void> record_ced(gpu::CommandBuffer& cmd, const LayerStep& s);

    gpu::Device*      device_ = nullptr;
    gpu::AttnRunner*  runner_ = nullptr;
    const TextConfig* cfg_    = nullptr;
    DecodeScratch     buf_{};
    LayerWeights      weights_{};
    gpu::CommandPool  pool_;
    // One command buffer, acquired once and re-begun per use. Acquiring one
    // per `run_attention` allocated a VkCommandBuffer every layer of every
    // token and never gave it back until the pool died.
    gpu::CommandBuffer cmd_{};
    uint64_t          moe_out_addr_ = 0;
    // The step design §2.1's candidate mask in the scratch was built for.
    uint32_t          cand_position_ = ~0u, cand_n_cmp_ = 0;
    uint64_t          ffn_in_addr_ = 0;
    float*            ffn_in_host_ = nullptr;
    const float*      moe_out_host_ = nullptr;
    uint32_t          hcdim_ = 0;
};

}  // namespace deepmoe::runtime
