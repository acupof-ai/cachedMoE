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

    Result<void> create(gpu::GpuScratch& s, const TextConfig& cfg);
};

// Compressed positions one `indexer.score` dispatch can be asked for. The
// score plane is the only per-step buffer whose size grows with the context,
// and at 4096 it is 16 KB; the KV store's `max_context` is checked against it.
inline constexpr uint32_t kMaxIndexPositions = 4096;

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
    // This layer's OWN index-key cache, which only a kv_source_layer has. The
    // one being scored against is `kv.idx_key`, and the two differ whenever a
    // ratio-2 source's group is incomplete: it publishes nothing and scores
    // against whatever cache was published last (docs/p2_attention.md §9.3).
    DeviceAddress idx_key_write = kNoDeviceAddress;
};

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

private:
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
    const float*      moe_out_host_ = nullptr;
    uint32_t          hcdim_ = 0;
};

}  // namespace deepmoe::runtime
