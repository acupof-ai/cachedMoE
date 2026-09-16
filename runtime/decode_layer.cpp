#include "runtime/decode_layer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <memory>

#include "runtime/rope.h"

namespace deepmoe::runtime {

Result<LayerWeights> LayerWeights::from_pinned(const store::PinnedStore& p, uint32_t layer) {
    const std::string pre = std::format("layers.{}", layer);
    LayerWeights w;
    auto get = [&](const char* suffix, uint64_t* data, uint64_t* scale) -> Result<void> {
        const std::string n = pre + "." + suffix;
        auto t = p.require(n);
        if (!t) return std::unexpected(t.error());
        *data = (*t)->data;
        if (scale) {
            if ((*t)->scale == kNoDeviceAddress)
                return fail(Err::FailedPrecondition,
                            std::format("'{}' has no block-scale plane", n));
            *scale = (*t)->scale;
        }
        return {};
    };
    struct Entry { const char* suffix; uint64_t* data; uint64_t* scale; };
    const Entry entries[] = {
        {"hc_attn_fn", &w.hc_attn_fn, nullptr},
        {"hc_attn_base", &w.hc_attn_base, nullptr},
        {"hc_attn_scale", &w.hc_attn_scale, nullptr},
        {"hc_ffn_fn", &w.hc_ffn_fn, nullptr},
        {"hc_ffn_base", &w.hc_ffn_base, nullptr},
        {"hc_ffn_scale", &w.hc_ffn_scale, nullptr},
        {"attn_norm.weight", &w.attn_norm, nullptr},
        {"ffn_norm.weight", &w.ffn_norm, nullptr},
        {"attn.wq_a.weight", &w.wq_a, &w.wq_a_scale},
        {"attn.q_norm.weight", &w.q_norm, nullptr},
        {"attn.wq_b.weight", &w.wq_b, &w.wq_b_scale},
        {"attn.wkv.weight", &w.wkv, &w.wkv_scale},
        {"attn.kv_norm.weight", &w.kv_norm, nullptr},
        {"attn.attn_sink", &w.attn_sink, nullptr},
        {"attn.wo_a.weight", &w.wo_a, &w.wo_a_scale},
        {"attn.wo_b.weight", &w.wo_b, &w.wo_b_scale},
        {"ffn.gate.weight", &w.gate_w, nullptr},
        {"ffn.gate.bias", &w.gate_bias, nullptr},
    };
    for (const Entry& e : entries)
        if (auto r = get(e.suffix, e.data, e.scale); !r) return std::unexpected(r.error());

    // design §7.4's tensors exist on four layers of forty (the compressor) and
    // eight (the indexer), so absence is the normal case and not an error. The
    // Engine decides whether to dispatch from the config's source lists, and
    // `bind` checks the two against each other.
    auto opt = [&](const char* suffix, uint64_t* data, uint64_t* scale) {
        auto t = p.find(pre + "." + suffix);
        if (!t) return;
        *data = t->data;
        if (scale && t->scale != kNoDeviceAddress) *scale = t->scale;
    };
    opt("attn.compressor.wkv.weight", &w.cmp_wkv, nullptr);
    opt("attn.compressor.wgate.weight", &w.cmp_wgate, nullptr);
    opt("attn.compressor.norm.weight", &w.cmp_norm, nullptr);
    opt("attn.indexer.wq_b.weight", &w.idx_wq_b, &w.idx_wq_b_scale);
    opt("attn.indexer.wk.weight", &w.idx_wk, nullptr);
    opt("attn.indexer.k_norm.weight", &w.idx_k_norm, nullptr);
    opt("attn.indexer.weights_proj.weight", &w.idx_wproj, nullptr);
    return w;
}

Result<void> DecodeScratch::create(gpu::GpuScratch& s, const TextConfig& cfg) {
    const uint32_t dim = cfg.hidden_size;
    const uint32_t hc  = cfg.hc_mult;
    const uint32_t qrows = cfg.num_attention_heads * cfg.head_dim;
    const uint32_t orows = cfg.o_groups * cfg.o_lora_rank;
    const uint32_t irows = cfg.index_n_heads * cfg.index_head_dim;
    // The score plane is [heads][stride]; 1024 covers the 128 + 512 of design
    // §7.5 with room for a longer window.
    const uint32_t score_stride = kAttnScoreStride;
    // §2.1's block planes, one uint32 per block of the largest scorable run.
    const uint32_t cand_bs = std::max<uint32_t>(1, cfg.candidate_block_size);
    const uint64_t cand_blocks = (uint64_t(kMaxIndexPositions) + cand_bs - 1) / cand_bs;

    struct Want { gpu::GpuScratch::View* v; uint64_t bytes; };
    const Want wants[] = {
        {&x,            uint64_t(hc) * dim * 4},
        {&xout,         uint64_t(hc) * dim * 4},
        {&utmp,         uint64_t(dim) * 4},
        {&partials,     512 * 4},          // 256 sum-of-squares pairs (design §7.2 stage 0)
        {&mix_raw,      128},              // the 24 raw hc_fn dot products
        {&mix_a,        128},              // pre[4] post[4] comb[16]
        {&mix_b,        128},
        {&u,            uint64_t(dim) * 4},
        {&qr,           uint64_t(cfg.q_lora_rank) * 4},
        {&q,            uint64_t(qrows) * 2},
        {&rope,         256},
        {&kv_raw,       uint64_t(cfg.head_dim) * 4},
        {&kv_partials,  256 * 4},
        {&kv,           uint64_t(cfg.head_dim) * 4},
        {&score,        uint64_t(cfg.num_attention_heads) * score_stride * 4},
        {&o,            uint64_t(qrows) * 4},
        {&woa,          uint64_t(orows) * 4},
        {&wob,          uint64_t(dim) * 4},
        {&gate_scores,  uint64_t(cfg.n_routed_experts) * 4},
        {&gate_ids,     64},
        {&gate_weights, 64},
        {&layer_done,   64},
        {&moe_x,        uint64_t(dim) * 4},
        {&moe_y,        uint64_t(dim) * 4},
        // §7.4. `irows` is index_n_heads * index_head_dim = 4096.
        {&rope_lat,     256},
        {&cmp_y,        uint64_t(cfg.head_dim) * 4},
        {&cmp_g,        uint64_t(cfg.head_dim) * 4},
        {&latent,       uint64_t(cfg.head_dim) * 4},
        {&latent_q,     uint64_t(cfg.head_dim) * 4},
        {&cmp_fp4,      uint64_t(cfg.head_dim) / 2},
        {&cmp_scale,    uint64_t(cfg.head_dim) / 16},
        {&idx_q_raw,    uint64_t(irows) * 4},
        {&idx_q,        uint64_t(irows) * 2},
        {&idx_q_fp4,    uint64_t(irows) / 2},
        {&idx_q_scale,  uint64_t(irows) / 32},
        {&idx_k_raw,    uint64_t(cfg.index_head_dim) * 4},
        {&idx_k_fp4,    uint64_t(cfg.index_head_dim) / 2},
        {&idx_k_scale,  256},
        {&idx_w,        uint64_t(cfg.index_n_heads) * 4},
        {&idx_score,    uint64_t(kMaxIndexPositions) * 4},
        {&idx_blk_key,  cand_blocks * 4},
        {&idx_cand,     cand_blocks * 4},
    };
    for (const Want& w : wants) {
        auto v = s.alloc(w.bytes);
        if (!v) return std::unexpected(v.error());
        *w.v = *v;
    }
    return {};
}

Result<void> DecodeLayer::create(gpu::Device& device, gpu::AttnRunner& runner,
                                 gpu::GpuScratch& scratch, const TextConfig& cfg) {
    device_ = &device;
    runner_ = &runner;
    cfg_    = &cfg;
    hcdim_  = cfg.hc_mult * cfg.hidden_size;
    if (auto r = buf_.create(scratch, cfg); !r) return r;
    if (auto r = pool_.create(device); !r) return r;
    auto cb = pool_.acquire();
    if (!cb) return std::unexpected(cb.error());
    cmd_ = *cb;
    return {};
}

namespace {

// The three mega_mhc stages share one address table, so it is written once and
// copied. The only per-half differences are the hc_fn / base / scale triple,
// the norm weight and which of the two mix buffers is the output.
void bind_mhc(gpu::AttnRunner& runner, gpu::AttnStage post, gpu::AttnStage mix,
              gpu::AttnStage final_, const DecodeScratch& b,
              uint64_t hc_fn, uint64_t hc_base, uint64_t hc_scale, uint64_t norm_w,
              const gpu::GpuScratch::View& mix_in, const gpu::GpuScratch::View& mix_out,
              const gpu::GpuScratch::View& a_in,
              const gpu::GpuScratch::View& x_in, const gpu::GpuScratch::View& x_out,
              uint64_t u_addr) {
    uint64_t* s = runner.slots(post);
    s[gpu::slot::kX]       = x_in.addr;
    s[gpu::slot::kA]       = a_in.addr;
    // MixOut is laid out [pre 4][post 4][comb 16], which is exactly the
    // PostIn / CombIn / PreMix triple the next sublayer reads (design §2.4's
    // handoff, expressed as three pointers into one buffer).
    s[gpu::slot::kPreMix]  = mix_in.addr;
    s[gpu::slot::kPostIn]  = mix_in.addr + 4 * sizeof(float);
    s[gpu::slot::kCombIn]  = mix_in.addr + 8 * sizeof(float);
    s[gpu::slot::kHcFn]    = hc_fn;
    s[gpu::slot::kHcBase]  = hc_base;
    s[gpu::slot::kHcScale] = hc_scale;
    s[gpu::slot::kNormW]   = norm_w;
    s[gpu::slot::kXout]    = x_out.addr;
    s[gpu::slot::kUtmp]    = b.utmp.addr;
    s[gpu::slot::kScratch] = b.partials.addr;
    s[gpu::slot::kMixRaw]  = b.mix_raw.addr;
    s[gpu::slot::kMixOut]  = mix_out.addr;
    s[gpu::slot::kU]       = u_addr;
    std::memcpy(runner.slots(mix), s, gpu::kAttnStageStride);
    std::memcpy(runner.slots(final_), s, gpu::kAttnStageStride);
}

}  // namespace

Result<void> DecodeLayer::bind(const LayerWeights& w, const LayerStep& st) {
    if (!runner_) return fail(Err::FailedPrecondition, "decode layer is not created");
    const TextConfig& c = *cfg_;
    DecodeScratch& b = buf_;

    // RoPE for this position. The base and whether YaRN is on come from the
    // layer's compress ratio, not from a global (design §2.1): layers 0 and 1
    // use rope_theta with YaRN off, everything else compress_rope_theta with
    // YaRN over original_max_position.
    const RopeConfig rc = rope_for_layer(st.compress_ratio, c.qk_rope_head_dim,
                                         c.rope_theta, c.compress_rope_theta,
                                         c.rope_scaling.original_max_position,
                                         c.rope_scaling.factor);
    const std::vector<float>& tab = rope_cached(rc, st.position);
    std::memcpy(b.rope.host, tab.data(), tab.size() * sizeof(float));

    // Attention half: the stream arrives in `x`, hc_post folds in whatever the
    // previous sublayer left (design §7.7), and the mixes for the FFN half go
    // to mix_b. The post-hc_post stream lands in `xout`.
    //
    // The sublayer before this one is the PREVIOUS LAYER's FFN, so the output
    // being folded in is `moe_y`, not `wob`. A layer-at-a-time validator never
    // sees the difference -- it runs with `apply_hc_post` off, so the slot is
    // not read -- and a forty-layer chain sees nothing else: feeding it `wob`
    // adds each layer's attention output to the stream twice and drops its MoE
    // output entirely, which decodes to the input token over and over.
    bind_mhc(*runner_, gpu::AttnStage::MhcPost, gpu::AttnStage::MhcMix,
             gpu::AttnStage::MhcFinal, b,
             w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, w.attn_norm,
             b.mix_a, b.mix_b, moe_view(), b.x, b.xout, b.u.addr);
    // FFN half: reads the stream the attention half wrote, folds in wo_b's
    // output, and writes back to `x`. In-place would also be safe -- a thread
    // reads all hc copies of its own element before writing any -- but keeping
    // the two apart makes a layer-at-a-time validator able to look at both.
    bind_mhc(*runner_, gpu::AttnStage::MhcPostB, gpu::AttnStage::MhcMixB,
             gpu::AttnStage::MhcFinalB, b,
             w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm,
             b.mix_b, b.mix_a, b.wob, b.xout, b.x, ffn_in_addr());
    // Closing hc_post: the MoE output into the stream. In the token loop this
    // is the next layer's MhcPost instead (design §7.7); it is bound here so a
    // single layer can be run and compared on its own.
    //
    // Skipped when the caller has ALREADY recorded a close for the previous
    // layer into the buffer this layer's attention joins: a stage owns one
    // slice of the address table, read when the buffer runs, so rebinding it
    // here would hand that close this layer's hc_ffn weights.
    if (st.bind_close)
        bind_mhc(*runner_, gpu::AttnStage::MhcClose, gpu::AttnStage::MhcClose,
                 gpu::AttnStage::MhcClose, b,
                 w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm,
                 b.mix_a, b.mix_b, moe_view(), b.x, b.xout, b.u.addr);

    uint64_t* qa = runner_->slots(gpu::AttnStage::WqA);
    qa[gpu::slot::kGemvW] = w.wq_a;
    qa[gpu::slot::kGemvS] = w.wq_a_scale;
    qa[gpu::slot::kGemvX] = b.u.addr;
    qa[gpu::slot::kGemvY] = b.qr.addr;

    uint64_t* qb = runner_->slots(gpu::AttnStage::WqB);
    qb[gpu::slot::kWqbW]     = w.wq_b;
    qb[gpu::slot::kWqbS]     = w.wq_b_scale;
    qb[gpu::slot::kWqbQr]    = b.qr.addr;
    qb[gpu::slot::kWqbNormW] = w.q_norm;
    qb[gpu::slot::kWqbRope]  = b.rope.addr;
    qb[gpu::slot::kWqbQ]     = b.q.addr;

    uint64_t* kv = runner_->slots(gpu::AttnStage::WkvGemv);
    kv[gpu::slot::kWkvW]       = w.wkv;
    kv[gpu::slot::kWkvS]       = w.wkv_scale;
    kv[gpu::slot::kWkvX]       = b.u.addr;
    kv[gpu::slot::kWkvNormW]   = w.kv_norm;
    kv[gpu::slot::kWkvRope]    = b.rope.addr;
    kv[gpu::slot::kWkvRaw]     = b.kv_raw.addr;
    kv[gpu::slot::kWkvScratch] = b.kv_partials.addr;
    kv[gpu::slot::kWkvVal]     = st.kv.win_val;
    kv[gpu::slot::kWkvScale]   = st.kv.win_scale;
    kv[gpu::slot::kWkvKv]      = b.kv.addr;
    std::memcpy(runner_->slots(gpu::AttnStage::WkvFinish), kv, gpu::kAttnStageStride);

    uint64_t* at = runner_->slots(gpu::AttnStage::AttnScore);
    at[gpu::slot::kAttnQ]        = b.q.addr;
    at[gpu::slot::kAttnWinVal]   = st.kv.win_val;
    at[gpu::slot::kAttnWinScale] = st.kv.win_scale;
    at[gpu::slot::kAttnCmpKv]    = st.kv.cmp_kv;
    at[gpu::slot::kAttnTopIdx]   = st.kv.top_idx;
    at[gpu::slot::kAttnSink]     = w.attn_sink;
    at[gpu::slot::kAttnRope]     = b.rope.addr;
    at[gpu::slot::kAttnScore]    = b.score.addr;
    at[gpu::slot::kAttnO]        = b.o.addr;
    std::memcpy(runner_->slots(gpu::AttnStage::AttnCombine), at, gpu::kAttnStageStride);

    uint64_t* wa = runner_->slots(gpu::AttnStage::WoA);
    wa[gpu::slot::kWoaW] = w.wo_a;
    wa[gpu::slot::kWoaS] = w.wo_a_scale;
    wa[gpu::slot::kWoaO] = b.o.addr;
    wa[gpu::slot::kWoaY] = b.woa.addr;

    uint64_t* wb = runner_->slots(gpu::AttnStage::WoB);
    wb[gpu::slot::kGemvW] = w.wo_b;
    wb[gpu::slot::kGemvS] = w.wo_b_scale;
    wb[gpu::slot::kGemvX] = b.woa.addr;
    wb[gpu::slot::kGemvY] = b.wob.addr;

    uint64_t* g = runner_->slots(gpu::AttnStage::GateScore);
    g[gpu::slot::kGateW]         = w.gate_w;
    g[gpu::slot::kGateBias]      = w.gate_bias;
    g[gpu::slot::kGateX]         = ffn_in_addr();
    g[gpu::slot::kGateScores]    = b.gate_scores.addr;
    g[gpu::slot::kGateIds]       = b.gate_ids.addr;
    g[gpu::slot::kGateWeights]   = b.gate_weights.addr;
    g[gpu::slot::kGateLayerDone] = b.layer_done.addr;
    std::memcpy(runner_->slots(gpu::AttnStage::GateTopK), g, gpu::kAttnStageStride);

    if (auto r = bind_ced(w, st, rc); !r) return r;
    weights_ = w;
    return {};
}

// design §7.4's compressor and indexer. Nothing here runs on a layer that is
// not a source; what it costs a reuse layer is the address writes, which are
// host memory and are skipped anyway.
Result<void> DecodeLayer::bind_ced(const LayerWeights& w, const LayerStep& st,
                                   const RopeConfig& rc) {
    if (!st.run_compressor && !st.run_indexer) return {};
    const TextConfig& c = *cfg_;
    DecodeScratch& b = buf_;

    // The SECOND RoPE table. A latent stands for the FIRST token of its group,
    // so it -- and the index key derived from it -- is rotated at
    // `start_pos + 1 - ratio`, while the indexer's queries are rotated at
    // `start_pos` like everything else (docs/p2_attention.md §9.3 item 3). At
    // ratio 1 the two tables coincide; at ratio 2 they are one position apart,
    // which is invisible in a cosine and wrong in the bytes.
    {
        const uint32_t ratio = st.compress_ratio ? st.compress_ratio : 1u;
        const uint32_t lat_pos = st.position + 1 - ratio;
        const std::vector<float>& tab = rope_cached(rc, lat_pos);
        std::memcpy(b.rope_lat.host, tab.data(), tab.size() * sizeof(float));
    }

    if (st.run_compressor) {
        if (!w.cmp_wkv || !w.cmp_norm)
            return fail(Err::FailedPrecondition,
                        std::format("layer {} is a kv_source_layer but its compressor "
                                    "weights are not pinned", st.layer));
        if (st.compress_ratio > 1 && !w.cmp_wgate)
            return fail(Err::FailedPrecondition,
                        std::format("layer {} pools at ratio {} but has no compressor "
                                    "gate", st.layer, st.compress_ratio));
        auto cmp = [&](gpu::AttnStage s, uint64_t weight, const gpu::GpuScratch::View& y) {
            uint64_t* p = runner_->slots(s);
            p[gpu::slot::kCmpW]          = weight;
            p[gpu::slot::kCmpX]          = b.u.addr;          // attn_norm output
            p[gpu::slot::kCmpY]          = y.addr;
            p[gpu::slot::kCmpG]          = b.cmp_g.addr;
            p[gpu::slot::kCmpKvState]    = st.kv.cmp_state_kv;
            p[gpu::slot::kCmpScoreState] = st.kv.cmp_state_score;
            p[gpu::slot::kCmpNormW]      = w.cmp_norm;
            p[gpu::slot::kCmpLatent]     = b.latent.addr;
            p[gpu::slot::kCmpRope]       = b.rope_lat.addr;
            p[gpu::slot::kCmpVal]        = st.kv.cmp_kv;
            p[gpu::slot::kCmpFp4]        = b.cmp_fp4.addr;
            p[gpu::slot::kCmpScaleB]     = b.cmp_scale.addr;
            p[gpu::slot::kCmpLatentQ]    = b.latent_q.addr;
        };
        cmp(gpu::AttnStage::CmpKvGemv, w.cmp_wkv, b.cmp_y);
        cmp(gpu::AttnStage::CmpGateGemv, w.cmp_wgate, b.cmp_g);
        cmp(gpu::AttnStage::CmpNorm, w.cmp_wkv, b.cmp_y);
        cmp(gpu::AttnStage::CmpStore, w.cmp_wkv, b.cmp_y);
    }

    if (st.run_indexer) {
        if (!w.idx_wq_b || !w.idx_wproj)
            return fail(Err::FailedPrecondition,
                        std::format("layer {} is an index_source_layer but its indexer "
                                    "weights are not pinned", st.layer));
        // `rope` and `kcache` are the two slots that are NOT the same for every
        // stage. The key is rotated at the latent's position and written into
        // the cache this layer owns; the score reads whatever cache was
        // published last, which need not be this layer's.
        auto idx = [&](gpu::AttnStage s, uint64_t rope, uint64_t kcache) {
            uint64_t* p = runner_->slots(s);
            p[gpu::slot::kIdxW]       = w.idx_wq_b;
            p[gpu::slot::kIdxS]       = w.idx_wq_b_scale;
            p[gpu::slot::kIdxQr]      = b.qr.addr;
            p[gpu::slot::kIdxQNormW]  = w.q_norm;
            p[gpu::slot::kIdxRope]    = rope;
            p[gpu::slot::kIdxQRaw]    = b.idx_q_raw.addr;
            p[gpu::slot::kIdxQ]       = b.idx_q.addr;
            p[gpu::slot::kIdxWk]      = w.idx_wk;
            p[gpu::slot::kIdxKNormW]  = w.idx_k_norm;
            p[gpu::slot::kIdxLatent]  = b.latent.addr;
            p[gpu::slot::kIdxKRaw]    = b.idx_k_raw.addr;
            p[gpu::slot::kIdxKCache]  = kcache;
            p[gpu::slot::kIdxKFp4]    = b.idx_k_fp4.addr;
            p[gpu::slot::kIdxKScale]  = b.idx_k_scale.addr;
            p[gpu::slot::kIdxWProjW]  = w.idx_wproj;
            p[gpu::slot::kIdxX]       = b.u.addr;
            p[gpu::slot::kIdxWeights] = b.idx_w.addr;
            p[gpu::slot::kIdxScore]   = b.idx_score.addr;
            p[gpu::slot::kIdxOut]     = st.kv.top_idx;
            p[gpu::slot::kIdxQFp4]    = b.idx_q_fp4.addr;
            p[gpu::slot::kIdxQScale]  = b.idx_q_scale.addr;
            p[gpu::slot::kIdxBlkKey]  = b.idx_blk_key.addr;
            p[gpu::slot::kIdxCand]    = b.idx_cand.addr;
        };
        idx(gpu::AttnStage::IdxQGemv,   b.rope.addr,     st.kv.idx_key);
        idx(gpu::AttnStage::IdxQFinish, b.rope.addr,     st.kv.idx_key);
        idx(gpu::AttnStage::IdxKey,     b.rope_lat.addr, st.idx_key_write);
        idx(gpu::AttnStage::IdxWeights, b.rope.addr,     st.kv.idx_key);
        idx(gpu::AttnStage::IdxScore,   b.rope.addr,     st.kv.idx_key);
        idx(gpu::AttnStage::IdxTopK,    b.rope.addr,     st.kv.idx_key);
        idx(gpu::AttnStage::IdxBlockKeys,   b.rope.addr, st.kv.idx_key);
        idx(gpu::AttnStage::IdxBlockSelect, b.rope.addr, st.kv.idx_key);
        idx(gpu::AttnStage::IdxApplyCand,   b.rope.addr, st.kv.idx_key);
    }
    (void)c;
    return {};
}

Result<void> DecodeLayer::record_attention(gpu::CommandBuffer& cmd, const LayerStep& st) {
    const TextConfig& c = *cfg_;
    DecodeScratch& b = buf_;
    const uint32_t dim   = c.hidden_size;
    (void)b;
    const uint32_t qrows = c.num_attention_heads * c.head_dim;
    const uint32_t orows = c.o_groups * c.o_lora_rank;
    const uint32_t ocols = qrows / c.o_groups;
    const uint32_t n_wg0 = (dim + 255) / 256;

    // The list sparse_attn walks. Its length is window + min(index_topk, n_cmp)
    // -- model.py's `topk = min(self.index_topk, end_pos // ratio)` -- and
    // every entry past the window must have been written by THIS step's
    // indexer. Both ways of getting that wrong have happened once (a list of
    // window + n_cmp, which read hundreds of never-written zeros and, past
    // 1,024 entries, ran off the end of the per-head score row into the next
    // head), so both are refused here, before anything is recorded.
    {
        const uint32_t n_kv = st.kv.n_kv;
        if (n_kv > kAttnScoreStride || n_kv > c.sliding_window + c.index_topk)
            return fail(Err::Internal,
                        std::format("layer {} position {}: a top-k list of {} entries; "
                                    "sparse_attn takes at most window {} + index_topk {} "
                                    "and its score row holds {}", st.layer, st.position,
                                    n_kv, c.sliding_window, c.index_topk, kAttnScoreStride));
        if (st.n_cmp > 0 &&
            n_kv != c.sliding_window + std::min(c.index_topk, st.n_cmp))
            return fail(Err::Internal,
                        std::format("layer {} position {}: {} compressed positions want a "
                                    "list of {} + {} entries, the store says {}", st.layer,
                                    st.position, st.n_cmp, c.sliding_window,
                                    std::min(c.index_topk, st.n_cmp), n_kv));
    }

    auto step = [&](gpu::AttnStage s, const void* push, uint32_t bytes,
                    uint32_t groups) -> Result<void> {
        if (auto r = runner_->record(cmd, s, push, bytes, groups); !r) return r;
        return cmd.barrier();
    };

    // 1. mega-mHC, attention half. hc_post is applied here for every layer but
    //    the first sublayer of a sequence, where there is no sublayer output
    //    to fold in yet (design §7.7).
    gpu::MhcPush mp{dim, c.hc_mult, (2 + c.hc_mult) * c.hc_mult, n_wg0,
                    c.hc_sinkhorn_iters,
                    st.apply_hc_post ? gpu::kMhcFlagPost : 0u,
                    static_cast<float>(c.rms_norm_eps), static_cast<float>(c.hc_eps)};
    if (auto r = step(gpu::AttnStage::MhcPost, &mp, sizeof mp, n_wg0); !r) return r;
    if (auto r = step(gpu::AttnStage::MhcMix, &mp, sizeof mp, mp.mix_rows); !r) return r;
    if (auto r = step(gpu::AttnStage::MhcFinal, &mp, sizeof mp, n_wg0); !r) return r;

    // 2-3. Q path.
    gpu::GemvPush qa{c.q_lora_rank, dim, dim / 32, 0};
    if (auto r = step(gpu::AttnStage::WqA, &qa, sizeof qa,
                      runner_->gemv_groups(gpu::AttnStage::WqA, c.q_lora_rank)); !r) return r;
    gpu::WqbPush qb{qrows, c.q_lora_rank, c.q_lora_rank / 32, c.head_dim,
                    c.qk_rope_head_dim, static_cast<float>(c.rms_norm_eps)};
    if (auto r = step(gpu::AttnStage::WqB, &qb, sizeof qb,
                      runner_->gemv_groups(gpu::AttnStage::WqB, qrows)); !r) return r;

    // 4. KV path and the ring write.
    const uint32_t kvg = runner_->gemv_groups(gpu::AttnStage::WkvGemv, c.head_dim);
    gpu::WkvPush kp{c.head_dim, dim, dim / 32, c.qk_rope_head_dim,
                    st.position % c.sliding_window, kvg,
                    static_cast<float>(c.rms_norm_eps)};
    if (auto r = step(gpu::AttnStage::WkvGemv, &kp, sizeof kp, kvg); !r) return r;
    if (auto r = step(gpu::AttnStage::WkvFinish, &kp, sizeof kp, 1); !r) return r;

    // 4b. design §7.4: the compressor and the indexer, on the four and eight
    //     layers that have them. `Attention.forward` runs them between the
    //     window-KV write and sparse_attn, and the order matters: the indexer
    //     needs the compressor's PRE-RoPE latent, which is why the cache write
    //     that rotates it comes last.
    if (auto r = record_ced(cmd, st); !r) return r;

    // 5. Sparse attention over the window plus the compressed picks.
    gpu::AttnPush ap{st.kv.n_kv, c.sliding_window, c.head_dim, c.qk_rope_head_dim,
                     kAttnScoreStride,
                     1.0f / std::sqrt(static_cast<float>(c.head_dim))};
    if (auto r = step(gpu::AttnStage::AttnScore, &ap, sizeof ap, c.num_attention_heads); !r)
        return r;
    if (auto r = step(gpu::AttnStage::AttnCombine, &ap, sizeof ap, c.num_attention_heads); !r)
        return r;

    // 6-7. Output projection.
    gpu::WoaPush wa{orows, ocols, ocols / 32, c.o_lora_rank};
    if (auto r = step(gpu::AttnStage::WoA, &wa, sizeof wa,
                      runner_->gemv_groups(gpu::AttnStage::WoA, orows)); !r) return r;
    gpu::GemvPush wb{dim, orows, orows / 32, 0};
    if (auto r = step(gpu::AttnStage::WoB, &wb, sizeof wb,
                      runner_->gemv_groups(gpu::AttnStage::WoB, dim)); !r) return r;

    // 8. mega-mHC, FFN half: hc_post folds the attention output into the
    //    stream and the mixes for the FFN come out of the same dispatch.
    gpu::MhcPush fp = mp;
    fp.flags = gpu::kMhcFlagPost;
    if (auto r = step(gpu::AttnStage::MhcPostB, &fp, sizeof fp, n_wg0); !r) return r;
    if (auto r = step(gpu::AttnStage::MhcMixB, &fp, sizeof fp, mp.mix_rows); !r) return r;
    if (auto r = step(gpu::AttnStage::MhcFinalB, &fp, sizeof fp, n_wg0); !r) return r;

    // 9. Gate. Its output lands in host-coherent memory so the planner can read
    //    ids[] without a fence (design §7.8).
    gpu::GatePush gp{c.n_routed_experts, dim, c.num_experts_per_tok, 16, 1.0f,
                     static_cast<float>(c.routed_scaling_factor)};
    if (auto r = step(gpu::AttnStage::GateScore, &gp, sizeof gp,
                      runner_->gemv_groups(gpu::AttnStage::GateScore, c.n_routed_experts)); !r)
        return r;
    return step(gpu::AttnStage::GateTopK, &gp, sizeof gp, 1);
}

// A decode step needs at most three distinct tables -- window-only layers at
// the position, compressed layers at the position, and a ratio-2 latent one
// position back -- but forty-odd `bind` calls asked for one each, and each is
// a vector allocation plus 32 pow/cos/sin. Four entries, replaced round-robin.
const std::vector<float>& DecodeLayer::rope_cached(const RopeConfig& rc, uint32_t position) {
    for (const RopeEntry& e : rope_cache_)
        if (e.valid && e.position == position && e.base == rc.base &&
            e.original_seq_len == rc.original_seq_len && e.factor == rc.factor &&
            e.dim == rc.rope_head_dim)
            return e.table;
    RopeEntry& e = rope_cache_[rope_next_++ % rope_cache_.size()];
    e.valid = true;
    e.position = position;
    e.base = rc.base;
    e.original_seq_len = rc.original_seq_len;
    e.factor = rc.factor;
    e.dim = rc.rope_head_dim;
    e.table = rope_table(rc, position);
    return e.table;
}

Result<void> DecodeLayer::record_ced(gpu::CommandBuffer& cmd, const LayerStep& st) {
    if (!st.run_compressor && !st.run_indexer) return {};
    const TextConfig& c = *cfg_;
    const uint32_t ratio = st.compress_ratio ? st.compress_ratio : 1u;

    auto step = [&](gpu::AttnStage s, const void* push, uint32_t bytes,
                    uint32_t groups) -> Result<void> {
        if (auto r = runner_->record(cmd, s, push, bytes, groups); !r) return r;
        return cmd.barrier();
    };

    const gpu::CmpPush cp{c.head_dim, c.hidden_size, ratio, st.position % ratio,
                          st.cmp_complete ? 1u : 0u, c.qk_rope_head_dim,
                          st.position / ratio, static_cast<float>(c.rms_norm_eps)};

    if (st.run_compressor) {
        const uint32_t gw = runner_->gemv_groups(gpu::AttnStage::CmpKvGemv, c.head_dim);
        if (auto r = step(gpu::AttnStage::CmpKvGemv, &cp, sizeof cp, gw); !r) return r;
        // No gate at ratio 1: `self.norm(self.wkv(x))` is a plain bf16 Linear
        // and there is nothing to pool (model.py, Compressor.forward).
        if (ratio > 1)
            if (auto r = step(gpu::AttnStage::CmpGateGemv, &cp, sizeof cp, gw); !r) return r;
        // Runs on EVERY step, complete or not: the state write is what carries
        // an incomplete group forward, and only the pooling is conditional.
        if (auto r = step(gpu::AttnStage::CmpNorm, &cp, sizeof cp, 1); !r) return r;
    }

    if (st.run_indexer) {
        const uint32_t irows = c.index_n_heads * c.index_head_dim;
        const float wscale = 1.0f / std::sqrt(static_cast<float>(c.index_head_dim)) /
                             std::sqrt(static_cast<float>(c.index_n_heads));
        gpu::IdxPush ip{irows, c.q_lora_rank, c.q_lora_rank / 32, c.index_n_heads,
                        c.index_head_dim, c.qk_rope_head_dim, 0, 0, c.sliding_window,
                        st.position / ratio, static_cast<float>(c.rms_norm_eps), wscale};

        // The key, and only when this layer's own group just completed: a
        // ratio-2 source at an incomplete position publishes nothing and the
        // Engine has already pointed `kv.idx_key` at whatever was published
        // last (docs/p2_attention.md §9.3 item 5).
        if (st.run_compressor && st.cmp_complete) {
            gpu::IdxPush ik = ip;
            ik.k = c.head_dim;
            if (auto r = step(gpu::AttnStage::IdxKey, &ik, sizeof ik, 1); !r) return r;
        }
        if (auto r = step(gpu::AttnStage::IdxQGemv, &ip, sizeof ip,
                          runner_->gemv_groups(gpu::AttnStage::IdxQGemv, irows)); !r)
            return r;
        if (auto r = step(gpu::AttnStage::IdxQFinish, &ip, sizeof ip, 1); !r) return r;
        gpu::IdxPush iw = ip;
        iw.rows = c.index_n_heads;
        iw.k    = c.hidden_size;
        if (auto r = step(gpu::AttnStage::IdxWeights, &iw, sizeof iw, 1); !r) return r;

        gpu::IdxPush is = ip;
        is.n_pos = st.n_cmp;
        is.topk  = std::min(c.index_topk, st.n_cmp);
        if (st.n_cmp > kMaxIndexPositions)
            return fail(Err::ResourceExhausted,
                        std::format("layer {} scores {} compressed positions; one indexer "
                                    "dispatch covers {}", st.layer, st.n_cmp,
                                    kMaxIndexPositions));
        const uint32_t sg = (st.n_cmp + gpu::kIdxScoreTile - 1) / gpu::kIdxScoreTile;
        if (sg) {
            if (auto r = step(gpu::AttnStage::IdxScore, &is, sizeof is, sg); !r) return r;

            // design §2.1. The source keeps the candidate_topk_blocks best
            // blocks of its own scores; a consumer masks its scores to them
            // before its own top-k. A consumer needs the SAME step's mask,
            // which the shared scratch holds from the source's dispatch on.
            gpu::IdxPush cb = is;
            cb.k    = c.candidate_block_size;
            cb.topk = c.candidate_topk_blocks;
            if (candidate_source(st)) {
                const uint32_t nb = (st.n_cmp + cb.k - 1) / cb.k;
                if (auto r = step(gpu::AttnStage::IdxBlockKeys, &cb, sizeof cb,
                                  (nb + 255) / 256); !r)
                    return r;
                if (auto r = step(gpu::AttnStage::IdxBlockSelect, &cb, sizeof cb, 1); !r)
                    return r;
                cand_position_ = st.position;
                cand_n_cmp_    = st.n_cmp;
            } else if (candidate_consumer(st)) {
                if (cand_position_ != st.position || cand_n_cmp_ != st.n_cmp)
                    return fail(Err::FailedPrecondition,
                                std::format("layer {} needs design 2.1's candidate blocks for "
                                            "position {} over {} positions, but layer {} last "
                                            "built them for position {} over {}", st.layer,
                                            st.position, st.n_cmp,
                                            c.candidate_source_layer_id, cand_position_,
                                            cand_n_cmp_));
                if (auto r = step(gpu::AttnStage::IdxApplyCand, &cb, sizeof cb, 1); !r)
                    return r;
            }

            // Poison the compressed half so `verify_after_attention` can tell a
            // list the kernel wrote from one it did not.
            if (st.kv.top_idx_host) {
                auto* out = reinterpret_cast<int32_t*>(st.kv.top_idx_host);
                std::fill(out + c.sliding_window, out + c.sliding_window + is.topk, -1);
            }
            if (auto r = step(gpu::AttnStage::IdxTopK, &is, sizeof is, 1); !r) return r;
        }
    }

    // The cache write last, because it rotates and quantises the latent the
    // indexer had to see unrotated.
    if (st.run_compressor && st.cmp_complete)
        if (auto r = step(gpu::AttnStage::CmpStore, &cp, sizeof cp, 1); !r) return r;
    return {};
}

Result<void> DecodeLayer::submit(gpu::CommandBuffer& cmd) {
    if (auto r = cmd.end(); !r) return r;
    return gpu::submit_and_wait(*device_, cmd);
}

Result<void> DecodeLayer::run_attention(const LayerStep& st) {
    if (auto r = cmd_.begin(); !r) return r;
    if (auto r = record_attention(cmd_, st); !r) return r;
    if (auto r = submit(cmd_); !r) return r;
    return verify_after_attention(st);
}

bool DecodeLayer::candidate_source(const LayerStep& st) const {
    const TextConfig& c = *cfg_;
    return st.run_indexer && c.candidate_topk_blocks && c.candidate_block_size &&
           st.layer == c.candidate_source_layer_id &&
           uint64_t(st.n_cmp) > uint64_t(c.candidate_topk_blocks) * c.candidate_block_size;
}

bool DecodeLayer::candidate_consumer(const LayerStep& st) const {
    const TextConfig& c = *cfg_;
    return st.run_indexer && c.candidate_topk_blocks && c.candidate_block_size &&
           st.layer > c.candidate_source_layer_id &&
           uint64_t(st.n_cmp) > uint64_t(c.candidate_topk_blocks) * c.candidate_block_size;
}

Result<void> DecodeLayer::verify_after_attention(const LayerStep& st) const {
    if (!st.run_indexer || st.n_cmp == 0 || !st.kv.top_idx_host) return {};
    const TextConfig& c = *cfg_;
    const uint32_t w = c.sliding_window;
    const uint32_t n_sel = std::min(c.index_topk, st.n_cmp);
    const auto* idx = reinterpret_cast<const int32_t*>(st.kv.top_idx_host);
    int32_t prev = int32_t(w) - 1;
    for (uint32_t i = 0; i < n_sel; ++i) {
        const int32_t v = idx[w + i];
        if (v <= prev || v >= int32_t(w + st.n_cmp))
            return fail(Err::Internal,
                        std::format("layer {} position {}: top-k entry {} of {} is {} (previous "
                                    "{}); the indexer must write {} strictly increasing "
                                    "compressed rows in [{}, {}){}", st.layer, st.position,
                                    i, n_sel, v, prev, n_sel, w, w + st.n_cmp,
                                    v == -1 ? " -- never written this step" : ""));
        prev = v;
    }
    return {};
}

Result<void> DecodeLayer::run_moe(MoeBridge& moe, const LayerStep& st, gpu::Timeline* timeline,
                                  const std::function<Result<void>(const uint32_t*, uint32_t)>& on_ready) {
    const TextConfig& c = *cfg_;
    const uint32_t topk = c.num_experts_per_tok;
    const auto* ids = static_cast<const uint32_t*>(buf_.gate_ids.host);
    const auto* wts = static_cast<const float*>(buf_.gate_weights.host);

    // design §7.1 / §7.8: the CPU reads the ids out of host-coherent memory,
    // makes the six experts resident, then host-signals the timeline value the
    // command buffer's MoE wait names. With everything pinned this is a
    // formality -- but it is the formality P3 turns into the NVMe fetch, so it
    // is here rather than bolted on later.
    if (on_ready)
        if (auto r = on_ready(ids, topk); !r) return r;
    if (timeline) {
        const TimelineValue v = gpu::timeline_value(st.position, st.layer);
        auto cur = timeline->value();
        if (cur && *cur < v)
            if (auto r = timeline->signal(v); !r) return r;
    }

    MoeCall call;
    call.layer   = st.layer;
    call.ids     = ids;
    call.weights = wts;
    call.topk    = topk;
    call.x       = ffn_norm_out();
    call.y       = static_cast<float*>(buf_.moe_y.host);
    call.hidden  = c.hidden_size;
    return moe.run(call);
}

Result<void> DecodeLayer::record_close(gpu::CommandBuffer& cmd, const LayerStep& st) {
    const TextConfig& c = *cfg_;
    const uint32_t n_wg0 = (c.hidden_size + 255) / 256;
    // Only hc_post is wanted here: the mixes this would compute belong to the
    // next layer's attention half, which recomputes them from the stream it
    // reads. Stage 1 is not dispatched and the Sinkhorn is skipped, so MixRaw
    // is never read. `bind` has already pointed MhcClose at the right buffers.
    gpu::MhcPush mp{c.hidden_size, c.hc_mult, (2 + c.hc_mult) * c.hc_mult, n_wg0,
                    c.hc_sinkhorn_iters, gpu::kMhcFlagPost | gpu::kMhcFlagSkipSinkhorn,
                    static_cast<float>(c.rms_norm_eps), static_cast<float>(c.hc_eps)};
    (void)st;
    if (auto r = runner_->record(cmd, gpu::AttnStage::MhcClose, &mp, sizeof mp, n_wg0); !r)
        return r;
    return cmd.barrier();
}

Result<void> DecodeLayer::run_close(const LayerStep& st) {
    if (auto r = cmd_.begin(); !r) return r;
    if (auto r = record_close(cmd_, st); !r) return r;
    return submit(cmd_);
}

const float*    DecodeLayer::block_out()    const { return static_cast<const float*>(buf_.xout.host); }
const float*    DecodeLayer::gate_scores()  const { return static_cast<const float*>(buf_.gate_scores.host); }
const uint32_t* DecodeLayer::gate_ids()     const { return static_cast<const uint32_t*>(buf_.gate_ids.host); }
const float*    DecodeLayer::gate_weights() const { return static_cast<const float*>(buf_.gate_weights.host); }
const float*    DecodeLayer::ffn_norm_out() const {
    return ffn_in_host_ ? ffn_in_host_ : static_cast<const float*>(buf_.u.host);
}

// ===========================================================================
// Track T: the verify batch (docs/p4_mgt1.md)
// ===========================================================================

uint32_t BatchStep::n_win_ext() const { return 128u + m - 1u; }

Result<void> BatchScratch::create(gpu::GpuScratch& s, const TextConfig& cfg, uint32_t mcap,
                                  uint32_t max_compressed) {
    if (mcap < 1 || mcap > gpu::kMgtMaxM)
        return fail(Err::InvalidArgument, std::format("batch size {} outside 1..{}", mcap,
                                                      gpu::kMgtMaxM));
    m_cap = mcap;
    const uint64_t M = mcap;
    const uint64_t dim = cfg.hidden_size, hc = cfg.hc_mult;
    const uint64_t qrows = uint64_t(cfg.num_attention_heads) * cfg.head_dim;
    const uint64_t orows = uint64_t(cfg.o_groups) * cfg.o_lora_rank;
    const uint64_t irows = uint64_t(cfg.index_n_heads) * cfg.index_head_dim;
    list_stride  = cfg.sliding_window + gpu::kMgtMaxM - 1 + cfg.index_topk;
    score_stride = std::max<uint32_t>(max_compressed, 1);
    const uint32_t bs = std::max<uint32_t>(1, cfg.candidate_block_size);
    blk_stride   = (score_stride + bs - 1) / bs;
    n_lists = 1;
    for (int64_t L = 0; L < int64_t(cfg.num_hidden_layers); ++L)
        if (cfg.is_index_source(static_cast<uint32_t>(L))) ++n_lists;
    // The shared split partial plane: the widest (KSplit x rows) of the six
    // GEMVs at the largest factors MgtRunner accepts for them.
    const uint64_t part_floats = std::max<uint64_t>({2 * qrows, 8 * dim, 4 * orows, 8 * 1280ull,
                                                     8 * 512ull, 2 * irows}) * M;
    struct Want { gpu::GpuScratch::View* v; uint64_t bytes; };
    const Want wants[] = {
        {&x, M * hc * dim * 4}, {&xout, M * hc * dim * 4}, {&utmp, M * dim * 4}, {&u, M * dim * 4},
        {&partials, M * 512 * 4}, {&mix_raw, M * 32 * 4}, {&mix_a, M * 32 * 4}, {&mix_b, M * 32 * 4},
        {&qr_raw, M * cfg.q_lora_rank * 4}, {&qr, M * cfg.q_lora_rank * 4}, {&q, M * qrows * 2},
        {&rope, M * 256}, {&rope_lat, M * 256}, {&kv_out, M * cfg.head_dim * 4},
        {&ovf_val, M * cfg.head_dim}, {&ovf_scale, M * (cfg.head_dim / 32)},
        {&score, M * cfg.num_attention_heads * uint64_t(kAttnScoreStride) * 4},
        {&tile_max, M * cfg.num_attention_heads * 64ull * 4},
        {&o, M * qrows * 4}, {&woa, M * orows * 4}, {&wob, M * dim * 4}, {&part, part_floats * 4},
        {&gate_scores, M * cfg.n_routed_experts * 4}, {&gate_ids, M * 16 * 4},
        {&gate_weights, M * 16 * 4}, {&layer_done, 64}, {&moe_y, M * dim * 4},
        {&cmp_y, M * cfg.head_dim * 4}, {&cmp_g, M * cfg.head_dim * 4},
        {&latent, M * cfg.head_dim * 4}, {&latent_q, M * cfg.head_dim * 4},
        {&cmp_fp4, M * cfg.head_dim / 2}, {&cmp_scale, M * cfg.head_dim / 16},
        {&idx_q_raw, M * irows * 4}, {&idx_q, M * irows * 2}, {&idx_q_fp4, M * irows / 2},
        {&idx_q_scale, M * irows / 32}, {&idx_k_raw, M * cfg.index_head_dim * 4},
        {&idx_k_fp4, M * cfg.index_head_dim / 2}, {&idx_k_scale, M * 16},
        {&idx_w, M * cfg.index_n_heads * 4}, {&idx_score, M * uint64_t(score_stride) * 4},
        {&idx_blk_key, M * uint64_t(blk_stride) * 4}, {&idx_cand, M * uint64_t(blk_stride) * 4},
        {&lists, uint64_t(n_lists) * M * list_stride * 4},
    };
    for (const Want& w : wants) {
        auto v = s.alloc(w.bytes);
        if (!v) return std::unexpected(v.error());
        *w.v = *v;
    }
    return {};
}

void write_batch_window_lists(BatchScratch& b, uint32_t window, uint32_t p0, uint32_t m) {
    const uint32_t q = p0 + m - 1;            // the newest position written
    for (uint32_t li = 0; li < b.n_lists; ++li) {
        int32_t* base = b.list_host(li);
        for (uint32_t mm = 0; mm < m; ++mm) {
            int32_t* row = base + uint64_t(mm) * b.list_stride;
            const uint32_t P = p0 + mm;
            const uint32_t oldest = P % window + 1;
            for (uint32_t i = 0; i < window; ++i) {
                const uint32_t slot = (oldest + i) % window;
                // the position slot `slot` holds once all m tokens are written
                const int64_t held = int64_t(q) - int64_t((q + window - slot) % window);
                row[i] = (held >= 0 && held <= int64_t(P)) ? int32_t(slot) : -1;
            }
            for (uint32_t j = 1; j < m; ++j)
                row[window + j - 1] = (j > mm && p0 + j >= window) ? int32_t(window + j - 1) : -1;
            std::fill(row + window + m - 1, row + b.list_stride, -1);
        }
    }
}

Result<void> DecodeLayer::create_batch(gpu::MgtRunner& mgt, gpu::GpuScratch& scratch,
                                       uint32_t m_cap, uint32_t max_compressed) {
    if (!cfg_) return fail(Err::FailedPrecondition, "create the decode layer first");
    mgt_ = &mgt;
    return bb_.create(scratch, *cfg_, m_cap, max_compressed);
}

namespace {

void bind_mhc_batch(gpu::MgtRunner& r, gpu::MgtStage post, gpu::MgtStage mix, gpu::MgtStage fin,
                    const BatchScratch& b, uint64_t hc_fn, uint64_t hc_base, uint64_t hc_scale,
                    uint64_t norm_w, uint64_t mix_in, uint64_t mix_out, uint64_t a_in,
                    uint64_t x_in, uint64_t x_out, uint64_t u) {
    uint64_t* s = r.slots(post);
    s[gpu::slot::kX]       = x_in;
    s[gpu::slot::kA]       = a_in;
    s[gpu::slot::kPreMix]  = mix_in;
    s[gpu::slot::kPostIn]  = mix_in + 4 * sizeof(float);
    s[gpu::slot::kCombIn]  = mix_in + 8 * sizeof(float);
    s[gpu::slot::kHcFn]    = hc_fn;
    s[gpu::slot::kHcBase]  = hc_base;
    s[gpu::slot::kHcScale] = hc_scale;
    s[gpu::slot::kNormW]   = norm_w;
    s[gpu::slot::kXout]    = x_out;
    s[gpu::slot::kUtmp]    = b.utmp.addr;
    s[gpu::slot::kScratch] = b.partials.addr;
    s[gpu::slot::kMixRaw]  = b.mix_raw.addr;
    s[gpu::slot::kMixOut]  = mix_out;
    s[gpu::slot::kU]       = u;
    if (mix != post) std::memcpy(r.slots(mix), s, gpu::kAttnStageStride);
    if (fin != post) std::memcpy(r.slots(fin), s, gpu::kAttnStageStride);
}

void bind_gemv(gpu::MgtRunner& r, gpu::MgtStage split, gpu::MgtStage tail, uint64_t w,
               uint64_t sc, uint64_t x, uint64_t y, const BatchScratch& b) {
    uint64_t* s = r.slots(split);
    s[gpu::mslot::kGW] = w;
    s[gpu::mslot::kGS] = sc;
    s[gpu::mslot::kGX] = x;
    s[gpu::mslot::kGY] = y;
    s[gpu::mslot::kGP] = b.part.addr;
    std::memcpy(r.slots(tail), s, gpu::kAttnStageStride);
}

}  // namespace

Result<void> DecodeLayer::bind_batch(const LayerWeights& w, const BatchStep& st) {
    if (!mgt_) return fail(Err::FailedPrecondition, "create_batch was not called");
    const TextConfig& c = *cfg_;
    BatchScratch& b = bb_;
    if (st.m < 1 || st.m > b.m_cap)
        return fail(Err::InvalidArgument, std::format("batch of {} on a scratch for {}", st.m, b.m_cap));
    const uint32_t hd = c.qk_rope_head_dim / 2;

    const RopeConfig rc = rope_for_layer(st.compress_ratio, c.qk_rope_head_dim, c.rope_theta,
                                         c.compress_rope_theta,
                                         c.rope_scaling.original_max_position,
                                         c.rope_scaling.factor);
    for (uint32_t m = 0; m < st.m; ++m) {
        const std::vector<float>& tab = rope_cached(rc, st.p0 + m);
        std::memcpy(static_cast<std::byte*>(b.rope.host) + uint64_t(m) * hd * 8, tab.data(),
                    tab.size() * sizeof(float));
    }
    if ((st.run_compressor || st.run_indexer) && st.compress_ratio) {
        const uint32_t r = st.compress_ratio;
        const uint32_t first = (r - 1 - st.p0 % r) % r;
        uint32_t c_i = 0;
        for (uint32_t mc = first; mc < st.m; mc += r, ++c_i) {
            const std::vector<float>& tab = rope_cached(rc, st.p0 + mc + 1 - r);
            std::memcpy(static_cast<std::byte*>(b.rope_lat.host) + uint64_t(c_i) * hd * 8,
                        tab.data(), tab.size() * sizeof(float));
        }
    }

    using S = gpu::MgtStage;
    gpu::MgtRunner& R = *mgt_;
    bind_mhc_batch(R, S::MhcPost, S::MhcMix, S::MhcFinal, b, w.hc_attn_fn, w.hc_attn_base,
                   w.hc_attn_scale, w.attn_norm, b.mix_a.addr, b.mix_b.addr, b.moe_y.addr,
                   b.x.addr, b.xout.addr, b.u.addr);
    bind_mhc_batch(R, S::MhcPostB, S::MhcMixB, S::MhcFinalB, b, w.hc_ffn_fn, w.hc_ffn_base,
                   w.hc_ffn_scale, w.ffn_norm, b.mix_b.addr, b.mix_a.addr, b.wob.addr,
                   b.xout.addr, b.x.addr, b.u.addr);
    if (st.bind_close)
        bind_mhc_batch(R, S::MhcClose, S::MhcClose, S::MhcClose, b, w.hc_ffn_fn, w.hc_ffn_base,
                       w.hc_ffn_scale, w.ffn_norm, b.mix_a.addr, b.mix_b.addr, b.moe_y.addr,
                       b.x.addr, b.xout.addr, b.utmp.addr);

    bind_gemv(R, S::WqASplit, S::WqACombine, w.wq_a, w.wq_a_scale, b.u.addr, b.qr_raw.addr, b);
    {
        uint64_t* n = R.slots(S::QNorm);
        n[gpu::mslot::kGX] = b.qr_raw.addr;
        n[gpu::mslot::kGY] = b.qr.addr;
        n[gpu::mslot::kGNormW] = w.q_norm;
    }
    bind_gemv(R, S::WqBSplit, S::WqBFinish, w.wq_b, w.wq_b_scale, b.qr.addr, b.q.addr, b);
    R.slots(S::WqBFinish)[gpu::mslot::kGRope] = b.rope.addr;
    bind_gemv(R, S::WkvSplit, S::WkvFinish, w.wkv, w.wkv_scale, b.u.addr, 0, b);
    {
        uint64_t* k = R.slots(S::WkvFinish);
        k[gpu::mslot::kGNormW]     = w.kv_norm;
        k[gpu::mslot::kGRope]      = b.rope.addr;
        k[gpu::mslot::kGRingVal]   = st.kv.win_val;
        k[gpu::mslot::kGRingScale] = st.kv.win_scale;
        k[gpu::mslot::kGOvfVal]    = b.ovf_val.addr;
        k[gpu::mslot::kGOvfScale]  = b.ovf_scale.addr;
        k[gpu::mslot::kGKvOut]     = b.kv_out.addr;
    }
    {
        uint64_t* a = R.slots(S::AttnScore);
        a[gpu::mslot::kAQ]        = b.q.addr;
        a[gpu::mslot::kAWinVal]   = st.kv.win_val;
        a[gpu::mslot::kAWinScale] = st.kv.win_scale;
        a[gpu::mslot::kACmpKv]    = st.kv.cmp_kv;
        a[gpu::mslot::kATopIdx]   = b.list_addr(st.list);
        a[gpu::mslot::kASink]     = w.attn_sink;
        a[gpu::mslot::kARope]     = b.rope.addr;
        a[gpu::mslot::kAScore]    = b.score.addr;
        a[gpu::mslot::kAO]        = b.o.addr;
        a[gpu::mslot::kATileMax]  = b.tile_max.addr;
        a[gpu::mslot::kAOvfVal]   = b.ovf_val.addr;
        a[gpu::mslot::kAOvfScale] = b.ovf_scale.addr;
        std::memcpy(R.slots(S::AttnPv), a, gpu::kAttnStageStride);
    }
    bind_gemv(R, S::WoASplit, S::WoACombine, w.wo_a, w.wo_a_scale, b.o.addr, b.woa.addr, b);
    bind_gemv(R, S::WoBSplit, S::WoBCombine, w.wo_b, w.wo_b_scale, b.woa.addr, b.wob.addr, b);
    {
        uint64_t* g = R.slots(S::GateScore);
        g[gpu::slot::kGateW]         = w.gate_w;
        g[gpu::slot::kGateBias]      = w.gate_bias;
        g[gpu::slot::kGateX]         = b.u.addr;
        g[gpu::slot::kGateScores]    = b.gate_scores.addr;
        g[gpu::slot::kGateIds]       = b.gate_ids.addr;
        g[gpu::slot::kGateWeights]   = b.gate_weights.addr;
        g[gpu::slot::kGateLayerDone] = b.layer_done.addr;
        std::memcpy(R.slots(S::GateTopK), g, gpu::kAttnStageStride);
    }

    if (st.run_compressor) {
        if (!w.cmp_wkv || !w.cmp_norm || (st.compress_ratio > 1 && !w.cmp_wgate))
            return fail(Err::FailedPrecondition,
                        std::format("layer {}: compressor weights are not pinned", st.layer));
        auto cmp = [&](S s, uint64_t weight, uint64_t y) {
            uint64_t* p = R.slots(s);
            p[gpu::slot::kCmpW]          = weight;
            p[gpu::slot::kCmpX]          = b.u.addr;
            p[gpu::slot::kCmpY]          = y;
            p[gpu::slot::kCmpG]          = b.cmp_g.addr;
            p[gpu::slot::kCmpKvState]    = st.kv.cmp_state_kv;
            p[gpu::slot::kCmpScoreState] = st.kv.cmp_state_score;
            p[gpu::slot::kCmpNormW]      = w.cmp_norm;
            p[gpu::slot::kCmpLatent]     = b.latent.addr;
            p[gpu::slot::kCmpRope]       = b.rope_lat.addr;
            p[gpu::slot::kCmpVal]        = st.kv.cmp_kv;
            p[gpu::slot::kCmpFp4]        = b.cmp_fp4.addr;
            p[gpu::slot::kCmpScaleB]     = b.cmp_scale.addr;
            p[gpu::slot::kCmpLatentQ]    = b.latent_q.addr;
        };
        cmp(S::CmpKvGemv, w.cmp_wkv, b.cmp_y.addr);
        cmp(S::CmpGateGemv, w.cmp_wgate, b.cmp_g.addr);
        cmp(S::CmpPool, w.cmp_wkv, b.cmp_y.addr);
        cmp(S::CmpState, w.cmp_wkv, b.cmp_y.addr);
        cmp(S::CmpStore, w.cmp_wkv, b.cmp_y.addr);
    }
    if (st.run_indexer) {
        if (!w.idx_wq_b || !w.idx_wproj)
            return fail(Err::FailedPrecondition,
                        std::format("layer {}: indexer weights are not pinned", st.layer));
        bind_gemv(R, S::IdxQSplit, S::IdxQCombine, w.idx_wq_b, w.idx_wq_b_scale, b.qr.addr,
                  b.idx_q_raw.addr, b);
        auto idx = [&](S s, uint64_t rope) {
            uint64_t* p = R.slots(s);
            p[gpu::mslot::kIQRaw]      = b.idx_q_raw.addr;
            p[gpu::mslot::kIQ]         = b.idx_q.addr;
            p[gpu::mslot::kIQFp4]      = b.idx_q_fp4.addr;
            p[gpu::mslot::kIQScale]    = b.idx_q_scale.addr;
            p[gpu::mslot::kIRope]      = rope;
            p[gpu::mslot::kIWk]        = w.idx_wk;
            p[gpu::mslot::kIKNormW]    = w.idx_k_norm;
            p[gpu::mslot::kILatent]    = b.latent.addr;
            p[gpu::mslot::kIKRaw]      = b.idx_k_raw.addr;
            p[gpu::mslot::kIKCache]    = st.idx_key_own;
            p[gpu::mslot::kIKFp4]      = b.idx_k_fp4.addr;
            p[gpu::mslot::kIKScale]    = b.idx_k_scale.addr;
            p[gpu::mslot::kIWProjW]    = w.idx_wproj;
            p[gpu::mslot::kIX]         = b.u.addr;
            p[gpu::mslot::kIWeights]   = b.idx_w.addr;
            p[gpu::mslot::kIScore]     = b.idx_score.addr;
            p[gpu::mslot::kIOut]       = b.list_addr(st.list);
            p[gpu::mslot::kIKCachePub] = st.idx_key_pub;
            p[gpu::mslot::kIBlkKey]    = b.idx_blk_key.addr;
            p[gpu::mslot::kICand]      = b.idx_cand.addr;
        };
        idx(S::IdxQFinish, b.rope.addr);
        idx(S::IdxKey, b.rope_lat.addr);
        idx(S::IdxWeights, b.rope.addr);
        idx(S::IdxScore, b.rope.addr);
        idx(S::IdxTopK, b.rope.addr);
        idx(S::IdxBlockKeys, b.rope.addr);
        idx(S::IdxBlockSelect, b.rope.addr);
        idx(S::IdxApplyCand, b.rope.addr);
    }
    weights_ = w;
    return {};
}

Result<void> DecodeLayer::record_attention_batch(gpu::CommandBuffer& cmd, const BatchStep& st) {
    if (!mgt_) return fail(Err::FailedPrecondition, "create_batch was not called");
    const TextConfig& c = *cfg_;
    const BatchScratch& b = bb_;
    const uint32_t M = st.m;
    const uint32_t dim = c.hidden_size;
    const uint32_t qrows = c.num_attention_heads * c.head_dim;
    const uint32_t orows = c.o_groups * c.o_lora_rank;
    const uint32_t n_wg0 = (dim + 255) / 256;
    const uint32_t win_ext = c.sliding_window + M - 1;
    using S = gpu::MgtStage;
    gpu::MgtRunner& R = *mgt_;

    // The list every query walks: the window plus min(index_topk, n_cmp) of the
    // LAST query's compressed positions (earlier queries have -1 past their own).
    const uint32_t n_sel = st.compress_ratio ? std::min(c.index_topk, st.n_cmp(M - 1)) : 0u;
    const uint32_t n_kv = win_ext + n_sel;
    if (n_kv > b.list_stride || n_kv > kAttnScoreStride)
        return fail(Err::Internal, std::format("layer {} p0 {} M {}: a list of {} entries; the "
                                               "batch lists hold {} and the score row {}",
                                               st.layer, st.p0, M, n_kv, b.list_stride,
                                               kAttnScoreStride));
    if (st.compress_ratio && st.n_cmp(M - 1) > b.score_stride)
        return fail(Err::ResourceExhausted,
                    std::format("layer {}: {} compressed positions, the batch score plane holds {}",
                                st.layer, st.n_cmp(M - 1), b.score_stride));
    if (auto r = R.ensure(M); !r) return r;

    auto step = [&](S s, const void* push, uint32_t bytes, uint32_t gx,
                    uint32_t gy) -> Result<void> {
        if (auto r = R.record(cmd, M, s, push, bytes, gx, gy); !r) return r;
        return cmd.barrier();
    };

    gpu::MhcPush mp{dim, c.hc_mult, (2 + c.hc_mult) * c.hc_mult, n_wg0, c.hc_sinkhorn_iters,
                    st.apply_hc_post ? gpu::kMhcFlagPost : 0u,
                    static_cast<float>(c.rms_norm_eps), static_cast<float>(c.hc_eps)};
    if (auto r = step(S::MhcPost, &mp, sizeof mp, n_wg0, M); !r) return r;
    if (auto r = step(S::MhcMix, &mp, sizeof mp, mp.mix_rows, M); !r) return r;
    if (auto r = step(S::MhcFinal, &mp, sizeof mp, n_wg0, M); !r) return r;

    // Q path.
    gpu::MgtGemvPush qa{};
    qa.rows = c.q_lora_rank; qa.k = dim; qa.scale_cols = dim / 32;
    qa.part_stride = c.q_lora_rank; qa.x_stride = dim; qa.y_stride = c.q_lora_rank;
    if (auto r = step(S::WqASplit, &qa, sizeof qa, R.split_groups(S::WqASplit, qa.rows), 1); !r)
        return r;
    if (auto r = step(S::WqACombine, &qa, sizeof qa, R.combine_groups(qa.rows), M); !r) return r;
    gpu::MgtGemvPush qn{};
    qn.head_dim = c.q_lora_rank; qn.x_stride = c.q_lora_rank; qn.y_stride = c.q_lora_rank;
    qn.eps = static_cast<float>(c.rms_norm_eps);
    if (auto r = step(S::QNorm, &qn, sizeof qn, 1, M); !r) return r;
    gpu::MgtGemvPush qb{};
    qb.rows = qrows; qb.k = c.q_lora_rank; qb.scale_cols = c.q_lora_rank / 32;
    qb.part_stride = qrows; qb.x_stride = c.q_lora_rank; qb.y_stride = qrows;
    qb.head_dim = c.head_dim; qb.rope_dim = c.qk_rope_head_dim;
    if (auto r = step(S::WqBSplit, &qb, sizeof qb, R.split_groups(S::WqBSplit, qrows), 1); !r)
        return r;
    if (auto r = step(S::WqBFinish, &qb, sizeof qb, R.combine_groups(qrows), M); !r) return r;

    // KV path: the overflow copy and the ring write for every token.
    gpu::MgtGemvPush kp{};
    kp.rows = c.head_dim; kp.k = dim; kp.scale_cols = dim / 32; kp.part_stride = c.head_dim;
    kp.x_stride = dim; kp.head_dim = c.head_dim; kp.rope_dim = c.qk_rope_head_dim;
    kp.p0 = st.p0; kp.window = c.sliding_window; kp.eps = static_cast<float>(c.rms_norm_eps);
    if (auto r = step(S::WkvSplit, &kp, sizeof kp, R.split_groups(S::WkvSplit, kp.rows), 1); !r)
        return r;
    if (auto r = step(S::WkvFinish, &kp, sizeof kp, 1, M); !r) return r;

    if (auto r = record_ced_batch(cmd, st); !r) return r;

    // Sparse attention, per query.
    gpu::MgtAttnPush ap{};
    ap.n_kv = n_kv; ap.n_win = c.sliding_window; ap.n_ovf = M - 1; ap.head_dim = c.head_dim;
    ap.rope_dim = c.qk_rope_head_dim; ap.score_stride = kAttnScoreStride;
    ap.softmax_scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
    ap.n_heads = c.num_attention_heads; ap.n_tiles = 32;
    ap.tile_len = (n_kv + ap.n_tiles - 1) / ap.n_tiles;
    ap.list_stride = b.list_stride;
    const uint32_t hg = (c.num_attention_heads + R.spec().tile_heads_per_wg - 1) /
                        R.spec().tile_heads_per_wg;
    if (auto r = step(S::AttnScore, &ap, sizeof ap, hg * ap.n_tiles, M); !r) return r;
    if (auto r = step(S::AttnPv, &ap, sizeof ap, c.num_attention_heads, M); !r) return r;

    // Output projection.
    gpu::MgtGemvPush wa{};
    wa.rows = orows; wa.k = qrows / c.o_groups; wa.scale_cols = wa.k / 32;
    wa.rows_per_group = c.o_lora_rank; wa.part_stride = orows; wa.x_stride = qrows;
    wa.y_stride = orows;
    if (auto r = step(S::WoASplit, &wa, sizeof wa, R.split_groups(S::WoASplit, orows), 1); !r)
        return r;
    if (auto r = step(S::WoACombine, &wa, sizeof wa, R.combine_groups(orows), M); !r) return r;
    gpu::MgtGemvPush wb{};
    wb.rows = dim; wb.k = orows; wb.scale_cols = orows / 32; wb.part_stride = dim;
    wb.x_stride = orows; wb.y_stride = dim;
    if (auto r = step(S::WoBSplit, &wb, sizeof wb, R.split_groups(S::WoBSplit, dim), 1); !r)
        return r;
    if (auto r = step(S::WoBCombine, &wb, sizeof wb, R.combine_groups(dim), M); !r) return r;

    gpu::MhcPush fp = mp;
    fp.flags = gpu::kMhcFlagPost;
    if (auto r = step(S::MhcPostB, &fp, sizeof fp, n_wg0, M); !r) return r;
    if (auto r = step(S::MhcMixB, &fp, sizeof fp, mp.mix_rows, M); !r) return r;
    if (auto r = step(S::MhcFinalB, &fp, sizeof fp, n_wg0, M); !r) return r;

    gpu::GatePush gp{c.n_routed_experts, dim, c.num_experts_per_tok, 16, 1.0f,
                     static_cast<float>(c.routed_scaling_factor)};
    if (auto r = step(S::GateScore, &gp, sizeof gp, R.row_groups(c.n_routed_experts), M); !r)
        return r;
    return step(S::GateTopK, &gp, sizeof gp, 1, M);
}

Result<void> DecodeLayer::record_ced_batch(gpu::CommandBuffer& cmd, const BatchStep& st) {
    if (!st.run_compressor && !st.run_indexer) return {};
    const TextConfig& c = *cfg_;
    const BatchScratch& b = bb_;
    const uint32_t M = st.m;
    const uint32_t ratio = st.compress_ratio ? st.compress_ratio : 1u;
    const uint32_t first = (ratio - 1 - st.p0 % ratio) % ratio;
    const uint32_t n_complete = first < M ? (M - 1 - first) / ratio + 1 : 0u;
    using S = gpu::MgtStage;
    gpu::MgtRunner& R = *mgt_;
    auto step = [&](S s, const void* push, uint32_t bytes, uint32_t gx,
                    uint32_t gy) -> Result<void> {
        if (auto r = R.record(cmd, M, s, push, bytes, gx, gy); !r) return r;
        return cmd.barrier();
    };

    gpu::MgtCmpPush cp{};
    cp.rows = c.head_dim; cp.k = c.hidden_size; cp.ratio = ratio; cp.p0 = st.p0;
    cp.rope_dim = c.qk_rope_head_dim; cp.norm_eps = static_cast<float>(c.rms_norm_eps);
    if (st.run_compressor) {
        const uint32_t gw = R.row_groups(c.head_dim);
        if (auto r = step(S::CmpKvGemv, &cp, sizeof cp, gw, M); !r) return r;
        if (ratio > 1)
            if (auto r = step(S::CmpGateGemv, &cp, sizeof cp, gw, M); !r) return r;
        if (n_complete)
            if (auto r = step(S::CmpPool, &cp, sizeof cp, 1, n_complete); !r) return r;
        if (ratio > 1)
            if (auto r = step(S::CmpState, &cp, sizeof cp, 1, 1); !r) return r;
    }

    if (st.run_indexer) {
        const uint32_t irows = c.index_n_heads * c.index_head_dim;
        gpu::MgtIdxPush ip{};
        ip.n_heads = c.index_n_heads; ip.head_dim = c.index_head_dim;
        ip.rope_dim = c.qk_rope_head_dim; ip.p0 = st.p0; ip.ratio = ratio;
        ip.topk = c.index_topk; ip.offset = c.sliding_window + M - 1;
        ip.score_stride = b.score_stride; ip.list_stride = b.list_stride;
        ip.key_sel = st.key_sel; ip.blk_stride = b.blk_stride;
        ip.norm_eps = static_cast<float>(c.rms_norm_eps);
        ip.wscale = 1.0f / std::sqrt(static_cast<float>(c.index_head_dim)) /
                    std::sqrt(static_cast<float>(c.index_n_heads));

        if (st.run_compressor && n_complete) {
            gpu::MgtIdxPush ik = ip;
            ik.k = c.head_dim;
            if (auto r = step(S::IdxKey, &ik, sizeof ik, 1, n_complete); !r) return r;
        }
        gpu::MgtGemvPush gq{};
        gq.rows = irows; gq.k = c.q_lora_rank; gq.scale_cols = c.q_lora_rank / 32;
        gq.part_stride = irows; gq.x_stride = c.q_lora_rank; gq.y_stride = irows;
        if (auto r = step(S::IdxQSplit, &gq, sizeof gq, R.split_groups(S::IdxQSplit, irows), 1); !r)
            return r;
        if (auto r = step(S::IdxQCombine, &gq, sizeof gq, R.combine_groups(irows), M); !r) return r;
        if (auto r = step(S::IdxQFinish, &ip, sizeof ip, 1, M); !r) return r;
        gpu::MgtIdxPush iw = ip;
        iw.k = c.hidden_size;
        if (auto r = step(S::IdxWeights, &iw, sizeof iw, 1, M); !r) return r;

        const uint32_t n_max = st.n_cmp(M - 1);
        if (n_max > kMaxIndexPositions)
            return fail(Err::ResourceExhausted,
                        std::format("layer {} scores {} compressed positions; one indexer "
                                    "dispatch covers {}", st.layer, n_max, kMaxIndexPositions));
        const uint32_t sg = (n_max + gpu::kIdxScoreTile - 1) / gpu::kIdxScoreTile;
        if (sg) {
            if (auto r = step(S::IdxScore, &ip, sizeof ip, sg, M); !r) return r;
            gpu::MgtIdxPush cb = ip;
            cb.k = c.candidate_block_size;
            cb.topk = c.candidate_topk_blocks;
            const bool cand = c.candidate_topk_blocks && c.candidate_block_size &&
                              uint64_t(n_max) > uint64_t(c.candidate_topk_blocks) *
                                                    c.candidate_block_size;
            if (cand && st.layer == c.candidate_source_layer_id) {
                const uint32_t nb = (n_max + cb.k - 1) / cb.k;
                if (nb > b.blk_stride)
                    return fail(Err::ResourceExhausted, "the batch block planes are too small");
                if (auto r = step(S::IdxBlockKeys, &cb, sizeof cb, (nb + 255) / 256, M); !r)
                    return r;
                if (auto r = step(S::IdxBlockSelect, &cb, sizeof cb, 1, M); !r) return r;
                cand_position_ = st.p0;
                cand_n_cmp_    = n_max;
            } else if (cand && st.layer > c.candidate_source_layer_id) {
                if (cand_position_ != st.p0 || cand_n_cmp_ != n_max)
                    return fail(Err::FailedPrecondition,
                                std::format("layer {} needs the batch's candidate blocks for p0 {} "
                                            "over {} positions; they were built for p0 {} over {}",
                                            st.layer, st.p0, n_max, cand_position_, cand_n_cmp_));
                if (auto r = step(S::IdxApplyCand, &cb, sizeof cb, 1, M); !r) return r;
            }
            // Poison every query's compressed half, so the verify can tell the
            // kernel's entries from stale ones.
            for (uint32_t mm = 0; mm < M; ++mm) {
                int32_t* row = bb_.list_host(st.list) + uint64_t(mm) * b.list_stride;
                std::fill(row + ip.offset, row + b.list_stride, -1);
            }
            if (auto r = step(S::IdxTopK, &ip, sizeof ip, 1, M); !r) return r;
        }
    }
    if (st.run_compressor && n_complete)
        if (auto r = step(S::CmpStore, &cp, sizeof cp, 1, n_complete); !r) return r;
    return {};
}

Result<void> DecodeLayer::run_attention_batch(const BatchStep& st) {
    if (auto r = cmd_.begin(); !r) return r;
    if (auto r = record_attention_batch(cmd_, st); !r) return r;
    if (auto r = submit(cmd_); !r) return r;
    return verify_after_attention_batch(st);
}

Result<void> DecodeLayer::verify_after_attention_batch(const BatchStep& st) const {
    if (!st.run_indexer || !st.compress_ratio) return {};
    const TextConfig& c = *cfg_;
    const uint32_t off = c.sliding_window + st.m - 1;
    for (uint32_t mm = 0; mm < st.m; ++mm) {
        const uint32_t n = st.n_cmp(mm);
        const uint32_t k = std::min(c.index_topk, n);
        const int32_t* row = bb_.list_host(st.list) + uint64_t(mm) * bb_.list_stride;
        int32_t prev = int32_t(off) - 1;
        for (uint32_t i = 0; i < k; ++i) {
            const int32_t v = row[off + i];
            if (v <= prev || v >= int32_t(off + n))
                return fail(Err::Internal,
                            std::format("layer {} p0 {} query {}: top-k entry {} of {} is {} "
                                        "(previous {}); want strictly increasing in [{}, {}){}",
                                        st.layer, st.p0, mm, i, k, v, prev, off, off + n,
                                        v == -1 ? " -- never written" : ""));
            prev = v;
        }
        for (uint32_t i = k; off + i < bb_.list_stride; ++i)
            if (row[off + i] != -1)
                return fail(Err::Internal,
                            std::format("layer {} query {}: entry {} past the {} picks is {}",
                                        st.layer, mm, i, k, row[off + i]));
    }
    return {};
}

Result<void> DecodeLayer::record_close_batch(gpu::CommandBuffer& cmd, const BatchStep& st) {
    const TextConfig& c = *cfg_;
    const uint32_t n_wg0 = (c.hidden_size + 255) / 256;
    gpu::MhcPush mp{c.hidden_size, c.hc_mult, (2 + c.hc_mult) * c.hc_mult, n_wg0,
                    c.hc_sinkhorn_iters, gpu::kMhcFlagPost | gpu::kMhcFlagSkipSinkhorn,
                    static_cast<float>(c.rms_norm_eps), static_cast<float>(c.hc_eps)};
    if (auto r = mgt_->ensure(st.m); !r) return r;
    if (auto r = mgt_->record(cmd, st.m, gpu::MgtStage::MhcClose, &mp, sizeof mp, n_wg0, st.m); !r)
        return r;
    return cmd.barrier();
}

Result<void> DecodeLayer::run_close_batch(const BatchStep& st) {
    if (auto r = cmd_.begin(); !r) return r;
    if (auto r = record_close_batch(cmd_, st); !r) return r;
    return submit(cmd_);
}

Result<void> DecodeLayer::record_tail_batch(gpu::CommandBuffer& cmd, const BatchStep& st,
                                            const BatchTail& t) {
    const TextConfig& c = *cfg_;
    const BatchScratch& b = bb_;
    const uint32_t M = st.m;
    const uint32_t n_wg0 = (c.hidden_size + 255) / 256;
    using S = gpu::MgtStage;
    gpu::MgtRunner& R = *mgt_;
    if (auto r = R.ensure(M); !r) return r;
    // hc_post of the last layer AND hc_pre (MhcClose writes xout and utmp), then
    // the model's own norm over utmp into u.
    uint64_t* close = R.slots(S::MhcClose);
    close[gpu::slot::kNormW] = t.norm_w;
    close[gpu::slot::kU] = b.u.addr;
    std::memcpy(R.slots(S::MhcFinal), close, gpu::kAttnStageStride);
    gpu::MhcPush mp{c.hidden_size, c.hc_mult, (2 + c.hc_mult) * c.hc_mult, n_wg0,
                    c.hc_sinkhorn_iters, gpu::kMhcFlagPost | gpu::kMhcFlagSkipSinkhorn,
                    static_cast<float>(c.rms_norm_eps), static_cast<float>(c.hc_eps)};
    auto step = [&](S s, const void* push, uint32_t bytes, uint32_t gx,
                    uint32_t gy) -> Result<void> {
        if (auto r = R.record(cmd, M, s, push, bytes, gx, gy); !r) return r;
        return cmd.barrier();
    };
    if (auto r = step(S::MhcClose, &mp, sizeof mp, n_wg0, M); !r) return r;
    if (auto r = step(S::MhcFinal, &mp, sizeof mp, n_wg0, M); !r) return r;

    uint64_t* h = R.slots(S::Head);
    h[gpu::mslot::kHW] = t.head_w;
    h[gpu::mslot::kHX] = b.u.addr;
    h[gpu::mslot::kHLogits] = t.logits;
    h[gpu::mslot::kHSample] = t.sample;
    h[gpu::mslot::kHTopOut] = t.topk_out;
    h[gpu::mslot::kHHist] = t.topk_hist;
    std::memcpy(R.slots(S::HeadArgmax), h, gpu::kAttnStageStride);
    std::memcpy(R.slots(S::HeadTopK), h, gpu::kAttnStageStride);
    gpu::MgtHeadPush hp{};
    hp.rows = c.vocab_size; hp.k = c.hidden_size; hp.slices = R.spec().head_slices;
    hp.x_stride = c.hidden_size; hp.topk_k = t.topk_k; hp.inv_t = t.inv_t;
    for (uint32_t sl = 0; sl < hp.slices; ++sl) {
        hp.slice = sl;
        if (auto r = step(S::Head, &hp, sizeof hp, R.row_groups(c.vocab_size), 1); !r) return r;
    }
    if (auto r = step(S::HeadArgmax, &hp, sizeof hp, 1, M); !r) return r;
    if (t.topk_out && t.topk_hist)
        if (auto r = step(S::HeadTopK, &hp, sizeof hp, 1, M); !r) return r;
    return {};
}

namespace {
class NullMoeBridge final : public MoeBridge {
public:
    const char* name() const override { return "null"; }
    Result<void> run(const MoeCall&) override {
        return fail(Err::Unavailable,
                    "no MoE bridge: the attention chain ran but the block output is not "
                    "the model's. Supply a bridge over the design 7.9 kernels.");
    }
};
}  // namespace

std::unique_ptr<MoeBridge> make_null_moe_bridge() { return std::make_unique<NullMoeBridge>(); }

}  // namespace deepmoe::runtime
