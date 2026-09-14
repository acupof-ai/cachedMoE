#include "runtime/decode_layer.h"

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
    const uint32_t score_stride = 1024;

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
              const gpu::GpuScratch::View& x_in, const gpu::GpuScratch::View& x_out) {
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
    s[gpu::slot::kU]       = b.u.addr;
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
             b.mix_a, b.mix_b, moe_view(), b.x, b.xout);
    // FFN half: reads the stream the attention half wrote, folds in wo_b's
    // output, and writes back to `x`. In-place would also be safe -- a thread
    // reads all hc copies of its own element before writing any -- but keeping
    // the two apart makes a layer-at-a-time validator able to look at both.
    bind_mhc(*runner_, gpu::AttnStage::MhcPostB, gpu::AttnStage::MhcMixB,
             gpu::AttnStage::MhcFinalB, b,
             w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm,
             b.mix_b, b.mix_a, b.wob, b.xout, b.x);
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
                 b.mix_a, b.mix_b, moe_view(), b.x, b.xout);

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
    g[gpu::slot::kGateX]         = b.u.addr;
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
        };
        idx(gpu::AttnStage::IdxQGemv,   b.rope.addr,     st.kv.idx_key);
        idx(gpu::AttnStage::IdxQFinish, b.rope.addr,     st.kv.idx_key);
        idx(gpu::AttnStage::IdxKey,     b.rope_lat.addr, st.idx_key_write);
        idx(gpu::AttnStage::IdxWeights, b.rope.addr,     st.kv.idx_key);
        idx(gpu::AttnStage::IdxScore,   b.rope.addr,     st.kv.idx_key);
        idx(gpu::AttnStage::IdxTopK,    b.rope.addr,     st.kv.idx_key);
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
    gpu::AttnPush ap{st.kv.n_kv, c.sliding_window, c.head_dim, c.qk_rope_head_dim, 1024,
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
        const uint32_t sg = (st.n_cmp + gpu::kIdxScoreTile - 1) / gpu::kIdxScoreTile;
        if (sg) {
            if (auto r = step(gpu::AttnStage::IdxScore, &is, sizeof is, sg); !r) return r;
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
    return submit(cmd_);
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
    call.x       = static_cast<const float*>(buf_.u.host);   // ffn_norm output
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
const float*    DecodeLayer::ffn_norm_out() const { return static_cast<const float*>(buf_.u.host); }

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
