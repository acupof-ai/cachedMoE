#include "runtime/dspark_runtime.h"
#include "runtime/rope.h"
#include "gpu/vulkan/dspark_mega.h"
#include "gpu/vulkan/dspark_onecb.h"
#include <cstdlib>
#include <bit>
#include "cpu/dequant.h"
#include "core/profiler.h"
#include "core/wc_read.h"
#include "core/log.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>

namespace cachedmoe::runtime {
namespace {
constexpr uint32_t D = layout::kHiddenSize, H = layout::kHcMult;
constexpr uint32_t M = layout::kDsparkBlockSize, V = layout::kVocabSize;
constexpr uint32_t HD = layout::kHeadDim, WIN = layout::kSlidingWindow;
using View = gpu::GpuScratch::View;
#define DS_TRY(expr)                                                                               \
    do {                                                                                           \
        auto ds_result = (expr);                                                                   \
        if (!ds_result) return std::unexpected(ds_result.error());                                 \
    } while (0)
} // namespace
struct DsparkRuntime::Impl {
    const store::PinnedStore *pinned = nullptr;
    TextConfig cfg;
    gpu::DsparkRunner ds;
    gpu::MgtRunner mgt;
    gpu::GpuScratch scratch;
    GpuMoeBridge moe;
    gpu::DsparkMegaRunner mega;
    gpu::DsparkOneCbRunner onecb;
    bool use_onecb = false;
    std::vector<std::string> op_labels;
    View route_save;
    gpu::Device *device = nullptr;
    gpu::MemoryAllocator *allocator = nullptr;
    bool use_mega = false, recording = false;
    uint32_t mega_groups = 120;
    bool profile_gpu = false;
    bool diagnostics = false;
    gpu::CommandPool profile_pool;
    gpu::CommandBuffer profile_cmd;
    gpu::QueryPool profile_queries;
    std::vector<gpu::DsparkMegaOp> ops;
    std::array<View, 3> mega_kv, mega_sink, expert_tables;
    View union_ids, union_list, union_w, union_count, xquant, hquant;
    std::array<View, 29> captures;
    struct Capture {
        std::string name;
        uint32_t count;
    };
    std::vector<Capture> capture_names;
    std::function<void(std::string_view, std::span<const float>)> *observer = nullptr;
    void inspect(std::string name, View src, uint32_t count) {
        if (!observer || !*observer || !diagnostics) return;
        if (recording)
            snapshot(std::move(name), src, count);
        else {
            std::vector<float> v(count);
            wc_readback(v.data(), src.host, count * 4);
            (*observer)(name, v);
        }
    }
    template <class Push>
    void op(uint32_t kind, const Push &push, std::initializer_list<uint64_t> ptr, uint32_t gx = 1,
            uint32_t gy = 1) {
        gpu::DsparkMegaOp o;
        o.kind = kind;
        o.gx = gx;
        o.gy = gy;
        static_assert(sizeof push <= 64);
        std::memcpy(o.push.data(), &push, sizeof push);
        std::copy(ptr.begin(), ptr.end(), o.ptr.begin());
        ops.push_back(o);
        op_labels.push_back(profile_stage + "helper." + std::to_string(kind));
    }
    void snapshot(std::string name, View src, uint32_t count, uint32_t mode = 0) {
        const auto dst = captures[capture_names.size()];
        std::array<uint32_t, 3> p{count, 16, mode};
        op(mode ? 22 : 21, p, {src.addr, dst.addr}, (count + 255) / 256);
        capture_names.push_back({std::move(name), count});
    }
    void moe_ops(uint32_t st) {
        uint32_t zero = 0;
        op(15, zero,
           {gate_ids.addr, gate_w.addr, union_ids.addr, union_list.addr, union_w.addr,
            union_count.addr});
        struct XQ {
            uint64_t src;
            uint32_t k, pad;
        } xq{u.addr, D, 0};
        op(16, xq, {0, 0, 0, 0, xquant.addr}, (M * D / 32 + 255) / 256);
        if (st == 0 && observer && *observer && diagnostics) {
            const auto dst = captures[capture_names.size()];
            uint32_t n = M * D;
            op(23, n, {xquant.addr, dst.addr}, (n + 255) / 256);
            capture_names.push_back({"diag.moe_xq", n});
        }
        const std::initializer_list<uint64_t> ptr{expert_tables[st].addr,
                                                  union_ids.addr,
                                                  union_list.addr,
                                                  union_w.addr,
                                                  xquant.addr,
                                                  hquant.addr,
                                                  attn.addr,
                                                  union_count.addr};
        struct Up {
            uint32_t layer, experts, slots, rows, k;
            float limit;
            uint32_t m;
        } up{0, 129, 16, 2304, D, float(cfg.swiglu_limit), M};
        op(17, up, ptr, 2304 / 8, 16);
        if (st == 0 && observer && *observer && diagnostics) {
            const auto dst = captures[capture_names.size()];
            uint32_t n = 2304;
            op(23, n, {hquant.addr, dst.addr}, (n + 255) / 256);
            capture_names.push_back({"diag.moe_h", n});
        }
        struct HQ {
            uint32_t slots, count, k;
        } hq{16, 16, 2304};
        op(18, hq, ptr,
           (layout::kMoeBatchColumns * (layout::kDsparkBlockSize * layout::kDsparkTopK + 1) *
                layout::kMoeIntermediate / 32 +
            255) /
               256);
        if (st == 0 && observer && *observer && diagnostics) {
            const auto dst = captures[capture_names.size()];
            uint32_t n = 2304;
            op(24, n, {hquant.addr, dst.addr}, (n + 255) / 256);
            capture_names.push_back({"diag.moe_hq", n});
        }
        struct Down {
            uint32_t layer, experts, slots, count, rows, k, flags, m;
        } down{0, 129, 16, 16, D, 2304, 0, M};
        op(19, down, ptr, D / 8);
        if (st == 0) inspect("diag.moe_y_rounded", attn, M * D);
    }
    View hidden, main, temp, x, xb, u, utmp, partial, raw, mix, nextmix, attn;
    View qa, qn, qb, q, kd, kv, tab, idx, sink, score, o, oi, woa;
    View gate_scores, gate_ids, gate_w, gate_done, logits, sample, bias, embeds, ids, conf;
    std::array<std::vector<uint16_t>, 3> ring;
    uint32_t next = 0;
    bool seeded = false;
    std::map<std::string, double> timing;
    std::map<std::string, double> gpu_timing;
    std::string profile_stage;
    void elapsed(const std::string &name, TimePoint start) {
        timing[name] += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    template <class Record> Result<void> profiled(const std::string &name, Record record) {
#if defined(CACHEDMOE_ENABLE_VULKAN)
        const auto start = Clock::now();
        auto cb = profile_pool.acquire();
        if (!cb) return std::unexpected(cb.error());
        profile_cmd = *cb;
        DS_TRY(profile_cmd.begin());
        DS_TRY(profile_cmd.reset_queries(profile_queries, 0, 2));
        DS_TRY(profile_cmd.write_timestamp(profile_queries, 0, false));
        DS_TRY(record(profile_cmd));
        DS_TRY(profile_cmd.write_timestamp(profile_queries, 1, true));
        DS_TRY(profile_cmd.end());
        DS_TRY(gpu::submit_and_wait(*device, profile_cmd));
        auto duration = profile_queries.elapsed_seconds(0, 1);
        if (!duration) return std::unexpected(duration.error());
        const auto label = name.starts_with("mtp.") ? name : profile_stage + name;
        gpu_timing[label] += *duration * 1e3;
        elapsed(label, start);
        return {};
#else
        return fail(Err::Unavailable, "DSpark timestamps require Vulkan");
#endif
    }
    Result<void> run_ds(gpu::DsparkStage stage, const void *push, uint32_t bytes, uint32_t groups,
                        std::string name = {}) {
        if (recording) {
            gpu::DsparkMegaOp o;
            o.kind = uint32_t(stage);
            o.gx = groups;
            // The first six enum entries match the mega opcodes.
            if (stage == gpu::DsparkStage::MarkovBias) o.kind = 12;
            if (stage == gpu::DsparkStage::AddBiasArgmax) o.kind = 13;
            if (stage == gpu::DsparkStage::Confidence) o.kind = 14;
            std::memcpy(o.push.data(), push, bytes);
            std::memcpy(o.ptr.data(), ds.slots(stage), 256);
            ops.push_back(o);
            op_labels.push_back(name.empty() ? profile_stage + gpu::dspark_stage_name(stage)
                                             : name);
            return {};
        }
        const auto label = name.empty() ? std::string(gpu::dspark_stage_name(stage)) : name;
        if (profile_gpu)
            return profiled(label,
                            [&](auto &cmd) { return ds.record(cmd, stage, push, bytes, groups); });
        auto start = Clock::now();
        auto r = ds.dispatch_now(stage, push, bytes, groups);
        elapsed(name.empty() ? gpu::dspark_stage_name(stage) : name, start);
        return r;
    }
    Result<void> run_mgt(uint32_t m, gpu::MgtStage stage, const void *push, uint32_t bytes,
                         uint32_t groups, uint32_t y = 1) {
        if (recording) {
            uint32_t kind = 0;
            switch (stage) {
            case gpu::MgtStage::MhcPost:
            case gpu::MgtStage::MhcClose:
                kind = 6;
                break;
            case gpu::MgtStage::MhcMix:
                kind = 7;
                break;
            case gpu::MgtStage::MhcFinal:
                kind = 8;
                break;
            case gpu::MgtStage::GateScore:
                kind = 9;
                break;
            case gpu::MgtStage::GateTopK:
                kind = 10;
                break;
            case gpu::MgtStage::Head:
                kind = 11;
                break;
            default:
                return fail(Err::Internal, "unsupported DSpark mega phase");
            }
            gpu::DsparkMegaOp o;
            o.kind = kind;
            o.gx = groups;
            o.gy = y;
            o.reserved = m | (uint32_t(stage) << 8);
            std::memcpy(o.push.data(), push, bytes);
            std::memcpy(o.ptr.data(), mgt.slots(stage), 256);
            ops.push_back(o);
            op_labels.push_back(profile_stage + gpu::mgt_stage_name(stage));
            return {};
        }
        if (profile_gpu)
            return profiled(gpu::mgt_stage_name(stage), [&](auto &cmd) {
                return mgt.record(cmd, m, stage, push, bytes, groups, y);
            });
        auto start = Clock::now();
        auto r = mgt.dispatch_now(m, stage, push, bytes, groups, y);
        elapsed(gpu::mgt_stage_name(stage), start);
        return r;
    }
    uint64_t A(const std::string &n) const {
        auto *t = pinned->find(n);
        return t ? t->data : 0;
    }
    uint64_t S(const std::string &n) const {
        auto *t = pinned->find(n);
        return t ? t->scale : 0;
    }
    Result<View> take(uint64_t n) {
        return scratch.alloc(n);
    }
    Result<void> gemv(const std::string &n, View in, View out, uint32_t m, uint32_t rows,
                      uint32_t k) {
        auto *s = ds.slots(gpu::DsparkStage::Gemv);
        s[0] = A(n);
        s[1] = S(n);
        s[2] = in.addr;
        s[3] = out.addr;
        if (!s[0]) return fail(Err::NotFound, "DSpark weight missing: " + n);
        gpu::DsparkGemvPush p{};
        p.m = m;
        p.rows = rows;
        p.k = k;
        p.scale_cols = k / 32;
        p.x_stride = k;
        p.y_stride = rows;
        p.flags = gpu::kDsFlagRoundOut;
        return run_ds(gpu::DsparkStage::Gemv, &p, sizeof p, ds.gemv_groups(rows), n);
    }
    Result<void> norm(const std::string &n, View in, View out, uint32_t m, uint32_t width) {
        auto *s = ds.slots(gpu::DsparkStage::RmsNorm);
        s[0] = in.addr;
        s[1] = A(n);
        s[2] = out.addr;
        gpu::DsparkGemvPush p{};
        p.m = m;
        p.k = width;
        p.x_stride = width;
        p.y_stride = width;
        p.eps = 1e-20f;
        p.flags = gpu::kDsFlagRoundIn;
        return run_ds(gpu::DsparkStage::RmsNorm, &p, sizeof p, m);
    }
    Result<void> rope(View in, View out, uint32_t p0, uint32_t m, uint32_t heads, bool quant,
                      bool inverse = false, bool bf = false, bool out_bf = false) {
        auto *t = static_cast<float *>(tab.host);
        for (uint32_t j = 0; j < m; ++j) {
            auto r = rope_table(rope_for_layer(0, 64), p0 + j);
            std::memcpy(t + j * 64, r.data(), 256);
        }
        auto *s = ds.slots(gpu::DsparkStage::RopeQuant);
        s[0] = in.addr;
        s[1] = tab.addr;
        s[2] = out.addr;
        gpu::DsparkGemvPush p{};
        p.m = m;
        p.rows = heads;
        p.k = HD;
        p.x_stride = heads * HD;
        p.y_stride = heads * HD;
        p.rope_dim = 64;
        p.quant = quant;
        p.inverse = inverse;
        p.flags = (bf ? gpu::kDsFlagInBf16 : 0) | (out_bf ? gpu::kDsFlagOutBf16 : 0);
        return run_ds(gpu::DsparkStage::RopeQuant, &p, sizeof p,
                      gpu::DsparkRunner::rope_groups(m, heads, HD));
    }
    Result<void> mhc(const std::string &prefix, const std::string &half, bool post) {
        auto *s = mgt.slots(gpu::MgtStage::MhcPost);
        s[gpu::slot::kX] = x.addr;
        s[gpu::slot::kA] = attn.addr;
        s[gpu::slot::kPreMix] = mix.addr;
        s[gpu::slot::kPostIn] = mix.addr + 16;
        s[gpu::slot::kCombIn] = mix.addr + 32;
        s[gpu::slot::kHcFn] = A(prefix + "hc_" + half + "_fn");
        s[gpu::slot::kHcBase] = A(prefix + "hc_" + half + "_base");
        s[gpu::slot::kHcScale] = A(prefix + "hc_" + half + "_scale");
        s[gpu::slot::kNormW] = A(prefix + half + "_norm.weight");
        s[gpu::slot::kXout] = xb.addr;
        s[gpu::slot::kUtmp] = utmp.addr;
        s[gpu::slot::kScratch] = partial.addr;
        s[gpu::slot::kMixRaw] = raw.addr;
        s[gpu::slot::kMixOut] = nextmix.addr;
        s[gpu::slot::kU] = u.addr;
        for (auto st : {gpu::MgtStage::MhcMix, gpu::MgtStage::MhcFinal})
            std::memcpy(mgt.slots(st), s, gpu::kAttnStageStride);
        gpu::MhcPush p{D,
                       H,
                       24,
                       20,
                       cfg.hc_sinkhorn_iters,
                       (post ? gpu::kMhcFlagPost : 0u) | gpu::kMhcFlagRoundResidual,
                       float(cfg.rms_norm_eps),
                       float(cfg.hc_eps)};
        DS_TRY(run_mgt(M, gpu::MgtStage::MhcPost, &p, sizeof p, 20, M));
        DS_TRY(run_mgt(M, gpu::MgtStage::MhcMix, &p, sizeof p, 24, M));
        DS_TRY(run_mgt(M, gpu::MgtStage::MhcFinal, &p, sizeof p, 20, M));
        std::swap(x, xb);
        std::swap(mix, nextmix);
        return {};
    }
    Result<void> attention(uint32_t stage, uint32_t pos) {
        const std::string p = std::format("mtp.{}.attn.", stage);
        if (recording) {
            kv = mega_kv[stage];
            sink = mega_sink[stage];
        }
        DS_TRY(gemv(p + "wq_a.weight", u, qa, M, 1280, D));
        if (stage == 0) inspect("diag.0.qa", qa, M * 1280);
        DS_TRY(norm(p + "q_norm.weight", qa, qn, M, 1280));
        if (stage == 0) inspect("diag.1.qn", qn, M * 1280);
        DS_TRY(gemv(p + "wq_b.weight", qn, qb, M, 32768, 1280));
        if (stage == 0) inspect("diag.2.qb", qb, M * 32768);
        // Produce attention's BF16 Q directly on the GPU. Reading 3 x 655 KB
        // of fp32 Q scalarly from WC memory cost 109 ms per draft cycle.
        DS_TRY(rope(qb, q, pos + 1, M, 64, false, false, false, true));
        DS_TRY(gemv(p + "wkv.weight", u, kd, M, HD, D));
        if (stage == 0) inspect("diag.3.kd", kd, M * 512);
        DS_TRY(norm(p + "kv_norm.weight", kd, qn, M, HD));
        if (stage == 0) inspect("diag.4.qn", qn, M * 512);
        if (recording) {
            auto tail = kv;
            tail.addr += WIN * HD * 2;
            tail.host = static_cast<uint16_t *>(tail.host) + WIN * HD;
            DS_TRY(rope(qn, tail, pos + 1, M, 1, true, false, false, true));
            if (stage == 0 && observer && *observer && diagnostics) {
                const auto dst = captures[capture_names.size()];
                uint32_t n = M * HD;
                op(25, n, {tail.addr, dst.addr}, (n + 255) / 256);
                capture_names.push_back({"diag.5.kd", n});
            }
        } else {
            DS_TRY(rope(qn, kd, pos + 1, M, 1, true));
            if (stage == 0) inspect("diag.5.kd", kd, M * 512);
        }
        auto *kh = static_cast<uint16_t *>(kv.host);
        if (!recording) std::memcpy(kh, ring[stage].data(), WIN * HD * 2);
        if (!recording) {
            std::vector<float> draft_kv(M * HD);
            wc_readback(draft_kv.data(), kd.host, M * HD * 4);
            for (uint32_t i = 0; i < M * HD; ++i)
                kh[WIN * HD + i] = cpu::float_to_bf16(draft_kv[i]);
        }
        auto *ix = static_cast<int32_t *>(idx.host);
        uint32_t count = 0;
        for (uint32_t i = 0; i < std::min(WIN, pos + 1); ++i)
            ix[count++] = int32_t(i);
        for (uint32_t i = 0; i < M; ++i)
            ix[count++] = int32_t(WIN + i);
        const auto *sw = pinned->find(p + "attn_sink");
        auto *sh = static_cast<float *>(sink.host);
        if (sw->dtype == QuantType::Bf16) {
            auto *v = static_cast<const uint16_t *>(sw->data_host);
            for (uint32_t i = 0; i < 64; ++i)
                sh[i] = cpu::bf16_to_float(v[i]);
        } else
            std::memcpy(sh, sw->data_host, 64 * 4);
        gpu::DsparkAttnPush ap{M, count, 64, HD, 256, 32768, 1 / std::sqrt(float(HD))};
        for (auto st : {gpu::DsparkStage::AttnScore, gpu::DsparkStage::AttnCombine}) {
            auto *s = ds.slots(st);
            s[0] = q.addr;
            s[1] = kv.addr;
            s[2] = idx.addr;
            s[3] = sink.addr;
            s[4] = score.addr;
            s[5] = o.addr;
            DS_TRY(run_ds(st, &ap, sizeof ap, 64));
        }
        DS_TRY(rope(o, oi, pos + 1, M, 64, false, true, true));
        if (stage == 0) inspect("diag.6.oi", oi, M * 32768);
        auto *sl = ds.slots(gpu::DsparkStage::WoA);
        sl[0] = A(p + "wo_a.weight");
        sl[1] = S(p + "wo_a.weight");
        sl[2] = oi.addr;
        sl[3] = woa.addr;
        gpu::DsparkGemvPush wp{};
        wp.m = M;
        wp.rows = 8192;
        wp.k = 4096;
        wp.scale_cols = 128;
        wp.x_stride = 32768;
        wp.y_stride = 8192;
        wp.rows_per_group = 1024;
        wp.flags = gpu::kDsFlagRoundOut;
        DS_TRY(run_ds(gpu::DsparkStage::WoA, &wp, sizeof wp, ds.gemv_groups(8192)));
        if (stage == 0) inspect("diag.7.woa", woa, M * 8192);
        return gemv(p + "wo_b.weight", woa, attn, M, D, 8192);
    }
};
DsparkRuntime::DsparkRuntime() : p_(std::make_unique<Impl>()) {}
DsparkRuntime::~DsparkRuntime() = default;
std::vector<std::string> DsparkRuntime::tensors(const Manifest &manifest) {
    std::vector<std::string> out;
    // Manifest tensor naming/selection follows PinnedStore: expert tensors are
    // supplied by the expert cache, never loaded twice into the pinned arena.
    for (uint32_t i = 0; i < 3; ++i) {
        const auto p = std::format("mtp.{}.", i);
        for (const char *n : {"hc_attn_fn",
                              "hc_attn_base",
                              "hc_attn_scale",
                              "hc_ffn_fn",
                              "hc_ffn_base",
                              "hc_ffn_scale",
                              "attn_norm.weight",
                              "ffn_norm.weight",
                              "attn.wq_a.weight",
                              "attn.wq_b.weight",
                              "attn.wkv.weight",
                              "attn.wo_a.weight",
                              "attn.wo_b.weight",
                              "attn.q_norm.weight",
                              "attn.kv_norm.weight",
                              "attn.attn_sink",
                              "ffn.gate.weight",
                              "ffn.gate.bias",
                              "ffn.shared_experts.w1.weight",
                              "ffn.shared_experts.w2.weight",
                              "ffn.shared_experts.w3.weight"})
            out.push_back(p + n);
    }
    for (const char *n : {"mtp.0.main_proj.weight", "mtp.0.main_norm.weight", "mtp.2.norm.weight",
                          "mtp.2.markov_head.embed.weight", "mtp.2.markov_head.head.weight",
                          "mtp.2.confidence_head.proj.weight"})
        out.emplace_back(n);
    (void)manifest;
    return out;
}
Result<void> DsparkRuntime::create(gpu::Device &dev, gpu::MemoryAllocator &alloc,
                                   const store::PinnedStore &pinned, store::ExpertStore &store,
                                   store::Planner &planner, const TextConfig &cfg,
                                   const GpuExecutionConfig &options) {
    auto &p = *p_;
    p.pinned = &pinned;
    p.cfg = cfg;
    p.device = &dev;
    p.allocator = &alloc;
    p.profile_gpu = options.draft_profile;
    p.diagnostics = options.draft_diagnostics;
    if (p.profile_gpu) {
        if (!dev.caps().timestamp_valid_bits)
            return fail(Err::Unavailable, "DSpark queue has no GPU timestamps");
        DS_TRY(p.profile_pool.create(dev));
        DS_TRY(p.profile_queries.create(dev, 2));
        auto cb = p.profile_pool.acquire();
        if (!cb) return std::unexpected(cb.error());
        p.profile_cmd = *cb;
    }
    log_info("DSpark draft: GPU timestamps {}", p.profile_gpu ? "on" : "off");
    p.use_mega = options.draft_mega;
    p.use_onecb = options.draft_onecb;
    if (p.use_mega && p.use_onecb)
        return fail(Err::InvalidArgument, "DSpark mega and one-CB are mutually exclusive");
    log_info("DSpark draft: one command buffer {}", p.use_onecb ? "on" : "off");
    gpu::MgtSpec spec;
    spec.pair_dot = options.mgt_pair_dot;
    DS_TRY(p.ds.create(dev, alloc, dev.environment().shader_dir));
    DS_TRY(p.mgt.create(dev, alloc, dev.environment().shader_dir, spec));
    DS_TRY(p.mgt.ensure(M));
    DS_TRY(p.scratch.create(alloc, 32ull << 20));
#define BUF(name, bytes)                                                                           \
    {                                                                                              \
        auto v = p.take(bytes);                                                                    \
        if (!v) return std::unexpected(v.error());                                                 \
        p.name = *v;                                                                               \
    }
    BUF(hidden, 4ull * M * D * 3);
    BUF(main, 4ull * M * D);
    BUF(temp, 4ull * M * 32768);
    BUF(x, 4ull * M * H * D);
    BUF(xb, 4ull * M * H * D);
    BUF(u, 4ull * M * D);
    BUF(utmp, 4ull * M * D);
    BUF(partial, 4ull * M * 512);
    BUF(raw, 4ull * M * 32);
    BUF(mix, 4ull * M * 32);
    BUF(nextmix, 4ull * M * 32);
    BUF(attn, 4ull * M * D);
    BUF(qa, 4ull * M * 1280);
    BUF(qn, 4ull * M * 1280);
    BUF(qb, 4ull * M * 32768);
    BUF(q, 2ull * M * 32768);
    BUF(kd, 4ull * M * HD);
    BUF(kv, 2ull * (WIN + M) * HD);
    BUF(tab, 4ull * M * 64);
    BUF(idx, 4ull * 256);
    BUF(sink, 4ull * 64);
    BUF(score, 4ull * M * 64 * 256);
    BUF(o, 2ull * M * 32768);
    BUF(oi, 4ull * M * 32768);
    BUF(woa, 4ull * M * 8192);
    BUF(gate_scores, 4ull * M * 128);
    BUF(gate_ids, 4ull * M * 16);
    BUF(gate_w, 4ull * M * 16);
    BUF(gate_done, 4ull * M);
    BUF(logits, 4ull * M * V);
    BUF(sample, 16ull * M);
    BUF(bias, 4ull * V);
    BUF(embeds, 4ull * M * 256);
    BUF(ids, 4ull * (M + 1));
    BUF(conf, 4ull * M);
#undef BUF
    DS_TRY(p.moe.create(dev, alloc, dev.environment().shader_dir, store, planner, pinned, cfg));
    for (uint32_t st = 0; st < 3; ++st) {
        auto kv = p.take(2ull * (WIN + M) * HD), sink = p.take(4ull * 64),
             table = p.take(129 * 6 * 8);
        if (!kv || !sink || !table) return fail(Err::ResourceExhausted, "DSpark mega tables");
        p.mega_kv[st] = *kv;
        p.mega_sink[st] = *sink;
        p.expert_tables[st] = *table;
        auto *t = static_cast<uint64_t *>(table->host);
        for (uint16_t e = 0; e < 128; ++e) {
            DS_TRY(store.table_row({uint16_t(40 + st), e}, t + e * 6));
        }
        const auto pre = std::format("mtp.{}.ffn.shared_experts.", st);
        for (uint32_t j = 0; j < 3; ++j) {
            const auto n = pre + std::format("w{}.weight", j + 1);
            t[128 * 6 + j * 2] = p.A(n);
            t[128 * 6 + j * 2 + 1] = p.S(n);
        }
    }
    auto take = [&](View &v, uint64_t bytes) -> Result<void> {
        auto r = p.take(bytes);
        if (!r) return std::unexpected(r.error());
        v = *r;
        return {};
    };
    DS_TRY(take(p.route_save, 3 * M * 16 * 4));
    DS_TRY(take(p.union_ids, 16 * 4));
    DS_TRY(take(p.union_list, 16 * 4));
    DS_TRY(take(p.union_w, M * 16 * 4));
    DS_TRY(take(p.union_count, 4));
    DS_TRY(take(p.xquant, 6ull * D * 2));
    DS_TRY(take(p.hquant, 6ull * 16 * 2304 * 4));
    for (auto &v : p.captures)
        DS_TRY(take(v, M * 32768 * 4));
    if (p.use_onecb)
        DS_TRY(p.onecb.create(dev, alloc, dev.environment().shader_dir));
    reset();
    return {};
}
void DsparkRuntime::set_mega(bool enabled, uint32_t groups) {
    auto &p = *p_;
    if (p.use_mega && !enabled)
        for (uint32_t st = 0; st < 3; ++st)
            wc_readback(p.ring[st].data(), p.mega_kv[st].host, WIN * HD * 2);
    p.use_mega = enabled;
    p.mega_groups = groups;
}
Result<void> DsparkRuntime::set_profile(bool enabled) {
    auto &p = *p_;
    if (enabled && !p.profile_queries.count()) {
        if (!p.device->caps().timestamp_valid_bits)
            return fail(Err::Unavailable, "DSpark queue has no timestamps");
        DS_TRY(p.profile_pool.create(*p.device));
        DS_TRY(p.profile_queries.create(*p.device, 2));
        auto cb = p.profile_pool.acquire();
        if (!cb) return std::unexpected(cb.error());
        p.profile_cmd = *cb;
    }
    p.profile_gpu = enabled;
    return {};
}
void DsparkRuntime::set_onecb(bool enabled) {
    auto &p = *p_;
    if (p.use_onecb && !enabled)
        for (uint32_t st = 0; st < 3; ++st)
            wc_readback(p.ring[st].data(), p.mega_kv[st].host, WIN * HD * 2);
    if (enabled && !p.use_onecb)
        for (uint32_t st = 0; st < 3; ++st)
            std::memcpy(p.mega_kv[st].host, p.ring[st].data(), WIN * HD * 2);
    p.use_onecb = enabled;
}
uint32_t DsparkRuntime::next_position() const {
    return p_->seeded ? p_->next : 0;
}
void DsparkRuntime::reset() {
    p_->next = 0;
    p_->seeded = false;
    for (uint32_t st = 0; st < 3; ++st) {
        p_->ring[st].assign(WIN * HD, 0);
        if (p_->mega_kv[st].host) std::memset(p_->mega_kv[st].host, 0, WIN * HD * 2);
    }
}
Result<void> DsparkRuntime::seed_window(uint32_t next,
                                        const std::array<std::span<const float>, 3> &rings) {
    for (auto r : rings)
        if (r.size() != WIN * HD)
            return fail(Err::InvalidArgument, "DSpark window must have 128x512 values");
    for (uint32_t st = 0; st < 3; ++st)
        for (uint32_t i = 0; i < WIN * HD; ++i)
            p_->ring[st][i] = cpu::float_to_bf16(rings[st][i]);
    for (uint32_t st = 0; st < 3; ++st)
        std::memcpy(p_->mega_kv[st].host, p_->ring[st].data(), WIN * HD * 2);
    p_->next = next;
    p_->seeded = true;
    return {};
}
Result<void> DsparkRuntime::append(uint32_t start, std::span<const float> hidden) {
    auto &p = *p_;
    if (hidden.size() % (3 * D))
        return fail(Err::InvalidArgument, "DSpark hidden rows must be 15360 wide");
    if (p.seeded && start != p.next)
        return fail(Err::FailedPrecondition,
                    "DSpark committed hidden positions are not contiguous");
    if (p.use_mega && !p.mega.valid())
        DS_TRY(p.mega.create(*p.device, *p.allocator, p.device->environment().shader_dir));
    struct ResetRecording {
        Impl &p;
        ~ResetRecording() {
            p.recording = false;
        }
    } reset_recording{p};
    if (p.use_onecb && !p.onecb.valid())
        DS_TRY(p.onecb.create(*p.device, *p.allocator, p.device->environment().shader_dir));
    p.recording = p.use_mega || p.use_onecb;
    const uint32_t count = uint32_t(hidden.size() / (3 * D));
    for (uint32_t at = 0; at < count; at += M) {
        const uint32_t n = std::min(M, count - at);
        p.ops.clear();
        p.op_labels.clear();
        std::memcpy(p.hidden.host, hidden.data() + size_t(at) * 3 * D, size_t(n) * 3 * D * 4);
        DS_TRY(p.gemv("mtp.0.main_proj.weight", p.hidden, p.temp, n, D, 3 * D));
        DS_TRY(p.norm("mtp.0.main_norm.weight", p.temp, p.main, n, D));
        for (uint32_t st = 0; st < 3; ++st) {
            const auto pre = std::format("mtp.{}.attn.", st);
            DS_TRY(p.gemv(pre + "wkv.weight", p.main, p.kd, n, HD, D));
            DS_TRY(p.norm(pre + "kv_norm.weight", p.kd, p.qn, n, HD));
            DS_TRY(p.rope(p.qn, p.kd, start + at, n, 1, true));
            if (p.recording) {
                std::array<uint32_t, 2> push{n, start + at};
                p.op(26, push, {p.kd.addr, p.mega_kv[st].addr}, (n * HD + 255) / 256);
            } else {
                std::vector<float> f(n * HD);
                wc_readback(f.data(), p.kd.host, size_t(n) * HD * 4);
                for (uint32_t j = 0; j < n; ++j) {
                    auto *row = p.ring[st].data() + ((start + at + j) % WIN) * HD;
                    for (uint32_t d = 0; d < HD; ++d)
                        row[d] = cpu::float_to_bf16(f[j * HD + d]);
                    std::memcpy(static_cast<uint16_t *>(p.mega_kv[st].host) +
                                    ((start + at + j) % WIN) * HD,
                                row, HD * 2);
                }
            }
        }
        if (p.use_onecb)
            DS_TRY(p.onecb.run(p.ops, p.ds, p.mgt));
        else if (p.recording)
            DS_TRY(p.mega.run(p.ops, p.mega_groups));
    }
    p.next = start + count;
    p.seeded = true;
    return {};
}
Result<DsparkRuntime::Output> DsparkRuntime::draft(uint32_t pos, uint32_t token,
                                                   uint32_t output_rows, bool read_logits) {
    if (output_rows < 1 || output_rows > M)
        return fail(Err::InvalidArgument, "DSpark output rows must be 1..5");
    auto &p = *p_;
    if (!p.seeded || p.next != pos + 1 || token >= V)
        return fail(Err::FailedPrecondition,
                    "DSpark draft needs committed main KV through position");
    if (p.use_mega && !p.mega.valid())
        DS_TRY(p.mega.create(*p.device, *p.allocator, p.device->environment().shader_dir));
    if (p.use_onecb && !p.onecb.valid())
        DS_TRY(p.onecb.create(*p.device, *p.allocator, p.device->environment().shader_dir));
    p.observer = &probe;
    p.recording = p.use_mega || p.use_onecb;
    p.ops.clear();
    p.op_labels.clear();
    p.capture_names.clear();
    struct ResetRecording {
        Impl &p;
        ~ResetRecording() {
            p.recording = false;
        }
    } reset_recording{p};
    p.timing.clear();
    p.gpu_timing.clear();
    const auto draft_start = Clock::now();
    std::array<uint32_t, 45> expert_ids{};
    if (!p.use_mega) DS_TRY(p.mgt.ensure(output_rows));
    const auto *emb = p.pinned->find("embed.weight");
    auto *x = static_cast<float *>(p.x.host);
    for (uint32_t m = 0; m < M; ++m) {
        const uint32_t id = m ? p.cfg.dspark_noise_token_id : token;
        std::vector<uint16_t> row(D);
        std::memcpy(row.data(), static_cast<const uint16_t *>(emb->data_host) + size_t(id) * D,
                    D * 2);
        for (uint32_t h = 0; h < H; ++h)
            for (uint32_t d = 0; d < D; ++d)
                x[(m * H + h) * D + d] = cpu::bf16_to_float(row[d]);
    }
    std::memset(p.mix.host, 0, M * 32 * 4);
    for (uint32_t m = 0; m < M; ++m)
        static_cast<float *>(p.mix.host)[m * 32] = 1;
    for (uint32_t st = 0; st < 3; ++st) {
        const auto pre = std::format("mtp.{}.", st);
        p.profile_stage = pre + "attn.";
        DS_TRY(p.mhc(pre, "attn", st != 0));
        if (probe) {
            if (p.recording)
                p.snapshot(pre + "attn_norm", p.u, M * D);
            else
                probe(pre + "attn_norm", {static_cast<float *>(p.u.host), M * D});
        }
        DS_TRY(p.attention(st, pos));
        if (probe) {
            if (p.recording)
                p.snapshot(pre + "wo_b", p.attn, M * D);
            else
                probe(pre + "wo_b", {static_cast<float *>(p.attn.host), M * D});
        }
        p.profile_stage = pre + "ffn.";
        DS_TRY(p.mhc(pre, "ffn", true));
        if (probe) {
            if (p.recording)
                p.snapshot(pre + "ffn_norm", p.u, M * D);
            else
                probe(pre + "ffn_norm", {static_cast<float *>(p.u.host), M * D});
        }
        auto *g = p.mgt.slots(gpu::MgtStage::GateScore);
        g[0] = p.A(pre + "ffn.gate.weight");
        g[1] = p.A(pre + "ffn.gate.bias");
        g[2] = p.u.addr;
        g[3] = p.gate_scores.addr;
        g[4] = p.gate_ids.addr;
        g[5] = p.gate_w.addr;
        g[6] = p.gate_done.addr;
        std::memcpy(p.mgt.slots(gpu::MgtStage::GateTopK), g, gpu::kAttnStageStride);
        gpu::GatePush gp{128, D, 3, 16, 1, 1.5f};
        DS_TRY(p.run_mgt(M, gpu::MgtStage::GateScore, &gp, sizeof gp, p.mgt.row_groups(128), M));
        DS_TRY(p.run_mgt(M, gpu::MgtStage::GateTopK, &gp, sizeof gp, 1, M));
        if (p.recording) {
            if (probe) {
                p.snapshot(pre + "route_i", p.gate_ids, M * 3, 1);
                p.snapshot(pre + "route_w", p.gate_w, M * 3, 2);
            }
            if (p.use_onecb) {
                uint32_t n = M * 16;
                p.op(21, n, {p.gate_ids.addr, p.route_save.addr + st * M * 16 * 4},
                     (n + 255) / 256);
            }
            p.moe_ops(st);
        } else {
            uint32_t ids[M * 3];
            float weights[M * 3];
            for (uint32_t m = 0; m < M; ++m)
                for (uint32_t i = 0; i < 3; ++i) {
                    ids[m * 3 + i] = static_cast<uint32_t *>(p.gate_ids.host)[m * 16 + i];
                    weights[m * 3 + i] = static_cast<float *>(p.gate_w.host)[m * 16 + i];
                }
            std::copy(ids, ids + M * 3, expert_ids.begin() + st * M * 3);
            if (probe) {
                std::vector<float> routed(M * 3);
                for (uint32_t i = 0; i < M * 3; ++i)
                    routed[i] = float(ids[i]);
                probe(pre + "route_i", routed);
                probe(pre + "route_w", weights);
            }
            GpuMoeBridge::BatchCall bc;
            bc.layer = 40 + st;
            bc.m = M;
            bc.topk = 3;
            bc.ids = ids;
            bc.weights = weights;
            bc.x = static_cast<float *>(p.u.host);
            bc.y = static_cast<float *>(p.attn.host);
            bc.hidden = D;
            auto moe_start = Clock::now();
            DS_TRY(p.moe.run_batch_union(bc));
            p.elapsed(p.profile_gpu ? pre + "moe" : "moe", moe_start);
            if (p.profile_gpu) p.gpu_timing[pre + "moe"] += p.moe.union_info().gpu_ms;
            if (st == 0 && probe && p.diagnostics) {
                std::vector<uint16_t> packed(M * D);
                wc_readback(packed.data(), p.moe.union_debug_x(), M * D * 2);
                std::vector<float> vals(M * D);
                for (uint32_t i = 0; i < M * D; ++i)
                    vals[i] = cpu::fp16_to_float(packed[i]);
                probe("diag.moe_xq", vals);
                const uint32_t slots = p.moe.union_debug_slots();
                std::vector<uint32_t> h(6 * slots * 2304);
                wc_readback(h.data(), p.moe.union_debug_h(), h.size() * 4);
                vals.resize(2304);
                for (uint32_t i = 0; i < 2304; ++i)
                    vals[i] = cpu::fp16_to_float(uint16_t(h[i / 2] >> (16 * (i % 2))));
                probe("diag.moe_h", vals);
                for (uint32_t i = 0; i < 2304; ++i) {
                    auto b = uint8_t(h[6 * slots * 2304 / 2 + i / 4] >> (8 * (i % 4)));
                    vals[i] = cpu::fp8_e4m3_to_float(b) *
                              std::bit_cast<float>(h[6 * slots * 2304 * 3 / 4 + i / 32]);
                }
                probe("diag.moe_hq", vals);
                std::vector<float> rounded(M * D);
                wc_readback(rounded.data(), p.attn.host, M * D * 4);
                for (auto &v : rounded)
                    v = cpu::bf16_to_float(cpu::float_to_bf16(v));
                probe("diag.moe_y_rounded", rounded);
            }
            auto *y = static_cast<float *>(p.attn.host);
            std::vector<float> rounded(M * D);
            wc_readback(rounded.data(), y, M * D * 4);
            for (auto &v : rounded)
                v = cpu::bf16_to_float(cpu::float_to_bf16(v));
            std::memcpy(y, rounded.data(), M * D * 4);
        }
    }
    p.profile_stage.clear();
    // Close final MTP residual using its FFN coefficients; collapse with ffn pre.
    auto *s = p.mgt.slots(gpu::MgtStage::MhcClose);
    s[gpu::slot::kX] = p.x.addr;
    s[gpu::slot::kA] = p.attn.addr;
    s[gpu::slot::kPreMix] = p.mix.addr;
    s[gpu::slot::kPostIn] = p.mix.addr + 16;
    s[gpu::slot::kCombIn] = p.mix.addr + 32;
    s[gpu::slot::kXout] = p.xb.addr;
    s[gpu::slot::kUtmp] = p.utmp.addr;
    s[gpu::slot::kScratch] = p.partial.addr;
    s[gpu::slot::kNormW] = p.A("mtp.2.norm.weight");
    s[gpu::slot::kU] = p.u.addr;
    std::memcpy(p.mgt.slots(gpu::MgtStage::MhcFinal), s, gpu::kAttnStageStride);
    gpu::MhcPush mp{D,
                    H,
                    24,
                    20,
                    p.cfg.hc_sinkhorn_iters,
                    gpu::kMhcFlagPost | gpu::kMhcFlagSkipSinkhorn | gpu::kMhcFlagRoundResidual,
                    float(p.cfg.rms_norm_eps),
                    float(p.cfg.hc_eps)};
    DS_TRY(p.run_mgt(M, gpu::MgtStage::MhcClose, &mp, sizeof mp, 20, M));
    DS_TRY(p.run_mgt(M, gpu::MgtStage::MhcFinal, &mp, sizeof mp, 20, M));
    if (probe) {
        if (p.recording)
            p.snapshot("head_norm_out", p.u, M * D);
        else
            probe("head_norm_out", {static_cast<float *>(p.u.host), M * D});
    }
    auto *hs = p.mgt.slots(gpu::MgtStage::Head);
    hs[gpu::mslot::kHW] = p.A("head.weight");
    hs[gpu::mslot::kHX] = p.u.addr;
    hs[gpu::mslot::kHLogits] = p.logits.addr;
    gpu::MgtHeadPush hp{};
    hp.rows = V;
    hp.k = D;
    hp.slices = p.mgt.spec().head_slices;
    hp.x_stride = D;
    // The mega body has a fixed five-column head; its Markov suffix still
    // truncates. Serial uses the existing specialised head for the needed rows.
    const uint32_t head_rows = p.use_mega ? M : output_rows;
    for (uint32_t i = 0; i < hp.slices; ++i) {
        hp.slice = i;
        DS_TRY(p.run_mgt(head_rows, gpu::MgtStage::Head, &hp, sizeof hp, p.mgt.row_groups(V)));
    }
    auto *id = static_cast<uint32_t *>(p.ids.host);
    id[0] = token;
    for (uint32_t i = 0; i < output_rows; ++i) {
        gpu::DsparkHeadPush dp{};
        dp.m = M;
        dp.rows = V;
        dp.k = 256;
        dp.rank = 256;
        dp.pos = i;
        dp.logit_stride = V;
        dp.flags = gpu::kDsFlagTokenFromBuf;
        auto *z = p.ds.slots(gpu::DsparkStage::MarkovBias);
        z[0] = p.A("mtp.2.markov_head.head.weight");
        z[1] = p.A("mtp.2.markov_head.embed.weight");
        z[2] = p.ids.addr;
        z[3] = p.bias.addr;
        z[4] = p.embeds.addr;
        DS_TRY(p.run_ds(gpu::DsparkStage::MarkovBias, &dp, sizeof dp, p.ds.gemv_groups(V)));
        z = p.ds.slots(gpu::DsparkStage::AddBiasArgmax);
        z[0] = p.logits.addr;
        z[1] = p.bias.addr;
        z[2] = p.sample.addr;
        z[3] = p.ids.addr;
        DS_TRY(p.run_ds(gpu::DsparkStage::AddBiasArgmax, &dp, sizeof dp, 1));
    }
    auto *z = p.ds.slots(gpu::DsparkStage::Confidence);
    z[0] = p.utmp.addr;
    z[1] = p.embeds.addr;
    z[2] = p.A("mtp.2.confidence_head.proj.weight");
    z[3] = p.conf.addr;
    gpu::DsparkHeadPush cp{};
    cp.m = output_rows;
    cp.k = D;
    cp.rank = 256;
    cp.x_stride = D;
    DS_TRY(p.run_ds(gpu::DsparkStage::Confidence, &cp, sizeof cp, 1));
    if (p.recording) {
        const auto start = Clock::now();
        if (p.use_onecb) {
            DS_TRY(p.onecb.run(p.ops, p.ds, p.mgt, p.op_labels, p.profile_gpu));
            p.gpu_timing = p.onecb.timing_ms;
            p.elapsed("onecb.record_submit", start);
        } else {
            DS_TRY(p.mega.run(p.ops, p.mega_groups));
            p.elapsed("mega.kernel", start);
        }
        for (size_t i = 0; i < p.capture_names.size(); ++i) {
            std::vector<float> v(p.capture_names[i].count);
            wc_readback(v.data(), p.captures[i].host, v.size() * 4);
            if (probe) probe(p.capture_names[i].name, v);
        }
    }
    Output out;
    out.gpu_submissions = p.recording ? 1 : 87 - 2 * (M - output_rows);
    out.gpu_dispatches = p.use_onecb   ? uint32_t(p.ops.size())
                         : p.recording ? 1
                                       : 87 - 2 * (M - output_rows);
    out.gpu_phases = p.recording ? uint32_t(p.ops.size()) : out.gpu_dispatches;
    if (p.use_onecb) {
        std::array<uint32_t, 3 * M * 16> raw_ids;
        wc_readback(raw_ids.data(), p.route_save.host, sizeof raw_ids);
        for (uint32_t st = 0; st < 3; ++st)
            for (uint32_t m = 0; m < M; ++m)
                for (uint32_t j = 0; j < 3; ++j)
                    expert_ids[(st * M + m) * 3 + j] = raw_ids[(st * M + m) * 16 + j];
    }
    out.expert_ids = expert_ids;
    out.expert_ids_valid = !p.use_mega;
    std::copy(id + 1, id + 1 + output_rows, out.tokens.begin());
    std::memcpy(out.confidence.data(), p.conf.host, output_rows * 4);
    auto copy_start = Clock::now();
    if (read_logits) {
        out.logits.resize(output_rows * V);
        wc_readback(out.logits.data(), p.logits.host, output_rows * V * 4);
    }
    p.elapsed("host.logits", copy_start);
    p.elapsed("wall.draft", draft_start);
    out.timing_ms = p.timing;
    out.gpu_timing_ms = p.gpu_timing;
    return out;
}
#undef DS_TRY
} // namespace cachedmoe::runtime
