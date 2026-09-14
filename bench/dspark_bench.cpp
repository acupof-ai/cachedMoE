// T_draft: the time and bandwidth of one DSpark draft cycle's own kernels
// (design §7.12, §10.1, §10.3), on the real checkpoint's mtp weights.
//
//   dspark_bench --model D:/models/DeepSeek-V4.1-Flash [--iters 16]
//                [--lanes 32] [--csv bench/results/dspark_p3.csv] [--note "..."]
//
// What is timed
// -------------
// Every stage of gpu/vulkan/dspark_kernels.h at the draft geometry -- M = 5
// draft positions, 133 KV rows, the 70-entry index list of the L3 position-64
// cycle -- first one stage at a time (N iterations in one command buffer, best of
// three resubmits, the attn_bench.cpp idiom) and then the WHOLE draft-kernel
// chain of a cycle recorded back to back into one command buffer, which is the
// number that goes into docs/p3_dspark.md §8:
//
//   main_proj + main_norm
//   3 x [ wq_a, q_norm, wq_b, RoPE(q), wkv(main) + kv_norm + RoPE/quant,
//         wkv(draft) + kv_norm + RoPE/quant, attn score + combine,
//         inverse RoPE, wo_a, wo_b ]
//   5 x [ Markov bias, bias add + argmax ]   (sequential, chained on the GPU)
//   confidence
//
// What is NOT timed here, and where its number comes from
// -------------------------------------------------------
// The head GEMV (1.32 GB bf16, head.slang, M = 1 today), and each mtp stage's
// Mega-mHC x 2, gate and FP4 MoE: those are existing kernels with existing
// measurements (docs/p2_attention.md §5, docs/kernel_p2_moe.md §3.5). The doc
// adds them to this benchmark's chain total and says which is which.
//
// One runner per mtp stage: a runner owns one address table, and a command
// buffer that re-read the same stage's weights would sit in the 32 MB MALL for
// the small tensors and report a cache number (docs/p2_attention.md §5's
// warning, which this benchmark inherits). Activations are synthetic: every
// kernel here is bandwidth- or latency-bound, not value-bound.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/config.h"
#include "gpu/vulkan/attn_kernels.h"     // GpuScratch
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/dspark_kernels.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/moe_kernels.h"      // default_shader_dir
#include "model/layout.h"
#include "model/manifest.h"
#include "runtime/rope.h"
#include "storage/backend.h"
#include "storage/io_engine.h"
#include "store/pinned.h"
#include "store/shard_set.h"

using namespace deepmoe;

namespace {

constexpr uint32_t kDim = 5120, kHeads = 64, kHeadDim = 512, kRope = 64;
constexpr uint32_t kQLora = 1280, kOLora = 1024, kGroups = 8;
constexpr uint32_t kVocab = 129280, kRank = 256, kM = 5, kStages = 3;
constexpr uint32_t kNKv = 133, kNIdx = 70;     // the L3 position-64 cycle

struct Opts {
    std::string model, csv, note;
    uint32_t iters = 16, lanes = 32;
};

struct Row {
    std::string name;
    double      ms;      // per iteration
    uint64_t    bytes;   // weight bytes read per iteration
    double gbps() const { return ms > 0 ? bytes / (ms * 1e-3) / 1e9 : 0.0; }
};

uint64_t fp8_bytes(uint64_t rows, uint64_t k) { return rows * k + ((rows + 31) / 32) * ((k + 31) / 32); }

std::string arg_after(int argc, char** argv, const char* flag, const char* dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return dflt;
}

}  // namespace

int main(int argc, char** argv) {
    Opts o;
    o.model = arg_after(argc, argv, "--model", "");
    o.csv   = arg_after(argc, argv, "--csv", "");
    o.note  = arg_after(argc, argv, "--note", "");
    o.iters = static_cast<uint32_t>(std::atoi(arg_after(argc, argv, "--iters", "16").c_str()));
    o.lanes = static_cast<uint32_t>(std::atoi(arg_after(argc, argv, "--lanes", "32").c_str()));
    if (o.model.empty()) {
        std::printf("usage: dspark_bench --model <checkpoint dir> [--iters N] [--lanes 16|32|64]\n"
                    "                    [--csv PATH] [--note TEXT]\n");
        return 2;
    }

    gpu::Device device;
    gpu::DeviceOptions dopts;
    dopts.enable_validation = std::getenv("VK_INSTANCE_LAYERS") != nullptr;
    if (auto r = device.create(dopts); !r) { std::printf("device: %s\n", r.error().str().c_str()); return 1; }
    if (auto r = device.caps().check_required(); !r) { std::printf("device: %s\n", r.error().str().c_str()); return 1; }
    gpu::MemoryAllocator alloc;
    if (auto r = alloc.init(device, MemoryPath::DeviceLocalHostVisible); !r) {
        std::printf("alloc: %s\n", r.error().str().c_str()); return 1;
    }
    auto mf = Manifest::load(store::ShardSet::join(o.model, layout::kManifestFile));
    if (!mf) { std::printf("manifest: %s\n", mf.error().str().c_str()); return 1; }
    store::ShardSet shards;
    if (auto r = shards.open_all(o.model, *mf, true); !r) { std::printf("shards: %s\n", r.error().str().c_str()); return 1; }
    IoConfig iocfg;
    auto backend = storage::make_default_backend(iocfg);
    if (!backend) { std::printf("backend: %s\n", backend.error().str().c_str()); return 1; }
    storage::IoEngine io;
    if (auto r = io.start(std::move(*backend), iocfg); !r) { std::printf("io: %s\n", r.error().str().c_str()); return 1; }

    store::PinnedStore pinned;
    auto backing = alloc.make_slab_backing();
    if (!backing) { std::printf("backing: %s\n", backing.error().str().c_str()); io.stop(); return 1; }
    store::PinnedConfig pcfg;
    pcfg.region_bytes = 512ull << 20;
    if (auto r = pinned.init(std::move(*backing), pcfg); !r) { std::printf("pinned: %s\n", r.error().str().c_str()); io.stop(); return 1; }
    std::vector<std::string> names;
    for (uint32_t s = 0; s < kStages; ++s)
        for (const char* t : {"attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight",
                              "attn.wo_a.weight", "attn.wo_b.weight", "attn.attn_sink",
                              "attn.q_norm.weight", "attn.kv_norm.weight"})
            names.push_back(std::format("mtp.{}.{}", s, t));
    for (const char* t : {"mtp.0.main_proj.weight", "mtp.0.main_norm.weight",
                          "mtp.2.markov_head.embed.weight", "mtp.2.markov_head.head.weight",
                          "mtp.2.confidence_head.proj.weight"})
        names.push_back(t);
    std::printf("loading the DSpark draft weights (%.2f GB)...\n", store::pinned_bytes(*mf, names) / 1e9);
    if (auto r = pinned.load(*mf, shards, io, names); !r) {
        std::printf("pinned load: %s\n", r.error().str().c_str()); io.stop(); return 1;
    }
    auto A = [&](const std::string& n) { auto* t = pinned.find(n); return t ? t->data : uint64_t(0); };
    auto S = [&](const std::string& n) { auto* t = pinned.find(n); return t ? t->scale : uint64_t(0); };

    gpu::DsparkSpec spec;
    spec.lanes_per_row = o.lanes;
    // 16 runners: a runner's table holds ONE slot set per stage kind, and a real
    // cycle issues Gemv five times, RmsNorm three and RopeQuant four per mtp
    // stage. Runner `5 s + j` is the j-th use in mtp stage s; runner 15 holds
    // main_proj / main_norm and the head tail. (A note for Track I: the runtime
    // either keeps this many tables or rewrites slots between submissions.)
    constexpr uint32_t kRunners = 16, kHeadRunner = 15;
    std::vector<std::unique_ptr<gpu::DsparkRunner>> rn(kRunners);
    for (uint32_t s = 0; s < kRunners; ++s) {
        rn[s] = std::make_unique<gpu::DsparkRunner>();
        if (auto r = rn[s]->create(device, alloc, gpu::default_shader_dir(), spec); !r) {
            std::printf("runner: %s\n", r.error().str().c_str()); io.stop(); return 1;
        }
    }
    gpu::GpuScratch scratch;
    if (auto r = scratch.create(alloc, 64ull << 20); !r) { std::printf("scratch: %s\n", r.error().str().c_str()); io.stop(); return 1; }
    auto take = [&](uint64_t b) { auto v = scratch.alloc(b); return v ? *v : gpu::GpuScratch::View{}; };

    // Activations: a plausible, deterministic signal.
    auto X15 = take(4ull * kDim * 3), Y5120 = take(4ull * kM * kDim), Xd = take(4ull * kM * kDim);
    auto Qa = take(4ull * kM * kQLora), Qn = take(4ull * kM * kQLora), Qb = take(4ull * kM * 32768);
    auto Qr = take(2ull * kM * 32768), Kv1 = take(4ull * kHeadDim), Kv1n = take(4ull * kHeadDim);
    auto Kvd = take(4ull * kM * kHeadDim), Kvdn = take(4ull * kM * kHeadDim);
    auto KvRing = take(2ull * 256 * kHeadDim), Idx = take(4ull * 256), Sink = take(4ull * kHeads);
    auto Score = take(4ull * kM * kHeads * 256), O16 = take(2ull * kM * 32768);
    auto Oinv = take(4ull * kM * 32768), Woa = take(4ull * kM * kGroups * kOLora);
    auto Wob = take(4ull * kM * kDim), TabD = take(4ull * kM * kRope), TabM = take(4ull * kRope);
    auto Logits = take(4ull * kM * kVocab), Bias = take(4ull * kVocab), Emb = take(4ull * kM * kRank);
    auto Sample = take(16), Ids = take(4ull * (kM + 1)), Conf = take(4ull * kM);
    if (!Conf.valid()) { std::printf("scratch exhausted\n"); io.stop(); return 1; }
    auto fill = [](gpu::GpuScratch::View& v, uint64_t n, float amp) {
        auto* p = static_cast<float*>(v.host);
        for (uint64_t i = 0; i < n; ++i) p[i] = amp * std::sin(0.37f * float(i % 9973) + 0.1f);
    };
    fill(X15, kDim * 3, 0.05f);
    fill(Xd, kM * kDim, 1.0f);
    fill(Qa, kM * kQLora, 1.0f); fill(Qn, kM * kQLora, 1.0f);
    fill(Kv1, kHeadDim, 1.0f); fill(Kvd, kM * kHeadDim, 1.0f);
    fill(Oinv, kM * 32768, 0.1f); fill(Woa, kM * kGroups * kOLora, 0.1f);
    fill(Logits, uint64_t(kM) * kVocab, 1.0f); fill(Sink, kHeads, 1.0f);
    std::memset(Qr.host, 0x11, Qr.bytes); std::memset(KvRing.host, 0x22, KvRing.bytes);
    std::memset(O16.host, 0x10, O16.bytes);
    { auto* p = static_cast<int32_t*>(Idx.host);
      for (uint32_t i = 0; i < kNIdx; ++i) p[i] = i < 65 ? int32_t(i) : int32_t(128 + i - 65); }
    { const auto cfg = runtime::rope_for_layer(0, kRope);
      auto* p = static_cast<float*>(TabD.host);
      for (uint32_t m = 0; m < kM; ++m) { auto r = runtime::rope_table(cfg, 65 + m); std::memcpy(p + m * kRope, r.data(), kRope * 4); }
      auto r = runtime::rope_table(cfg, 64); std::memcpy(TabM.host, r.data(), kRope * 4); }
    static_cast<uint32_t*>(Ids.host)[0] = 223;

    // ---- the stage recipes: set slots on runner `s`, then return (push, groups)
    struct Disp { gpu::DsparkStage st; std::vector<uint8_t> push; uint32_t groups; };
    auto gemv_push = [&](uint32_t m, uint32_t rows, uint32_t k) {
        gpu::DsparkGemvPush p{}; p.m = m; p.rows = rows; p.k = k; p.scale_cols = k / 32;
        p.x_stride = k; p.y_stride = rows; p.eps = 1e-20f; return p;
    };
    auto pack = [](const auto& p) { std::vector<uint8_t> v(sizeof p); std::memcpy(v.data(), &p, sizeof p); return v; };
    auto gemv = [&](uint32_t s, const std::string& w, uint64_t x, uint64_t y, uint32_t m, uint32_t rows, uint32_t k) {
        uint64_t* sl = rn[s]->slots(gpu::DsparkStage::Gemv);
        sl[gpu::dkslot::kW] = A(w); sl[gpu::dkslot::kS] = S(w); sl[gpu::dkslot::kX] = x; sl[gpu::dkslot::kY] = y;
        return Disp{gpu::DsparkStage::Gemv, pack(gemv_push(m, rows, k)), rn[s]->gemv_groups(rows)};
    };
    auto norm = [&](uint32_t s, const std::string& w, uint64_t x, uint64_t y, uint32_t m, uint32_t n) {
        uint64_t* sl = rn[s]->slots(gpu::DsparkStage::RmsNorm);
        sl[gpu::dkslot::kNormX] = x; sl[gpu::dkslot::kNormW] = A(w); sl[gpu::dkslot::kNormY] = y;
        gpu::DsparkGemvPush p{}; p.m = m; p.k = n; p.x_stride = n; p.y_stride = n; p.eps = 1e-20f;
        p.flags = gpu::kDsFlagRoundIn;
        return Disp{gpu::DsparkStage::RmsNorm, pack(p), rn[s]->norm_groups(m)};
    };
    auto rope = [&](uint32_t s, uint64_t x, uint64_t tab, uint64_t y, uint32_t m, uint32_t rpp, bool inv, bool q, uint32_t fl) {
        uint64_t* sl = rn[s]->slots(gpu::DsparkStage::RopeQuant);
        sl[gpu::dkslot::kRopeX] = x; sl[gpu::dkslot::kRopeTab] = tab; sl[gpu::dkslot::kRopeY] = y;
        gpu::DsparkGemvPush p{}; p.m = m; p.rows = rpp; p.k = kHeadDim; p.x_stride = rpp * kHeadDim;
        p.y_stride = rpp * kHeadDim; p.rope_dim = kRope; p.inverse = inv; p.quant = q; p.flags = fl;
        return Disp{gpu::DsparkStage::RopeQuant, pack(p), gpu::DsparkRunner::rope_groups(m, rpp, kHeadDim)};
    };
    auto attn = [&](uint32_t s, gpu::DsparkStage st) {
        uint64_t* sl = rn[s]->slots(st);
        sl[gpu::dkslot::kAttnQ] = Qr.addr; sl[gpu::dkslot::kAttnKv] = KvRing.addr;
        sl[gpu::dkslot::kAttnTopIdx] = Idx.addr; sl[gpu::dkslot::kAttnSink] = Sink.addr;
        sl[gpu::dkslot::kAttnScore] = Score.addr; sl[gpu::dkslot::kAttnO] = O16.addr;
        gpu::DsparkAttnPush p{}; p.m = kM; p.n_kv = kNIdx; p.n_heads = kHeads; p.head_dim = kHeadDim;
        p.score_stride = 256; p.q_stride = 32768; p.softmax_scale = 1.0f / std::sqrt(512.0f);
        return Disp{st, pack(p), rn[s]->attn_groups(kHeads)};
    };
    auto woa = [&](uint32_t s, const std::string& w) {
        uint64_t* sl = rn[s]->slots(gpu::DsparkStage::WoA);
        sl[gpu::dkslot::kW] = A(w); sl[gpu::dkslot::kS] = S(w); sl[gpu::dkslot::kX] = Oinv.addr; sl[gpu::dkslot::kY] = Woa.addr;
        gpu::DsparkGemvPush p{}; p.m = kM; p.rows = kGroups * kOLora; p.k = 4096; p.scale_cols = 128;
        p.x_stride = 32768; p.y_stride = kGroups * kOLora; p.rows_per_group = kOLora;
        return Disp{gpu::DsparkStage::WoA, pack(p), rn[s]->gemv_groups(kGroups * kOLora)};
    };
    auto markov = [&](uint32_t pos) {
        uint64_t* sl = rn[kHeadRunner]->slots(gpu::DsparkStage::MarkovBias);
        sl[gpu::dkslot::kMkW] = A("mtp.2.markov_head.head.weight");
        sl[gpu::dkslot::kMkEmbed] = A("mtp.2.markov_head.embed.weight");
        sl[gpu::dkslot::kMkTokenIn] = Ids.addr; sl[gpu::dkslot::kMkBias] = Bias.addr; sl[gpu::dkslot::kMkEmbedOut] = Emb.addr;
        gpu::DsparkHeadPush p{}; p.m = kM; p.rows = kVocab; p.k = kRank; p.rank = kRank; p.pos = pos;
        p.logit_stride = kVocab; p.flags = gpu::kDsFlagTokenFromBuf;
        return Disp{gpu::DsparkStage::MarkovBias, pack(p), rn[kHeadRunner]->gemv_groups(kVocab)};
    };
    auto argmax = [&](uint32_t pos) {
        uint64_t* sl = rn[kHeadRunner]->slots(gpu::DsparkStage::AddBiasArgmax);
        sl[gpu::dkslot::kAbLogits] = Logits.addr; sl[gpu::dkslot::kAbBias] = Bias.addr;
        sl[gpu::dkslot::kAbSample] = Sample.addr; sl[gpu::dkslot::kAbOutIds] = Ids.addr;
        gpu::DsparkHeadPush p{}; p.m = kM; p.rows = kVocab; p.pos = pos; p.logit_stride = kVocab;
        return Disp{gpu::DsparkStage::AddBiasArgmax, pack(p), 1u};
    };
    auto conf = [&]() {
        uint64_t* sl = rn[kHeadRunner]->slots(gpu::DsparkStage::Confidence);
        sl[gpu::dkslot::kCfX] = Xd.addr; sl[gpu::dkslot::kCfEmbed] = Emb.addr;
        sl[gpu::dkslot::kCfProj] = A("mtp.2.confidence_head.proj.weight"); sl[gpu::dkslot::kCfOut] = Conf.addr;
        gpu::DsparkHeadPush p{}; p.m = kM; p.k = kDim; p.rank = kRank; p.x_stride = kDim;
        return Disp{gpu::DsparkStage::Confidence, pack(p), 1u};
    };

    auto p = [](uint32_t s, const char* t) { return std::format("mtp.{}.attn.{}", s, t); };

    gpu::CommandPool pool;
    if (auto r = pool.create(device); !r) { std::printf("pool: %s\n", r.error().str().c_str()); io.stop(); return 1; }
    gpu::QueryPool queries;
    const bool timed = queries.create(device, 2).has_value() && device.caps().timestamp_valid_bits > 0;

    // Records `n` repetitions of `make()` (which rewrites slots and returns the
    // dispatch list) into one command buffer; best of three resubmits.
    auto time_it = [&](const std::string& name, uint64_t bytes, uint32_t n,
                       const std::function<std::vector<std::pair<uint32_t, Disp>>(uint32_t)>& make) -> Row {
        // warm
        for (auto& [r, d] : make(0)) (void)rn[r]->dispatch_now(d.st, d.push.data(), uint32_t(d.push.size()), d.groups);
        auto cb = pool.acquire();
        if (!cb) return Row{name, 0, bytes};
        gpu::CommandBuffer cmd = *cb;
        (void)cmd.begin();
        if (timed) { (void)cmd.reset_queries(queries, 0, 2); (void)cmd.write_timestamp(queries, 0, false); }
        // NOTE: slots are host-visible and read at execution time, so every
        // iteration of one row must use the same slot contents. Rows that cycle
        // weights do so by cycling RUNNERS, whose tables are independent.
        for (uint32_t i = 0; i < n; ++i) {
            for (auto& [r, d] : make(i)) {
                (void)rn[r]->record(cmd, d.st, d.push.data(), uint32_t(d.push.size()), d.groups);
                (void)cmd.barrier();
            }
        }
        if (timed) (void)cmd.write_timestamp(queries, 1, true);
        (void)cmd.end();
        double best = 1e9;
        for (uint32_t rep = 0; rep < 3; ++rep) {
            if (auto r = gpu::submit_and_wait(device, cmd); !r) return Row{name, 0, bytes};
            if (timed) if (auto s = queries.elapsed_seconds(0, 1); s) best = std::min(best, *s * 1e3 / n);
        }
        return Row{name, timed ? best : 0.0, bytes};
    };

    std::vector<Row> rows;
    auto one = [&](Disp d, uint32_t r) { return std::vector<std::pair<uint32_t, Disp>>{{r, std::move(d)}}; };
    const uint32_t N = o.iters;
    // Slots for runner s are written once per row (i = 0..2 cover all three
    // runners); make(i) for i >= 3 re-uses runner i % 3, whose slots are already
    // set -- the lambdas below rewrite identical values, which is harmless.
    rows.push_back(time_it("main_proj (K=15360, M=1)", fp8_bytes(kDim, 15360), N, [&](uint32_t) {
        return one(gemv(0, "mtp.0.main_proj.weight", X15.addr, Y5120.addr, 1, kDim, 15360), 0); }));
    rows.push_back(time_it("main_norm", 0, N, [&](uint32_t) {
        return one(norm(0, "mtp.0.main_norm.weight", Y5120.addr, Xd.addr, 1, kDim), 0); }));
    rows.push_back(time_it("wq_a (M=5)", fp8_bytes(kQLora, kDim), N, [&](uint32_t i) {
        uint32_t s = i % 3; return one(gemv(s, p(s, "wq_a.weight"), Xd.addr, Qa.addr, kM, kQLora, kDim), s); }));
    rows.push_back(time_it("q_norm (M=5)", 0, N, [&](uint32_t i) {
        uint32_t s = i % 3; return one(norm(s, p(s, "q_norm.weight"), Qa.addr, Qn.addr, kM, kQLora), s); }));
    rows.push_back(time_it("wq_b (M=5)", fp8_bytes(32768, kQLora), N, [&](uint32_t i) {
        uint32_t s = i % 3; return one(gemv(s, p(s, "wq_b.weight"), Qn.addr, Qb.addr, kM, 32768, kQLora), s); }));
    rows.push_back(time_it("rope(q) (M=5)", 0, N, [&](uint32_t i) {
        uint32_t s = i % 3; return one(rope(s, Qb.addr, TabD.addr, Oinv.addr, kM, kHeads, false, false, 0), s); }));
    rows.push_back(time_it("wkv (M=1, main_x)", fp8_bytes(kHeadDim, kDim), N, [&](uint32_t i) {
        uint32_t s = i % 3; return one(gemv(s, p(s, "wkv.weight"), Y5120.addr, Kv1.addr, 1, kHeadDim, kDim), s); }));
    rows.push_back(time_it("wkv (M=5)", fp8_bytes(kHeadDim, kDim), N, [&](uint32_t i) {
        uint32_t s = i % 3; return one(gemv(s, p(s, "wkv.weight"), Xd.addr, Kvd.addr, kM, kHeadDim, kDim), s); }));
    rows.push_back(time_it("kv_norm (M=5)", 0, N, [&](uint32_t i) {
        uint32_t s = i % 3; return one(norm(s, p(s, "kv_norm.weight"), Kvd.addr, Kvdn.addr, kM, kHeadDim), s); }));
    rows.push_back(time_it("rope+quant(kv) (M=5)", 0, N, [&](uint32_t i) {
        uint32_t s = i % 3; return one(rope(s, Kvdn.addr, TabD.addr, Kvd.addr, kM, 1, false, true, 0), s); }));
    rows.push_back(time_it("attn.score (M=5, 70 kv)", 2ull * kNKv * kHeadDim, N, [&](uint32_t i) {
        uint32_t s = i % 3; return one(attn(s, gpu::DsparkStage::AttnScore), s); }));
    rows.push_back(time_it("attn.combine (M=5)", 2ull * kNKv * kHeadDim, N, [&](uint32_t i) {
        uint32_t s = i % 3; return one(attn(s, gpu::DsparkStage::AttnCombine), s); }));
    rows.push_back(time_it("inverse rope (M=5)", 0, N, [&](uint32_t i) {
        uint32_t s = i % 3; return one(rope(s, O16.addr, TabD.addr, Oinv.addr, kM, kHeads, true, false, gpu::kDsFlagInBf16), s); }));
    rows.push_back(time_it("wo_a (M=5)", fp8_bytes(kGroups * kOLora, 4096), N, [&](uint32_t i) {
        uint32_t s = i % 3; return one(woa(s, p(s, "wo_a.weight")), s); }));
    rows.push_back(time_it("wo_b (M=5)", fp8_bytes(kDim, kGroups * kOLora), N, [&](uint32_t i) {
        uint32_t s = i % 3; return one(gemv(s, p(s, "wo_b.weight"), Woa.addr, Wob.addr, kM, kDim, kGroups * kOLora), s); }));
    rows.push_back(time_it("markov bias (1 position)", 2ull * kVocab * kRank, N, [&](uint32_t i) {
        return one(markov(i % kM), kHeadRunner); }));
    rows.push_back(time_it("bias add + argmax", 0, N, [&](uint32_t i) {
        return one(argmax(i % kM), kHeadRunner); }));
    rows.push_back(time_it("confidence (M=5)", 2ull * (kDim + kRank), N, [&](uint32_t) {
        return one(conf(), kHeadRunner); }));

    // ---- the whole chain of one draft cycle, back to back -----------------
    auto cycle = [&](uint32_t) {
        std::vector<std::pair<uint32_t, Disp>> v;
        const uint32_t H = kHeadRunner;
        v.push_back({H, gemv(H, "mtp.0.main_proj.weight", X15.addr, Y5120.addr, 1, kDim, 15360)});
        v.push_back({H, norm(H, "mtp.0.main_norm.weight", Y5120.addr, Xd.addr, 1, kDim)});
        for (uint32_t s = 0; s < kStages; ++s) {
            auto R = [&](uint32_t j) { return 5 * s + j; };
            // Q
            v.push_back({R(0), gemv(R(0), p(s, "wq_a.weight"), Xd.addr, Qa.addr, kM, kQLora, kDim)});
            v.push_back({R(0), norm(R(0), p(s, "q_norm.weight"), Qa.addr, Qn.addr, kM, kQLora)});
            v.push_back({R(1), gemv(R(1), p(s, "wq_b.weight"), Qn.addr, Qb.addr, kM, 32768, kQLora)});
            v.push_back({R(0), rope(R(0), Qb.addr, TabD.addr, Oinv.addr, kM, kHeads, false, false, 0)});
            // main_x KV (the ring write)
            v.push_back({R(2), gemv(R(2), p(s, "wkv.weight"), Y5120.addr, Kv1.addr, 1, kHeadDim, kDim)});
            v.push_back({R(1), norm(R(1), p(s, "kv_norm.weight"), Kv1.addr, Kv1n.addr, 1, kHeadDim)});
            v.push_back({R(1), rope(R(1), Kv1n.addr, TabM.addr, Kv1.addr, 1, 1, false, true, 0)});
            // draft KV
            v.push_back({R(3), gemv(R(3), p(s, "wkv.weight"), Xd.addr, Kvd.addr, kM, kHeadDim, kDim)});
            v.push_back({R(2), norm(R(2), p(s, "kv_norm.weight"), Kvd.addr, Kvdn.addr, kM, kHeadDim)});
            v.push_back({R(2), rope(R(2), Kvdn.addr, TabD.addr, Kvd.addr, kM, 1, false, true, 0)});
            // attention, un-rope, O
            v.push_back({R(0), attn(R(0), gpu::DsparkStage::AttnScore)});
            v.push_back({R(0), attn(R(0), gpu::DsparkStage::AttnCombine)});
            v.push_back({R(3), rope(R(3), O16.addr, TabD.addr, Oinv.addr, kM, kHeads, true, false, gpu::kDsFlagInBf16)});
            v.push_back({R(0), woa(R(0), p(s, "wo_a.weight"))});
            v.push_back({R(4), gemv(R(4), p(s, "wo_b.weight"), Woa.addr, Wob.addr, kM, kDim, kGroups * kOLora)});
        }
        for (uint32_t i = 0; i < kM; ++i) { v.push_back({H, markov(i)}); v.push_back({H, argmax(i)}); }
        v.push_back({H, conf()});
        return v;
    };
    {
        uint64_t b = fp8_bytes(kDim, 15360) + 5ull * 2 * kVocab * kRank + 2ull * (kDim + kRank);
        b += 3 * (fp8_bytes(kQLora, kDim) + fp8_bytes(32768, kQLora) + 2 * fp8_bytes(kHeadDim, kDim) +
                  fp8_bytes(kGroups * kOLora, 4096) + fp8_bytes(kDim, kGroups * kOLora));
        rows.push_back(time_it("CYCLE: every DSpark-kernel dispatch of one draft", b, N, cycle));
    }

    // ---- report -------------------------------------------------------------
    std::printf("\n%-62s %10s %12s %8s\n", "kernel", "ms", "bytes", "GB/s");
    auto ms_of = [&](const std::string& n) { for (auto& r : rows) if (r.name == n) return r.ms; return 0.0; };
    for (const Row& r : rows)
        std::printf("%-62s %10.3f %12llu %8.1f\n", r.name.c_str(), r.ms,
                    static_cast<unsigned long long>(r.bytes), r.gbps());
    const double per_stage =
        ms_of("wq_a (M=5)") + ms_of("q_norm (M=5)") + ms_of("wq_b (M=5)") + ms_of("rope(q) (M=5)") +
        ms_of("wkv (M=1, main_x)") + ms_of("kv_norm (M=5)") * (1.0 / 1.0) + ms_of("rope+quant(kv) (M=5)") +
        ms_of("wkv (M=5)") + ms_of("kv_norm (M=5)") + ms_of("rope+quant(kv) (M=5)") +
        ms_of("attn.score (M=5, 70 kv)") + ms_of("attn.combine (M=5)") + ms_of("inverse rope (M=5)") +
        ms_of("wo_a (M=5)") + ms_of("wo_b (M=5)");
    const double tail = 5 * (ms_of("markov bias (1 position)") + ms_of("bias add + argmax")) +
                        ms_of("confidence (M=5)");
    const double front = ms_of("main_proj (K=15360, M=1)") + ms_of("main_norm");
    std::printf("\nsum of per-stage rows: main %.3f + 3 x %.3f + head tail %.3f = %.3f ms "
                "(DSpark kernels only; add head GEMV, 3x(mHC x2, gate, MoE M=5))\n",
                front, per_stage, tail, front + 3 * per_stage + tail);

    if (!o.csv.empty()) {
        if (std::FILE* f = std::fopen(o.csv.c_str(), "w")) {
            std::fprintf(f, "kernel,lanes,iters,ms,bytes,gbps,note\n");
            for (const Row& r : rows)
                std::fprintf(f, "\"%s\",%u,%u,%.4f,%llu,%.2f,\"%s\"\n", r.name.c_str(), o.lanes, N, r.ms,
                             static_cast<unsigned long long>(r.bytes), r.gbps(), o.note.c_str());
            std::fprintf(f, "\"sum: main + 3 x stage + head tail\",%u,%u,%.4f,0,0,\"%s\"\n", o.lanes, N,
                         front + 3 * per_stage + tail, o.note.c_str());
            std::fclose(f);
            std::printf("-> %s\n", o.csv.c_str());
        }
    }
    scratch.destroy();
    for (auto& r : rn) r->destroy();
    queries.destroy();
    pool.destroy();
    pinned.reset();
    io.stop();
    return 0;
}
