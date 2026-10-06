#include "core/env.h"
#include "core/wc_read.h"
#include "cpu/dequant.h"
// The design §7.12 DSpark draft kernels against tools/oracle_dspark.py's golden
// tensors, on the real checkpoint.
//
// What is checked
// ---------------
// One draft cycle at the L3 prompt's first decode position (main position 64,
// draft positions 65..69), exported per stage by `tools/oracle_dspark.py` from
// the reference's own `DSparkBlock` / `DSparkAttention` / `DSparkMarkovHead` /
// `DSparkConfidenceHead` behind tools/dsref.py's CPU kernel shims. Every stage of
// gpu/vulkan/dspark_kernels.h is fed the reference's own input and compared with
// the reference's own output, so a failure names one kernel:
//
//   main_proj (fp8 GEMV K=15360, M=1) -> main_norm
//   per mtp stage:  wq_a -> q_norm -> wq_b -> RoPE(q)
//                   wkv(main_x, M=1) -> kv_norm -> RoPE + act_quant  (the ring write)
//                   wkv(x, M=5)      -> kv_norm -> RoPE + act_quant
//                   attention(score, combine) -> inverse RoPE -> wo_a -> wo_b
//   head tail:      Markov bias x5 (sequential) -> bias add + argmax -> confidence
//
// Stages 1 and 2 have no golden attention output (the export keeps the two
// 327 KB attention tensors for stage 0 only, to stay under the 5 MB data
// budget), so there the attention is CHAINED: GPU RoPE(q) -> GPU attention ->
// GPU inverse RoPE -> GPU wo_a, compared at wo_a's output. That doubles as the
// composition check.
//
// The head GEMV (129,280 x 5120 bf16, M = 5) is not a DSpark kernel -- it is
// head.slang's, which is M = 1 today (docs/p3_dspark.md §6) -- so the logits the
// Markov stages add their bias to are computed here on the CPU from the golden
// `head_norm_out` (bf16 table decode, fp32 partial sums of 256 folded into fp64).
//
// Criterion: cosine >= 0.9999 per stage (Track K's bar) plus relative L2, the
// same pair tests/test_gpu_attn.cpp uses and for the same reason: the reference
// is bf16 and E4M3, so max-element error measures rounding boundaries, not
// kernels. The draft TOKENS must match exactly.
//
// Gated on CACHEDMOE_MODEL_DIR, tests/data/dspark, and a working Vulkan device.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>
#include <thread>
#include <vector>

#include "core/config.h"
#include "gpu/vulkan/attn_kernels.h"     // GpuScratch
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/dspark_kernels.h"
#include "gpu/vulkan/memory.h"
#include "model/layout.h"
#include "model/manifest.h"
#include "runtime/rope.h"
#include "runtime/speculate.h"
#include "runtime/sampling.h"
#include "runtime/engine.h"
#include "storage/backend.h"
#include "storage/io_engine.h"
#include "store/pinned.h"
#include "store/shard_set.h"
#include "gpu/vulkan/moe_kernels.h"      // default_shader_dir
#include "tests/l1_golden.h"              // model_dir / skip_without_model
#include "tests/l2_golden.h"              // load_l2 / agree -- the index shape is shared
#include "tests/test_framework.h"

#ifndef CACHEDMOE_TEST_DATA_DIR
#define CACHEDMOE_TEST_DATA_DIR "tests/data"
#endif

using namespace cachedmoe;
using namespace cachedmoe::testing;

namespace {

std::string ds_dir() { return std::string(CACHEDMOE_TEST_DATA_DIR) + "/dspark"; }

constexpr uint32_t kDim = 5120, kHeads = 64, kHeadDim = 512, kRope = 64;
constexpr uint32_t kQLora = 1280, kOLora = 1024, kGroups = 8, kWin = 128;
constexpr uint32_t kVocab = 129280, kRank = 256, kM = 5, kStages = 3;
constexpr uint32_t kCtxTargets = 3;              // layers 37, 38, 39
constexpr float    kEps = 1e-20f;

// Every DSpark weight a draft cycle reads, per mtp stage, plus the main head
// for the CPU logits. store::pinned_layer_tensors only knows `layers.N`.
std::vector<std::string> dspark_tensors() {
    std::vector<std::string> out;
    for (uint32_t s = 0; s < kStages; ++s) {
        const std::string p = std::format("mtp.{}", s);
        for (const char* t : {"attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight",
                              "attn.wo_a.weight", "attn.wo_b.weight", "attn.attn_sink",
                              "attn.q_norm.weight", "attn.kv_norm.weight"})
            out.push_back(std::format("{}.{}", p, t));
    }
    for (const char* t : {"mtp.0.main_proj.weight", "mtp.0.main_norm.weight",
                          "mtp.2.norm.weight", "mtp.2.markov_head.embed.weight",
                          "mtp.2.markov_head.head.weight",
                          "mtp.2.confidence_head.proj.weight", "head.weight"})
        out.push_back(t);
    return out;
}

struct Rig {
    gpu::Device           device;
    gpu::MemoryAllocator  alloc;
    Manifest              manifest;
    store::ShardSet       shards;
    storage::IoEngine     io;
    store::PinnedStore    pinned;
    gpu::DsparkRunner     runner;
    gpu::GpuScratch       scratch;
    bool                  io_started = false;
    std::string           why;

    // Same teardown order as tests/test_gpu_attn.cpp's Rig, for the same
    // reason: PinnedStore's regions outlive vkDestroyDevice unless reset.
    ~Rig() {
        scratch.destroy();
        runner.destroy();
        pinned.reset();
        if (io_started) io.stop();
    }

    bool bring_up() {
        const std::string dir = model_dir() ? model_dir() : "";
        gpu::DeviceOptions dopts;
        dopts.enable_validation = std::getenv("VK_INSTANCE_LAYERS") != nullptr;
        if (auto r = device.create(dopts); !r) { why = r.error().str(); return false; }
        if (auto r = device.caps().check_required(); !r) { why = r.error().str(); return false; }
        if (auto r = alloc.init(device, MemoryPath::DeviceLocalHostVisible); !r) {
            why = r.error().str(); return false;
        }
        auto mf = Manifest::load(store::ShardSet::join(dir, layout::kManifestFile));
        if (!mf) { why = mf.error().str(); return false; }
        manifest = std::move(*mf);
        if (auto r = shards.open_all(dir, manifest, true); !r) { why = r.error().str(); return false; }
        IoConfig cfg;
        auto backend = storage::make_default_backend(cfg);
        if (!backend) { why = backend.error().str(); return false; }
        if (auto r = io.start(std::move(*backend), cfg); !r) { why = r.error().str(); return false; }
        io_started = true;
        auto backing = alloc.make_slab_backing();
        if (!backing) { why = backing.error().str(); return false; }
        store::PinnedConfig pc;
        pc.region_bytes = 512ull << 20;
        if (auto r = pinned.init(std::move(*backing), pc); !r) { why = r.error().str(); return false; }
        if (auto r = runner.create(device, alloc, gpu::default_shader_dir()); !r) {
            why = r.error().str(); return false;
        }
        if (auto r = scratch.create(alloc, 64ull << 20); !r) { why = r.error().str(); return false; }
        if (auto r = pinned.load(manifest, shards, io, dspark_tensors()); !r) {
            why = r.error().str(); return false;
        }
        return true;
    }

    const store::PinnedTensor* t(const std::string& n) { return pinned.find(n); }
    uint64_t addr(const std::string& n) { auto* p = t(n); return p ? p->data : 0; }
    uint64_t scale_addr(const std::string& n) { auto* p = t(n); return p ? p->scale : 0; }
};

struct Buf {
    gpu::GpuScratch::View v{};
    uint64_t a() const { return v.addr; }
    void set(const std::vector<float>& src) { std::memcpy(v.host, src.data(), src.size() * 4); }
    void set_bf16(const std::vector<float>& src) {
        auto* p = static_cast<uint16_t*>(v.host);
        for (size_t i = 0; i < src.size(); ++i) p[i] = f32_to_bf16(src[i]);
    }
    void set_i32(const std::vector<float>& src) {
        auto* p = static_cast<int32_t*>(v.host);
        for (size_t i = 0; i < src.size(); ++i) p[i] = static_cast<int32_t>(src[i]);
    }
    std::vector<float> read(size_t n) const {
        std::vector<float> out(n);
        std::memcpy(out.data(), v.host, n * 4);
        return out;
    }
    std::vector<float> read_bf16(size_t n) const {
        std::vector<float> out(n);
        const auto* p = static_cast<const uint16_t*>(v.host);
        for (size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
        return out;
    }
    uint32_t* u32() { return static_cast<uint32_t*>(v.host); }

    // bf16 is IEEE half with the fp32 exponent width: the top 16 bits of the fp32
    // word, rounded to nearest-even on the dropped 16. The golden values are
    // already bf16, so this is exact for everything the test writes.
    static uint16_t f32_to_bf16(float f) {
        uint32_t b;
        std::memcpy(&b, &f, 4);
        uint32_t hi = b >> 16;
        const uint32_t lo = b & 0xFFFFu;
        if (lo > 0x8000u || (lo == 0x8000u && (hi & 1u))) ++hi;
        return static_cast<uint16_t>(hi);
    }
};

Buf take(gpu::GpuScratch& s, uint64_t bytes) {
    Buf b;
    auto v = s.alloc(bytes);
    if (v) b.v = *v;
    return b;
}

bool ok(const char* what, const Agreement& g, double cos_min = 0.9999, double rel_l2_max = 2e-2) {
    const bool pass = g.cos >= cos_min && g.rel_l2 <= rel_l2_max && std::isfinite(g.max_abs);
    std::printf("      %-26s %s%s\n", what, g.str().c_str(), pass ? "" : "   <-- FAIL");
    return pass;
}

// [n_pos][32] (cos, sin), positions pos0..pos0+n-1: rope_theta 10000 and YaRN
// off, which is what every compress_ratio == 0 layer -- the mtp stages included --
// uses (design §2.4 v0.7).
std::vector<float> rope_rows(uint32_t pos0, uint32_t n) {
    const auto cfg = runtime::rope_for_layer(0, kRope);
    std::vector<float> out;
    for (uint32_t i = 0; i < n; ++i) {
        auto r = runtime::rope_table(cfg, pos0 + i);
        out.insert(out.end(), r.begin(), r.end());
    }
    return out;
}

// fp8 GEMV: kW/kS from the pinned tensor, x fp32 [m][k], y fp32 [m][rows].
Result<void> gemv(Rig& rig, const std::string& w, Buf& x, Buf& y, uint32_t m,
                  uint32_t rows, uint32_t k) {
    uint64_t* s = rig.runner.slots(gpu::DsparkStage::Gemv);
    s[gpu::dkslot::kW] = rig.addr(w);
    s[gpu::dkslot::kS] = rig.scale_addr(w);
    s[gpu::dkslot::kX] = x.a();
    s[gpu::dkslot::kY] = y.a();
    gpu::DsparkGemvPush p{};
    p.m = m; p.rows = rows; p.k = k; p.scale_cols = k / 32; p.row_base = 0;
    p.x_stride = k; p.y_stride = rows; p.eps = kEps;
    return rig.runner.dispatch_now(gpu::DsparkStage::Gemv, &p, sizeof p,
                                   rig.runner.gemv_groups(rows));
}

Result<void> rmsnorm(Rig& rig, const std::string& w, Buf& x, Buf& y, uint32_t m, uint32_t n) {
    uint64_t* s = rig.runner.slots(gpu::DsparkStage::RmsNorm);
    s[gpu::dkslot::kNormX] = x.a();
    s[gpu::dkslot::kNormW] = rig.addr(w);
    s[gpu::dkslot::kNormY] = y.a();
    gpu::DsparkGemvPush p{};
    p.m = m; p.k = n; p.x_stride = n; p.y_stride = n; p.eps = kEps;
    // The reference hands RMSNorm a bf16 tensor (every Linear output is bf16).
    p.flags = gpu::kDsFlagRoundIn;
    return rig.runner.dispatch_now(gpu::DsparkStage::RmsNorm, &p, sizeof p,
                                   rig.runner.norm_groups(m));
}

Result<void> rope(Rig& rig, Buf& x, Buf& tab, Buf& y, uint32_t m, uint32_t rows_per_pos,
                  bool inverse, bool quant, uint32_t flags = 0) {
    uint64_t* s = rig.runner.slots(gpu::DsparkStage::RopeQuant);
    s[gpu::dkslot::kRopeX] = x.a();
    s[gpu::dkslot::kRopeTab] = tab.a();
    s[gpu::dkslot::kRopeY] = y.a();
    gpu::DsparkGemvPush p{};
    p.m = m; p.rows = rows_per_pos; p.k = kHeadDim;
    p.x_stride = rows_per_pos * kHeadDim; p.y_stride = rows_per_pos * kHeadDim;
    p.rope_dim = kRope; p.inverse = inverse ? 1u : 0u; p.quant = quant ? 1u : 0u;
    p.flags = flags;
    return rig.runner.dispatch_now(gpu::DsparkStage::RopeQuant, &p, sizeof p,
                                   gpu::DsparkRunner::rope_groups(m, rows_per_pos, kHeadDim));
}

}  // namespace

// ---------------------------------------------------------------------------

CACHEDMOE_TEST(gpu_dspark, golden_per_stage) {
    if (skip_without_model("gpu_dspark")) return;
    auto set = load_l2(ds_dir());
    if (!set) {
        CACHEDMOE_SKIP_PRINTF("      SKIP gpu_dspark: no DSpark data (%s). Run "
                    "`tools/oracle_dspark.py --out tests/data/dspark`\n",
                    set.error().str().c_str());
        return;
    }
    const L2Step* gp = nullptr;
    for (const L2Step& s : set->steps) if (s.step == "golden_pos64") gp = &s;
    if (!gp) {
        CACHEDMOE_SKIP_PRINTF("      SKIP gpu_dspark: tests/data/dspark has no golden_pos64 record\n");
        return;
    }
    const L2Step& g = *gp;
    Rig rig;
    if (!rig.bring_up()) {
        CACHEDMOE_SKIP_PRINTF("      SKIP gpu_dspark: %s\n", rig.why.c_str());
        return;
    }
    const uint32_t main_pos = static_cast<uint32_t>(set->cfg("golden_pos", 64));
    const uint32_t pos = main_pos == 0 ? 64 : main_pos;   // index.json keeps it top-level
    std::printf("      draft cycle at main position %u, draft positions %u..%u, %s\n",
                pos, pos + 1, pos + kM, rig.runner.spec().name().c_str());

    auto& sc = rig.scratch;
    Buf bX   = take(sc, 16ull * kM * 32768);     // generic input, fp32 [5][<=32768]
    Buf bY   = take(sc, 16ull * kM * 32768);
    Buf bY2  = take(sc, 16ull * kM * 32768);
    Buf bTab = take(sc, 4ull * 32 * 2 * (kM + 1));
    Buf bQ   = take(sc, 2ull * kM * 32768);      // bf16 q
    Buf bKv  = take(sc, 2ull * 256 * kHeadDim);  // bf16 kv
    Buf bIdx = take(sc, 4ull * 256);
    Buf bSink = take(sc, 4ull * kHeads);
    Buf bScore = take(sc, 4ull * kM * kHeads * 256);
    Buf bO   = take(sc, 2ull * kM * 32768);      // bf16 attention output
    REQUIRE(bX.a() && bY.a() && bY2.a() && bQ.a() && bO.a() && bScore.a());

    const std::vector<float> tab_draft = rope_rows(pos + 1, kM);
    const std::vector<float> tab_main  = rope_rows(pos, 1);
    const std::vector<float>& main_x   = g.f("main_norm_out");

    // --- main_proj + main_norm (M = 1, K = 15360) -----------------------------
    {
        bX.set(g.f("main_proj_in"));
        REQUIRE_OK(gemv(rig, "mtp.0.main_proj.weight", bX, bY, 1, kDim, kDim * kCtxTargets));
        CHECK(ok("main_proj", agree(bY.read(kDim), g.f("main_proj_out"))));
        bX.set(g.f("main_proj_out"));
        REQUIRE_OK(rmsnorm(rig, "mtp.0.main_norm.weight", bX, bY, 1, kDim));
        CHECK(ok("main_norm", agree(bY.read(kDim), main_x)));
    }

    for (uint32_t st = 0; st < kStages; ++st) {
        const std::string p = std::format("mtp.{}.attn.", st);
        const std::string n = std::format("s{}.", st);
        auto G = [&](const char* s) -> const std::vector<float>& { return g.f(n + s); };
        std::printf("    -- mtp stage %u --\n", st);

        // Q path: wq_a -> q_norm -> wq_b -> RoPE
        bX.set(G("attn_norm"));
        REQUIRE_OK(gemv(rig, p + "wq_a.weight", bX, bY, kM, kQLora, kDim));
        CHECK(ok("wq_a", agree(bY.read(kM * kQLora), G("wq_a"))));
        bX.set(G("wq_a"));
        REQUIRE_OK(rmsnorm(rig, p + "q_norm.weight", bX, bY, kM, kQLora));
        CHECK(ok("q_norm", agree(bY.read(kM * kQLora), G("q_norm"))));
        bX.set(G("q_norm"));
        REQUIRE_OK(gemv(rig, p + "wq_b.weight", bX, bY, kM, kHeads * kHeadDim, kQLora));
        CHECK(ok("wq_b", agree(bY.read(kM * 32768), G("wq_b_prerope"))));

        bTab.set(tab_draft);
        bX.set(G("wq_b_prerope"));
        REQUIRE_OK(rope(rig, bX, bTab, bY, kM, kHeads, false, false));
        const std::vector<float> q_roped = bY.read(kM * 32768);
        if (st == 0) CHECK(ok("rope(q)", agree(q_roped, G("sparse_q"))));

        // main_x KV: the one ring write per committed main position
        bX.set(main_x);
        REQUIRE_OK(gemv(rig, p + "wkv.weight", bX, bY, 1, kHeadDim, kDim));
        CHECK(ok("wkv(main_x)", agree(bY.read(kHeadDim), G("wkv_main"))));
        bX.set(G("wkv_main"));
        REQUIRE_OK(rmsnorm(rig, p + "kv_norm.weight", bX, bY, 1, kHeadDim));
        CHECK(ok("kv_norm(main)", agree(bY.read(kHeadDim), G("kv_norm_main"))));
        bTab.set(tab_main);
        bX.set(G("kv_norm_main"));
        REQUIRE_OK(rope(rig, bX, bTab, bY, 1, 1, false, true));
        CHECK(ok("rope+quant(main_kv)", agree(bY.read(kHeadDim), G("main_kv"))));

        // draft KV
        bX.set(G("attn_norm"));
        REQUIRE_OK(gemv(rig, p + "wkv.weight", bX, bY, kM, kHeadDim, kDim));
        CHECK(ok("wkv(draft)", agree(bY.read(kM * kHeadDim), G("wkv_draft"))));
        bX.set(G("wkv_draft"));
        REQUIRE_OK(rmsnorm(rig, p + "kv_norm.weight", bX, bY, kM, kHeadDim));
        CHECK(ok("kv_norm(draft)", agree(bY.read(kM * kHeadDim), G("kv_norm_draft"))));
        bTab.set(tab_draft);
        bX.set(G("kv_norm_draft"));
        REQUIRE_OK(rope(rig, bX, bTab, bY, kM, 1, false, true));
        CHECK(ok("rope+quant(draft_kv)", agree(bY.read(kM * kHeadDim), G("draft_kv"))));

        // attention: stage 0 from the golden q, stages 1-2 chained from our RoPE
        {
            const L2Tensor* it = g.find(n + "sparse_idx");
            REQUIRE(it != nullptr);
            const uint32_t n_idx = static_cast<uint32_t>(it->shape.back());
            std::vector<float> idx0(it->f.begin(), it->f.begin() + n_idx);
            bIdx.set_i32(idx0);
            bKv.set_bf16(G("sparse_kv"));
            bSink.set(G("attn_sink"));
            bQ.set_bf16(st == 0 ? G("sparse_q") : q_roped);

            gpu::DsparkAttnPush ap{};
            ap.m = kM; ap.n_kv = n_idx; ap.n_heads = kHeads; ap.head_dim = kHeadDim;
            ap.score_stride = 256; ap.q_stride = kHeads * kHeadDim;
            ap.softmax_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
            for (gpu::DsparkStage stg : {gpu::DsparkStage::AttnScore, gpu::DsparkStage::AttnCombine}) {
                uint64_t* s = rig.runner.slots(stg);
                s[gpu::dkslot::kAttnQ] = bQ.a();
                s[gpu::dkslot::kAttnKv] = bKv.a();
                s[gpu::dkslot::kAttnTopIdx] = bIdx.a();
                s[gpu::dkslot::kAttnSink] = bSink.a();
                s[gpu::dkslot::kAttnScore] = bScore.a();
                s[gpu::dkslot::kAttnO] = bO.a();
                REQUIRE_OK(rig.runner.dispatch_now(stg, &ap, sizeof ap,
                                                   rig.runner.attn_groups(kHeads)));
            }
            bTab.set(tab_draft);
            REQUIRE_OK(rope(rig, bO, bTab, bY2, kM, kHeads, true, false, gpu::kDsFlagInBf16));
            if (st == 0)
                CHECK(ok("attn + inverse rope", agree(bY2.read(kM * 32768), G("attn_out_inv"))));
        }

        // wo_a (fp8 bytes, bf16 arithmetic, no act_quant), chained from bY2
        {
            uint64_t* s = rig.runner.slots(gpu::DsparkStage::WoA);
            s[gpu::dkslot::kW] = rig.addr(p + "wo_a.weight");
            s[gpu::dkslot::kS] = rig.scale_addr(p + "wo_a.weight");
            s[gpu::dkslot::kX] = bY2.a();
            s[gpu::dkslot::kY] = bY.a();
            gpu::DsparkGemvPush wp{};
            wp.m = kM; wp.rows = kGroups * kOLora; wp.k = 32768 / kGroups;
            wp.scale_cols = wp.k / 32; wp.x_stride = 32768; wp.y_stride = kGroups * kOLora;
            wp.rows_per_group = kOLora;
            REQUIRE_OK(rig.runner.dispatch_now(gpu::DsparkStage::WoA, &wp, sizeof wp,
                                               rig.runner.gemv_groups(kGroups * kOLora)));
            CHECK(ok(st == 0 ? "wo_a" : "wo_a (chained attn)",
                     agree(bY.read(kM * kGroups * kOLora), G("wo_a_out"))));
        }

        bX.set(G("wo_a_out"));
        REQUIRE_OK(gemv(rig, p + "wo_b.weight", bX, bY, kM, kDim, kGroups * kOLora));
        CHECK(ok("wo_b", agree(bY.read(kM * kDim), G("wo_b"))));
    }

    // --- head tail --------------------------------------------------------------
    std::printf("    -- head tail --\n");
    const store::PinnedTensor* head = rig.t("head.weight");
    REQUIRE(head != nullptr && head->data_host != nullptr);
    const std::vector<float>& hn = g.f("head_norm_out");
    Buf bLogits = take(sc, 4ull * kM * kVocab);
    Buf bBias   = take(sc, 4ull * kVocab);
    Buf bEmb    = take(sc, 4ull * kM * kRank);
    Buf bSample = take(sc, 16);
    Buf bIds    = take(sc, 4ull * (kM + 1));
    Buf bConf   = take(sc, 4ull * kM);
    REQUIRE(bLogits.a() && bBias.a() && bEmb.a() && bSample.a() && bIds.a() && bConf.a());
    {
        // `ParallelHead`: F.linear(x.float(), weight_fp32), weight bf16 on disk.
        // 3.3e9 multiply-adds, so: a bf16 decode table, each row decoded once and
        // dotted against all five positions, and one thread per vocabulary chunk.
        static std::vector<float> lut;
        if (lut.empty()) { lut.resize(65536); for (uint32_t i = 0; i < 65536; ++i) lut[i] = bf16_to_f32(uint16_t(i)); }
        const auto* w = static_cast<const uint16_t*>(head->data_host);
        auto* L = static_cast<float*>(bLogits.v.host);
        const uint32_t nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
        std::vector<std::thread> th;
        for (uint32_t t = 0; t < nt; ++t) {
            th.emplace_back([&, t] {
                std::vector<float> row(kDim);
                const uint32_t lo = kVocab * t / nt, hi = kVocab * (t + 1) / nt;
                for (uint32_t v = lo; v < hi; ++v) {
                    const uint16_t* wr = w + uint64_t(v) * kDim;
                    for (uint32_t d = 0; d < kDim; ++d) row[d] = lut[wr[d]];
                    for (uint32_t m = 0; m < kM; ++m) {
                        const float* x = hn.data() + m * kDim;
                        double acc = 0.0;
                        float part = 0.0f;
                        for (uint32_t d = 0; d < kDim; ++d) {
                            part += row[d] * x[d];
                            if ((d & 255u) == 255u) { acc += part; part = 0.0f; }
                        }
                        L[uint64_t(m) * kVocab + v] = static_cast<float>(acc + part);
                    }
                }
            });
        }
        for (auto& x : th) x.join();
    }
    const std::vector<float>& ids = g.f("draft_ids");          // [6]: input + 5 drafts
    const std::vector<float>& top_ids = g.f("draft_logits.top_ids");
    const std::vector<float>& top_val = g.f("draft_logits.top_logits");
    const uint32_t topk = static_cast<uint32_t>(top_ids.size() / kM);
    bIds.u32()[0] = static_cast<uint32_t>(ids[0]);
    std::vector<uint32_t> got_ids;
    for (uint32_t i = 0; i < kM; ++i) {
        gpu::DsparkHeadPush hp{};
        hp.m = kM; hp.rows = kVocab; hp.k = kRank; hp.rank = kRank; hp.pos = i;
        hp.logit_stride = kVocab; hp.flags = gpu::kDsFlagTokenFromBuf;
        {
            uint64_t* s = rig.runner.slots(gpu::DsparkStage::MarkovBias);
            s[gpu::dkslot::kMkW] = rig.addr("mtp.2.markov_head.head.weight");
            s[gpu::dkslot::kMkEmbed] = rig.addr("mtp.2.markov_head.embed.weight");
            s[gpu::dkslot::kMkTokenIn] = bIds.a();
            s[gpu::dkslot::kMkBias] = bBias.a();
            s[gpu::dkslot::kMkEmbedOut] = bEmb.a();
            REQUIRE_OK(rig.runner.dispatch_now(gpu::DsparkStage::MarkovBias, &hp, sizeof hp,
                                               rig.runner.gemv_groups(kVocab)));
        }
        {
            uint64_t* s = rig.runner.slots(gpu::DsparkStage::AddBiasArgmax);
            s[gpu::dkslot::kAbLogits] = bLogits.a();
            s[gpu::dkslot::kAbBias] = bBias.a();
            s[gpu::dkslot::kAbSample] = bSample.a();
            s[gpu::dkslot::kAbOutIds] = bIds.a();
            REQUIRE_OK(rig.runner.dispatch_now(gpu::DsparkStage::AddBiasArgmax, &hp, sizeof hp,
                                               gpu::DsparkRunner::single_group()));
        }
        gpu::DsparkSampleOut so;
        std::memcpy(&so, bSample.v.host, sizeof so);
        got_ids.push_back(so.token);
        // the biased logits at the golden top-k ids
        const auto* L = static_cast<const float*>(bLogits.v.host) + uint64_t(i) * kVocab;
        std::vector<float> ours(topk), want(topk);
        for (uint32_t r = 0; r < topk; ++r) {
            ours[r] = L[static_cast<uint32_t>(top_ids[i * topk + r])];
            want[r] = top_val[i * topk + r];
        }
        const std::string what = std::format("draft pos {} logits@top{}", i, topk);
        CHECK(ok(what.c_str(), agree(ours, want)));
        std::printf("      draft pos %u: token %u (want %u) margin %.4f\n", i, so.token,
                    static_cast<uint32_t>(ids[i + 1]), so.margin());
        CHECK_EQ(so.token, static_cast<uint32_t>(ids[i + 1]));
        CHECK_EQ(so.rows, kVocab);
    }
    CHECK(ok("markov embeds", agree(bEmb.read(kM * kRank), g.f("markov_embed")), 0.9999999, 1e-6));
    {
        bX.set(g.f("head_norm_in"));
        gpu::DsparkHeadPush hp{};
        hp.m = kM; hp.k = kDim; hp.rank = kRank; hp.x_stride = kDim;
        uint64_t* s = rig.runner.slots(gpu::DsparkStage::Confidence);
        s[gpu::dkslot::kCfX] = bX.a();
        s[gpu::dkslot::kCfEmbed] = bEmb.a();
        s[gpu::dkslot::kCfProj] = rig.addr("mtp.2.confidence_head.proj.weight");
        s[gpu::dkslot::kCfOut] = bConf.a();
        REQUIRE_OK(rig.runner.dispatch_now(gpu::DsparkStage::Confidence, &hp, sizeof hp,
                                           gpu::DsparkRunner::single_group()));
        const std::vector<float> c = bConf.read(kM);
        const std::vector<float>& want = g.f("confidence");
        CHECK(ok("confidence", agree(c, want), 0.9999, 1e-3));
        for (uint32_t i = 0; i < kM; ++i)
            std::printf("      confidence[%u] %.6f (want %.6f)\n", i, c[i], want[i]);
    }
}

// ============================================================================
// dspark_verify: the four numbers exact acceptance needs, per verify row
// (docs/p4_dspark_runtime.md §2.4, docs/p3_dspark.md §3.3 rule 2).
//
// This one needs no checkpoint: the kernel reads a logit row out of a buffer and
// a candidate list out of another, so the row is synthesised here. What is
// checked, per row, against `runtime::emulate_verify_row` (the host mirror, with
// the same 256-lane partition and the same fixed-shape reduction tree):
//
//   * the candidates' logits -- pure gathers, so bit for bit;
//   * the full-vocabulary logsumexp -- the GPU's exp/log are its own, so a
//     relative tolerance, and the value is also checked against a float64
//     recomputation so a systematically wrong lse cannot hide inside agreement
//     between two copies of the same mistake;
//   * the two draws. Gumbel-max means the masked draw must never be a candidate
//     and the full draw may be; the token ids are compared with the mirror and a
//     mismatch is reported with the key margin, because two keys within a few
//     ULP of each other is a near-tie and not a kernel fault;
//   * the masked draw's DISTRIBUTION: over many seeds, the empirical frequency
//     against softmax restricted to the non-candidates. That is the property
//     `accept_sampling_exact` actually relies on, and it does not depend on the
//     mirror being right.
// ============================================================================
CACHEDMOE_TEST(gpu_dspark, verify_rows_give_exact_acceptance_its_four_numbers) {
    struct VRig {
        gpu::Device          device;
        gpu::MemoryAllocator alloc;
        gpu::DsparkRunner    runner;
        gpu::GpuScratch      scratch;
        std::string          why;
        ~VRig() { scratch.destroy(); runner.destroy(); }
        bool bring_up(uint32_t m) {
            gpu::DeviceOptions dopts;
            dopts.enable_validation = std::getenv("VK_INSTANCE_LAYERS") != nullptr;
            if (auto r = device.create(dopts); !r) { why = r.error().str(); return false; }
            if (auto r = device.caps().check_required(); !r) { why = r.error().str(); return false; }
            if (auto r = alloc.init(device, MemoryPath::DeviceLocalHostVisible); !r) {
                why = r.error().str(); return false;
            }
            gpu::DsparkSpec spec;
            spec.m = m;
            if (auto r = runner.create(device, alloc, gpu::default_shader_dir(), spec); !r) {
                why = r.error().str(); return false;
            }
            if (auto r = scratch.create(alloc, 64ull << 20); !r) { why = r.error().str(); return false; }
            return true;
        }
    };

    constexpr uint32_t kM = 6;
    constexpr uint32_t kVocab = 129280;
    constexpr uint32_t kK = 16;
    VRig rig;
    if (!rig.bring_up(kM)) {
        CACHEDMOE_SKIP_PRINTF("       SKIP gpu_dspark.verify_rows: %s\n", rig.why.c_str());
        return;
    }

    // Six rows that look like verify rows: a few dominant tokens and a long
    // tail, different per row, with the candidate list drawn from the row's own
    // top so the mask actually removes mass.
    std::vector<float> logits(size_t(kM) * kVocab);
    std::vector<int32_t> cand(size_t(kM) * gpu::kDsVerifyMaxCand, -1);
    for (uint32_t j = 0; j < kM; ++j) {
        float* row = logits.data() + size_t(j) * kVocab;
        for (uint32_t i = 0; i < kVocab; ++i) {
            const float t = float((i * 2654435761u + j * 40503u) % 100003u) / 100003.0f;
            row[i] = -6.0f + 4.0f * t - 0.000002f * float(i);
        }
        // A handful of peaks, so the row is not uniform and the top set is
        // well separated from the tail.
        for (uint32_t c = 0; c < 40; ++c)
            row[(c * 977u + j * 131u) % kVocab] += 6.0f + 0.15f * float(c);
        // The candidates: the row's top kK, best first, which is what the draft
        // lattice hands the verify batch.
        std::vector<uint32_t> idx(kVocab);
        for (uint32_t i = 0; i < kVocab; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + kK, idx.end(),
                          [&](uint32_t a, uint32_t b) {
                              return row[a] != row[b] ? row[a] > row[b] : a < b;
                          });
        for (uint32_t c = 0; c < kK; ++c) cand[size_t(j) * gpu::kDsVerifyMaxCand + c] =
            static_cast<int32_t>(idx[c]);
    }

    Buf bl = take(rig.scratch, uint64_t(kM) * kVocab * 4);
    Buf bc = take(rig.scratch, uint64_t(kM) * gpu::kDsVerifyMaxCand * 4);
    Buf bo = take(rig.scratch, uint64_t(kM) * gpu::kDsVerifyRecordWords * 4);
    REQUIRE(bl.v.host && bc.v.host && bo.v.host);
    std::memcpy(bl.v.host, logits.data(), logits.size() * 4);
    std::memcpy(bc.v.host, cand.data(), cand.size() * 4);
    std::memset(bo.v.host, 0, size_t(kM) * gpu::kDsVerifyRecordWords * 4);

    uint64_t* sl = rig.runner.slots(gpu::DsparkStage::VerifyRows);
    REQUIRE(sl != nullptr);
    sl[gpu::dvslot::kVLogits] = bl.a();
    sl[gpu::dvslot::kVCand]   = bc.a();
    sl[gpu::dvslot::kVOut]    = bo.a();

    const uint32_t seed = 0xA5A5C3C3u;
    gpu::DsparkVerifyPush push{kM, kVocab, kK, gpu::kDsVerifyMaxCand, kVocab, seed, 0, 1.0f};
    REQUIRE_OK(rig.runner.dispatch_now(gpu::DsparkStage::VerifyRows, &push, sizeof push,
                                       gpu::DsparkRunner::verify_groups(kM)));

    std::vector<gpu::DsparkVerifyRow> got(kM);
    std::memcpy(got.data(), bo.v.host, size_t(kM) * sizeof(gpu::DsparkVerifyRow));

    uint32_t token_mismatch = 0;
    for (uint32_t j = 0; j < kM; ++j) {
        const std::span<const float> row(logits.data() + size_t(j) * kVocab, kVocab);
        const std::span<const int32_t> cj(cand.data() + size_t(j) * gpu::kDsVerifyMaxCand, kK);
        gpu::DsparkVerifyRow want{};
        runtime::emulate_verify_row(row, cj, seed, j, 1.0f, want);

        CHECK(got[j].rows == kVocab);
        CHECK(got[j].n_cand == kK);

        // (a) the candidates' logits: gathers, so exactly equal.
        uint32_t cand_bad = 0;
        for (uint32_t c = 0; c < kK; ++c)
            if (got[j].cand_logit[c] != want.cand_logit[c]) ++cand_bad;
        CHECK(cand_bad == 0);

        // (b) the logsumexp, against the mirror AND against a float64 sum that
        //     shares nothing with either.
        double s64 = 0.0;
        double m64 = -1e300;
        for (uint32_t i = 0; i < kVocab; ++i) m64 = std::max(m64, double(row[i]));
        for (uint32_t i = 0; i < kVocab; ++i) s64 += std::exp(double(row[i]) - m64);
        const double lse64 = m64 + std::log(s64);
        const double d_mirror = std::fabs(double(got[j].lse) - double(want.lse));
        const double d_true   = std::fabs(double(got[j].lse) - lse64);
        CHECK(d_mirror <= 1e-4);
        CHECK(d_true <= 1e-3);

        // (c) the draws. The masked one must never be a candidate.
        bool masked_is_cand = false;
        for (uint32_t c = 0; c < kK; ++c)
            if (int32_t(got[j].masked_token) == cj[c]) masked_is_cand = true;
        CHECK(!masked_is_cand);
        if (got[j].masked_token != want.masked_token) ++token_mismatch;
        if (got[j].full_token != want.full_token) ++token_mismatch;

        std::printf("       row %u: lse gpu %.6f mirror %.6f f64 %.6f | masked %u/%u "
                    "full %u/%u | cand logits %s\n",
                    j, double(got[j].lse), double(want.lse), lse64,
                    got[j].masked_token, want.masked_token,
                    got[j].full_token, want.full_token,
                    cand_bad ? "DIFFER" : "identical");
        if (got[j].masked_token != want.masked_token || got[j].full_token != want.full_token)
            std::printf("         (near-tie: gpu keys %.7f/%.7f, mirror %.7f/%.7f)\n",
                        double(got[j].masked_key), double(got[j].full_key),
                        double(want.masked_key), double(want.full_key));
    }
    // A handful of near-ties over 776k Gumbel keys is the fp32 exp/log gap; a
    // systematic difference is not.
    CHECK(token_mismatch <= 2);

    // (d) the masked draw's distribution. A short row (so softmax over the
    //     complement is cheap and exact in float64) drawn many times with
    //     different seeds, against the analytic frequencies.
    {
        constexpr uint32_t kShort = 512;
        constexpr uint32_t kDraws = 4000;
        constexpr uint32_t kCandS = 8;
        std::vector<float> srow(kShort);
        for (uint32_t i = 0; i < kShort; ++i)
            srow[i] = 2.0f * std::sin(0.037f * float(i)) + 0.004f * float(i % 61);
        std::vector<int32_t> scand(gpu::kDsVerifyMaxCand, -1);
        for (uint32_t c = 0; c < kCandS; ++c) scand[c] = int32_t(c * 37 + 5);

        std::memcpy(bl.v.host, srow.data(), srow.size() * 4);
        std::memcpy(bc.v.host, scand.data(), scand.size() * 4);
        std::vector<uint32_t> hist(kShort, 0);
        for (uint32_t d = 0; d < kDraws; ++d) {
            gpu::DsparkVerifyPush p{1, kShort, kCandS, gpu::kDsVerifyMaxCand, kShort,
                                    0x1000u + d * 2654435761u, 0, 1.0f};
            REQUIRE_OK(rig.runner.dispatch_now(gpu::DsparkStage::VerifyRows, &p, sizeof p, 1));
            gpu::DsparkVerifyRow r{};
            std::memcpy(&r, bo.v.host, sizeof r);
            REQUIRE(r.masked_token < kShort);
            ++hist[r.masked_token];
        }
        double z = 0.0;
        std::vector<double> want_p(kShort, 0.0);
        for (uint32_t i = 0; i < kShort; ++i) {
            bool c = false;
            for (uint32_t s = 0; s < kCandS; ++s) if (int32_t(i) == scand[s]) c = true;
            if (c) continue;
            want_p[i] = std::exp(double(srow[i]));
            z += want_p[i];
        }
        double tv = 0.0, chi2 = 0.0, floor_tv = 0.0;
        uint32_t chi_cells = 0;
        for (uint32_t i = 0; i < kShort; ++i) {
            const double p = want_p[i] / z;
            tv += std::fabs(double(hist[i]) / double(kDraws) - p);
            // The noise floor an EXACT sampler still shows at this many draws:
            // E|hat p - p| = sqrt(2 p (1-p) / (pi n)) per cell, halved and summed.
            // docs/p3_dspark.md §13.3 reports its offline check the same way (TV
            // 0.0033 against a 0.0039 floor), because a bare TV over a 500-way
            // support says nothing on its own.
            floor_tv += std::sqrt(2.0 * p * (1.0 - p) / (3.14159265358979 * double(kDraws)));
            const double e = p * double(kDraws);
            if (e >= 5.0) { chi2 += (double(hist[i]) - e) * (double(hist[i]) - e) / e; ++chi_cells; }
        }
        tv *= 0.5;
        floor_tv *= 0.5;
        uint32_t drawn_cand = 0;
        for (uint32_t s = 0; s < kCandS; ++s) drawn_cand += hist[scand[s]];
        std::printf("       masked draw over %u draws: total variation %.4f against a %.4f "
                    "noise floor, chi2 %.1f over %u cells; candidates drawn %u times, "
                    "must be 0\n",
                    kDraws, tv, floor_tv, chi2, chi_cells, drawn_cand);
        for (uint32_t s = 0; s < kCandS; ++s) CHECK(hist[scand[s]] == 0);
        // Within a quarter of the floor is what an exact sampler looks like; a
        // biased one is a multiple of it.
        CHECK(tv <= 1.25 * floor_tv);
        // chi2 over `chi_cells` degrees of freedom: the far tail is roughly
        // cells + 4 sqrt(cells).
        CHECK(chi2 <= double(chi_cells) + 4.0 * std::sqrt(double(chi_cells)));
    }
}

CACHEDMOE_TEST(gpu_dspark, main_path_gpu_readout) {
    gpu::Device device;
    REQUIRE(device.create({}));
    gpu::MemoryAllocator alloc;
    REQUIRE(alloc.init(device, MemoryPath::DeviceLocalHostVisible));
    gpu::MgtRunner runner;
    REQUIRE(runner.create(device,alloc,gpu::default_shader_dir()));
    gpu::GpuScratch scratch;
    REQUIRE(scratch.create(alloc,16ull<<20));
    auto logits=take(scratch,6ull*kVocab*4);
    auto ids=take(scratch,6*4);
    auto ranks=take(scratch,6*sizeof(gpu::MgtRankOut));
    auto candidates=take(scratch,6ull*gpu::kMgtTopKRecordWords*4);
    auto hist=take(scratch,6ull*256*256*4);
    REQUIRE(logits.v.addr && ids.v.addr && ranks.v.addr && candidates.v.addr && hist.v.addr);
    const std::array<uint32_t,5> path{256,1023,5,17,32};
    std::memcpy(ids.v.host,path.data(),path.size()*4);
    auto* rp=runner.slots(gpu::MgtStage::HeadRank);
    rp[gpu::mslot::kHLogits]=logits.v.addr;
    rp[gpu::mslot::kHDraft]=ids.v.addr;
    rp[gpu::mslot::kHRank]=ranks.v.addr;
    auto* tp=runner.slots(gpu::MgtStage::HeadTopK);
    tp[gpu::mslot::kHLogits]=logits.v.addr;
    tp[gpu::mslot::kHTopOut]=candidates.v.addr;
    tp[gpu::mslot::kHHist]=hist.v.addr;
    tp[gpu::mslot::kHRank]=ranks.v.addr;
    uint32_t exact=0,fallback=0;
    for(uint32_t m=1;m<=6;++m) {
        std::vector<float> matrix(size_t(m)*kVocab,-32.0f);
        for(uint32_t row=0;row+1<m;++row)
            for(uint32_t id:{5u,17u,32u,256u,1023u})matrix[size_t(row)*kVocab+id]=4.0f;
        // The bonus row is flat and exceeds the per-thread candidate cap.
        std::fill(matrix.end()-kVocab,matrix.end(),0.0f);
        std::memcpy(logits.v.host,matrix.data(),matrix.size()*4);
        REQUIRE(runner.ensure(m));
        gpu::MgtHeadPush p{};p.rows=kVocab;p.k=m-1;
        if(m>1) {
            REQUIRE(runner.dispatch_now(m,gpu::MgtStage::HeadRank,&p,sizeof p,1,m-1));
            std::array<gpu::MgtRankOut,5> got{};
            wc_readback(got.data(),ranks.v.host,(m-1)*sizeof(gpu::MgtRankOut));
            uint32_t accepted=0;
            for(uint32_t row=0;row+1<m;++row) {
                uint32_t better=0;
                const float v=matrix[size_t(row)*kVocab+path[row]];
                for(uint32_t id=0;id<kVocab;++id) {
                    const float x=matrix[size_t(row)*kVocab+id];
                    better+=x>v || (x==v && id<path[row]);
                }
                CHECK_EQ(got[row].rows,kVocab);CHECK_EQ(got[row].token,path[row]);
                CHECK_EQ(got[row].error,0u);CHECK_EQ(got[row].better,better);
                if(accepted==row && better<4)++accepted;
            }
            auto cpu=runtime::accept_topk_prefix(std::span(path).first(m-1),matrix,kVocab,4);
            REQUIRE(cpu);CHECK_EQ(accepted,*cpu);
        }
        p.topk_k=runtime::kTopKDefaultK;
        for(float temperature:{0.7f,1.0f,1.5f}) {
            p.inv_t=1.0f/temperature;
            std::memset(candidates.v.host,0xcd,size_t(m)*gpu::kMgtTopKRecordWords*4);
            REQUIRE(runner.dispatch_now(m,gpu::MgtStage::HeadTopK,&p,sizeof p,1,m));
            std::vector<uint32_t> words(size_t(m)*gpu::kMgtTopKRecordWords);
            wc_readback(words.data(),candidates.v.host,words.size()*4);
            for(uint32_t row=0;row<m;++row) {
                const auto* w=words.data()+size_t(row)*gpu::kMgtTopKRecordWords;
                CHECK_EQ(w[0],kVocab);CHECK_EQ(w[7],row);
                runtime::TopKLogits tk;tk.rows=w[0];tk.max_logit=std::bit_cast<float>(w[1]);
                tk.tail=std::bit_cast<float>(w[2]);tk.bin=w[3];tk.overflow=w[5]!=0;
                if(!tk.overflow)for(uint32_t t=0;t<256;++t) {
                    const uint32_t n=w[8+t];REQUIRE(n<=32);
                    const uint32_t* seg=w+8+256+t*64;
                    for(uint32_t j=0;j<n;++j)tk.cand.push_back({seg[j*2],std::bit_cast<float>(seg[j*2+1])});
                }
                const auto full=std::span(matrix).subspan(size_t(row)*kVocab,kVocab);
                for(float top_p:{0.95f,1.0f}) {
                    const auto a=runtime::nucleus_from_topk(tk,temperature,top_p);
                    const auto b=runtime::nucleus_from_full(full,temperature,top_p);
                    if(!a.exact){++fallback;continue;}
                    ++exact;CHECK(a.ids==b.ids);
                    for(double u:{0.0,0.17,0.51,0.99})
                        CHECK_EQ(runtime::sample_nucleus(a,u),runtime::sample_nucleus(b,u));
                }
            }
            // A single matrix feeds both rank and sampling. Poison every row:
            // only the rejection/bonus row may change, including M=1 (k=0).
            for(uint32_t accept_k:{1u,4u,kVocab}) {
                auto cpu=runtime::accept_topk_prefix(std::span(path).first(m-1),matrix,kVocab,accept_k);
                REQUIRE(cpu);
                std::memset(candidates.v.host,0xcd,words.size()*4);
                p.slice=gpu::kMgtHeadSpecReadout;p.x_stride=accept_k;
                REQUIRE(runner.dispatch_now(m,gpu::MgtStage::HeadTopK,&p,sizeof p,1,m));
                std::vector<uint32_t> selected(words.size());
                wc_readback(selected.data(),candidates.v.host,selected.size()*4);
                for(uint32_t row=0;row<m;++row) {
                    const size_t start=size_t(row)*gpu::kMgtTopKRecordWords;
                    const auto got=std::span(selected).subspan(start,gpu::kMgtTopKRecordWords);
                    if(row==*cpu) CHECK(std::equal(got.begin(),got.end(),words.begin()+start));
                    else CHECK(std::all_of(got.begin(),got.end(),[](uint32_t w){return w==0xcdcdcdcdu;}));
                }
            }
            p.slice=0;p.x_stride=0;
        }
    }
    // Invalid ids must never become out-of-bounds device reads; nonfinite
    // values must be reported even when the offending id is in another wave.
    gpu::MgtHeadPush p{};p.rows=kVocab;p.k=1;
    static_cast<uint32_t*>(ids.v.host)[0]=kVocab;
    REQUIRE(runner.dispatch_now(2,gpu::MgtStage::HeadRank,&p,sizeof p,1,1));
    gpu::MgtRankOut bad{};wc_readback(&bad,ranks.v.host,sizeof bad);CHECK_EQ(bad.error,1u);
    static_cast<uint32_t*>(ids.v.host)[0]=256;
    static_cast<float*>(logits.v.host)[257]=std::numeric_limits<float>::quiet_NaN();
    REQUIRE(runner.dispatch_now(2,gpu::MgtStage::HeadRank,&p,sizeof p,1,1));
    wc_readback(&bad,ranks.v.host,sizeof bad);CHECK_EQ(bad.error,2u);
    CHECK(exact>0);CHECK(fallback>0);
    std::printf("main-path readout: M=1..6 ranks/ties exact, %u nuclei exact, %u fallback cases\n",exact,fallback);
}

CACHEDMOE_TEST(gpu_dspark, batch_cm_causal_rows_match_decode) {
    gpu::Device device;REQUIRE(device.create({}));
    if(!device.caps().cooperative_matrix){CACHEDMOE_SKIP_PRINTF("no cooperative matrices\n");return;}
    gpu::MemoryAllocator alloc;REQUIRE(alloc.init(device,MemoryPath::DeviceLocalHostVisible));
    gpu::MgtSpec spec;spec.attn_cm=true;gpu::MgtRunner batch;
    REQUIRE(batch.create(device,alloc,gpu::default_shader_dir(),spec));
    gpu::AttnRunner decode;REQUIRE(decode.create(device,alloc,gpu::default_shader_dir()));
    REQUIRE(decode.has_attn_cm());
    gpu::GpuScratch scratch;REQUIRE(scratch.create(alloc,20ull<<20));
    constexpr uint32_t heads=64,dim=512,list_stride=160,score_stride=256,part_stride=262144;
    auto q=take(scratch,6ull*heads*dim*2),win=take(scratch,133ull*dim),sc=take(scratch,133ull*16);
    auto cmp=take(scratch,23ull*dim*2),lists=take(scratch,6ull*list_stride*4);
    auto sink=take(scratch,heads*4),rope=take(scratch,6ull*64*4),score=take(scratch,6ull*heads*score_stride*4);
    auto out=take(scratch,6ull*heads*dim*4),g16=take(scratch,6ull*list_stride*dim*2);
    auto q16=take(scratch,6ull*heads*dim*2),p16=take(scratch,6ull*heads*list_stride*2);
    auto inv=take(scratch,6ull*heads*4),part=take(scratch,6ull*part_stride*4),reference=take(scratch,heads*dim*4);
    REQUIRE(q.v.addr && win.v.addr && sc.v.addr && cmp.v.addr && lists.v.addr && sink.v.addr && rope.v.addr);
    REQUIRE(score.v.addr && out.v.addr && g16.v.addr && q16.v.addr && p16.v.addr && inv.v.addr && part.v.addr && reference.v.addr);
    for(uint32_t i=0;i<6*heads*dim;++i)static_cast<uint16_t*>(q.v.host)[i]=cpu::float_to_bf16(std::sin(float(i%977)*.03f)*.2f);
    for(uint32_t i=0;i<133*dim;++i)static_cast<uint8_t*>(win.v.host)[i]=uint8_t(32+(i*13)%64)|uint8_t((i&1)<<7);
    for(uint32_t i=0;i<133*16;++i)static_cast<uint8_t*>(sc.v.host)[i]=uint8_t(123+i%3);
    for(uint32_t i=0;i<23*dim;++i)static_cast<uint16_t*>(cmp.v.host)[i]=cpu::float_to_bf16(std::sin(float(i%997)*.01f)*.1f);
    for(uint32_t i=0;i<heads;++i)static_cast<float*>(sink.v.host)[i]=float(i%7)*.1f;
    for(uint32_t m=0;m<6;++m)for(uint32_t i=0;i<32;++i){
        auto* r=static_cast<float*>(rope.v.host)+m*64;r[2*i]=std::cos(float(m+i)*.07f);r[2*i+1]=std::sin(float(m+i)*.07f);
    }
    using S=gpu::MgtStage;using A=gpu::AttnStage;
    const std::array stages{std::pair{S::AttnCmGather,A::AttnCmGather},std::pair{S::AttnCmScore,A::AttnCmScore},
        std::pair{S::AttnCmSoftmax,A::AttnCmSoftmax},std::pair{S::AttnCmPv,A::AttnCmPv},std::pair{S::AttnCmFinish,A::AttnCmFinish}};
    for(uint32_t m=1;m<=6;++m) {
        REQUIRE(batch.ensure(m));const uint32_t nkv=128+m-1+23,e=gpu::AttnRunner::attn_cm_e(nkv);
        for(uint32_t row=0;row<m;++row)for(uint32_t t=0;t<list_stride;++t){
            int32_t id=-1;
            if(t<128 && (t>6 || t<=row))id=int32_t(t);
            else if(t>=128 && t<128+m-1 && t-128+1>row)id=int32_t(t);
            else if(t>=128+m-1 && t<nkv && t-128-m+1<=row)id=int32_t(t);
            static_cast<int32_t*>(lists.v.host)[row*list_stride+t]=id;
        }
        uint64_t ptr[32]{};ptr[0]=q.v.addr;ptr[1]=win.v.addr;ptr[2]=sc.v.addr;ptr[3]=cmp.v.addr;
        ptr[4]=lists.v.addr;ptr[5]=sink.v.addr;ptr[6]=rope.v.addr;ptr[7]=score.v.addr;ptr[8]=out.v.addr;
        ptr[10]=win.v.addr+128*dim;ptr[11]=sc.v.addr+128*16;
        ptr[12]=g16.v.addr;ptr[13]=q16.v.addr;ptr[14]=p16.v.addr;ptr[15]=inv.v.addr;ptr[16]=part.v.addr;
        for(auto [b,d]:stages)std::memcpy(batch.slots(b),ptr,sizeof ptr);
        gpu::MgtAttnCmPush push{{nkv,e,128,dim,64,heads,score_stride,1/std::sqrt(float(dim)),1,4},m-1,list_stride,part_stride};
        for(auto [b,d]:stages)REQUIRE(batch.dispatch_now(m,b,&push,sizeof push,gpu::AttnRunner::attn_cm_groups(d,push.cm),m));
        std::vector<float> got(size_t(m)*heads*dim),ref(heads*dim);
        wc_readback(got.data(),out.v.host,got.size()*4);
        for(uint32_t row=0;row<m;++row) {
            ptr[0]=q.v.addr+uint64_t(row)*heads*dim*2;ptr[4]=lists.v.addr+uint64_t(row)*list_stride*4;
            ptr[6]=rope.v.addr+row*64*4;ptr[8]=reference.v.addr;
            for(auto [b,d]:stages)std::memcpy(decode.slots(d),ptr,sizeof ptr);
            auto single=push.cm;single.n_win=128+m-1;
            for(auto [b,d]:stages)REQUIRE(decode.dispatch_now(d,&single,sizeof single,gpu::AttnRunner::attn_cm_groups(d,single)));
            wc_readback(ref.data(),reference.v.host,ref.size()*4);
            CHECK(std::equal(ref.begin(),ref.end(),got.begin()+size_t(row)*heads*dim));
        }
        std::printf("batch CM M=%u causal/overflow/compressed/padding/RoPE: decode rows bit-identical\n",m);
    }
}

CACHEDMOE_TEST(gpu_dspark, batch_engram_prefetch_planes) {
    if(skip_without_model("gpu_dspark.batch_engram_prefetch_planes"))return;
    Rig rig;REQUIRE(rig.bring_up());
    gpu::DecodeRunner dec;REQUIRE(dec.create(rig.device,rig.alloc,gpu::default_shader_dir()));
    auto tables=runtime::EngramTables::load(std::string(CACHEDMOE_TEST_DATA_DIR)+"/l3");REQUIRE(tables);
    auto model=V41Config::load(std::string(model_dir())+"/config.json");REQUIRE(model);
    TextConfig cfg=model->text;
    runtime::EngramRunner eg;
    REQUIRE(eg.create(rig.device,rig.alloc,dec,rig.manifest,rig.shards,rig.io,rig.pinned,cfg,std::move(*tables)));
    constexpr uint32_t HCD=4*5120;
    auto x=rig.scratch.alloc(6*HCD*4);auto y=rig.scratch.alloc(6*HCD*4);REQUIRE(x && y);
    auto* input=static_cast<float*>(x->host);
    for(uint32_t i=0;i<6*HCD;++i)input[i]=std::sin(float(i)*.013f)*.5f;
    std::vector<uint32_t> history(150);
    for(uint32_t i=0;i<history.size();++i)history[i]=(i*191+500)%kVocab;
    gpu::CommandPool pool;REQUIRE(pool.create(rig.device));auto cb=pool.acquire();REQUIRE(cb);
    for(uint32_t L:{1u,14u}) {
        std::vector<std::string> names;
        for(const char* suffix:{"wkv.weight","q_weight","k_weight"})names.push_back(std::format("layers.{}.engram.{}",L,suffix));
        REQUIRE(rig.pinned.load(rig.manifest,rig.shards,rig.io,names));
        for(uint32_t M=1;M<=6;++M) {
            std::vector<float> expected(size_t(M)*HCD);
            for(uint32_t m=0;m<M;++m) {
                REQUIRE_OK(eg.run(L,history,128+m,x->addr+m*HCD*4,y->addr+m*HCD*4));
                wc_readback(expected.data()+size_t(m)*HCD,static_cast<float*>(y->host)+size_t(m)*HCD,HCD*4);
            }
            for(uint32_t m=0;m<M;++m)REQUIRE(eg.fetch(L,history,128+m,m));
            REQUIRE(cb->begin());
            for(uint32_t m=0;m<M;++m)REQUIRE(eg.record(*cb,L,x->addr+m*HCD*4,y->addr+m*HCD*4,m));
            REQUIRE(cb->end());REQUIRE(gpu::submit_and_wait(rig.device,*cb));
            std::vector<float> actual(expected.size());wc_readback(actual.data(),y->host,actual.size()*4);
            CHECK(actual==expected);
            std::printf("batch engram L%u M%u: bit-identical across one submission\n",L,M);
        }
    }
    CHECK(!eg.fetch(1,history,128,6));
    eg.destroy();CHECK(!eg.fetch(1,history,128));
}

CACHEDMOE_TEST(gpu_dspark, batch_projection_fold_scale) {
    gpu::Device device;REQUIRE(device.create({}));
    gpu::MemoryAllocator alloc;REQUIRE(alloc.init(device,MemoryPath::DeviceLocalHostVisible));
    gpu::MgtRunner original,folded;gpu::MgtSpec spec;
    REQUIRE(original.create(device,alloc,gpu::default_shader_dir(),spec));
    spec.fold_scale=true;REQUIRE(folded.create(device,alloc,gpu::default_shader_dir(),spec));
    gpu::GpuScratch scratch;REQUIRE(scratch.create(alloc,80ull<<20));
    auto w=take(scratch,32768ull*1280);auto scale=take(scratch,32768ull*1280/1024);
    auto x=take(scratch,6ull*32768*4);auto parts=take(scratch,8ull*6*32768*4);
    REQUIRE(w.v.addr && scale.v.addr && x.v.addr && parts.v.addr);
    auto* wb=static_cast<uint8_t*>(w.v.host);
    for(size_t i=0;i<32768ull*1280;++i)wb[i]=uint8_t((i*17+31)%126)|uint8_t((i&1)<<7);
    auto* sb=static_cast<uint8_t*>(scale.v.host);
    for(size_t i=0;i<32768ull*1280/1024;++i)sb[i]=uint8_t(121+i%12);
    auto* xb=static_cast<float*>(x.v.host);
    for(uint32_t i=0;i<6*32768;++i)xb[i]=std::sin(float(i%977)*0.01f)*(1+float(i%13)*0.03f);
    struct Shape {gpu::MgtStage stage;uint32_t rows,k;};
    using S=gpu::MgtStage;
    for(const auto shape:{Shape{S::WqASplit,1280,5120},Shape{S::WqBSplit,32768,1280},
                          Shape{S::WkvSplit,512,5120},Shape{S::WoASplit,8192,4096},Shape{S::WoBSplit,5120,8192}}) {
        for(uint32_t m:{1u,3u,6u}) {
            const size_t n=size_t(original.ksplit(shape.stage))*m*shape.rows;
            std::vector<float> ref(n),got(n);double elapsed[2]{};
            uint32_t arm=0;
            for(auto* r:{&original,&folded}) {
                REQUIRE(r->ensure(m));auto* p=r->slots(shape.stage);
                p[gpu::mslot::kGW]=w.v.addr;p[gpu::mslot::kGS]=scale.v.addr;
                p[gpu::mslot::kGX]=x.v.addr;p[gpu::mslot::kGP]=parts.v.addr;
                gpu::MgtGemvPush push{};push.rows=shape.rows;push.k=shape.k;
                push.scale_cols=shape.k/32;push.part_stride=shape.rows;push.x_stride=shape.k;
                if(shape.stage==S::WoASplit){push.rows_per_group=1024;push.x_stride=32768;}
                const auto start=std::chrono::steady_clock::now();
                REQUIRE(r->dispatch_now(m,shape.stage,&push,sizeof push,r->split_groups(shape.stage,shape.rows)));
                elapsed[arm++]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                auto& values=r==&original?ref:got;
                wc_readback(values.data(),parts.v.host,values.size()*4);
            }
            CHECK(ref==got);
            std::printf("projection fold %s M=%u old_ms=%.6f fold_ms=%.6f partials=%zu bit-identical\n",
                        gpu::mgt_stage_name(shape.stage),m,elapsed[0],elapsed[1],n);
        }
    }
}

CACHEDMOE_TEST(gpu_dspark, batch_projection_pair_dot) {
    gpu::Device device;REQUIRE(device.create({}));
    gpu::MemoryAllocator alloc;REQUIRE(alloc.init(device,MemoryPath::DeviceLocalHostVisible));
    gpu::MgtRunner original,folded;gpu::MgtSpec spec;
    REQUIRE(original.create(device,alloc,gpu::default_shader_dir(),spec));
    spec.pair_dot=true;REQUIRE(folded.create(device,alloc,gpu::default_shader_dir(),spec));
    gpu::GpuScratch scratch;REQUIRE(scratch.create(alloc,80ull<<20));
    auto w=take(scratch,32768ull*1280);auto scale=take(scratch,32768ull*1280/1024);
    auto x=take(scratch,6ull*32768*4);auto parts=take(scratch,8ull*6*32768*4);
    REQUIRE(w.v.addr && scale.v.addr && x.v.addr && parts.v.addr);
    auto* wb=static_cast<uint8_t*>(w.v.host);
    for(size_t i=0;i<32768ull*1280;++i)wb[i]=uint8_t((i*17+31)%126)|uint8_t((i&1)<<7);
    auto* sb=static_cast<uint8_t*>(scale.v.host);
    for(size_t i=0;i<32768ull*1280/1024;++i)sb[i]=uint8_t(121+i%12);
    auto* xb=static_cast<float*>(x.v.host);
    for(uint32_t i=0;i<6*32768;++i)xb[i]=std::sin(float(i%977)*0.01f)*(1+float(i%13)*0.03f);
    gpu::CommandPool pool;REQUIRE(pool.create(device));gpu::QueryPool query;REQUIRE(query.create(device,2));
    struct Shape {gpu::MgtStage stage;uint32_t rows,k;};
    using S=gpu::MgtStage;
    for(const auto shape:{Shape{S::WqASplit,1280,5120},Shape{S::WqBSplit,32768,1280},
                          Shape{S::WkvSplit,512,5120},Shape{S::WoASplit,8192,4096},Shape{S::WoBSplit,5120,8192}}) {
        for(uint32_t m:{1u,3u,6u}) {
            const size_t n=size_t(original.ksplit(shape.stage))*m*shape.rows;
            std::vector<float> ref(n),got(n);double elapsed[2]{};
            uint32_t arm=0;
            for(auto* r:{&original,&folded}) {
                REQUIRE(r->ensure(m));auto* p=r->slots(shape.stage);
                p[gpu::mslot::kGW]=w.v.addr;p[gpu::mslot::kGS]=scale.v.addr;
                p[gpu::mslot::kGX]=x.v.addr;p[gpu::mslot::kGP]=parts.v.addr;
                gpu::MgtGemvPush push{};push.rows=shape.rows;push.k=shape.k;
                push.scale_cols=shape.k/32;push.part_stride=shape.rows;push.x_stride=shape.k;
                if(shape.stage==S::WoASplit){push.rows_per_group=1024;push.x_stride=32768;}
                pool.reset();auto cb=pool.acquire();REQUIRE(cb);REQUIRE(cb->begin());
                REQUIRE(cb->reset_queries(query,0,2));REQUIRE(cb->write_timestamp(query,0,false));
                REQUIRE(r->record(*cb,m,shape.stage,&push,sizeof push,r->split_groups(shape.stage,shape.rows)));
                REQUIRE(cb->write_timestamp(query,1,true));REQUIRE(cb->end());REQUIRE(gpu::submit_and_wait(device,*cb));
                auto ticks=query.read_range(0,2);REQUIRE(ticks);
                const auto bits=device.caps().timestamp_valid_bits;const uint64_t mask=bits>=64?~0ull:(1ull<<bits)-1;
                elapsed[arm++]=double(((*ticks)[1]-(*ticks)[0])&mask)*device.caps().timestamp_period_ns*1e-6;
                auto& values=r==&original?ref:got;
                wc_readback(values.data(),parts.v.host,values.size()*4);
            }
            CHECK(ref==got);
            std::printf("projection pair %s M=%u old_ms=%.6f fold_ms=%.6f partials=%zu bit-identical\n",
                        gpu::mgt_stage_name(shape.stage),m,elapsed[0],elapsed[1],n);
        }
    }
}

CACHEDMOE_TEST(gpu_dspark, full_runtime_chain) {
    if (skip_without_model("gpu_dspark.full_runtime_chain")) return;
    auto set=load_l2(ds_dir()); REQUIRE(set);
    const L2Step* gold=nullptr;
    for (auto& g:set->steps) if(g.step=="golden_pos64") gold=&g;
    REQUIRE(gold); const auto& g=*gold;
    RuntimeConfig cfg; cfg.model_dir=model_dir();
    cfg.cache.budget_bytes=512ull*layout::kExpertSlotBytes;
    cfg.cache.slots_per_slab=64;cfg.speculation.enabled=true;
    runtime::Engine engine;
    REQUIRE(engine.init(cfg)); REQUIRE(engine.init_gpu());
    auto& draft=*engine.dspark_runtime();
    const bool onecb=::cachedmoe::environment::get("CACHEDMOE_DSPARK_TEST_ONECB")!=nullptr;
    draft.set_onecb(false);
    auto set_fused=[&](bool enabled){if(onecb)draft.set_onecb(enabled);else draft.set_mega(enabled);};
    std::printf("draft comparison path=%s\n",onecb?"onecb":"mega");
    std::array<std::span<const float>,3> rings;
    for(uint32_t st=0;st<3;++st) rings[st]={g.f(std::format("s{}.sparse_kv",st)).data(),kWin*kHeadDim};
    REQUIRE(draft.seed_window(64,rings));
    REQUIRE(draft.append(64,g.f("main_hidden")));
    set_fused(false);
    std::map<std::string,std::vector<float>> serial_values;
    bool agree_all=true;
    draft.probe=[&](std::string_view name,std::span<const float> val) {
        auto& saved=serial_values[std::string(name)];saved.resize(val.size());
        wc_readback(saved.data(),val.data(),val.size_bytes());
        if(name.starts_with("diag."))return;
        std::string ref(name);
        if(ref.starts_with("mtp.")) ref="s"+ref.substr(4,1)+"."+ref.substr(6);
        const auto& expected=g.f(ref);
        const auto ag=agree(val.data(),expected.data(),expected.size());
        agree_all &= ok(ref.c_str(),ag,0.999,0.05);
    };
    const auto input=uint32_t(g.f("draft_ids")[0]);
    auto out=draft.draft(64,input);
    if(!out) std::printf("DSpark runtime: %s\n",out.error().str().c_str());
    REQUIRE(out);
    for(const auto& [name,t]:out->timing_ms)std::printf("draft timing %-32s %.6f ms\n",name.c_str(),t);
    for(uint32_t j=0;j<5;++j) {
        std::printf("draft[%u] %u expected %.0f confidence %.5f expected %.5f\n",j,out->tokens[j],g.f("draft_ids")[j+1],out->confidence[j],g.f("confidence")[j]);
        const auto& top=g.f("draft_logits.top_ids");
        const auto& val=g.f("draft_logits.top_logits");
        const size_t width=top.size()/5;
        if(out->tokens[j]!=uint32_t(g.f("draft_ids")[j+1])) {
            // The existing MoE path keeps intermediate activations in fp16.
            // Permit only a top-two reference tie below 0.1 logit, never a broad
            // token disagreement. Drafts are proposals; target acceptance owns correctness.
            std::printf("      near-tie: reference margin %.6f, fp16 MoE draft differs\n",val[j*width]-val[j*width+1]);
            CHECK(val[j*width]-val[j*width+1]<0.1f);
            CHECK(out->tokens[j]==uint32_t(top[j*width+1]));
        }
        CHECK(std::abs(out->confidence[j]-g.f("confidence")[j])<0.1f);
    }
    CHECK(agree_all);
    set_fused(true);
    draft.probe=[&](std::string_view name,std::span<const float> val) {
        const auto& expected=serial_values.at(std::string(name));
        const auto ag=agree(val.data(),expected.data(),expected.size());
        std::printf("fused vs serial %s %s max=%.8g expectedmax=%.8g\n",std::string(name).c_str(),ag.str().c_str(),*std::max_element(val.begin(),val.end()),*std::max_element(expected.begin(),expected.end()));
        CHECK(std::memcmp(val.data(),expected.data(),val.size_bytes())==0);
    };
    auto fused=draft.draft(64,input);
    if(!fused)std::printf("mega: %s\n",fused.error().str().c_str());
    REQUIRE(fused);
    CHECK_EQ(fused->gpu_submissions,1u);
    if(!onecb)CHECK_EQ(fused->gpu_dispatches,1u);
    else CHECK(fused->gpu_dispatches>80u);
    CHECK(fused->tokens==out->tokens);
    CHECK(fused->confidence==out->confidence);
    const auto ag=agree(fused->logits.data(),out->logits.data(),out->logits.size());
    std::printf("fused logits %s\n",ag.str().c_str());
    CHECK(fused->logits==out->logits);
    for(const auto& [name,t]:fused->timing_ms)std::printf("mega timing %-32s %.6f ms\n",name.c_str(),t);
    for(const auto& [name,t]:fused->gpu_timing_ms)std::printf("draft gpu %-42s %.6f ms\n",name.c_str(),t);
    std::printf("fused dispatches=%u phases=%u\n",fused->gpu_dispatches,fused->gpu_phases);
    // Compare wall time without oracle callbacks or snapshot phases.
    draft.probe={};set_fused(false);
    auto plain=draft.draft(64,input);REQUIRE(plain);
    set_fused(true);auto fast=draft.draft(64,input);REQUIRE(fast);
    CHECK(fast->logits==plain->logits);CHECK(fast->tokens==plain->tokens);
    std::printf("draft bench serial_ms=%.6f mega_ms=%.6f kernel_ms=%.6f dispatches=%u phases=%u\n",
        plain->timing_ms.at("wall.draft"),fast->timing_ms.at("wall.draft"),fast->timing_ms.at(onecb?"onecb.record_submit":"mega.kernel"),fast->gpu_dispatches,fast->gpu_phases);
    for(uint32_t rows=1;rows<5;++rows) {
        for(bool mega:{false,true}) {
            set_fused(mega);
            auto prefix=draft.draft(64,input,rows);REQUIRE(prefix);
            CHECK_EQ(prefix->logits.size(),size_t(rows)*kVocab);
            CHECK(std::equal(prefix->logits.begin(),prefix->logits.end(),plain->logits.begin()));
            CHECK(std::equal(prefix->tokens.begin(),prefix->tokens.begin()+rows,plain->tokens.begin()));
            CHECK(std::equal(prefix->confidence.begin(),prefix->confidence.begin()+rows,plain->confidence.begin()));
            std::printf("draft prefix rows=%u mega=%u wall_ms=%.6f phases=%u bit-identical\n",rows,mega,
                        prefix->timing_ms.at("wall.draft"),prefix->gpu_phases);
        }
    }
    auto compact=draft.draft(64,input,2,false);REQUIRE(compact);CHECK(compact->logits.empty());
    CHECK_EQ(compact->tokens[0],plain->tokens[0]);CHECK_EQ(compact->tokens[1],plain->tokens[1]);
    // KV updates stay on the GPU, including a five-row append across the ring wrap.
    std::vector<float> hidden;
    for(uint32_t i=0;i<5;++i)hidden.insert(hidden.end(),g.f("main_hidden").begin(),g.f("main_hidden").end());
    set_fused(false);REQUIRE(draft.seed_window(127,rings));REQUIRE(draft.append(127,g.f("main_hidden")));
    set_fused(true);REQUIRE(draft.append(128,hidden));
    auto gpu_kv=draft.draft(132,input);REQUIRE(gpu_kv);
    set_fused(false);REQUIRE(draft.seed_window(127,rings));REQUIRE(draft.append(127,g.f("main_hidden")));
    REQUIRE(draft.append(128,hidden));auto cpu_kv=draft.draft(132,input);REQUIRE(cpu_kv);
    CHECK(gpu_kv->logits==cpu_kv->logits);CHECK(gpu_kv->tokens==cpu_kv->tokens);
    std::printf("fused committed KV wrap: logits bit-identical\n");
    if(onecb){
        set_fused(false);draft.probe={};REQUIRE_OK(draft.set_profile(false));
        auto untimed=draft.draft(132,input,2,false);REQUIRE(untimed);
        REQUIRE_OK(draft.set_profile(true));auto timed=draft.draft(132,input,2,false);REQUIRE(timed);
        CHECK(untimed->tokens==timed->tokens);CHECK(untimed->confidence==timed->confidence);
        std::printf("timestamp comparison same state: off=%.6f on=%.6f delta=%.3f%%\n",untimed->timing_ms.at("wall.draft"),timed->timing_ms.at("wall.draft"),100*(timed->timing_ms.at("wall.draft")/untimed->timing_ms.at("wall.draft")-1));
    }

}

CACHEDMOE_TEST(gpu_dspark, adaptive_zero_keeps_one_target_forward) {
    if(skip_without_model("gpu_dspark.adaptive_zero_keeps_one_target_forward"))return;
    auto state=runtime::DecodeState::load(std::string(CACHEDMOE_TEST_DATA_DIR)+"/l3");REQUIRE(state);
    RuntimeConfig cfg;cfg.model_dir=model_dir();cfg.cache.budget_bytes=512ull*layout::kExpertSlotBytes;
    cfg.cache.slots_per_slab=64;cfg.speculation.enabled=true;cfg.speculation.max_draft=5;
    cfg.speculation.min_confidence=1e6f;
    runtime::Engine engine;REQUIRE(engine.init(cfg));REQUIRE(engine.init_gpu());
    runtime::SessionConfig sc;sc.max_context=256;
    sc.engram_tables_dir=std::string(CACHEDMOE_TEST_DATA_DIR)+"/l3";
    REQUIRE(engine.begin_session(sc));engine.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    auto first=engine.feed(std::span(state->prompt_ids()).first(8));REQUIRE(first);
    uint32_t layers=0;const bool gpu_route=::cachedmoe::environment::get("CACHEDMOE_BATCH_GPU_ROUTE")&&std::string_view(::cachedmoe::environment::get("CACHEDMOE_BATCH_GPU_ROUTE"))=="1";
    if(!gpu_route)engine.batch_probe=[&](uint32_t,const auto&){++layers;};
    uint32_t root=first->token;
    uint32_t observed = 0;
    engine.spec_verify_probe = [&](std::span<const uint32_t> input,
                                   std::span<const runtime::Engine::BatchRow> rows,
                                   std::span<const float> logits) {
        ++observed;
        const uint32_t vocab = engine.model().text.vocab_size;
        CHECK_EQ(input.size(), size_t(1)); // adaptive k=0 still verifies its root once
        CHECK_EQ(logits.size(), size_t(vocab));
        const auto best = std::max_element(logits.begin(), logits.end());
        CHECK_EQ(uint32_t(best - logits.begin()), rows[0].argmax);
    };
    for(uint32_t i=0;i<2;++i) {
        const uint32_t before=engine.context_length();layers=0;const auto calls=engine.batch_forward_calls();
        auto cycle=engine.speculative_step(root,6);REQUIRE(cycle);
        CHECK_EQ(cycle->cycle.k,0u);CHECK_EQ(cycle->cycle.accepted,0u);
        CHECK_EQ(cycle->rows.size(),size_t(1));CHECK_EQ(engine.batch_forward_calls()-calls,1ull);
        CHECK_EQ(engine.last_batch_layers(),40u);
        if(gpu_route)CHECK_EQ(engine.last_batch_submits(),1u);else CHECK_EQ(layers,40u);
        CHECK_EQ(engine.context_length(),before+1);CHECK_EQ(engine.history().back(),root);
        CHECK_EQ(engine.dspark_runtime()->next_position(),before+1);
        CHECK(cycle->cycle.draft_ms>0);root=cycle->rows.back().token;
    }
    CHECK_EQ(observed, 2u);
}

CACHEDMOE_TEST(gpu_dspark, committed_prefix_survives_window_wrap) {
    if (skip_without_model("gpu_dspark.committed_prefix_survives_window_wrap")) return;
    auto state=runtime::DecodeState::load(std::string(CACHEDMOE_TEST_DATA_DIR)+"/l3");
    REQUIRE(state);REQUIRE(!state->prompt_ids().empty());
    RuntimeConfig cfg;cfg.model_dir=model_dir();cfg.speculation.enabled=true;
    cfg.cache.budget_bytes=2000ull*layout::kExpertSlotBytes;cfg.cache.slots_per_slab=100;
    runtime::Engine engine;REQUIRE(engine.init(cfg));REQUIRE(engine.init_gpu());
    runtime::SessionConfig sc;sc.max_context=512;
    sc.engram_tables_dir=std::string(CACHEDMOE_TEST_DATA_DIR)+"/l3";
    REQUIRE(engine.begin_session(sc));
    engine.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    std::vector<uint32_t> prompt(128);
    for(size_t i=0;i<prompt.size();++i)prompt[i]=state->prompt_ids()[i%state->prompt_ids().size()];
    auto first=engine.feed(prompt);REQUIRE_OK(first);
    // Every token is a stop: a six-row verification must retain only row 0.
    // Rows 1..5 wrap over committed positions 1..5 and must be restored.
    std::vector<uint32_t> stops(engine.model().text.vocab_size);
    for(uint32_t i=0;i<stops.size();++i)stops[i]=i;
    std::vector<uint32_t> slots{1,2,3,4,5},layers(40);
    for(uint32_t i=0;i<layers.size();++i)layers[i]=i;
    auto ring_before=engine.kv_store().snapshot_ring(slots,layers);REQUIRE(ring_before);
    auto carry_before=engine.kv_store().backup_rows(128,128);REQUIRE(carry_before);
    auto cycle=engine.speculative_step(first->token,6,stops);REQUIRE(cycle);
    CHECK_EQ(cycle->cycle.k,5u);CHECK_EQ(cycle->rows.size(),size_t(1));
    CHECK_EQ(engine.context_length(),129u);
    CHECK(std::equal(prompt.begin(),prompt.end(),engine.history().begin()));
    CHECK_EQ(engine.history().back(),first->token);
    CHECK_EQ(engine.dspark_runtime()->next_position(),129u);
    auto ring_after=engine.kv_store().snapshot_ring(slots,layers);REQUIRE(ring_after);
    CHECK(ring_before->val==ring_after->val);CHECK(ring_before->scale==ring_after->scale);
    auto carry_after=engine.kv_store().backup_rows(129,129);REQUIRE(carry_after);
    REQUIRE(carry_before->planes.size()==carry_after->planes.size());
    for(size_t i=0;i<carry_before->planes.size();++i) {
        const auto& before=carry_before->planes[i];const auto& after=carry_after->planes[i];
        const uint32_t changed=128%before.ratio;
        for(size_t j=0;j<before.carry_kv.size();++j)if(j/512!=changed) {
            CHECK_EQ(before.carry_kv[j],after.carry_kv[j]);
            CHECK_EQ(before.carry_score[j],after.carry_score[j]);
        }
    }
    // Force the real reseed path to consume the restored hidden window.
    engine.dspark_runtime()->reset();
    auto tail=engine.speculative_step(cycle->rows.back().token,1);REQUIRE(tail);
    CHECK_EQ(tail->cycle.k,0u);CHECK_EQ(tail->rows.size(),size_t(1));
    CHECK_EQ(engine.context_length(),130u);CHECK_EQ(engine.dspark_runtime()->next_position(),130u);
    engine.reset_context();CHECK_EQ(engine.context_length(),0u);
    auto reset=engine.feed(std::span(prompt).first(4));REQUIRE(reset);
    auto fresh=engine.speculative_step(reset->token,2);REQUIRE(fresh);
    CHECK(engine.context_length()==4+fresh->rows.size());
}
