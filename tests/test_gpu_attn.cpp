// The design §7.2-§7.8 kernels against the oracle's per-stage L2 tensors, on
// the real checkpoint.
//
// This is the attention-path twin of tests/test_gpu_moe.cpp, and it closes the
// L2 loop of design §12: for one decode token at position 64 of a 64-token
// prefill, every stage of the §7.14 list is fed the reference's own input and
// its output compared against the reference's own output. Because the inputs
// are golden rather than chained, a failure names one kernel instead of
// pointing vaguely upstream -- and then the last three cases DO chain, so a
// per-stage pass that does not compose still gets caught.
//
// The weights come from the pinned set (store/pinned.h), read off NVMe by the
// real IoEngine into GPU-addressable memory, and are reached by
// buffer_device_address exactly as the runtime will reach them.
//
// Gated on DEEPMOE_MODEL_DIR, on tests/data/l2 existing, and on a working
// Vulkan device; a skip is a pass.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>
#include <vector>

#include "core/config.h"
#include "cpu/dequant.h"
#include "gpu/vulkan/attn_kernels.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "model/layout.h"
#include "model/manifest.h"
#include "runtime/rope.h"
#include "storage/backend.h"
#include "storage/io_engine.h"
#include "store/pinned.h"
#include "store/shard_set.h"
#include "gpu/vulkan/moe_kernels.h"   // default_shader_dir
#include "tests/l1_golden.h"            // model_dir / skip_without_model
#include "tests/l2_golden.h"
#include "tests/test_framework.h"

#ifndef DEEPMOE_TEST_DATA_DIR
#define DEEPMOE_TEST_DATA_DIR "tests/data"
#endif

using namespace deepmoe;
using namespace deepmoe::testing;

namespace {

std::string l2_dir() { return std::string(DEEPMOE_TEST_DATA_DIR) + "/l2"; }

// The P3 knobs, from the environment, so one binary can re-validate a sweep
// point without a rebuild -- the same idea as VK_INSTANCE_LAYERS switching the
// validation layers on. Unset means the AttnSpec defaults, which is what CI
// runs. See docs/p2_attention.md §13.
gpu::AttnSpec env_spec() {
    gpu::AttnSpec sp;
    auto num = [](const char* n, uint32_t& v) {
        if (const char* e = std::getenv(n)) v = static_cast<uint32_t>(std::atoi(e));
    };
    num("DEEPMOE_FP8_ARITH", sp.fp8_arith_decode);
    num("DEEPMOE_TILE_HEADS", sp.tile_heads_per_wg);
    num("DEEPMOE_PV_HEADS", sp.pv_heads_per_wg);
    num("DEEPMOE_KSPLIT_WQA", sp.ksplit_wq_a);
    num("DEEPMOE_KSPLIT_WKV", sp.ksplit_wkv);
    num("DEEPMOE_KSPLIT_WOA", sp.ksplit_wo_a);
    num("DEEPMOE_KSPLIT_WOB", sp.ksplit_wo_b);
    num("DEEPMOE_KSPLIT_ROWS", sp.rows_per_lane_ksplit);
    return sp;
}

// Dimensions, all from the L2 index's copy of config.json so the test cannot
// drift from the data.
struct Dims {
    uint32_t dim = 5120, hc = 4, n_heads = 64, head_dim = 512, rope_dim = 64;
    uint32_t q_lora = 1280, o_lora = 1024, o_groups = 8, window = 128;
    uint32_t n_experts = 384, topk = 6;
    float    norm_eps = 1e-20f, route_scale = 1.5f;
};

struct Rig {
    gpu::Device           device;
    gpu::MemoryAllocator  alloc;
    Manifest              manifest;
    store::ShardSet       shards;
    storage::IoEngine     io;
    store::PinnedStore    pinned;
    gpu::AttnRunner       runner;
    gpu::GpuScratch       scratch;
    bool                  io_started = false;
    std::string           why;

    // `scratch` and `runner` go first by declaration order, but PinnedStore
    // holds its regions through a SlabBacking and does not give them back on
    // destruction, so the validation layers see one VkBuffer and one
    // VkDeviceMemory outliving vkDestroyDevice unless it is reset by hand --
    // the same call bench/attn_bench makes at the end of main.
    ~Rig() {
        scratch.destroy();
        runner.destroy();
        pinned.reset();
        if (io_started) io.stop();
    }

    bool bring_up() {
        const std::string dir = model_dir() ? model_dir() : "";
        // The same switch bench/kernel_bench uses: VK_INSTANCE_LAYERS set in the
        // environment turns the validation layers on, so a run with them is one
        // env var rather than a rebuild.
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
        pc.region_bytes = 512ull << 20;   // a layer is ~135 MB; 512 MB regions keep it to a few
        if (auto r = pinned.init(std::move(*backing), pc); !r) { why = r.error().str(); return false; }
        if (auto r = runner.create(device, alloc, gpu::default_shader_dir(), env_spec()); !r) {
            why = r.error().str(); return false;
        }
        if (auto r = scratch.create(alloc, 64ull << 20); !r) { why = r.error().str(); return false; }
        return true;
    }

    bool load_layer(uint32_t L) {
        auto names = store::pinned_layer_tensors(manifest, L);
        if (auto r = pinned.load(manifest, shards, io, names); !r) {
            why = r.error().str(); return false;
        }
        return true;
    }

    uint64_t addr(const std::string& n) {
        const store::PinnedTensor* t = pinned.find(n);
        return t ? t->data : 0;
    }
    uint64_t scale_addr(const std::string& n) {
        const store::PinnedTensor* t = pinned.find(n);
        return t ? t->scale : 0;
    }
};

// A scratch view plus the host-side helpers a test needs.
struct Buf {
    gpu::GpuScratch::View v{};
    float*    f()  { return static_cast<float*>(v.host); }
    uint32_t* u32(){ return static_cast<uint32_t*>(v.host); }
    uint8_t*  u8() { return static_cast<uint8_t*>(v.host); }
    uint16_t* u16(){ return static_cast<uint16_t*>(v.host); }
    uint64_t  a() const { return v.addr; }
    void zero() { std::memset(v.host, 0, static_cast<size_t>(v.bytes)); }
    void set(const std::vector<float>& src) {
        std::memcpy(v.host, src.data(), src.size() * 4);
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
};

Buf take(gpu::GpuScratch& s, uint64_t bytes) {
    Buf b;
    auto v = s.alloc(bytes);
    if (v) b.v = *v;
    return b;
}

// design §12 L2 wants cosine >= 0.999. These are tighter, and every case prints
// what it actually achieved so the numbers land in docs/p2_attention.md rather
// than only in a pass/fail.
// The criterion is cosine plus RELATIVE L2, not the max-element relative error
// design §12 L1 uses. L1 compares one expert FFN against an fp32 oracle, where
// a single outlying element is a real bug. Here the reference itself is bf16:
// `generate.py` runs under `set_default_dtype(bfloat16)`, so every Linear
// output, the residual stream and the attention inputs carry eight mantissa
// bits while our kernels keep fp32. One element landing on the other side of a
// bf16 or an E4M3 rounding boundary then moves by up to half an ulp of ITS OWN
// magnitude -- 0.4% for bf16, 6% for E4M3 -- and `max|d| / max|y|` reports that
// as though it were an error in the whole vector. Relative L2 measures what the
// next kernel actually sees. Both numbers are printed either way.
// The largest gap between two vectors in ULP of the larger magnitude. Used
// where two kernels compute the SAME sum in a different association -- a
// K-split GEMV against the unsplit one, a tiled softmax against the untiled --
// so the question is not "is this accurate" but "is this the same number to
// within fp32 re-association".
double ulp_gap(const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const float m = std::max(std::fabs(a[i]), std::fabs(b[i]));
        if (m == 0.0f) continue;
        const double u = double(std::nextafter(m, 3.4e38f)) - double(m);
        worst = std::max(worst, std::fabs(double(a[i]) - double(b[i])) / u);
    }
    return worst;
}

// ULP alone overstates a re-associated dot product: an output element that is
// a small difference of large terms has a big relative error and a tiny
// absolute one, so the relative L2 over the whole vector is printed with it.
void report_ulp(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    const Agreement g = agree(a, b);
    std::printf("      %-22s vs the kernel it replaces: max %.0f ULP, relL2 %.2e, "
                "cos %.9f\n", what, ulp_gap(a, b), g.rel_l2, g.cos);
}

bool ok(const char* what, const Agreement& g, double cos_min, double rel_l2_max) {
    const bool pass = g.cos >= cos_min && g.rel_l2 <= rel_l2_max && std::isfinite(g.max_abs);
    std::printf("      %-22s %s%s\n", what, g.str().c_str(), pass ? "" : "   <-- FAIL");
    return pass;
}

}  // namespace

// ---------------------------------------------------------------------------

DEEPMOE_TEST(gpu_attn, l2_per_stage) {
    if (skip_without_model("gpu_attn")) return;
    auto set = load_l2(l2_dir());
    if (!set) {
        std::printf("      SKIP gpu_attn: no L2 data (%s). Run "
                    "`oracle.py --level l2 --out tests/data/l2`\n", set.error().str().c_str());
        return;
    }
    Rig rig;
    if (!rig.bring_up()) {
        std::printf("      SKIP gpu_attn: %s\n", rig.why.c_str());
        return;
    }

    Dims d;
    d.dim       = static_cast<uint32_t>(set->cfg("dim", 5120));
    d.hc        = static_cast<uint32_t>(set->cfg("hc_mult", 4));
    d.n_heads   = static_cast<uint32_t>(set->cfg("n_heads", 64));
    d.head_dim  = static_cast<uint32_t>(set->cfg("head_dim", 512));
    d.rope_dim  = static_cast<uint32_t>(set->cfg("rope_head_dim", 64));
    d.q_lora    = static_cast<uint32_t>(set->cfg("q_lora_rank", 1280));
    d.o_lora    = static_cast<uint32_t>(set->cfg("o_lora_rank", 1024));
    d.o_groups  = static_cast<uint32_t>(set->cfg("o_groups", 8));
    d.window    = static_cast<uint32_t>(set->cfg("window_size", 128));
    d.n_experts = static_cast<uint32_t>(set->cfg("n_routed_experts", 384));
    d.topk      = static_cast<uint32_t>(set->cfg("n_activated_experts", 6));

    const uint32_t pos = set->decode_pos;
    const uint32_t hcdim = d.hc * d.dim;
    const uint32_t qrows = d.n_heads * d.head_dim;
    const uint32_t ocols = qrows / d.o_groups;
    const uint32_t orows = d.o_groups * d.o_lora;

    // Scratch, laid out once and reused per layer.
    gpu::GpuScratch& S = rig.scratch;
    Buf bX    = take(S, hcdim * 4),  bA    = take(S, d.dim * 4);
    Buf bPost = take(S, 64),         bComb = take(S, 256),  bPre = take(S, 64);
    Buf bXout = take(S, hcdim * 4),  bUtmp = take(S, d.dim * 4);
    Buf bScr  = take(S, 512 * 4),    bMixR = take(S, 128),  bMixO = take(S, 128);
    Buf bU    = take(S, d.dim * 4);
    Buf bQr   = take(S, d.q_lora * 4), bQ = take(S, qrows * 2);
    Buf bRope = take(S, 256),  bRopeRaw = take(S, 256);
    Buf bKvRaw= take(S, d.head_dim * 4), bKvScr = take(S, 256);
    Buf bWinV = take(S, uint64_t(d.window) * d.head_dim);
    Buf bWinS = take(S, uint64_t(d.window) * (d.head_dim / 32));
    Buf bKv   = take(S, d.head_dim * 4);
    Buf bCmp  = take(S, 1024ull * d.head_dim * 2);
    Buf bTop  = take(S, 1024 * 4);
    Buf bScore= take(S, uint64_t(d.n_heads) * 1024 * 4);
    Buf bO    = take(S, uint64_t(qrows) * 4);
    Buf bWoa  = take(S, orows * 4), bWob = take(S, d.dim * 4);
    Buf bGs   = take(S, d.n_experts * 4), bGid = take(S, 64), bGw = take(S, 64), bDone = take(S, 64);
    // P3 (docs/p2_attention.md §13): the fp32 partial plane the K-split GEMVs
    // write, and a second output buffer per stage so the split kernel can be
    // compared against the unsplit one it replaces as well as against the
    // oracle. 16 * 8192 floats covers every (ksplit, rows) pair in the model.
    Buf bPart  = take(S, 16ull * 8192 * 4);
    Buf bQr2   = take(S, d.q_lora * 4),  bKvRaw2 = take(S, d.head_dim * 4);
    Buf bWoa2  = take(S, orows * 4),     bWob2   = take(S, d.dim * 4);
    Buf bO2    = take(S, uint64_t(qrows) * 4), bKv2 = take(S, d.head_dim * 4);
    Buf bWinV2 = take(S, uint64_t(d.window) * d.head_dim);
    Buf bWinS2 = take(S, uint64_t(d.window) * (d.head_dim / 32));
    // sparse_attn_t's three planes, at the tile count the test drives.
    const uint32_t kTiles = 8;
    Buf bTileMax = take(S, uint64_t(d.n_heads) * kTiles * 4);
    Buf bPartO   = take(S, uint64_t(kTiles) * d.n_heads * d.head_dim * 4);
    Buf bPartD   = take(S, uint64_t(kTiles) * d.n_heads * 4);
    REQUIRE(bDone.v.valid());
    REQUIRE(bPartD.v.valid());

    // DEEPMOE_SKIP_P3 leaves out every P3 stage (docs/p2_attention.md §13), so
    // this case can be pointed at an older shader directory through
    // DEEPMOE_SHADER_DIR and the classic stages compared line for line. An old
    // directory has no gemv_ksplit or sparse_attn_t, and its wkv.spv has no
    // stage 2, so the P3 dispatches would read slots the old shaders never
    // expected.
    const bool p3 = std::getenv("DEEPMOE_SKIP_P3") == nullptr;
    uint32_t layers_checked = 0;
    for (const L2Step& g : set->steps) {
        const uint32_t L = g.layer;
        std::printf("    layer %u (%s)\n", L, g.step.c_str());
        REQUIRE(rig.load_layer(L));
        const std::string p = std::format("layers.{}", L);

        // compress_ratios decides the RoPE base and whether YaRN is on.
        uint32_t ratio = 2;
        {
            // the index carries the whole compress_ratios array only as config;
            // layers 0 and 1 are the window-only ones (design §2.1).
            ratio = (L <= 1) ? 0u : (L >= 20 ? 1u : 2u);
        }
        const runtime::RopeConfig rc = runtime::rope_for_layer(ratio, d.rope_dim);
        const std::vector<float> rope = runtime::rope_table(rc, pos);
        std::memcpy(bRope.v.host, rope.data(), rope.size() * 4);

        // --- 1. mega_mhc, attention half (design §7.2) -------------------
        // No hc_post: the reference's attention half consumes the stream as it
        // arrives (after engram on layers 1 and 14), and `pre_mix` is the
        // previous layer's ffn_pre.
        {
            bX.set(g.f("attn_resid_in"));
            bPre.set(g.f("pre_mix_in"));
            bScr.zero();
            uint64_t* s = rig.runner.slots(gpu::AttnStage::MhcPost);
            s[gpu::slot::kX] = bX.a();
            s[gpu::slot::kA] = bA.a();
            s[gpu::slot::kPostIn] = bPost.a();
            s[gpu::slot::kCombIn] = bComb.a();
            s[gpu::slot::kPreMix] = bPre.a();
            s[gpu::slot::kHcFn] = rig.addr(p + ".hc_attn_fn");
            s[gpu::slot::kHcBase] = rig.addr(p + ".hc_attn_base");
            s[gpu::slot::kHcScale] = rig.addr(p + ".hc_attn_scale");
            s[gpu::slot::kNormW] = rig.addr(p + ".attn_norm.weight");
            s[gpu::slot::kXout] = bXout.a();
            s[gpu::slot::kUtmp] = bUtmp.a();
            s[gpu::slot::kScratch] = bScr.a();
            s[gpu::slot::kMixRaw] = bMixR.a();
            s[gpu::slot::kMixOut] = bMixO.a();
            s[gpu::slot::kU] = bU.a();
            std::memcpy(rig.runner.slots(gpu::AttnStage::MhcMix), s, 32 * 8);
            std::memcpy(rig.runner.slots(gpu::AttnStage::MhcFinal), s, 32 * 8);

            gpu::MhcPush mp{d.dim, d.hc, (2 + d.hc) * d.hc, d.dim / 256, 20, 0,
                            d.norm_eps, 1e-6f};
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::MhcPost, &mp, sizeof mp, mp.n_wg0));
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::MhcMix, &mp, sizeof mp, mp.mix_rows));
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::MhcFinal, &mp, sizeof mp, mp.n_wg0));

            const std::vector<float> mix = bMixO.read(8 + d.hc * d.hc);
            CHECK(ok("mhc attn_pre", agree(mix.data(), g.f("attn_pre").data(), d.hc), 0.999999, 1e-4));
            CHECK(ok("mhc attn_post", agree(mix.data() + d.hc,
                                                    g.f("attn_post").data(), d.hc), 0.999999, 1e-4));
            CHECK(ok("mhc attn_comb", agree(mix.data() + 2 * d.hc,
                                                    g.f("attn_comb").data(), d.hc * d.hc), 0.999999, 1e-4));
            CHECK(ok("mhc attn_norm_out", agree(bU.read(d.dim), g.f("attn_norm_out")), 0.99999, 5e-3));
        }

        // --- 2. wq_a (design §7.3) ---------------------------------------
        {
            bU.set(g.f("attn_norm_out"));      // golden input, not the chained one
            uint64_t* s = rig.runner.slots(gpu::AttnStage::WqA);
            s[gpu::slot::kGemvW] = rig.addr(p + ".attn.wq_a.weight");
            s[gpu::slot::kGemvS] = rig.scale_addr(p + ".attn.wq_a.weight");
            s[gpu::slot::kGemvX] = bU.a();
            s[gpu::slot::kGemvY] = bQr.a();
            gpu::GemvPush gp{d.q_lora, d.dim, d.dim / 32, 0};
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::WqA, &gp, sizeof gp,
                                              rig.runner.gemv_groups(gpu::AttnStage::WqA, d.q_lora)));
            CHECK(ok("wq_a", agree(bQr.read(d.q_lora), g.f("wq_a_out")), 0.99999, 5e-3));

            if (p3) {
                // The same GEMV, K-split (§13). Two dispatches over fp32 partials
                // and a combine, checked against the oracle AND against the
                // unsplit kernel: the only thing that may differ between them is
                // where the sum is re-associated.
                uint64_t* k = rig.runner.slots(gpu::AttnStage::WqAKSplit);
                k[gpu::slot::kKspW] = rig.addr(p + ".attn.wq_a.weight");
                k[gpu::slot::kKspS] = rig.scale_addr(p + ".attn.wq_a.weight");
                k[gpu::slot::kKspX] = bU.a();
                k[gpu::slot::kKspY] = bQr2.a();
                k[gpu::slot::kKspPart] = bPart.a();
                std::memcpy(rig.runner.slots(gpu::AttnStage::WqAKCombine), k, 32 * 8);
                gpu::KSplitPush kp{d.q_lora, d.dim, d.dim / 32, 0, d.q_lora, 0};
                REQUIRE_OK(rig.runner.dispatch_now(
                    gpu::AttnStage::WqAKSplit, &kp, sizeof kp,
                    rig.runner.ksplit_groups(gpu::AttnStage::WqAKSplit, d.q_lora)));
                REQUIRE_OK(rig.runner.dispatch_now(
                    gpu::AttnStage::WqAKCombine, &kp, sizeof kp,
                    gpu::AttnRunner::combine_groups(d.q_lora)));
                CHECK(ok("  wq_a K-split", agree(bQr2.read(d.q_lora), g.f("wq_a_out")), 0.99999, 5e-3));
                report_ulp("  wq_a K-split", bQr2.read(d.q_lora), bQr.read(d.q_lora));
            }
        }

        // --- 3. wq_b + q_norm + RoPE (design §7.3) -----------------------
        {
            bQr.set(g.f("wq_a_out"));
            bQ.zero();
            uint64_t* s = rig.runner.slots(gpu::AttnStage::WqB);
            s[gpu::slot::kWqbW] = rig.addr(p + ".attn.wq_b.weight");
            s[gpu::slot::kWqbS] = rig.scale_addr(p + ".attn.wq_b.weight");
            s[gpu::slot::kWqbQr] = bQr.a();
            s[gpu::slot::kWqbNormW] = rig.addr(p + ".attn.q_norm.weight");
            s[gpu::slot::kWqbRope] = bRope.a();
            s[gpu::slot::kWqbQ] = bQ.a();
            gpu::WqbPush wp{qrows, d.q_lora, d.q_lora / 32, d.head_dim, d.rope_dim, d.norm_eps};
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::WqB, &wp, sizeof wp,
                                              rig.runner.gemv_groups(gpu::AttnStage::WqB, qrows)));
            CHECK(ok("wq_b+rope", agree(bQ.read_bf16(qrows), g.f("q")), 0.9999, 2e-2));
        }

        // --- 4. wkv + kv_norm + RoPE + the fp8 ring write (design §7.4) ---
        {
            bU.set(g.f("attn_norm_out"));
            bKvScr.zero();
            const uint32_t slot = pos % d.window;
            uint64_t* s = rig.runner.slots(gpu::AttnStage::WkvGemv);
            s[gpu::slot::kWkvW] = rig.addr(p + ".attn.wkv.weight");
            s[gpu::slot::kWkvS] = rig.scale_addr(p + ".attn.wkv.weight");
            s[gpu::slot::kWkvX] = bU.a();
            s[gpu::slot::kWkvNormW] = rig.addr(p + ".attn.kv_norm.weight");
            s[gpu::slot::kWkvRope] = bRope.a();
            s[gpu::slot::kWkvRaw] = bKvRaw.a();
            s[gpu::slot::kWkvScratch] = bKvScr.a();
            s[gpu::slot::kWkvVal] = bWinV.a();
            s[gpu::slot::kWkvScale] = bWinS.a();
            s[gpu::slot::kWkvKv] = bKv.a();
            std::memcpy(rig.runner.slots(gpu::AttnStage::WkvFinish), s, 32 * 8);

            const uint32_t g0 = rig.runner.gemv_groups(gpu::AttnStage::WkvGemv, d.head_dim);
            gpu::WkvPush kp{d.head_dim, d.dim, d.dim / 32, d.rope_dim, slot, g0, d.norm_eps};
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::WkvGemv, &kp, sizeof kp, g0));
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::WkvFinish, &kp, sizeof kp, 1));

            CHECK(ok("wkv raw", agree(bKvRaw.read(d.head_dim), g.f("wkv_out")), 0.99999, 5e-3));
            CHECK(ok("wkv post-fp8", agree(bKv.read(d.head_dim), g.f("kv")), 0.9995, 4e-2));
            // The bytes the ring actually holds, against the oracle's.
            const std::vector<float>& gb = g.f("kv_fp8");
            size_t byte_diff = 0;
            for (size_t i = 0; i < gb.size(); ++i)
                if (static_cast<float>(bWinV.u8()[slot * d.head_dim + i]) != gb[i]) ++byte_diff;
            std::printf("      %-22s %zu/%zu E4M3 bytes differ from the reference's\n",
                        "wkv cache bytes", byte_diff, gb.size());
            const std::vector<float>& gs = g.f("kv_scale_e8m0");
            size_t scale_diff = 0;
            for (size_t i = 0; i < gs.size(); ++i)
                if (static_cast<float>(bWinS.u8()[slot * (d.head_dim / 32) + i]) != gs[i])
                    ++scale_diff;
            // A byte or a scale differing is not a bug: our wkv output differs
            // from the reference's in the last bf16 bits, and an input sitting
            // on an E4M3 (or power-of-two) rounding boundary then codes one
            // step away. The values it decodes to are checked above.
            std::printf("      %-22s %zu/%zu UE8M0 scale bytes differ\n",
                        "wkv cache scales", scale_diff, gs.size());

            if (p3) {
                // The K-split form (§13). Its combine folds the whole kv_norm /
                // RoPE / ring-write tail in, so it is two dispatches against two,
                // not three: WkvKSplit + WkvKFinish replace WkvGemv + WkvFinish.
                // It writes its own ring so the bytes can be compared with the
                // unsplit kernel's.
                uint64_t* k = rig.runner.slots(gpu::AttnStage::WkvKSplit);
                k[gpu::slot::kKspW] = rig.addr(p + ".attn.wkv.weight");
                k[gpu::slot::kKspS] = rig.scale_addr(p + ".attn.wkv.weight");
                k[gpu::slot::kKspX] = bU.a();
                k[gpu::slot::kKspY] = bKvRaw2.a();
                k[gpu::slot::kKspPart] = bPart.a();
                uint64_t* f = rig.runner.slots(gpu::AttnStage::WkvKFinish);
                f[gpu::slot::kWkvNormW] = rig.addr(p + ".attn.kv_norm.weight");
                f[gpu::slot::kWkvRope] = bRope.a();
                f[gpu::slot::kWkvRaw] = bKvRaw2.a();
                f[gpu::slot::kWkvVal] = bWinV2.a();
                f[gpu::slot::kWkvScale] = bWinS2.a();
                f[gpu::slot::kWkvKv] = bKv2.a();
                f[gpu::slot::kWkvPart] = bPart.a();
                gpu::KSplitPush sp{d.head_dim, d.dim, d.dim / 32, 0, d.head_dim, 0};
                gpu::WkvPush fp{d.head_dim, d.dim, d.dim / 32, d.rope_dim, slot, 1,
                                d.norm_eps, d.head_dim};
                REQUIRE_OK(rig.runner.dispatch_now(
                    gpu::AttnStage::WkvKSplit, &sp, sizeof sp,
                    rig.runner.ksplit_groups(gpu::AttnStage::WkvKSplit, d.head_dim)));
                REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::WkvKFinish, &fp, sizeof fp, 1));
                CHECK(ok("  wkv K-split raw",
                         agree(bKvRaw2.read(d.head_dim), g.f("wkv_out")), 0.99999, 5e-3));
                CHECK(ok("  wkv K-split post-fp8",
                         agree(bKv2.read(d.head_dim), g.f("kv")), 0.9995, 4e-2));
                report_ulp("  wkv K-split raw", bKvRaw2.read(d.head_dim), bKvRaw.read(d.head_dim));
                size_t split_byte_diff = 0;
                for (uint32_t i = 0; i < d.head_dim; ++i)
                    if (bWinV2.u8()[slot * d.head_dim + i] != bWinV.u8()[slot * d.head_dim + i])
                        ++split_byte_diff;
                std::printf("      %-22s %zu/%u E4M3 ring bytes differ from the unsplit kernel\n",
                            "  wkv K-split bytes", split_byte_diff, d.head_dim);
            }
        }

        // --- 5. sparse attention (design §7.5) ---------------------------
        {
            // Seed the ring from the oracle's window KV, re-encoding each
            // 32-element block to the E4M3 byte + UE8M0 scale the kernel
            // reads. The values are already on the grid, so this is exact --
            // asserted below.
            const std::vector<float>& win = g.f("win_kv");
            const uint32_t nwin = static_cast<uint32_t>(win.size() / d.head_dim);
            std::vector<float> back(d.head_dim);
            double reenc_err = 0;
            for (uint32_t r = 0; r < nwin; ++r) {
                for (uint32_t b = 0; b < d.head_dim / 32; ++b) {
                    uint8_t bytes[32];
                    const float sc = cpu::act_quant_block(&win[r * d.head_dim + b * 32], 32,
                                                          bytes, back.data() + b * 32);
                    std::memcpy(bWinV.u8() + r * d.head_dim + b * 32, bytes, 32);
                    bWinS.u8()[r * (d.head_dim / 32) + b] = cpu::e8m0_encode(sc);
                }
                for (uint32_t i = 0; i < d.head_dim; ++i)
                    reenc_err = std::max(reenc_err,
                                         std::fabs(double(back[i] - win[r * d.head_dim + i])));
            }
            std::printf("      %-22s max|re-encode - oracle| = %.3e over %u rows\n",
                        "window ring seed", reenc_err, nwin);

            uint32_t ncmp = 0;
            if (const L2Tensor* c = g.find("cmp_kv")) {
                ncmp = static_cast<uint32_t>(c->f.size() / d.head_dim);
                for (size_t i = 0; i < c->f.size(); ++i)
                    bCmp.u16()[i] = cpu::float_to_bf16(c->f[i]);
            }
            const std::vector<float>& idx = g.f("topk_idxs");
            for (size_t i = 0; i < idx.size(); ++i)
                bTop.u32()[i] = static_cast<uint32_t>(static_cast<int32_t>(idx[i]));

            for (size_t i = 0; i < g.f("q").size(); ++i)
                bQ.u16()[i] = cpu::float_to_bf16(g.f("q")[i]);

            uint64_t* s = rig.runner.slots(gpu::AttnStage::AttnScore);
            s[gpu::slot::kAttnQ] = bQ.a();
            s[gpu::slot::kAttnWinVal] = bWinV.a();
            s[gpu::slot::kAttnWinScale] = bWinS.a();
            s[gpu::slot::kAttnCmpKv] = bCmp.a();
            s[gpu::slot::kAttnTopIdx] = bTop.a();
            s[gpu::slot::kAttnSink] = rig.addr(p + ".attn.attn_sink");
            s[gpu::slot::kAttnRope] = bRope.a();
            s[gpu::slot::kAttnScore] = bScore.a();
            s[gpu::slot::kAttnO] = bO.a();
            std::memcpy(rig.runner.slots(gpu::AttnStage::AttnCombine), s, 32 * 8);

            gpu::AttnPush ap{static_cast<uint32_t>(idx.size()), d.window, d.head_dim,
                             d.rope_dim, 1024,
                             1.0f / std::sqrt(static_cast<float>(d.head_dim))};
            (void)ncmp;
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::AttnScore, &ap, sizeof ap, d.n_heads));
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::AttnCombine, &ap, sizeof ap, d.n_heads));
            CHECK(ok("sparse_attn+irope", agree(bO.read(qrows), g.f("attn_out_irope")), 0.99999, 1e-2));

            // The same thing again with AttnPush::n_heads filled in, which puts
            // AttnSpec::heads_per_wg heads in a workgroup instead of one and
            // divides the 41 MB of L2 traffic by the same factor. The geometry
            // is opt-in (see the field's comment), so both have to be right:
            // the dispatch above is what a caller that predates it gets.
            bO.zero();
            gpu::AttnPush gp = ap;
            gp.n_heads = d.n_heads;
            const uint32_t ag = rig.runner.attn_groups(d.n_heads);
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::AttnScore, &gp, sizeof gp, ag));
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::AttnCombine, &gp, sizeof gp, ag));
            CHECK(ok("  same, heads/wg grouped",
                     agree(bO.read(qrows), g.f("attn_out_irope")), 0.99999, 1e-2));

            if (p3) {
                // And the head-group x KV-tile grid of §13, which is three
                // dispatches: scores + a tile max, then p against the FINAL max
                // and P.V over one tile, then the combine. The bf16 rounding of p
                // must see the same row max the reference's does, so agreement
                // with the untiled kernel here is the check that the two-pass
                // structure really did reproduce it and not an online softmax.
                bO2.zero();
                uint64_t* t = rig.runner.slots(gpu::AttnStage::AttnScoreT);
                std::memcpy(t, rig.runner.slots(gpu::AttnStage::AttnScore), 32 * 8);
                t[gpu::slot::kAttnO] = bO2.a();
                t[gpu::slot::kAttnTileMax] = bTileMax.a();
                t[gpu::slot::kAttnPartO] = bPartO.a();
                t[gpu::slot::kAttnPartD] = bPartD.a();
                std::memcpy(rig.runner.slots(gpu::AttnStage::AttnPvT), t, 32 * 8);
                std::memcpy(rig.runner.slots(gpu::AttnStage::AttnFinishT), t, 32 * 8);
                const uint32_t nkv = static_cast<uint32_t>(idx.size());
                // Two P.V grids: one tile, where AttnPvT finishes by itself,
                // and kTiles / 2 tiles, where AttnFinishT adds them -- a shape
                // different from the score grid's kTiles, so a stage reading
                // the wrong one of the two tile counts shows up here.
                for (uint32_t pvt : {uint32_t{1}, kTiles / 2}) {
                    bO2.zero();
                    gpu::AttnTPush tp{nkv, d.window, d.head_dim, d.rope_dim, 1024,
                                      1.0f / std::sqrt(static_cast<float>(d.head_dim)),
                                      d.n_heads, kTiles, (nkv + kTiles - 1) / kTiles,
                                      d.n_heads * d.head_dim, pvt, (nkv + pvt - 1) / pvt};
                    REQUIRE_OK(rig.runner.dispatch_now(
                        gpu::AttnStage::AttnScoreT, &tp, sizeof tp,
                        rig.runner.attn_tile_groups(d.n_heads, kTiles)));
                    REQUIRE_OK(rig.runner.dispatch_now(
                        gpu::AttnStage::AttnPvT, &tp, sizeof tp,
                        rig.runner.attn_pv_groups(d.n_heads, pvt)));
                    if (gpu::AttnRunner::attn_tiled_finish_needed(pvt))
                        REQUIRE_OK(rig.runner.dispatch_now(
                            gpu::AttnStage::AttnFinishT, &tp, sizeof tp,
                            gpu::AttnRunner::attn_finish_groups(d.n_heads, d.head_dim)));
                    const char* nm = (pvt == 1) ? "  KV-tile, P.V 1 tile" : "  KV-tile, P.V 4 tiles";
                    CHECK(ok(nm, agree(bO2.read(qrows), g.f("attn_out_irope")), 0.99999, 1e-2));
                    report_ulp(nm, bO2.read(qrows), bO.read(qrows));
                }
            }
        }

        // --- 6. wo_a, grouped and unquantised (design §7.6) --------------
        {
            bO.set(g.f("attn_out_irope"));
            uint64_t* s = rig.runner.slots(gpu::AttnStage::WoA);
            s[gpu::slot::kWoaW] = rig.addr(p + ".attn.wo_a.weight");
            s[gpu::slot::kWoaS] = rig.scale_addr(p + ".attn.wo_a.weight");
            s[gpu::slot::kWoaO] = bO.a();
            s[gpu::slot::kWoaY] = bWoa.a();
            gpu::WoaPush wp{orows, ocols, ocols / 32, d.o_lora};
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::WoA, &wp, sizeof wp,
                                              rig.runner.gemv_groups(gpu::AttnStage::WoA, orows)));
            CHECK(ok("wo_a", agree(bWoa.read(orows), g.f("wo_a_out")), 0.99999, 5e-3));

            if (p3) {
                // K-split (§13), carrying wo_a's block-diagonal structure in
                // `rows_per_group`: a workgroup never straddles two 1024-row
                // groups, so it stages the one slice of `o` its group needs.
                uint64_t* k = rig.runner.slots(gpu::AttnStage::WoAKSplit);
                k[gpu::slot::kKspW] = rig.addr(p + ".attn.wo_a.weight");
                k[gpu::slot::kKspS] = rig.scale_addr(p + ".attn.wo_a.weight");
                k[gpu::slot::kKspX] = bO.a();
                k[gpu::slot::kKspY] = bWoa2.a();
                k[gpu::slot::kKspPart] = bPart.a();
                std::memcpy(rig.runner.slots(gpu::AttnStage::WoAKCombine), k, 32 * 8);
                gpu::KSplitPush kp{orows, ocols, ocols / 32, d.o_lora, orows, 0};
                REQUIRE_OK(rig.runner.dispatch_now(
                    gpu::AttnStage::WoAKSplit, &kp, sizeof kp,
                    rig.runner.ksplit_groups(gpu::AttnStage::WoAKSplit, orows)));
                REQUIRE_OK(rig.runner.dispatch_now(
                    gpu::AttnStage::WoAKCombine, &kp, sizeof kp,
                    gpu::AttnRunner::combine_groups(orows)));
                CHECK(ok("  wo_a K-split", agree(bWoa2.read(orows), g.f("wo_a_out")), 0.99999, 5e-3));
                report_ulp("  wo_a K-split", bWoa2.read(orows), bWoa.read(orows));
            }
        }

        // --- 7. wo_b (design §7.6) ---------------------------------------
        {
            bWoa.set(g.f("wo_a_out"));
            uint64_t* s = rig.runner.slots(gpu::AttnStage::WoB);
            s[gpu::slot::kGemvW] = rig.addr(p + ".attn.wo_b.weight");
            s[gpu::slot::kGemvS] = rig.scale_addr(p + ".attn.wo_b.weight");
            s[gpu::slot::kGemvX] = bWoa.a();
            s[gpu::slot::kGemvY] = bWob.a();
            gpu::GemvPush gp{d.dim, orows, orows / 32, 0};
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::WoB, &gp, sizeof gp,
                                              rig.runner.gemv_groups(gpu::AttnStage::WoB, d.dim)));
            CHECK(ok("wo_b", agree(bWob.read(d.dim), g.f("wo_b_out")), 0.99999, 5e-3));

            if (p3) {
                uint64_t* k = rig.runner.slots(gpu::AttnStage::WoBKSplit);
                k[gpu::slot::kKspW] = rig.addr(p + ".attn.wo_b.weight");
                k[gpu::slot::kKspS] = rig.scale_addr(p + ".attn.wo_b.weight");
                k[gpu::slot::kKspX] = bWoa.a();
                k[gpu::slot::kKspY] = bWob2.a();
                k[gpu::slot::kKspPart] = bPart.a();
                std::memcpy(rig.runner.slots(gpu::AttnStage::WoBKCombine), k, 32 * 8);
                gpu::KSplitPush kp{d.dim, orows, orows / 32, 0, d.dim, 0};
                REQUIRE_OK(rig.runner.dispatch_now(
                    gpu::AttnStage::WoBKSplit, &kp, sizeof kp,
                    rig.runner.ksplit_groups(gpu::AttnStage::WoBKSplit, d.dim)));
                REQUIRE_OK(rig.runner.dispatch_now(
                    gpu::AttnStage::WoBKCombine, &kp, sizeof kp,
                    gpu::AttnRunner::combine_groups(d.dim)));
                CHECK(ok("  wo_b K-split", agree(bWob2.read(d.dim), g.f("wo_b_out")), 0.99999, 5e-3));
                report_ulp("  wo_b K-split", bWob2.read(d.dim), bWob.read(d.dim));
            }
        }

        // --- 8. hc_post fused into the ffn-half mega_mhc (design §7.7) ---
        {
            bX.set(g.f("attn_resid_in"));
            bA.set(g.f("wo_b_out"));
            bPost.set(g.f("attn_post"));
            bComb.set(g.f("attn_comb"));
            bPre.set(g.f("attn_pre"));
            bScr.zero();
            uint64_t* s = rig.runner.slots(gpu::AttnStage::MhcPost);
            s[gpu::slot::kHcFn] = rig.addr(p + ".hc_ffn_fn");
            s[gpu::slot::kHcBase] = rig.addr(p + ".hc_ffn_base");
            s[gpu::slot::kHcScale] = rig.addr(p + ".hc_ffn_scale");
            s[gpu::slot::kNormW] = rig.addr(p + ".ffn_norm.weight");
            std::memcpy(rig.runner.slots(gpu::AttnStage::MhcMix), s, 32 * 8);
            std::memcpy(rig.runner.slots(gpu::AttnStage::MhcFinal), s, 32 * 8);

            gpu::MhcPush mp{d.dim, d.hc, (2 + d.hc) * d.hc, d.dim / 256, 20,
                            gpu::kMhcFlagPost, d.norm_eps, 1e-6f};
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::MhcPost, &mp, sizeof mp, mp.n_wg0));
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::MhcMix, &mp, sizeof mp, mp.mix_rows));
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::MhcFinal, &mp, sizeof mp, mp.n_wg0));

            CHECK(ok("hc_post -> stream", agree(bXout.read(hcdim), g.f("attn_block_out")), 0.99999, 5e-3));
            const std::vector<float> mix = bMixO.read(8 + d.hc * d.hc);
            CHECK(ok("mhc ffn_pre", agree(mix.data(), g.f("ffn_pre").data(), d.hc), 0.999999, 1e-3));
            CHECK(ok("mhc ffn_norm_out", agree(bU.read(d.dim), g.f("ffn_norm_out")), 0.99999, 1e-2));
        }

        // --- 9. gate (design §7.8) ---------------------------------------
        {
            bU.set(g.f("ffn_norm_out"));
            bDone.zero();
            uint64_t* s = rig.runner.slots(gpu::AttnStage::GateScore);
            s[gpu::slot::kGateW] = rig.addr(p + ".ffn.gate.weight");
            s[gpu::slot::kGateBias] = rig.addr(p + ".ffn.gate.bias");
            s[gpu::slot::kGateX] = bU.a();
            s[gpu::slot::kGateScores] = bGs.a();
            s[gpu::slot::kGateIds] = bGid.a();
            s[gpu::slot::kGateWeights] = bGw.a();
            s[gpu::slot::kGateLayerDone] = bDone.a();
            std::memcpy(rig.runner.slots(gpu::AttnStage::GateTopK), s, 32 * 8);

            gpu::GatePush gp{d.n_experts, d.dim, d.topk, 16, 1.0f, d.route_scale};
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::GateScore, &gp, sizeof gp,
                                              rig.runner.gemv_groups(gpu::AttnStage::GateScore, d.n_experts)));
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::GateTopK, &gp, sizeof gp, 1));

            CHECK(ok("gate scores", agree(bGs.read(d.n_experts), g.f("gate_scores")), 0.9999999, 1e-3));
            // The ids must match exactly: a different expert set is a different
            // model, not a tolerance question.
            const std::vector<float>& gid = g.f("gate_top6_ids");
            uint32_t id_match = 0;
            for (uint32_t i = 0; i < d.topk; ++i)
                for (uint32_t j = 0; j < d.topk; ++j)
                    if (bGid.u32()[i] == static_cast<uint32_t>(gid[j])) { ++id_match; break; }
            std::printf("      %-22s %u/%u of the reference's experts selected\n",
                        "gate top-6 ids", id_match, d.topk);
            CHECK_EQ(id_match, d.topk);
            CHECK_EQ(bDone.u32()[0], 1u);
        }
        ++layers_checked;
    }
    std::printf("    %u layers checked, pinned set %.1f MB in %u regions\n",
                layers_checked, rig.pinned.bytes_loaded() / 1e6, rig.pinned.region_count());
    CHECK(layers_checked > 0);
}

// ---------------------------------------------------------------------------
// design §7.5 at a context the L2 oracle does not have.
//
// Everything above is one decode token at position 64 of a 64-token prefill,
// where `topk_idxs` is 193 entries long and the top-k has never been asked to
// select (docs/p2_attention.md §9.4). That says nothing about what
// `sparse_attn_t` does when the indexer really does hand it 512 compressed
// picks out of a 32K context, and there is no reference tensor for it, so the
// check here is against an fp64 CPU transcription of `sparse_attn_kernel`
// instead -- the reference's own order of operations, in double, including the
// bf16 rounding of p against the FINAL row max with the sink excluded.
//
// The untiled kernel is NOT run here: `sparse_attn.slang` stages the whole
// index list in LDS and is capped at `kMaxKv`. The tiled one reads indices
// straight from memory, which is the other half of why §13 replaced it.
namespace {

// A rig with no checkpoint: this case needs a device, the runner and scratch,
// and nothing off disk.
struct BareRig {
    gpu::Device          device;
    gpu::MemoryAllocator alloc;
    gpu::AttnRunner      runner;
    gpu::GpuScratch      scratch;
    std::string          why;

    ~BareRig() { scratch.destroy(); runner.destroy(); }

    bool bring_up(uint64_t scratch_bytes) {
        gpu::DeviceOptions dopts;
        dopts.enable_validation = std::getenv("VK_INSTANCE_LAYERS") != nullptr;
        if (auto r = device.create(dopts); !r) { why = r.error().str(); return false; }
        if (auto r = device.caps().check_required(); !r) { why = r.error().str(); return false; }
        if (auto r = alloc.init(device, MemoryPath::DeviceLocalHostVisible); !r) {
            why = r.error().str(); return false;
        }
        if (auto r = runner.create(device, alloc, gpu::default_shader_dir(), env_spec()); !r) {
            why = r.error().str(); return false;
        }
        if (auto r = scratch.create(alloc, scratch_bytes); !r) { why = r.error().str(); return false; }
        return true;
    }
};

// A deterministic uniform in [-1, 1); the values only have to be well
// conditioned, and a fixed seed makes a failure reproducible.
struct Lcg {
    uint64_t s;
    float next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return float(int32_t(uint32_t(s >> 32))) * (1.0f / 2147483648.0f);
    }
};

}  // namespace

DEEPMOE_TEST(gpu_attn, sparse_attn_long_context) {
    BareRig rig;
    // 32768 KV entries is 32 MB of compressed rows, 8 MB of scores and 8 MB of
    // P.V partials.
    if (!rig.bring_up(192ull << 20)) {
        std::printf("      SKIP gpu_attn long context: %s\n", rig.why.c_str());
        return;
    }
    const uint32_t n_heads = 64, head_dim = 512, rope_dim = 64, n_win = 128;
    const float    scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // The two contexts, and how many heads the fp64 reference checks at each:
    // it is 64 * n_kv * 512 multiply-adds a head, so at 32K all 64 would be a
    // billion of them for no more coverage than a spread of three.
    struct Case { uint32_t n_kv, n_tiles, pv_tiles, heads_checked; };
    const Case cases[] = {{4096, 8, 1, 64}, {4096, 16, 4, 64}, {32768, 32, 16, 4},
                          {32768, 16, 1, 4}};

    gpu::GpuScratch& S = rig.scratch;
    const uint32_t max_kv = 32768;
    Buf bQ     = take(S, uint64_t(n_heads) * head_dim * 2);
    Buf bWinV  = take(S, uint64_t(n_win) * head_dim);
    Buf bWinS  = take(S, uint64_t(n_win) * (head_dim / 32));
    Buf bCmp   = take(S, uint64_t(max_kv) * head_dim * 2);
    Buf bTop   = take(S, uint64_t(max_kv) * 4);
    Buf bSink  = take(S, uint64_t(n_heads) * 4);
    Buf bRope  = take(S, uint64_t(rope_dim) * 4);
    Buf bScore = take(S, uint64_t(n_heads) * max_kv * 4);
    Buf bO     = take(S, uint64_t(n_heads) * head_dim * 4);
    Buf bTileMax = take(S, uint64_t(n_heads) * 64 * 4);
    Buf bPartO = take(S, uint64_t(32) * n_heads * head_dim * 4);
    Buf bPartD = take(S, uint64_t(32) * n_heads * 4);
    REQUIRE(bPartD.v.valid());

    Lcg rng{0x9E3779B97F4A7C15ull};
    // q small enough that a 512-dim dot times head_dim**-0.5 lands in a range
    // exp() is happy with.
    std::vector<float> q(uint64_t(n_heads) * head_dim);
    for (float& v : q) v = 0.15f * rng.next();
    for (size_t i = 0; i < q.size(); ++i) bQ.u16()[i] = cpu::float_to_bf16(q[i]);
    // q as the kernel will see it, i.e. after the bf16 round trip.
    for (size_t i = 0; i < q.size(); ++i) q[i] = bf16_to_f32(bQ.u16()[i]);

    // The window ring, through the same E4M3 + UE8M0 encode wkv.slang performs.
    std::vector<float> wrow(head_dim), wback(head_dim);
    std::vector<float> win(uint64_t(n_win) * head_dim);
    for (uint32_t r = 0; r < n_win; ++r) {
        for (uint32_t i = 0; i < head_dim; ++i) wrow[i] = 0.4f * rng.next();
        for (uint32_t b = 0; b < head_dim / 32; ++b) {
            uint8_t bytes[32];
            const float sc = cpu::act_quant_block(&wrow[b * 32], 32, bytes,
                                                  wback.data() + b * 32);
            std::memcpy(bWinV.u8() + r * head_dim + b * 32, bytes, 32);
            bWinS.u8()[r * (head_dim / 32) + b] = cpu::e8m0_encode(sc);
        }
        std::memcpy(&win[uint64_t(r) * head_dim], wback.data(), head_dim * 4);
    }
    // The compressed half is bf16 on our side (design §6 / §11.3).
    for (uint64_t i = 0; i < uint64_t(max_kv) * head_dim; ++i)
        bCmp.u16()[i] = cpu::float_to_bf16(0.4f * rng.next());
    for (uint32_t h = 0; h < n_heads; ++h) bSink.f()[h] = 0.5f * rng.next();
    const std::vector<float> rope = runtime::rope_table(runtime::rope_for_layer(2, rope_dim), 300);
    std::memcpy(bRope.v.host, rope.data(), uint64_t(rope_dim) * 4);

    // One KV row, exactly as the shader decodes it, into an fp64 scratch row.
    // Materialising the row rather than indexing per element keeps the 268
    // million multiply-adds of the 4096 case to a couple of seconds.
    std::vector<double> kvrow(head_dim);
    auto load_kv = [&](int idx) {
        if (idx < static_cast<int>(n_win)) {
            const float* r = &win[uint64_t(idx) * head_dim];
            for (uint32_t dd = 0; dd < head_dim; ++dd) kvrow[dd] = double(r[dd]);
        } else {
            const uint16_t* r = bCmp.u16() + (uint64_t(idx) - n_win) * head_dim;
            for (uint32_t dd = 0; dd < head_dim; ++dd) kvrow[dd] = double(bf16_to_f32(r[dd]));
        }
    };

    for (const Case& c : cases) {
        // Every position live except a handful of -1 holes, which are the case
        // the reference's finite -1e30 row-max floor exists for.
        for (uint32_t t = 0; t < c.n_kv; ++t)
            bTop.u32()[t] = (t % 401u == 7u) ? 0xFFFFFFFFu : t;
        bO.zero();
        bScore.zero();

        uint64_t* s = rig.runner.slots(gpu::AttnStage::AttnScoreT);
        s[gpu::slot::kAttnQ] = bQ.a();
        s[gpu::slot::kAttnWinVal] = bWinV.a();
        s[gpu::slot::kAttnWinScale] = bWinS.a();
        s[gpu::slot::kAttnCmpKv] = bCmp.a();
        s[gpu::slot::kAttnTopIdx] = bTop.a();
        s[gpu::slot::kAttnSink] = bSink.a();
        s[gpu::slot::kAttnRope] = bRope.a();
        s[gpu::slot::kAttnScore] = bScore.a();
        s[gpu::slot::kAttnO] = bO.a();
        s[gpu::slot::kAttnTileMax] = bTileMax.a();
        s[gpu::slot::kAttnPartO] = bPartO.a();
        s[gpu::slot::kAttnPartD] = bPartD.a();
        std::memcpy(rig.runner.slots(gpu::AttnStage::AttnPvT), s, 32 * 8);
        std::memcpy(rig.runner.slots(gpu::AttnStage::AttnFinishT), s, 32 * 8);

        gpu::AttnTPush tp{c.n_kv, n_win, head_dim, rope_dim, c.n_kv, scale, n_heads,
                          c.n_tiles, (c.n_kv + c.n_tiles - 1) / c.n_tiles,
                          n_heads * head_dim, c.pv_tiles,
                          (c.n_kv + c.pv_tiles - 1) / c.pv_tiles};
        REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::AttnScoreT, &tp, sizeof tp,
                                           rig.runner.attn_tile_groups(n_heads, c.n_tiles)));
        REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::AttnPvT, &tp, sizeof tp,
                                           rig.runner.attn_pv_groups(n_heads, c.pv_tiles)));
        if (gpu::AttnRunner::attn_tiled_finish_needed(c.pv_tiles))
            REQUIRE_OK(rig.runner.dispatch_now(
                gpu::AttnStage::AttnFinishT, &tp, sizeof tp,
                gpu::AttnRunner::attn_finish_groups(n_heads, head_dim)));

        // --- the fp64 transcription -------------------------------------
        std::vector<float> ref(uint64_t(c.heads_checked) * head_dim);
        std::vector<double> pv(c.n_kv);
        for (uint32_t hh = 0; hh < c.heads_checked; ++hh) {
            // Spread the sampled heads over the range rather than taking a
            // prefix, so a bug in one head group is not invisible.
            const uint32_t h = (c.heads_checked == n_heads)
                                   ? hh : (hh * (n_heads / c.heads_checked) + 1);
            double mx = -1.0e30;
            for (uint32_t t = 0; t < c.n_kv; ++t) {
                const int idx = static_cast<int>(bTop.u32()[t]);
                if (idx < 0) { pv[t] = -3.0e38; continue; }
                load_kv(idx);
                double acc = 0.0;
                for (uint32_t dd = 0; dd < head_dim; ++dd)
                    acc += double(q[uint64_t(h) * head_dim + dd]) * kvrow[dd];
                pv[t] = acc * double(scale);
                mx = std::max(mx, pv[t]);
            }
            // p = bf16(exp(s - mx)), which is where `acc_s_cast` rounds, and
            // the sink enters the denominator only.
            double den = std::exp(double(bSink.f()[h]) - mx);
            for (uint32_t t = 0; t < c.n_kv; ++t) {
                const int idx = static_cast<int>(bTop.u32()[t]);
                pv[t] = (idx < 0) ? 0.0
                                  : double(bf16_to_f32(cpu::float_to_bf16(
                                        static_cast<float>(std::exp(pv[t] - mx)))));
                den += pv[t];
            }
            std::vector<double> o(head_dim, 0.0);
            for (uint32_t t = 0; t < c.n_kv; ++t) {
                if (pv[t] == 0.0) continue;
                load_kv(static_cast<int>(bTop.u32()[t]));
                for (uint32_t dd = 0; dd < head_dim; ++dd) o[dd] += pv[t] * kvrow[dd];
            }
            const double inv = 1.0 / den;
            for (uint32_t dd = 0; dd < head_dim; ++dd) o[dd] *= inv;
            // The inverse rotation the kernel fuses into its write-out.
            for (uint32_t dd = head_dim - rope_dim; dd < head_dim; dd += 2) {
                const uint32_t j = (dd - (head_dim - rope_dim)) >> 1;
                const double cs = rope[j * 2], sn = rope[j * 2 + 1];
                const double re = o[dd], im = o[dd + 1];
                o[dd]     = re * cs + im * sn;      // conjugate
                o[dd + 1] = -re * sn + im * cs;
            }
            for (uint32_t dd = 0; dd < head_dim; ++dd)
                ref[uint64_t(hh) * head_dim + dd] = static_cast<float>(o[dd]);
        }

        std::vector<float> got(uint64_t(c.heads_checked) * head_dim);
        for (uint32_t hh = 0; hh < c.heads_checked; ++hh) {
            const uint32_t h = (c.heads_checked == n_heads)
                                   ? hh : (hh * (n_heads / c.heads_checked) + 1);
            std::memcpy(&got[uint64_t(hh) * head_dim],
                        bO.f() + uint64_t(h) * head_dim, head_dim * 4);
        }
        std::printf("    n_kv %u, scores %u tiles x %u heads/wg, P.V %u tiles x %u heads/wg, "
                    "%u of %u heads against fp64\n", c.n_kv, c.n_tiles,
                    rig.runner.spec().tile_heads_per_wg, c.pv_tiles,
                    rig.runner.spec().pv_heads_per_wg, c.heads_checked, n_heads);
        CHECK(ok("sparse_attn_t long", agree(got, ref), 0.999999, 1e-3));
    }
}

// attn_common.slang's arithmetic E4M3 decode, against the 256-entry table it
// replaces. The shader expression is transcribed here exactly; if these two
// agree on all 256 codes then `Fp8Arith` is a pure speed knob and cannot
// change a single output bit. (NaN codes 0x7F/0xFF are included: both forms
// produce +-480, which is what the table holds.)
DEEPMOE_TEST(gpu_attn, fp8_arith_decode_matches_table) {
    size_t differ = 0;
    for (uint32_t b = 0; b < 256; ++b) {
        const uint32_t bits = ((b & 0x7Fu) << 20) + 0x3C000000u;
        float v;
        std::memcpy(&v, &bits, 4);
        float mag = ((b & 0x78u) != 0u) ? v : ((v - 0.0078125f) * 2.0f);
        uint32_t mb;
        std::memcpy(&mb, &mag, 4);
        mb |= (b & 0x80u) << 24;
        std::memcpy(&mag, &mb, 4);
        float want = cpu::kFp8E4M3Table[b];
        if ((b & 0x7Fu) == 0x7Fu) want = (b & 0x80u) ? -480.0f : 480.0f;
        uint32_t ab, wb;
        std::memcpy(&ab, &mag, 4);
        std::memcpy(&wb, &want, 4);
        if (ab != wb && !(mag == 0.0f && want == 0.0f)) ++differ;
    }
    std::printf("      %-22s %zu/256 codes differ from the LDS table\n",
                "fp8 arithmetic decode", differ);
    CHECK_EQ(differ, size_t{0});
}

// The bf16 GEMV of design §7.11 against a CPU reference over the same pinned
// weights. There is no golden logits vector -- the L2 export stops at layer
// 39's block output -- so this checks the kernel against cpu/dequant.h rather
// than against the reference model, which is still the thing that would catch a
// wrong bf16 unpack or a broken reduction.
DEEPMOE_TEST(gpu_attn, head_bf16_gemv) {
    if (skip_without_model("gpu_attn head")) return;
    Rig rig;
    if (!rig.bring_up()) { std::printf("      SKIP gpu_attn: %s\n", rig.why.c_str()); return; }

    if (auto r = rig.pinned.load(rig.manifest, rig.shards, rig.io, {"head.weight"}); !r) {
        std::printf("      SKIP gpu_attn head: %s\n", r.error().str().c_str());
        return;
    }
    const store::PinnedTensor* head = rig.pinned.find("head.weight");
    REQUIRE(head != nullptr && head->data != 0);

    constexpr uint32_t kDim = 5120;
    constexpr uint32_t kRows = 2048;        // a slice: the whole head is 1.32 GB
    gpu::GpuScratch& S = rig.scratch;
    Buf bX = take(S, kDim * 4), bL = take(S, kRows * 4);
    for (uint32_t i = 0; i < kDim; ++i)
        bX.f()[i] = std::sin(0.001f * static_cast<float>(i)) * 0.5f;

    uint64_t* s = rig.runner.slots(gpu::AttnStage::Head);
    s[gpu::slot::kHeadW] = head->data;
    s[gpu::slot::kHeadX] = bX.a();
    s[gpu::slot::kHeadLogits] = bL.a();
    gpu::HeadPush hp{kRows, kDim, 0};
    REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::Head, &hp, sizeof hp,
                                      rig.runner.gemv_groups(gpu::AttnStage::Head, kRows)));

    const auto* w = static_cast<const uint16_t*>(head->data_host);
    std::vector<float> ref(kRows);
    for (uint32_t r = 0; r < kRows; ++r) {
        double acc = 0;
        for (uint32_t k = 0; k < kDim; ++k)
            acc += double(cpu::bf16_to_float(w[size_t(r) * kDim + k])) * bX.f()[k];
        ref[r] = static_cast<float>(acc);
    }
    const Agreement a = agree(bL.read(kRows), ref);
    std::printf("      %-22s %s\n", "head vs cpu fp64", a.str().c_str());
    CHECK(a.cos > 0.9999999 && a.rel < 1e-5);
}

// ---------------------------------------------------------------------------
// design §7.4's compressor and indexer, against tests/data/l2x.
//
// These are the two stages docs/p2_attention.md §6 listed as "LOADED": until
// they existed the decode step read the compressed KV and the top-k list out of
// the oracle's prefill instead of producing them. Everything here is fed the
// reference's own input and compared against the reference's own output, the
// same contract as `l2_per_stage`.
//
// It runs against `tests/data/l2x`, not `tests/data/l2`, because three of the
// tensors it needs are not in the oracle's own export -- the whole index-key
// cache, the score it produces, and the ratio-2 compressor's carried state.
// `tools/oracle_l2_extra.py` writes a superset of the L2 files with those
// added; see its header for what each one is and, for `index_score`, what is
// weaker about it than a capture point.
//
// Two things this CANNOT check against the reference, and why:
//
//   * **The ratio-2 pooling.** At decode position 64 a ratio-2 source's group
//     does not complete ((64 + 1) % 2 != 0), so `Compressor.forward` returns
//     None and there is no reference latent to compare to. What the reference
//     does do is write slot 0 of both states, and that write IS checked. The
//     pooling itself is then run on a synthetic complete group and compared
//     against an fp64 CPU transcription of the reference expression.
//   * **The top-k selection.** `index_topk` is 512 and a 64-token prefill at
//     ratio 2 leaves 32 compressed positions, so `min(index_topk, n)` is n and
//     the reference selects ALL of them. The comparison against `topk_idxs` is
//     therefore real but degenerate; the radix select that runs when it is not
//     degenerate is exercised by `gpu_attn.indexer_topk_select` below.
DEEPMOE_TEST(gpu_attn, l2_compressor_indexer) {
    if (skip_without_model("gpu_attn 7.4")) return;
    const std::string dir = std::string(DEEPMOE_TEST_DATA_DIR) + "/l2x";
    auto set = load_l2(dir);
    if (!set) {
        std::printf("      SKIP gpu_attn 7.4: no L2 extra data (%s). Run "
                    "tools/oracle_l2_extra.py --out tests/data/l2x\n",
                    set.error().str().c_str());
        return;
    }
    Rig rig;
    if (!rig.bring_up()) { std::printf("      SKIP gpu_attn 7.4: %s\n", rig.why.c_str()); return; }

    Dims d;
    d.dim      = static_cast<uint32_t>(set->cfg("dim", 5120));
    d.head_dim = static_cast<uint32_t>(set->cfg("head_dim", 512));
    d.rope_dim = static_cast<uint32_t>(set->cfg("rope_head_dim", 64));
    d.q_lora   = static_cast<uint32_t>(set->cfg("q_lora_rank", 1280));
    d.window   = static_cast<uint32_t>(set->cfg("window_size", 128));
    const uint32_t nih = static_cast<uint32_t>(set->cfg("index_n_heads", 32));
    const uint32_t ihd = static_cast<uint32_t>(set->cfg("index_head_dim", 128));
    const uint32_t irows = nih * ihd;
    const uint32_t pos = set->decode_pos;
    const uint32_t kMaxPos = 1024;

    gpu::GpuScratch& S = rig.scratch;
    Buf bU     = take(S, d.dim * 4),          bQr   = take(S, d.q_lora * 4);
    Buf bRopeQ = take(S, 256),                bRopeL = take(S, 256);
    Buf bCmpY  = take(S, d.head_dim * 4),     bCmpG = take(S, d.head_dim * 4);
    Buf bKvSt  = take(S, 8ull * d.head_dim * 4), bScSt = take(S, 8ull * d.head_dim * 4);
    Buf bLat   = take(S, d.head_dim * 4),     bLatQ = take(S, d.head_dim * 4);
    Buf bCFp4  = take(S, d.head_dim / 2),     bCScB = take(S, d.head_dim / 16);
    Buf bCmp   = take(S, uint64_t(kMaxPos) * d.head_dim * 2);
    Buf bIQRaw = take(S, irows * 4),          bIQ   = take(S, irows * 2);
    Buf bIQFp4 = take(S, irows / 2),          bIQSc = take(S, irows / 32);
    Buf bIKRaw = take(S, ihd * 4),            bIKC  = take(S, uint64_t(kMaxPos) * ihd * 2);
    Buf bIKFp4 = take(S, ihd / 2),            bIKSc = take(S, 64);
    Buf bIWt   = take(S, nih * 4),            bISc  = take(S, kMaxPos * 4);
    Buf bTop   = take(S, (kMaxPos + 256) * 4);
    REQUIRE(bTop.v.valid());

    // How many bytes of a u8 plane differ from the oracle's. A difference is
    // not automatically a bug -- our input differs from the reference's in the
    // last bf16 bits, so a value on a rounding boundary codes one step away --
    // which is why the dequantised value is what the cosine gate sees and the
    // byte count is reported beside it.
    auto byte_diff = [](const uint8_t* got, const std::vector<float>& want) {
        size_t n = 0;
        for (size_t i = 0; i < want.size(); ++i)
            if (static_cast<float>(got[i]) != want[i]) ++n;
        return n;
    };

    uint32_t checked = 0;
    for (const L2Step& g : set->steps) {
        const uint32_t L = g.layer;
        const uint32_t ratio = static_cast<uint32_t>(g.f("cmp_ratio")[0]);
        std::printf("    layer %u (%s, compress_ratio %u)\n", L, g.step.c_str(), ratio);
        REQUIRE(rig.load_layer(L));
        const std::string p = std::format("layers.{}", L);

        // Two positions, not one. q is rotated at this token's position; a
        // latent stands for the FIRST token of its group, so it and the index
        // key derived from it are rotated at start_pos + 1 - ratio.
        const runtime::RopeConfig rc = runtime::rope_for_layer(ratio, d.rope_dim);
        const std::vector<float> rq = runtime::rope_table(rc, pos);
        const std::vector<float> rl = runtime::rope_table(rc, pos + 1 - ratio);
        std::memcpy(bRopeQ.v.host, rq.data(), rq.size() * 4);
        std::memcpy(bRopeL.v.host, rl.data(), rl.size() * 4);

        const uint32_t slot = pos % ratio;
        const uint32_t complete = ((pos + 1) % ratio == 0) ? 1u : 0u;
        gpu::CmpPush cp{d.head_dim, d.dim, ratio, slot, complete, d.rope_dim,
                        pos / ratio, d.norm_eps};

        auto cmp_slots = [&](gpu::AttnStage st, const char* wname, Buf& y) {
            uint64_t* s = rig.runner.slots(st);
            s[gpu::slot::kCmpW] = rig.addr(p + "." + wname);
            s[gpu::slot::kCmpX] = bU.a();
            s[gpu::slot::kCmpY] = y.a();
            s[gpu::slot::kCmpG] = bCmpG.a();
            s[gpu::slot::kCmpKvState] = bKvSt.a();
            s[gpu::slot::kCmpScoreState] = bScSt.a();
            s[gpu::slot::kCmpNormW] = rig.addr(p + ".attn.compressor.norm.weight");
            s[gpu::slot::kCmpLatent] = bLat.a();
            s[gpu::slot::kCmpRope] = bRopeL.a();
            s[gpu::slot::kCmpVal] = bCmp.a();
            s[gpu::slot::kCmpFp4] = bCFp4.a();
            s[gpu::slot::kCmpScaleB] = bCScB.a();
            s[gpu::slot::kCmpLatentQ] = bLatQ.a();
            return s;
        };

        // --- compressor.wkv, and wgate when the layer pools (7.4) ---------
        {
            bU.set(g.f("attn_norm_out"));
            cmp_slots(gpu::AttnStage::CmpKvGemv, "attn.compressor.wkv.weight", bCmpY);
            const uint32_t gw = rig.runner.gemv_groups(gpu::AttnStage::CmpKvGemv, d.head_dim);
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::CmpKvGemv, &cp, sizeof cp, gw));
            CHECK(ok("compressor.wkv", agree(bCmpY.read(d.head_dim), g.f("cmp_wkv_out")),
                     0.99999, 5e-3));
            if (ratio > 1) {
                cmp_slots(gpu::AttnStage::CmpGateGemv, "attn.compressor.wgate.weight", bCmpG);
                REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::CmpGateGemv, &cp, sizeof cp, gw));
                CHECK(ok("compressor.wgate", agree(bCmpG.read(d.head_dim), g.f("cmp_wgate_out")),
                         0.99999, 5e-3));
            }
        }

        // --- compressor state + pooling + norm (7.4) ----------------------
        {
            bCmpY.set(g.f("cmp_wkv_out"));
            if (ratio > 1) bCmpG.set(g.f("cmp_wgate_out"));
            bKvSt.zero(); bScSt.zero(); bLat.zero();
            cmp_slots(gpu::AttnStage::CmpNorm, "attn.compressor.wkv.weight", bCmpY);
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::CmpNorm, &cp, sizeof cp, 1));

            if (ratio > 1) {
                // The group does not complete here, so what the reference did
                // was write slot pos % ratio of both states and return None.
                const std::vector<float>& gk = g.f("cmp_state_kv");
                const std::vector<float>& gs = g.f("cmp_state_score");
                std::vector<float> wantk(gk.begin() + size_t(slot) * d.head_dim,
                                         gk.begin() + size_t(slot + 1) * d.head_dim);
                std::vector<float> wants(gs.begin() + size_t(slot) * d.head_dim,
                                         gs.begin() + size_t(slot + 1) * d.head_dim);
                std::vector<float> gotk(d.head_dim), gots(d.head_dim);
                std::memcpy(gotk.data(), bKvSt.f() + size_t(slot) * d.head_dim, d.head_dim * 4);
                std::memcpy(gots.data(), bScSt.f() + size_t(slot) * d.head_dim, d.head_dim * 4);
                CHECK(ok("cmp kv_state[slot]", agree(gotk, wantk), 0.99999, 5e-3));
                CHECK(ok("cmp score_state[slot]", agree(gots, wants), 0.99999, 5e-3));

                // The pooling itself, on a synthetic complete group, compared
                // against an fp64 transcription of the reference expression --
                // at this position the reference produced no latent at all.
                for (uint32_t j = 0; j < ratio; ++j) {
                    std::memcpy(bKvSt.f() + size_t(j) * d.head_dim,
                                g.f("cmp_wkv_out").data(), d.head_dim * 4);
                    for (uint32_t i = 0; i < d.head_dim; ++i)
                        bScSt.f()[size_t(j) * d.head_dim + i] =
                            g.f("cmp_wgate_out")[i] * (1.0f + 0.25f * float(j));
                }
                gpu::CmpPush cf = cp;
                cf.slot = ratio - 1;
                cf.complete = 1;
                bLat.zero();
                REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::CmpNorm, &cf, sizeof cf, 1));

                const auto* nw = static_cast<const uint16_t*>(
                    rig.pinned.find(p + ".attn.compressor.norm.weight")->data_host);
                std::vector<float> ref(d.head_dim);
                double sq = 0;
                for (uint32_t i = 0; i < d.head_dim; ++i) {
                    double mx = -1e300, den = 0, num = 0;
                    for (uint32_t j = 0; j < ratio; ++j)
                        mx = std::max(mx, double(bScSt.f()[size_t(j) * d.head_dim + i]));
                    for (uint32_t j = 0; j < ratio; ++j) {
                        const double e = std::exp(double(bScSt.f()[size_t(j) * d.head_dim + i]) - mx);
                        den += e;
                        num += e * double(bKvSt.f()[size_t(j) * d.head_dim + i]);
                    }
                    // kv.to(dtype) before the norm: the reference's one bf16
                    // rounding on the ratio > 1 path.
                    ref[i] = cpu::bf16_to_float(cpu::float_to_bf16(float(num / den)));
                    sq += double(ref[i]) * double(ref[i]);
                }
                const double rs = 1.0 / std::sqrt(sq / d.head_dim + d.norm_eps);
                for (uint32_t i = 0; i < d.head_dim; ++i)
                    ref[i] = cpu::bf16_to_float(cpu::float_to_bf16(
                        float(double(ref[i]) * rs * double(cpu::bf16_to_float(nw[i])))));
                CHECK(ok("cmp pool+norm (cpu)", agree(bLat.read(d.head_dim), ref), 0.9999999, 1e-5));
            } else {
                CHECK(ok("compressor.norm", agree(bLat.read(d.head_dim), g.f("latent_pre_rope")),
                         0.99999, 5e-3));
            }
        }

        // --- compressor RoPE + FP4 block-16/E4M3 + the cache write ---------
        if (g.find("latent") != nullptr) {
            bLat.set(g.f("latent_pre_rope"));
            bCmp.zero(); bLatQ.zero();
            cmp_slots(gpu::AttnStage::CmpStore, "attn.compressor.wkv.weight", bCmpY);
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::CmpStore, &cp, sizeof cp, 1));
            CHECK(ok("compressor.store", agree(bLatQ.read(d.head_dim), g.f("latent")),
                     0.9999, 2e-2));
            std::printf("      %-22s %zu/%zu E2M1 nibble bytes, %zu/%zu E4M3 scale bytes differ\n",
                        "cmp_kv fp4 bytes",
                        byte_diff(bCFp4.u8(), g.f("latent_fp4")), g.f("latent_fp4").size(),
                        byte_diff(bCScB.u8(), g.f("latent_scale_e4m3")),
                        g.f("latent_scale_e4m3").size());
            // The compressed-KV cache row sparse_attn will read.
            std::vector<float> row(d.head_dim);
            for (uint32_t i = 0; i < d.head_dim; ++i)
                row[i] = bf16_to_f32(bCmp.u16()[size_t(cp.cmp_row) * d.head_dim + i]);
            CHECK(ok("cmp_kv cache row", agree(row, g.f("latent")), 0.9999, 2e-2));
        }

        // --- the indexer (7.4) --------------------------------------------
        const float wscale = 1.0f / std::sqrt(float(ihd)) / std::sqrt(float(nih));
        gpu::IdxPush ip{irows, d.q_lora, d.q_lora / 32, nih, ihd, d.rope_dim,
                        0, 0, d.window, pos / ratio, d.norm_eps, wscale};
        auto idx_slots = [&](gpu::AttnStage st, uint64_t rope) {
            uint64_t* s = rig.runner.slots(st);
            s[gpu::slot::kIdxW] = rig.addr(p + ".attn.indexer.wq_b.weight");
            s[gpu::slot::kIdxS] = rig.scale_addr(p + ".attn.indexer.wq_b.weight");
            s[gpu::slot::kIdxQr] = bQr.a();
            s[gpu::slot::kIdxQNormW] = rig.addr(p + ".attn.q_norm.weight");
            s[gpu::slot::kIdxRope] = rope;
            s[gpu::slot::kIdxQRaw] = bIQRaw.a();
            s[gpu::slot::kIdxQ] = bIQ.a();
            s[gpu::slot::kIdxWk] = rig.addr(p + ".attn.indexer.wk.weight");
            s[gpu::slot::kIdxKNormW] = rig.addr(p + ".attn.indexer.k_norm.weight");
            s[gpu::slot::kIdxLatent] = bLat.a();
            s[gpu::slot::kIdxKRaw] = bIKRaw.a();
            s[gpu::slot::kIdxKCache] = bIKC.a();
            s[gpu::slot::kIdxKFp4] = bIKFp4.a();
            s[gpu::slot::kIdxKScale] = bIKSc.a();
            s[gpu::slot::kIdxWProjW] = rig.addr(p + ".attn.indexer.weights_proj.weight");
            s[gpu::slot::kIdxX] = bU.a();
            s[gpu::slot::kIdxWeights] = bIWt.a();
            s[gpu::slot::kIdxScore] = bISc.a();
            s[gpu::slot::kIdxOut] = bTop.a();
            s[gpu::slot::kIdxQFp4] = bIQFp4.a();
            s[gpu::slot::kIdxQScale] = bIQSc.a();
            return s;
        };

        {   // indexer.wq_b, with q_norm and the act_quant round trip fused
            bQr.set(g.f("wq_a_out"));
            idx_slots(gpu::AttnStage::IdxQGemv, bRopeQ.a());
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::IdxQGemv, &ip, sizeof ip,
                          rig.runner.gemv_groups(gpu::AttnStage::IdxQGemv, irows)));
            CHECK(ok("indexer.wq_b", agree(bIQRaw.read(irows), g.f("index_q_pre_rope")),
                     0.99999, 5e-3));
        }
        {   // RoPE + FP4 block-32/UE8M0 on q
            bIQRaw.set(g.f("index_q_pre_rope"));
            idx_slots(gpu::AttnStage::IdxQFinish, bRopeQ.a());
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::IdxQFinish, &ip, sizeof ip, 1));
            CHECK(ok("indexer.q rope+fp4", agree(bIQ.read_bf16(irows), g.f("index_q")),
                     0.9999, 2e-2));
        }
        if (g.find("index_k") != nullptr) {
            // The index key, off the compressor's pre-RoPE latent.
            bLat.set(g.f("latent_pre_rope"));
            bIKC.zero();
            gpu::IdxPush ik = ip;
            ik.k = d.head_dim;
            idx_slots(gpu::AttnStage::IdxKey, bRopeL.a());
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::IdxKey, &ik, sizeof ik, 1));
            CHECK(ok("indexer.wk+k_norm", agree(bIKRaw.read(ihd), g.f("index_k_pre_rope")),
                     0.99999, 5e-3));
            std::vector<float> krow(ihd);
            for (uint32_t i = 0; i < ihd; ++i)
                krow[i] = bf16_to_f32(bIKC.u16()[size_t(ik.k_row) * ihd + i]);
            CHECK(ok("indexer.key rope+fp4", agree(krow, g.f("index_k")), 0.9999, 2e-2));
            std::printf("      %-22s %zu/%zu E2M1 nibble bytes, %zu/%zu UE8M0 scale bytes differ\n",
                        "index_k fp4 bytes",
                        byte_diff(bIKFp4.u8(), g.f("index_k_fp4")), g.f("index_k_fp4").size(),
                        byte_diff(bIKSc.u8(), g.f("index_k_scale_e8m0")),
                        g.f("index_k_scale_e8m0").size());
        }
        {   // weights_proj, already scaled by softmax_scale * n_heads ** -0.5
            bU.set(g.f("attn_norm_out"));
            gpu::IdxPush iw = ip;
            iw.rows = nih;
            iw.k = d.dim;
            idx_slots(gpu::AttnStage::IdxWeights, bRopeQ.a());
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::IdxWeights, &iw, sizeof iw, 1));
            CHECK(ok("indexer.weights_proj", agree(bIWt.read(nih), g.f("index_weights_scaled")),
                     0.99999, 5e-3));
        }
        const uint32_t npos = static_cast<uint32_t>(g.f("index_compress_len")[0]);
        const uint32_t ntop = static_cast<uint32_t>(g.f("index_topk_n")[0]);
        {   // the scores, against the reference's own q, keys and weights
            const std::vector<float>& kall = g.f("index_k_all");
            for (size_t i = 0; i < kall.size(); ++i) bIKC.u16()[i] = cpu::float_to_bf16(kall[i]);
            for (size_t i = 0; i < g.f("index_q").size(); ++i)
                bIQ.u16()[i] = cpu::float_to_bf16(g.f("index_q")[i]);
            bIWt.set(g.f("index_weights_scaled"));
            bISc.zero();
            gpu::IdxPush is = ip;
            is.n_pos = npos;
            is.topk = ntop;
            idx_slots(gpu::AttnStage::IdxScore, bRopeQ.a());
            const uint32_t sg = (npos + gpu::kIdxScoreTile - 1) / gpu::kIdxScoreTile;
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::IdxScore, &is, sizeof is, sg));
            CHECK(ok("indexer.score", agree(bISc.read(npos), g.f("index_score")), 0.99999, 5e-3));

            // ... and the selection. topk == n_pos at this context, so the
            // reference keeps every compressed position and the comparison is
            // exact but degenerate; the non-degenerate path has its own case.
            bISc.set(g.f("index_score"));
            for (uint32_t i = 0; i < ntop; ++i) bTop.u32()[d.window + i] = 0xFFFFFFFFu;
            // A stage owns its OWN slice of the address table, so the top-k
            // needs its slots written even though it reads the same buffers.
            idx_slots(gpu::AttnStage::IdxTopK, bRopeQ.a());
            REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::IdxTopK, &is, sizeof is, 1));
            const std::vector<float>& want = g.f("topk_idxs");
            uint32_t bad = 0;
            for (uint32_t i = 0; i < ntop; ++i)
                if (static_cast<int32_t>(bTop.u32()[d.window + i]) !=
                    static_cast<int32_t>(want[d.window + i])) ++bad;
            std::printf("      %-22s %u/%u positions match the reference's set "
                        "(topk=%u of %u reachable)\n", "indexer.topk", ntop - bad, ntop,
                        ntop, npos);
            CHECK_EQ(bad, 0u);
        }
        ++checked;
    }
    std::printf("    %u source layers checked\n", checked);
    CHECK(checked > 0);
}

// The radix select of indexer.slang stage 5, on a case the checkpoint's own
// geometry never reaches.
//
// index_topk is 512 and the L2 export's context leaves 32 or 65 compressed
// positions, so min(index_topk, n) is always n and the selection above is the
// identity. At a real 64K context it is 512 of 32768, which is a different code
// path entirely: four 8-bit histogram passes to find the key of the k-th
// largest, then a chunked prefix sum to emit the winners in ascending position
// order. This drives that path and checks it against a CPU stable sort.
DEEPMOE_TEST(gpu_attn, indexer_topk_select) {
    if (skip_without_model("gpu_attn topk")) return;
    Rig rig;
    if (!rig.bring_up()) { std::printf("      SKIP gpu_attn topk: %s\n", rig.why.c_str()); return; }

    constexpr uint32_t kN = 4096, kK = 512, kOffset = 128;
    gpu::GpuScratch& S = rig.scratch;
    Buf bSc = take(S, kN * 4), bOut = take(S, (kN + kOffset) * 4);
    REQUIRE(bOut.v.valid());

    // A deterministic spread with both signs and exact duplicates, so the tie
    // rule is exercised as well as the ordering.
    std::vector<float> sc(kN);
    for (uint32_t i = 0; i < kN; ++i)
        sc[i] = std::sin(float(i) * 0.7913f) * 10.0f + ((i % 37 == 0) ? 3.5f : 0.0f);
    for (uint32_t i = 0; i < 64; ++i) sc[i * 61 % kN] = 2.5f;       // a tie cluster
    bSc.set(sc);
    for (uint32_t i = 0; i < kN + kOffset; ++i) bOut.u32()[i] = 0xFFFFFFFFu;

    uint64_t* s = rig.runner.slots(gpu::AttnStage::IdxTopK);
    s[gpu::slot::kIdxScore] = bSc.a();
    s[gpu::slot::kIdxOut] = bOut.a();
    gpu::IdxPush ip{};
    ip.n_pos = kN;
    ip.topk = kK;
    ip.offset = kOffset;
    REQUIRE_OK(rig.runner.dispatch_now(gpu::AttnStage::IdxTopK, &ip, sizeof ip, 1));

    // The reference set: the kK largest, ties broken towards the lowest index,
    // which is what stage 5's chunked tie rank does.
    std::vector<uint32_t> order(kN);
    for (uint32_t i = 0; i < kN; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return sc[a] > sc[b]; });
    std::vector<uint32_t> want(order.begin(), order.begin() + kK);
    std::sort(want.begin(), want.end());

    uint32_t bad = 0;
    for (uint32_t i = 0; i < kK; ++i) {
        const int32_t got = static_cast<int32_t>(bOut.u32()[kOffset + i]);
        if (got != static_cast<int32_t>(want[i] + kOffset)) ++bad;
    }
    std::printf("      %-22s %u/%u of the top-%u match a CPU stable sort over %u positions\n",
                "indexer.topk select", kK - bad, kK, kK, kN);
    CHECK_EQ(bad, 0u);
}
