// The batched prefill kernels of docs/p3_prefill.md against the reference's own
// `start_pos == 0` tensors (tools/oracle_prefill.py -> tests/data/prefill), and
// then a whole forty-layer GPU prefill handed to the decode engine.
//
// gpu_prefill.stages
// ------------------
// Every stage of a prefill layer, fed the reference's OWN input, so a failure
// names one kernel: the mHC halves and hc_post at two positions, the engram,
// the window KV, the compressor and the index keys over all 32 / 64 groups,
// the Q path and the indexer for all 64 queries, the band attention with the
// reference's own index matrix and KV, the output projection, the gate for
// every token and the expert-major MoE with the reference's own routing.
// Wide tensors the oracle stores at two positions are also checked at every
// position through a sketch (L2 norm and a signed sum, oracle_prefill.py).
//
// gpu_prefill.forty_layers
// ------------------------
// The 64-token L2 prompt through all forty layers and the head on the GPU,
// probed at the exported layers, with all forty layers' routing compared token
// by token; the state it leaves compared against tests/data/l3's prefill
// record; and then that state handed to runtime::Engine through
// Prefill::write_l3_dir + Engine::load_decode_state for eight decode steps,
// teacher-forced and free-running, against the reference's tokens.
//
// Gated on DEEPMOE_MODEL_DIR, tests/data/prefill (and l3), and a Vulkan device.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/json.h"
#include "cpu/dequant.h"
#include "cpu/gate.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/moe_kernels.h"   // default_shader_dir
#include "gpu/vulkan/prefill_kernels.h"
#include "model/layout.h"
#include "model/manifest.h"
#include "model/v41_config.h"
#include "runtime/engine.h"
#include "runtime/engram.h"
#include "storage/backend.h"
#include "storage/io_engine.h"
#include "store/pinned.h"
#include "store/shard_set.h"
#include "tests/l1_golden.h"
#include "tests/l2_golden.h"
#include "tests/test_framework.h"

#ifndef DEEPMOE_TEST_DATA_DIR
#define DEEPMOE_TEST_DATA_DIR "tests/data"
#endif

using namespace deepmoe;
using namespace deepmoe::testing;

namespace {

std::string pf_dir() { return std::string(DEEPMOE_TEST_DATA_DIR) + "/prefill"; }
std::string l3_dir() { return std::string(DEEPMOE_TEST_DATA_DIR) + "/l3"; }

constexpr uint32_t kN = 64, kDim = 5120, kHc = 4, kHd = 512;

// The oracle's sketch sign (tools/oracle_prefill.py `sketch_sign`).
float sketch_sign(uint64_t i) {
    const uint32_t u = static_cast<uint32_t>(i) * 2654435761u + 12345u;
    return ((u >> 16) & 1u) ? -1.0f : 1.0f;
}

// Worst relative disagreement of (norm, signed sum) over rows, against the
// reference's [rows][3] sketch. The signed sum is a random projection, so
// |delta| / norm bounds roughly how far the row moved.
struct SketchAgreement { double norm = 0.0, proj = 0.0; };
SketchAgreement sketch_agree(const float* rows, uint32_t n, uint64_t width,
                             const std::vector<float>& ref) {
    SketchAgreement a;
    for (uint32_t r = 0; r < n; ++r) {
        double nn = 0.0, pr = 0.0;
        const float* v = rows + uint64_t(r) * width;
        for (uint64_t i = 0; i < width; ++i) {
            nn += double(v[i]) * v[i];
            pr += double(v[i]) * sketch_sign(i);
        }
        nn = std::sqrt(nn);
        const double rn = ref[size_t(r) * 3], rp = ref[size_t(r) * 3 + 2];
        const double den = std::max(rn, 1e-30);
        a.norm = std::max(a.norm, std::fabs(nn - rn) / den);
        a.proj = std::max(a.proj, std::fabs(pr - rp) / den);
    }
    return a;
}

// Rows `pos` of a [n][width] tensor.
std::vector<float> pick(const std::vector<float>& v, uint64_t width, std::initializer_list<uint32_t> pos) {
    std::vector<float> out;
    for (uint32_t p : pos) out.insert(out.end(), v.begin() + p * width, v.begin() + (p + 1) * width);
    return out;
}

struct Tally {
    uint32_t checked = 0, failed = 0;
    double worst_cos = 1.0;
    std::string worst;
    bool check(const char* what, uint32_t L, const Agreement& g, double cos_min = 0.9999) {
        ++checked;
        const bool pass = std::isfinite(g.cos) && g.cos >= cos_min;
        if (g.cos < worst_cos) { worst_cos = g.cos; worst = std::format("L{} {}", L, what); }
        std::printf("      L%-2u %-26s %s%s\n", L, what, g.str().c_str(), pass ? "" : "   <-- FAIL");
        if (!pass) ++failed;
        return pass;
    }
    bool sketch(const char* what, uint32_t L, const SketchAgreement& s, double tol = 2e-2) {
        ++checked;
        const bool pass = s.norm <= tol && s.proj <= tol;
        std::printf("      L%-2u %-26s sketch: |dnorm|/norm %.2e  |dproj|/norm %.2e%s\n", L, what,
                    s.norm, s.proj, pass ? "" : "   <-- FAIL");
        if (!pass) ++failed;
        return pass;
    }
};

struct Rig {
    gpu::Device          device;
    gpu::MemoryAllocator alloc;
    Manifest             manifest;
    store::ShardSet      shards;
    storage::IoEngine    io;
    store::PinnedStore   pinned;
    V41Config            cfg;
    runtime::EngramTables engram;
    gpu::PrefillRunner   runner;
    gpu::Prefill         prefill;
    bool                 io_started = false;
    std::string          why;

    ~Rig() {
        prefill.destroy();
        runner.destroy();
        pinned.reset();
        if (io_started) io.stop();
    }
    bool up(const std::vector<uint32_t>& layers, const gpu::PrefillConfig& pc) {
        const std::string dir = model_dir() ? model_dir() : "";
        gpu::DeviceOptions dopts;
        dopts.enable_validation = std::getenv("VK_INSTANCE_LAYERS") != nullptr;
        if (auto r = device.create(dopts); !r) { why = r.error().str(); return false; }
        if (auto r = alloc.init(device, MemoryPath::DeviceLocalHostVisible); !r) { why = r.error().str(); return false; }
        auto mf = Manifest::load(store::ShardSet::join(dir, layout::kManifestFile));
        if (!mf) { why = mf.error().str(); return false; }
        manifest = std::move(*mf);
        auto c = V41Config::load(store::ShardSet::join(dir, "config.json"));
        if (!c) { why = c.error().str(); return false; }
        cfg = std::move(*c);
        if (auto r = shards.open_all(dir, manifest, true); !r) { why = r.error().str(); return false; }
        IoConfig ioc;
        auto be = storage::make_default_backend(ioc);
        if (!be) { why = be.error().str(); return false; }
        if (auto r = io.start(std::move(*be), ioc); !r) { why = r.error().str(); return false; }
        io_started = true;
        auto backing = alloc.make_slab_backing();
        if (!backing) { why = backing.error().str(); return false; }
        store::PinnedConfig pcfg;
        pcfg.region_bytes = 512ull << 20;
        if (auto r = pinned.init(std::move(*backing), pcfg); !r) { why = r.error().str(); return false; }
        std::vector<std::string> names = store::pinned_global_tensors(manifest);
        for (uint32_t L : layers) {
            auto n = store::pinned_layer_tensors(manifest, L);
            names.insert(names.end(), n.begin(), n.end());
        }
        if (auto r = pinned.load(manifest, shards, io, names); !r) { why = r.error().str(); return false; }
        auto t = runtime::EngramTables::load(l3_dir());
        if (!t) { why = t.error().str(); return false; }
        engram = std::move(*t);
        if (auto r = runner.create(device, alloc, gpu::default_shader_dir()); !r) { why = r.error().str(); return false; }
        if (auto r = prefill.create(device, alloc, runner, manifest, shards, io, pinned, cfg.text,
                                    &engram, pc); !r) { why = r.error().str(); return false; }
        return true;
    }
};

struct GBuf {
    gpu::GpuBuffer b{};
    uint64_t a() const { return b.dev_addr; }
    float* f() const { return static_cast<float*>(b.host_ptr); }
    void set(const std::vector<float>& v) { std::memcpy(b.host_ptr, v.data(), v.size() * sizeof(float)); }
    std::vector<float> read(size_t n) const { return std::vector<float>(f(), f() + n); }
};
GBuf take(gpu::Prefill& p, uint64_t bytes) {
    GBuf g;
    auto b = p.scratch(bytes);
    if (b) g.b = *b;
    return g;
}

// DEEPMOE_PF_COOP_MOE / DEEPMOE_PF_COOP_DENSE override the option (a)/(b)
// thresholds (PrefillConfig::coopmat_min_rows / coopmat_dense_min_rows; -1 =
// tiled only, 0 = cooperative matrix everywhere it applies) for an A/B run.
void apply_kernel_env(gpu::PrefillConfig& pc) {
    if (const char* e = std::getenv("DEEPMOE_PF_COOP_MOE"))
        pc.coopmat_min_rows = static_cast<uint32_t>(std::atoi(e));
    if (const char* e = std::getenv("DEEPMOE_PF_COOP_DENSE"))
        pc.coopmat_dense_min_rows = static_cast<uint32_t>(std::atoi(e));
    std::printf("    kernels: MoE coopmat at n >= %d, dense coopmat at n >= %d (-1 = never)\n",
                static_cast<int32_t>(pc.coopmat_min_rows), static_cast<int32_t>(pc.coopmat_dense_min_rows));
}

std::vector<uint32_t> prompt_ids(const std::string& dir) {
    std::vector<uint32_t> out;
    auto doc = json_parse_file(dir + "/index.json");
    if (!doc) return out;
    if (const JsonValue* a = doc->find("prompt_ids"))
        if (auto arr = a->as_array())
            for (const JsonValue& e : **arr) out.push_back(static_cast<uint32_t>(e.as_int().value_or(0)));
    return out;
}

const std::vector<float>& rows_of(const L2Step& g, const std::string& name, bool* full) {
    if (g.find(name)) { *full = true; return g.f(name); }
    *full = false;
    return g.f(name + ".rows");
}

}  // namespace

DEEPMOE_TEST(gpu_prefill, stages) {
    if (skip_without_model("gpu_prefill")) return;
    auto set = load_l2(pf_dir());
    if (!set) {
        std::printf("      SKIP gpu_prefill: no prefill data (%s); run tools/oracle_prefill.py\n",
                    set.error().str().c_str());
        return;
    }
    const std::vector<uint32_t> prompt = prompt_ids(pf_dir());
    REQUIRE(prompt.size() == kN);
    gpu::PrefillConfig pc;
    pc.max_tokens = kN;
    pc.transit_slots = 16;
    apply_kernel_env(pc);
    Rig rig;
    std::vector<uint32_t> layers;
    for (const L2Step& s : set->steps) if (s.layer < 40) layers.push_back(s.layer);
    if (!rig.up(layers, pc)) {
        std::printf("      SKIP gpu_prefill: %s\n", rig.why.c_str());
        return;
    }
    gpu::Prefill& P = rig.prefill;
    const TextConfig& c = rig.cfg.text;
    Tally tally;
    const uint32_t round = gpu::kPfFlagRound;

    // scratch, reused per layer
    GBuf h2 = take(P, 2ull * kHc * kDim * 4), h2o = take(P, 2ull * kHc * kDim * 4);
    GBuf coeff = take(P, 2ull * 24 * 4), raw = take(P, 2ull * 24 * 4), mix = take(P, 2ull * 24 * 4);
    GBuf u2 = take(P, 2ull * kDim * 4), rs = take(P, kN * 4ull), a2 = take(P, 2ull * kDim * 4);
    GBuf x = take(P, kN * kDim * 4ull), xq = take(P, kN * kDim * 2ull), xs = take(P, kN * 160 * 4ull);
    GBuf kvraw = take(P, kN * kHd * 4ull), kvn = take(P, kN * kHd * 4ull), kv = take(P, kN * kHd * 4ull);
    GBuf ckv = take(P, kN * kHd * 4ull), csc = take(P, kN * kHd * 4ull), lat = take(P, kN * kHd * 4ull);
    GBuf latn = take(P, kN * kHd * 4ull), cache = take(P, kN * kHd * 4ull);
    GBuf kraw = take(P, kN * 128 * 4ull), kn = take(P, kN * 128 * 4ull), keys = take(P, kN * 128 * 4ull);
    GBuf qraw = take(P, kN * 1280 * 4ull), qr = take(P, kN * 1280 * 4ull);
    GBuf qrq = take(P, kN * 1280 * 2ull), qrs = take(P, kN * 40 * 4ull);
    GBuf q = take(P, kN * 32768 * 4ull), iq = take(P, kN * 4096 * 4ull), iw = take(P, kN * 32 * 4ull);
    GBuf isc = take(P, kN * kN * 4ull), idx = take(P, kN * 640 * 4ull), gkv = take(P, kN * kHd * 4ull);
    GBuf gcmp = take(P, kN * kHd * 4ull), gkeys = take(P, kN * 128 * 4ull), o = take(P, kN * 32768 * 4ull);
    GBuf woa = take(P, kN * 8192 * 4ull), woaq = take(P, kN * 8192 * 2ull), woas = take(P, kN * 256 * 4ull);
    GBuf wob = take(P, kN * kDim * 4ull), gate = take(P, kN * 384 * 4ull), y = take(P, kN * kDim * 4ull);
    GBuf exq = take(P, kN * 6144 * 2ull), exs = take(P, kN * 192 * 4ull), ekv = take(P, kN * 25600 * 4ull);
    GBuf ekv2 = take(P, 2ull * 25600 * 4);
    REQUIRE(ekv2.b.valid());

    auto W = [&](uint32_t L, const std::string& n, uint32_t fmt) {
        return P.weight(std::format("layers.{}.{}", L, n), fmt);
    };
    auto addr = [&](uint32_t L, const std::string& n) -> uint64_t {
        const store::PinnedTensor* t = rig.pinned.find(std::format("layers.{}.{}", L, n));
        return t ? t->data : 0;
    };
    // Coefficient rows [2][24] from the oracle's [64][4] / [64][4] / [64][16] at rows 0 and 63.
    auto coeffs = [&](const L2Step& g, const char* half) {
        std::vector<float> v(2 * 24, 0.0f);
        const auto& pre = g.f(std::format("{}_pre", half));
        const auto& post = g.f(std::format("{}_post", half));
        const auto& comb = g.f(std::format("{}_comb", half));
        uint32_t i = 0;
        for (uint32_t p : {0u, kN - 1}) {
            std::memcpy(v.data() + i * 24, pre.data() + p * 4, 16);
            std::memcpy(v.data() + i * 24 + 4, post.data() + p * 4, 16);
            std::memcpy(v.data() + i * 24 + 8, comb.data() + p * 16, 64);
            ++i;
        }
        return v;
    };

    for (const L2Step& g : set->steps) {
        const uint32_t L = g.layer;
        if (L >= 40) continue;
        const uint32_t ratio = c.compress_ratio(L);
        std::printf("    layer %u (ratio %u%s%s)\n", L, ratio, c.is_kv_source(L) ? ", kv source" : "",
                    c.is_engram_layer(L) ? ", engram" : "");

        // --- engram: the hash rows of every token, then the GEMM and the gate at rows 0, 63
        std::vector<float> stream_in = g.f("block_in");       // [2][4][5120]
        if (c.is_engram_layer(L)) {
            const auto& hashes = g.find("engram_hashes")->raw;   // i32 [64][24]
            uint32_t mism = 0;
            std::vector<uint64_t> rows(24);
            for (uint32_t p = 0; p < kN; ++p) {
                REQUIRE_OK(rig.engram.hash_rows(L, prompt, p, rows));
                for (uint32_t r = 0; r < 24; ++r) {
                    int32_t ref;
                    std::memcpy(&ref, hashes.data() + (p * 24 + r) * 4, 4);
                    mism += uint64_t(ref) != rows[r];
                }
            }
            std::printf("      L%-2u engram hash rows            %u / %u differ\n", L, mism, kN * 24);
            CHECK_EQ(mism, 0u);
            REQUIRE_OK(P.op_engram_rows(L, prompt));
            REQUIRE_OK(P.op_act_quant(P.engram_x(), kN, 6144, exq.a(), exs.a()));
            auto wkv = W(L, "engram.wkv.weight", gpu::kPfFp8);
            REQUIRE_OK(wkv);
            REQUIRE_OK(P.op_gemm(*wkv, gpu::kPfActQ, exq.a(), exs.a(), kN, 6144, ekv.a(), round));
            std::memcpy(ekv2.f(), ekv.f(), 25600 * 4);
            std::memcpy(ekv2.f() + 25600, ekv.f() + (kN - 1) * 25600ull, 25600 * 4);
            h2.set(stream_in);
            REQUIRE_OK(P.op_engram_gate(h2.a(), ekv2.a(), addr(L, "engram.q_weight"),
                                        addr(L, "engram.k_weight"), h2o.a(), 2));
            tally.check("engram_out", L, agree(h2o.read(2 * kHc * kDim), g.f("engram_out")));
            stream_in = g.f("engram_out");
        }

        // --- mHC, attention half, at positions 0 and 63 --------------------------------
        h2.set(stream_in);
        coeff.set(coeffs(g, "attn"));     // pre of this half's coefficients is not what hc_pre uses:
        {
            // hc_pre uses the PREVIOUS sublayer's pre, which the oracle exports as pre_mix_in
            std::vector<float> pm(2 * 24, 0.0f);
            const auto& in = g.f("pre_mix_in");
            std::memcpy(pm.data(), in.data(), 16);
            std::memcpy(pm.data() + 24, in.data() + (kN - 1) * 4, 16);
            GBuf pmb = take(P, 2 * 24 * 4);
            pmb.set(pm);
            REQUIRE_OK(P.op_mhc_pre_norm(h2.a(), 2, pmb.a(), 24, addr(L, "attn_norm.weight"), u2.a(), rs.a()));
        }
        bool full = false;
        const auto& an = rows_of(g, "attn_norm_out", &full);
        tally.check("attn_norm_out (rows)", L, agree(u2.read(2 * kDim), full ? pick(an, kDim, {0, kN - 1}) : an));
        {
            auto hcf = W(L, "hc_attn_fn", gpu::kPfFp32);
            REQUIRE_OK(hcf);
            REQUIRE_OK(P.op_gemm(*hcf, gpu::kPfActF32, h2.a(), 0, 2, kHc * kDim, raw.a(),
                                 gpu::kPfFlagRowScale, 1.0f, 0, rs.a()));
            REQUIRE_OK(P.op_sinkhorn(raw.a(), mix.a(), 2, addr(L, "hc_attn_base"), addr(L, "hc_attn_scale")));
            const std::vector<float> ref = coeffs(g, "attn");
            tally.check("attn mixes (pre,post,comb)", L, agree(mix.read(48), ref), 0.999999);
        }

        // --- attention over all 64 queries, on the layers with the full input ------------
        if (full) {
            x.set(an);
            REQUIRE_OK(P.op_act_quant(x.a(), kN, kDim, xq.a(), xs.a()));
            auto wkv = W(L, "attn.wkv.weight", gpu::kPfFp8);
            REQUIRE_OK(wkv);
            REQUIRE_OK(P.op_gemm(*wkv, gpu::kPfActQ, xq.a(), xs.a(), kN, kDim, kvraw.a(), round));
            tally.check("wkv_out (rows)", L, agree(pick(kvraw.read(kN * kHd), kHd, {0, kN - 1}), g.f("wkv_out")));
            REQUIRE_OK(P.op_rmsnorm(kvraw.a(), kvn.a(), kN, kHd, addr(L, "attn.kv_norm.weight")));
            REQUIRE_OK(P.op_rope(kvn.a(), kv.a(), kN, kHd, kHd, 1, 32, ratio > 0, 0, 1, false));
            tally.check("kv (fp8, all 64)", L, agree(kv.read(kN * kHd), g.f("kv")));

            uint32_t G = 0;
            if (c.is_kv_source(L)) {
                G = kN / ratio;
                auto cw = W(L, "attn.compressor.wkv.weight", gpu::kPfBf16);
                REQUIRE_OK(cw);
                if (ratio > 1) {
                    auto cg = W(L, "attn.compressor.wgate.weight", gpu::kPfBf16);
                    REQUIRE_OK(cg);
                    REQUIRE_OK(P.op_gemm(*cw, gpu::kPfActF32, x.a(), 0, kN, kDim, ckv.a(), 0));
                    REQUIRE_OK(P.op_gemm(*cg, gpu::kPfActF32, x.a(), 0, kN, kDim, csc.a(), 0));
                    tally.check("compressor.wkv (rows)", L,
                                agree(pick(ckv.read(kN * kHd), kHd, {0, kN - 1}), g.f("cmp_wkv_out")));
                    REQUIRE_OK(P.op_cmp_pool(ckv.a(), csc.a(), lat.a(), G, ratio));
                } else {
                    REQUIRE_OK(P.op_gemm(*cw, gpu::kPfActF32, x.a(), 0, kN, kDim, lat.a(), round));
                    tally.check("compressor.wkv (rows)", L,
                                agree(pick(lat.read(kN * kHd), kHd, {0, kN - 1}), g.f("cmp_wkv_out")));
                }
                REQUIRE_OK(P.op_rmsnorm(lat.a(), latn.a(), G, kHd, addr(L, "attn.compressor.norm.weight")));
                tally.check("latent (pre-RoPE, all G)", L, agree(latn.read(G * kHd), g.f("cmp_latent_pre_rope")));
                REQUIRE_OK(P.op_rope(latn.a(), cache.a(), G, kHd, kHd, 3, 16, true, 0, ratio, false));
                tally.check("cmp_cache (FP4/16, all G)", L, agree(cache.read(G * kHd), g.f("cmp_cache")));
                auto wk = W(L, "attn.indexer.wk.weight", gpu::kPfBf16);
                REQUIRE_OK(wk);
                REQUIRE_OK(P.op_gemm(*wk, gpu::kPfActF32, latn.a(), 0, G, kHd, kraw.a(), round));
                REQUIRE_OK(P.op_rmsnorm(kraw.a(), kn.a(), G, 128, addr(L, "attn.indexer.k_norm.weight")));
                tally.check("index_k pre-RoPE", L, agree(kn.read(G * 128), g.f("index_k_pre_rope")));
                REQUIRE_OK(P.op_rope(kn.a(), keys.a(), G, 128, 128, 2, 32, true, 0, ratio, false));
                tally.check("index_k (FP4/32, all G)", L, agree(keys.read(G * 128), g.f("index_k")));
                const auto& cref = g.f("cmp_cache");
                const auto ours = cache.read(G * kHd);
                uint32_t exact = 0;
                for (size_t i = 0; i < ours.size(); ++i) exact += ours[i] == cref[i];
                std::printf("      L%-2u cmp_cache values identical   %u / %zu\n", L, exact, ours.size());
            } else if (ratio) {
                G = kN / ratio;
            }

            auto wqa = W(L, "attn.wq_a.weight", gpu::kPfFp8);
            auto wqb = W(L, "attn.wq_b.weight", gpu::kPfFp8);
            REQUIRE_OK(wqa);
            REQUIRE_OK(wqb);
            REQUIRE_OK(P.op_gemm(*wqa, gpu::kPfActQ, xq.a(), xs.a(), kN, kDim, qraw.a(), round));
            tally.check("wq_a_out (rows)", L, agree(pick(qraw.read(kN * 1280), 1280, {0, kN - 1}), g.f("wq_a_out")));
            REQUIRE_OK(P.op_rmsnorm(qraw.a(), qr.a(), kN, 1280, addr(L, "attn.q_norm.weight")));
            tally.check("qr (rows)", L, agree(pick(qr.read(kN * 1280), 1280, {0, kN - 1}), g.f("qr")));
            tally.sketch("qr (all 64)", L, sketch_agree(qr.f(), kN, 1280, g.f("sk.qr")));
            REQUIRE_OK(P.op_act_quant(qr.a(), kN, 1280, qrq.a(), qrs.a()));
            REQUIRE_OK(P.op_gemm(*wqb, gpu::kPfActQ, qrq.a(), qrs.a(), kN, 1280, q.a(), round));
            REQUIRE_OK(P.op_rope(q.a(), q.a(), kN, 32768, kHd, 0, 32, ratio > 0, 0, 1, false));
            tally.check("q post-RoPE (31, 63)", L, agree(pick(q.read(kN * 32768ull), 32768, {31, 63}), g.f("q")));
            tally.sketch("q (all 64)", L, sketch_agree(q.f(), kN, 32768, g.f("sk.q")));

            const auto& topk_ref = g.find("topk_idxs")->raw;          // i32 [64][n_idx]
            const uint32_t n_idx = static_cast<uint32_t>(g.find("topk_idxs")->shape[1]);
            if (c.is_index_source(L) && ratio) {
                auto iqb = W(L, "attn.indexer.wq_b.weight", gpu::kPfFp8);
                auto iwp = W(L, "attn.indexer.weights_proj.weight", gpu::kPfBf16);
                REQUIRE_OK(iqb);
                REQUIRE_OK(iwp);
                REQUIRE_OK(P.op_gemm(*iqb, gpu::kPfActQ, qrq.a(), qrs.a(), kN, 1280, iq.a(), round));
                tally.check("index q pre-RoPE (31, 63)", L,
                            agree(pick(iq.read(kN * 4096ull), 4096, {31, 63}), g.f("index_q_pre_rope")));
                REQUIRE_OK(P.op_rope(iq.a(), iq.a(), kN, 4096, 128, 2, 32, true, 0, 1, false));
                tally.check("index q FP4 (31, 63)", L, agree(pick(iq.read(kN * 4096ull), 4096, {31, 63}), g.f("index_q")));
                REQUIRE_OK(P.op_gemm(*iwp, gpu::kPfActF32, x.a(), 0, kN, kDim, iw.a(), round | gpu::kPfFlagRoundPre,
                                     static_cast<float>(1.0 / std::sqrt(128.0) / std::sqrt(32.0))));
                tally.check("index weights (all 64)", L, agree(iw.read(kN * 32), g.f("index_weights")));
                gkeys.set(g.f("index_k_all"));
                REQUIRE_OK(P.op_index_score(iq.a(), gkeys.a(), G, iw.a(), isc.a(), kN, ratio, 0));
                const auto ours = isc.read(kN * G);
                const auto& ref = g.f("index_score");
                std::vector<float> fo, fr;
                uint32_t mask_mism = 0;
                for (size_t i = 0; i < ours.size(); ++i) {
                    const bool io = std::isfinite(ours[i]), ir = std::isfinite(ref[i]);
                    if (io != ir) { ++mask_mism; continue; }
                    if (io) { fo.push_back(ours[i]); fr.push_back(ref[i]); }
                }
                std::printf("      L%-2u index_score mask            %u / %zu entries differ\n", L,
                            mask_mism, ours.size());
                CHECK_EQ(mask_mism, 0u);
                tally.check("index_score (visible)", L, agree(fo, fr));
                std::vector<int32_t> rows(kN * n_idx);
                gpu::Prefill::topk_rows(kN, 0, 0, kN, c.sliding_window, ratio, G, c.index_topk, ours.data(),
                                        rows.data(), n_idx);
                uint32_t diff = 0;
                for (size_t i = 0; i < rows.size(); ++i) {
                    int32_t r;
                    std::memcpy(&r, topk_ref.data() + i * 4, 4);
                    diff += r != rows[i];
                }
                std::printf("      L%-2u topk_idxs (ours)            %u / %zu entries differ\n", L, diff, rows.size());
                CHECK_EQ(diff, 0u);
            } else {
                std::vector<int32_t> rows(kN * n_idx, -1);
                gpu::Prefill::topk_rows(kN, 0, 0, kN, c.sliding_window, 0, 0, 0, nullptr, rows.data(),
                                        std::min(kN, c.sliding_window));
                uint32_t diff = 0;
                const uint32_t nw = std::min(kN, c.sliding_window);
                for (uint32_t j = 0; j < kN; ++j)
                    for (uint32_t i = 0; i < nw; ++i) {
                        int32_t r;
                        std::memcpy(&r, topk_ref.data() + (j * n_idx + i) * 4, 4);
                        diff += r != rows[j * nw + i];
                    }
                std::printf("      L%-2u window band (ours)          %u / %u entries differ\n", L, diff, kN * nw);
                CHECK_EQ(diff, 0u);
            }
            // the attention itself, on the reference's KV, compressed rows and index matrix
            std::memcpy(idx.f(), topk_ref.data(), topk_ref.size());
            gkv.set(g.f("kv"));
            if (g.find("cmp_cache")) gcmp.set(g.f("cmp_cache"));
            REQUIRE_OK(P.op_attention(q.a(), gkv.a(), kN, g.find("cmp_cache") ? gcmp.a() : 0, idx.a(), n_idx,
                                      addr(L, "attn.attn_sink"), o.a(), kN));
            REQUIRE_OK(P.op_rope(o.a(), o.a(), kN, 32768, kHd, 0, 32, ratio > 0, 0, 1, true));
            tally.check("attn_out inv-RoPE (31, 63)", L,
                        agree(pick(o.read(kN * 32768ull), 32768, {31, 63}), g.f("attn_out_irope")));
            tally.sketch("attn_out (all 64)", L, sketch_agree(o.f(), kN, 32768, g.f("sk.attn_out_irope")));
            auto woaw = W(L, "attn.wo_a.weight", gpu::kPfFp8);
            auto wobw = W(L, "attn.wo_b.weight", gpu::kPfFp8);
            REQUIRE_OK(woaw);
            REQUIRE_OK(wobw);
            REQUIRE_OK(P.op_gemm(*woaw, gpu::kPfActF32, o.a(), 0, kN, 32768, woa.a(), round, 1.0f, 1024));
            tally.check("wo_a_out (rows)", L, agree(pick(woa.read(kN * 8192), 8192, {0, kN - 1}), g.f("wo_a_out")));
            REQUIRE_OK(P.op_act_quant(woa.a(), kN, 8192, woaq.a(), woas.a()));
            REQUIRE_OK(P.op_gemm(*wobw, gpu::kPfActQ, woaq.a(), woas.a(), kN, 8192, wob.a(), round));
            tally.check("wo_b_out (rows)", L, agree(pick(wob.read(kN * kDim), kDim, {0, kN - 1}), g.f("wo_b_out")));
            tally.sketch("wo_b_out (all 64)", L, sketch_agree(wob.f(), kN, kDim, g.f("sk.wo_b_out")));
        }

        // --- hc_post (attention half) and the FFN mHC at positions 0 and 63 --------------
        h2.set(stream_in);
        a2.set(g.f("wo_b_out"));
        coeff.set(coeffs(g, "attn"));
        REQUIRE_OK(P.op_mhc_post(h2.a(), a2.a(), coeff.a(), h2o.a(), 2));
        tally.check("attn_block_out (rows)", L, agree(h2o.read(2 * kHc * kDim), g.f("attn_block_out")));
        h2.set(g.f("attn_block_out"));
        REQUIRE_OK(P.op_mhc_pre_norm(h2.a(), 2, coeff.a(), 24, addr(L, "ffn_norm.weight"), u2.a(), rs.a()));
        bool ffull = false;
        const auto& fn = rows_of(g, "ffn_norm_out", &ffull);
        tally.check("ffn_norm_out (rows)", L, agree(u2.read(2 * kDim), ffull ? pick(fn, kDim, {0, kN - 1}) : fn));
        {
            auto hcf = W(L, "hc_ffn_fn", gpu::kPfFp32);
            REQUIRE_OK(hcf);
            REQUIRE_OK(P.op_gemm(*hcf, gpu::kPfActF32, h2.a(), 0, 2, kHc * kDim, raw.a(),
                                 gpu::kPfFlagRowScale, 1.0f, 0, rs.a()));
            REQUIRE_OK(P.op_sinkhorn(raw.a(), mix.a(), 2, addr(L, "hc_ffn_base"), addr(L, "hc_ffn_scale")));
            tally.check("ffn mixes (pre,post,comb)", L, agree(mix.read(48), coeffs(g, "ffn")), 0.999999);
        }

        // --- the gate for every token and the MoE on the reference's routing ---------------
        if (ffull) {
            x.set(fn);
            auto gw = W(L, "ffn.gate.weight", gpu::kPfBf16);
            REQUIRE_OK(gw);
            REQUIRE_OK(P.op_gemm(*gw, gpu::kPfActF32, x.a(), 0, kN, kDim, gate.a(), 0));
            const auto& ref_ids = g.find("gate_ids")->raw;
            const auto& bias = g.f("gate_bias");
            std::vector<uint32_t> ids(kN * 6);
            std::vector<float> wts(kN * 6);
            uint32_t equal_sets = 0;
            for (uint32_t t = 0; t < kN; ++t) {
                std::vector<float> row(gate.f() + t * 384, gate.f() + (t + 1) * 384);
                for (float& v : row) v = cpu::sqrt_softplus(v);
                auto gr = cpu::gate_topk(row, bias, 6, 0, 1.5f);
                REQUIRE_OK(gr);
                std::set<uint32_t> a, b;
                for (uint32_t s = 0; s < 6; ++s) {
                    a.insert(gr->ids[s]);
                    int32_t r;
                    std::memcpy(&r, ref_ids.data() + (t * 6 + s) * 4, 4);
                    b.insert(uint32_t(r));
                    ids[t * 6 + s] = uint32_t(r);
                    wts[t * 6 + s] = g.f("gate_weights")[t * 6 + s];
                }
                equal_sets += a == b;
            }
            std::printf("      L%-2u gate top-6 sets             %u / %u tokens identical\n", L, equal_sets, kN);
            CHECK_EQ(equal_sets, kN);
            std::memset(y.f(), 0, kN * kDim * 4ull);
            REQUIRE_OK(P.op_moe(L, kN, x.a(), ids, wts, y.a(), true, false));
            tally.check("moe_shared_out (rows)", L, agree(pick(y.read(kN * kDim), kDim, {0, kN - 1}), g.f("moe_shared_out")));
            tally.sketch("moe_shared_out (all 64)", L, sketch_agree(y.f(), kN, kDim, g.f("sk.moe_shared_out")));
            std::memset(y.f(), 0, kN * kDim * 4ull);
            REQUIRE_OK(P.op_moe(L, kN, x.a(), ids, wts, y.a(), true, true));
            tally.check("moe_out (rows)", L, agree(pick(y.read(kN * kDim), kDim, {0, kN - 1}), g.f("moe_out")));
            tally.sketch("moe_out (all 64)", L, sketch_agree(y.f(), kN, kDim, g.f("sk.moe_out")));
        }

        // --- hc_post (FFN half) -------------------------------------------------------------
        h2.set(g.f("attn_block_out"));
        a2.set(g.f("moe_out"));
        coeff.set(coeffs(g, "ffn"));
        REQUIRE_OK(P.op_mhc_post(h2.a(), a2.a(), coeff.a(), h2o.a(), 2));
        tally.check("block_out (rows)", L, agree(h2o.read(2 * kHc * kDim), g.f("block_out")));
    }
    std::printf("    %u checks, %u failed; worst cos %.9f at %s\n", tally.checked, tally.failed,
                tally.worst_cos, tally.worst.c_str());
    CHECK_EQ(tally.failed, 0u);
}

namespace {

// Spearman rho of the reference's top-k ordering under our logits (as test_decode).
double rank_rho(const std::vector<uint32_t>& ids, const float* ours) {
    const size_t n = ids.size();
    if (n < 2) return 1.0;
    std::vector<uint32_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = static_cast<uint32_t>(i);
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return ours[ids[a]] > ours[ids[b]]; });
    double d2 = 0;
    for (size_t r = 0; r < n; ++r) {
        const double d = double(r) - double(order[r]);
        d2 += d * d;
    }
    return 1.0 - 6.0 * d2 / (double(n) * (double(n) * double(n) - 1.0));
}

}  // namespace

DEEPMOE_TEST(gpu_prefill, forty_layers) {
    if (skip_without_model("gpu_prefill")) return;
    auto set = load_l2(pf_dir());
    if (!set || !std::fopen((l3_dir() + "/index.json").c_str(), "rb")) {
        std::printf("      SKIP gpu_prefill: needs tests/data/prefill and tests/data/l3\n");
        return;
    }
    const std::vector<uint32_t> prompt = prompt_ids(pf_dir());
    REQUIRE(prompt.size() == kN);
    const L2Step* routing = nullptr;
    for (const L2Step& s : set->steps) if (s.step == "routing") routing = &s;
    REQUIRE(routing != nullptr);

    // --- the decode engine, whose device / pinned set / IoEngine the prefill borrows
    runtime::Engine engine;
    {
        RuntimeConfig cfg;
        cfg.model_dir = model_dir();
        cfg.cache.budget_bytes = 8ull << 30;
        cfg.cache.slots_per_slab = 100;
        if (auto r = engine.init(cfg); !r) {
            std::printf("      SKIP gpu_prefill: %s\n", r.error().str().c_str());
            return;
        }
        if (auto r = engine.init_gpu(); !r) {
            std::printf("      SKIP gpu_prefill: %s\n", r.error().str().c_str());
            return;
        }
    }
    const TextConfig& c = engine.model().text;
    gpu::MemoryAllocator alloc;
    REQUIRE_OK(alloc.init(engine.device(), MemoryPath::DeviceLocalHostVisible));
    store::ShardSet shards;
    REQUIRE_OK(shards.open_all(model_dir(), engine.manifest(), true));
    auto tables = runtime::EngramTables::load(l3_dir());
    REQUIRE_OK(tables);
    gpu::PrefillRunner runner;
    REQUIRE_OK(runner.create(engine.device(), alloc, gpu::default_shader_dir()));
    gpu::PrefillConfig pc;
    pc.max_tokens = kN;
    pc.transit_slots = 16;
    apply_kernel_env(pc);
    pc.probe_layers = true;
    gpu::Prefill prefill;
    REQUIRE_OK(prefill.create(engine.device(), alloc, runner, engine.manifest(), shards, engine.io(),
                              engine.pinned(), c, &*tables, pc));

    // --- probes: the exported layers per stage, every layer's routing -------------
    double worst_block = 1.0, worst_attn_in = 1.0, worst_ffn_in = 1.0;
    uint32_t worst_block_layer = 0;
    uint32_t route_tokens = 0, route_equal = 0, route_layers_all = 0;
    std::vector<uint32_t> divergent(40, 0);
    prefill.probe = [&](const gpu::PrefillProbe& pv) {
        const uint32_t L = pv.layer;
        if (const L2Tensor* ids = routing->find(std::format("L{:02d}.gate_ids", L))) {
            uint32_t eq = 0;
            for (uint32_t t = 0; t < pv.attn_rows; ++t) {
                std::set<uint32_t> a, b;
                for (uint32_t s = 0; s < 6; ++s) {
                    a.insert(pv.gate_ids[t * 6 + s]);
                    int32_t r;
                    std::memcpy(&r, ids->raw.data() + (t * 6 + s) * 4, 4);
                    b.insert(uint32_t(r));
                }
                eq += a == b;
            }
            route_tokens += pv.attn_rows;
            route_equal += eq;
            route_layers_all += eq == pv.attn_rows;
            divergent[L] = pv.attn_rows - eq;
        }
        const L2Step* g = set->layer(L);
        if (!g) return;
        auto rows2 = [&](const float* p, uint64_t w) {
            std::vector<float> v(p, p + w);
            v.insert(v.end(), p + (kN - 1) * w, p + kN * w);
            return v;
        };
        bool full = false;
        const auto& an = rows_of(*g, "attn_norm_out", &full);
        const Agreement a_in = agree(rows2(pv.attn_norm_out, kDim), full ? pick(an, kDim, {0, kN - 1}) : an);
        bool ffull = false;
        const auto& fn = rows_of(*g, "ffn_norm_out", &ffull);
        const Agreement f_in = agree(rows2(pv.ffn_norm_out, kDim), ffull ? pick(fn, kDim, {0, kN - 1}) : fn);
        const Agreement kvg = agree(std::vector<float>(pv.kv, pv.kv + kN * kHd), g->f("kv"));
        const Agreement moe = agree(rows2(pv.moe_out, kDim), g->f("moe_out"));
        const Agreement blk = agree(rows2(pv.block_out, uint64_t(kHc) * kDim), g->f("block_out"));
        const SketchAgreement sk = sketch_agree(pv.block_out, kN, uint64_t(kHc) * kDim, g->f("sk.block_out"));
        std::printf("      L%-2u attn_norm %.7f  kv %.7f  ffn_norm %.7f  moe_out %.7f  block_out %.7f "
                    "(all 64: |dnorm| %.1e)  routing %u/%u\n",
                    L, a_in.cos, kvg.cos, f_in.cos, moe.cos, blk.cos, sk.norm, kN - divergent[L], kN);
        if (pv.cmp_cache) {
            const Agreement cc = agree(std::vector<float>(pv.cmp_cache, pv.cmp_cache + pv.n_cmp * kHd),
                                       g->f("cmp_cache"));
            const Agreement ik = agree(std::vector<float>(pv.index_k, pv.index_k + pv.n_cmp * 128),
                                       g->f("index_k"));
            std::printf("          cmp_cache %.7f  index_k %.7f\n", cc.cos, ik.cos);
        }
        if (pv.topk_first && g->find("topk_idxs")) {
            const auto& ref = g->find("topk_idxs")->raw;
            uint32_t diff = 0;
            for (uint32_t i = 0; i < kN * pv.n_idx; ++i) {
                int32_t r;
                std::memcpy(&r, ref.data() + i * 4, 4);
                diff += r != pv.topk_first[i];
            }
            std::printf("          topk_idxs %u / %u entries differ\n", diff, kN * pv.n_idx);
        }
        worst_attn_in = std::min(worst_attn_in, a_in.cos);
        worst_ffn_in = std::min(worst_ffn_in, f_in.cos);
        if (blk.cos < worst_block) { worst_block = blk.cos; worst_block_layer = L; }
    };

    std::printf("    GPU prefill of %zu tokens, oracle mode (replay >= N)\n", prompt.size());
    auto out = prefill.run(prompt);
    REQUIRE_OK(out);
    const gpu::PrefillTimes& tm = prefill.times();
    std::printf("    %.2f s: embed %.0f  engram io %.0f / gpu %.0f  mhc %.0f  attention %.0f  gate %.0f  "
                "shared %.0f  expert io %.0f / gpu %.0f  head %.0f  other %.0f ms; %u experts, %.2f GB, "
                "%u dispatches, %u submits\n",
                tm.total / 1e3, tm.embed, tm.engram_io, tm.engram, tm.mhc, tm.attention, tm.gate,
                tm.shared_expert, tm.expert_io, tm.expert_gpu, tm.head, tm.host, tm.experts_read,
                tm.expert_bytes / 1e9, tm.dispatches, tm.submits);
    std::printf("    routing: %u / %u (token, layer) top-6 sets identical; %u / 40 layers identical at "
                "every token\n    tokens routed differently, per layer:", route_equal, route_tokens,
                route_layers_all);
    for (uint32_t L = 0; L < 40; ++L) std::printf(" %u", divergent[L]);
    std::printf("\n    worst probe: attn_norm %.7f, ffn_norm %.7f, block_out %.7f (L%u)\n", worst_attn_in,
                worst_ffn_in, worst_block, worst_block_layer);

    // --- the logits and the first token against L3 ---------------------------------
    auto st = runtime::DecodeState::load(l3_dir());
    REQUIRE_OK(st);
    const runtime::RefLogits& ref0 = st->logits(0);
    const double rho = rank_rho(ref0.top_ids, out->logits.data());
    double maxd = 0;
    for (size_t i = 0; i < ref0.top_ids.size(); ++i)
        maxd = std::max(maxd, std::fabs(double(out->logits[ref0.top_ids[i]]) - ref0.top_logits[i]));
    std::printf("    first token %u (reference %u) %s, margin ours %.4f ref %.4f, rho %.4f, max|dlogit| %.3f\n",
                out->first_token, ref0.argmax, out->first_token == ref0.argmax ? "MATCH" : "DIFFER",
                out->top1 - out->top2, ref0.margin(), rho, maxd);
    CHECK_EQ(out->first_token, ref0.argmax);

    // --- the handoff state against L3's prefill record ------------------------------
    {
        double worst_win = 1.0, worst_cmp = 1.0, worst_key = 1.0;
        uint32_t wl = 0, cl = 0, kl = 0;
        for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
            const gpu::PrefillHandoff::Layer& hl = out->layers[L];
            if (const runtime::StateTensor* w = st->tensor(0, std::format("L{:02d}.win_kv", L))) {
                const Agreement a = agree(std::vector<float>(hl.win_kv.begin(), hl.win_kv.begin() + kN * kHd),
                                          std::vector<float>(w->f.begin(), w->f.begin() + kN * kHd));
                if (a.cos < worst_win) { worst_win = a.cos; wl = L; }
            }
            if (hl.n_cmp) {
                if (const runtime::StateTensor* t = st->tensor(0, std::format("L{:02d}.cmp_cache", L))) {
                    const Agreement a = agree(hl.cmp_cache,
                                              std::vector<float>(t->f.begin(), t->f.begin() + hl.cmp_cache.size()));
                    if (a.cos < worst_cmp) { worst_cmp = a.cos; cl = L; }
                }
                if (const runtime::StateTensor* t = st->tensor(0, std::format("L{:02d}.index_k", L))) {
                    const Agreement a = agree(hl.index_k,
                                              std::vector<float>(t->f.begin(), t->f.begin() + hl.index_k.size()));
                    if (a.cos < worst_key) { worst_key = a.cos; kl = L; }
                }
            }
        }
        std::printf("    handoff vs L3 prefill record: window KV worst cos %.6f (L%u), compressed KV %.6f "
                    "(L%u), index keys %.6f (L%u)\n", worst_win, wl, worst_cmp, cl, worst_key, kl);
        CHECK(worst_win > 0.9);
    }

    // --- hand it to the decode engine and decode eight steps --------------------------
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "deepmoe_prefill_handoff";
    std::filesystem::create_directories(tmp);
    REQUIRE_OK(gpu::Prefill::write_l3_dir(*out, c, l3_dir(), tmp.string()));
    prefill.destroy();
    runner.destroy();
    REQUIRE_OK(engine.load_decode_state(tmp.string()));
    const runtime::DecodeState* ds = engine.decode_state();
    REQUIRE(ds != nullptr);
    const std::vector<uint32_t>& ref = ds->greedy_tokens();
    const uint32_t base = ds->decode_pos();
    uint32_t forced = 0;
    std::printf("    decode from OUR prefill state: %u steps teacher-forced\n", ds->steps());
    for (uint32_t s = 0; s < ds->steps(); ++s) {
        auto r = engine.decode_step(ref[s], base + s, static_cast<int32_t>(s));
        REQUIRE_OK(r);
        const runtime::RefLogits& rl = ds->logits(s + 1);
        const bool ok = r->token == rl.argmax;
        forced += ok;
        std::printf("      step %u in %6u -> %6u (reference %6u) %s  margin ours %.4f ref %.4f rho %.3f\n",
                    s, ref[s], r->token, rl.argmax, ok ? "match" : "DIFFER", r->margin(), rl.margin(),
                    rank_rho(rl.top_ids, engine.last_logits().data()));
    }
    std::printf("    teacher-forced: %u/%u\n", forced, ds->steps());
    uint32_t token = out->first_token, matched = 0;
    bool diverged = false;
    for (uint32_t s = 0; s < ds->steps(); ++s) {
        auto r = engine.decode_step(token, base + s, static_cast<int32_t>(s));
        REQUIRE_OK(r);
        const bool ok = !diverged && r->token == ref[s + 1];
        if (ok) ++matched; else diverged = true;
        std::printf("      free step %u in %6u -> %6u (reference %6u) %s margin %.4f\n", s, token,
                    r->token, ref[s + 1], r->token == ref[s + 1] ? "match" : "DIFFER", r->margin());
        token = r->token;
    }
    std::printf("    free-running: %u/%u before divergence\n", matched, ds->steps());
    CHECK(forced * 8 >= ds->steps() * 7);
    std::filesystem::remove_all(tmp);
}
