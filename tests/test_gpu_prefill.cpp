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
    // DEEPMOE_PF_ATTN=legacy: the per-(head, query) band attention instead of
    // the cooperative-matrix one; DEEPMOE_PF_ATTN_HT: its head tiles per workgroup.
    if (const char* e = std::getenv("DEEPMOE_PF_ATTN")) pc.attn_coop = std::string(e) != "legacy";
    if (const char* e = std::getenv("DEEPMOE_PF_ATTN_HT")) pc.attn_head_tiles = static_cast<uint32_t>(std::atoi(e));
    const std::string attn = pc.attn_coop ? std::format("coopmat, {} head tiles", pc.attn_head_tiles) : "legacy";
    std::printf("    kernels: MoE coopmat at n >= %d, dense coopmat at n >= %d (-1 = never), attention %s\n",
                static_cast<int32_t>(pc.coopmat_min_rows), static_cast<int32_t>(pc.coopmat_dense_min_rows),
                attn.c_str());
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

// An i32 tensor of an L3-format record, read straight off the container (the
// committed tests/data/longctx subset has no KV buffers, so DecodeState cannot
// load it). Empty when absent.
std::vector<int32_t> record_i32(const std::string& dir, size_t record, const std::string& name) {
    std::vector<int32_t> out;
    auto doc = json_parse_file(dir + "/index.json");
    if (!doc) return out;
    const JsonValue* steps = doc->find("steps");
    if (!steps) return out;
    auto arr = steps->as_array();
    if (!arr || record >= (*arr)->size()) return out;
    const JsonValue& rec = (**arr)[record];
    const JsonValue* tens = rec.find("tensors");
    if (!tens) return out;
    auto ta = tens->as_array();
    if (!ta) return out;
    for (const JsonValue& e : **ta) {
        if (e.string_or("name", "") != name) continue;
        std::FILE* f = std::fopen((dir + "/" + rec.string_or("file", "")).c_str(), "rb");
        if (!f) return out;
        const int64_t off = rec.int_or("data_offset", 12) + e.int_or("offset", 0);
        out.resize(static_cast<size_t>(e.int_or("bytes", 0) / 4));
        std::fseek(f, static_cast<long>(off), SEEK_SET);
        if (std::fread(out.data(), 4, out.size(), f) != out.size()) out.clear();
        std::fclose(f);
        break;
    }
    return out;
}

// Our handoff against a reference prefill record (DecodeState step 0), one
// line per layer: the window ring, the compressed KV and index keys, the
// compressor state, and -- where the reference exported them -- the last
// position's index row and top-6. Returns the worst cosines.
struct HandoffAgreement {
    double win = 1.0, cmp = 1.0, key = 1.0, state = 1.0;
    uint32_t win_l = 0, cmp_l = 0, key_l = 0;
    uint32_t topk_same = 0, topk_total = 0, gate_same = 0, gate_layers = 0;
};
HandoffAgreement compare_handoff(const gpu::PrefillHandoff& out, const runtime::DecodeState& st,
                                 const TextConfig& c, uint32_t N, const std::string& small_dir) {
    HandoffAgreement h;
    const uint32_t win_rows = std::min<uint32_t>(N, c.sliding_window);
    std::printf("    handoff vs the reference prefill record, per layer (cosines; topk = the last "
                "position's compressed picks in common; gate = its top-6 in common)\n");
    for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
        const gpu::PrefillHandoff::Layer& hl = out.layers[L];
        std::string line = std::format("      L{:02d} r{}", L, c.compress_ratio(L));
        // with N < 128 only slots 0..N-1 hold anything; at N >= 128 all of them
        if (const runtime::StateTensor* w = st.tensor(0, std::format("L{:02d}.win_kv", L))) {
            const Agreement a = agree(std::vector<float>(hl.win_kv.begin(), hl.win_kv.begin() + win_rows * kHd),
                                      std::vector<float>(w->f.begin(), w->f.begin() + win_rows * kHd));
            line += std::format("  win {:.6f}", a.cos);
            if (a.cos < h.win) { h.win = a.cos; h.win_l = L; }
        }
        if (hl.n_cmp) {
            if (const runtime::StateTensor* t = st.tensor(0, std::format("L{:02d}.cmp_cache", L))) {
                const Agreement a = agree(hl.cmp_cache,
                                          std::vector<float>(t->f.begin(), t->f.begin() + hl.cmp_cache.size()));
                line += std::format("  cmp {:.6f} ({} rows)", a.cos, hl.n_cmp);
                if (a.cos < h.cmp) { h.cmp = a.cos; h.cmp_l = L; }
            }
            if (const runtime::StateTensor* t = st.tensor(0, std::format("L{:02d}.index_k", L))) {
                const Agreement a = agree(hl.index_k,
                                          std::vector<float>(t->f.begin(), t->f.begin() + hl.index_k.size()));
                line += std::format("  key {:.6f}", a.cos);
                if (a.cos < h.key) { h.key = a.cos; h.key_l = L; }
            }
        }
        if (!hl.cmp_state_kv.empty()) {
            const runtime::StateTensor* sk = st.tensor(0, std::format("L{:02d}.cmp_state_kv", L));
            const runtime::StateTensor* ss = st.tensor(0, std::format("L{:02d}.cmp_state_score", L));
            if (sk && ss && sk->f.size() == hl.cmp_state_kv.size() && ss->f.size() == hl.cmp_state_score.size()) {
                // the -inf pattern must be identical; the written slots compared by cosine
                uint32_t pattern = 0;
                std::vector<float> a, b, as, bs;
                for (size_t i = 0; i < hl.cmp_state_score.size(); ++i) {
                    const bool oi = std::isinf(hl.cmp_state_score[i]), ri = std::isinf(ss->f[i]);
                    pattern += oi != ri;
                    if (!oi && !ri) {
                        a.push_back(hl.cmp_state_kv[i]); b.push_back(sk->f[i]);
                        as.push_back(hl.cmp_state_score[i]); bs.push_back(ss->f[i]);
                    }
                }
                if (a.empty()) {
                    line += pattern ? "  state PATTERN DIFFERS" : "  state empty (both)";
                } else {
                    const double ck = agree(a, b).cos, cs = agree(as, bs).cos;
                    line += std::format("  state kv {:.6f} score {:.6f}{}", ck, cs,
                                        pattern ? " PATTERN DIFFERS" : "");
                    h.state = std::min({h.state, ck, cs});
                }
                if (pattern) h.state = 0.0;
            }
        }
        if (const runtime::StateTensor* t = st.tensor(0, std::format("L{:02d}.topk_idxs_last", L));
            t && !t->i.empty() && hl.topk_last.size() == t->i.size()) {
            const uint32_t nw = std::min<uint32_t>(N, c.sliding_window);
            uint32_t wsame = 0;
            for (uint32_t i = 0; i < nw && i < t->i.size(); ++i) wsame += hl.topk_last[i] == t->i[i];
            std::set<int32_t> ours, ref;
            for (size_t i = nw; i < t->i.size(); ++i) {
                if (hl.topk_last[i] >= 0) ours.insert(hl.topk_last[i]);
                if (t->i[i] >= 0) ref.insert(t->i[i]);
            }
            uint32_t common = 0;
            for (int32_t v : ours) common += static_cast<uint32_t>(ref.count(v));
            line += std::format("  window {}/{}", wsame, nw);
            if (!ref.empty()) {
                line += std::format("  topk {}/{}", common, ref.size());
                if (c.is_index_source(L)) { h.topk_same += common; h.topk_total += uint32_t(ref.size()); }
            }
        }
        if (!small_dir.empty() && hl.gate_ids_last.size() == 6) {
            const std::vector<int32_t> ids = record_i32(small_dir, 0, std::format("L{:02d}.gate_ids_last", L));
            if (ids.size() == 6) {
                uint32_t same = 0;
                for (uint32_t a : hl.gate_ids_last)
                    for (int32_t b : ids) same += int32_t(a) == b;
                line += std::format("  gate {}/6", same);
                h.gate_same += same;
                ++h.gate_layers;
            }
        }
        std::printf("%s\n", line.c_str());
    }
    std::printf("    worst: window KV %.6f (L%u), compressed KV %.6f (L%u), index keys %.6f (L%u), "
                "compressor state %.6f\n", h.win, h.win_l, h.cmp, h.cmp_l, h.key, h.key_l, h.state);
    if (h.topk_total)
        std::printf("    last position's compressed picks at the index sources: %u / %u in common\n",
                    h.topk_same, h.topk_total);
    if (h.gate_layers)
        std::printf("    last position's top-6: %u / %u in common over %u layers\n", h.gate_same,
                    h.gate_layers * 6, h.gate_layers);
    return h;
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
        const HandoffAgreement h = compare_handoff(*out, *st, c, kN, "");
        CHECK(h.win > 0.9);
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

// gpu_prefill.longctx
// -------------------
// DEEPMOE_PF_LONGCTX=traces/longctx/ctx4k (or ctx16k): a Track M export
// (docs/p3_longctx.md §4.1). The GPU prefill of its prompt -- oracle mode unless
// DEEPMOE_PF_REPLAY sets a replay length -- its first token and handoff state
// against the export's prefill record, then eight decode steps from our state
// through runtime::Engine. DEEPMOE_PF_DECODE: free (default: free-running from
// our first token, the salt retrieval), forced (teacher-forced), none, or
// ref-free / ref-forced (no prefill: the same steps from the export's own
// state, the engine's baseline). One decode mode per process: a second pass
// over the same positions would pool a ratio-2 group with the first pass's
// state at odd N. DEEPMOE_PF_TRUNCATE=n prefills only the first n prompt
// tokens and decodes from there with no reference to compare against -- an
// engine sanity check at a context between 64 and the export's (a garbage
// step shows as token 0 at margin 0).
DEEPMOE_TEST(gpu_prefill, longctx) {
    const char* dir_env = std::getenv("DEEPMOE_PF_LONGCTX");
    if (!dir_env) {
        std::printf("      SKIP gpu_prefill.longctx: set DEEPMOE_PF_LONGCTX=traces/longctx/ctx4k\n");
        return;
    }
    if (skip_without_model("gpu_prefill")) return;
    const std::string dir = dir_env;
    const std::string name = std::filesystem::path(dir).filename().string();
    const std::string small = std::string(DEEPMOE_TEST_DATA_DIR) + "/longctx/" + name;
    const std::string mode = std::getenv("DEEPMOE_PF_DECODE") ? std::getenv("DEEPMOE_PF_DECODE") : "free";
    auto st = runtime::DecodeState::load(dir);
    REQUIRE_OK(st);
    std::vector<uint32_t> prompt = prompt_ids(dir);
    const uint32_t truncate = std::getenv("DEEPMOE_PF_TRUNCATE")
                                  ? static_cast<uint32_t>(std::atoi(std::getenv("DEEPMOE_PF_TRUNCATE"))) : 0;
    if (truncate && truncate < prompt.size()) prompt.resize(truncate);
    const bool truncated = truncate && truncate == prompt.size();
    const uint32_t N = static_cast<uint32_t>(prompt.size());
    REQUIRE(N > 0);
    const std::vector<uint32_t>& greedy = st->greedy_tokens();
    std::printf("    %s: %u prompt tokens, reference continuation", name.c_str(), N);
    for (uint32_t t : greedy) std::printf(" %u", t);
    std::printf("\n");

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
    const bool from_ref = mode.starts_with("ref");
    const bool forced = mode.ends_with("forced");
    uint32_t first = greedy.empty() ? 0 : greedy[0];
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / ("deepmoe_prefill_" + name);

    if (!from_ref) {
        gpu::MemoryAllocator alloc;
        REQUIRE_OK(alloc.init(engine.device(), MemoryPath::DeviceLocalHostVisible));
        store::ShardSet shards;
        REQUIRE_OK(shards.open_all(model_dir(), engine.manifest(), true));
        auto tables = runtime::EngramTables::load(l3_dir());
        REQUIRE_OK(tables);
        gpu::PrefillRunner runner;
        REQUIRE_OK(runner.create(engine.device(), alloc, gpu::default_shader_dir()));
        gpu::PrefillConfig pc;
        pc.max_tokens = N;
        pc.replay = std::getenv("DEEPMOE_PF_REPLAY") ? static_cast<uint32_t>(std::atoi(std::getenv("DEEPMOE_PF_REPLAY"))) : 0;
        if (pc.replay == 0) pc.replay = N;
        apply_kernel_env(pc);
        gpu::Prefill prefill;
        REQUIRE_OK(prefill.create(engine.device(), alloc, runner, engine.manifest(), shards, engine.io(),
                                  engine.pinned(), c, &*tables, pc));
        std::printf("    GPU prefill of %u tokens, %s\n", N,
                    pc.replay >= N ? "oracle mode (replay >= N)" : std::format("replay {}", pc.replay).c_str());
        auto out = prefill.run(prompt);
        REQUIRE_OK(out);
        const gpu::PrefillTimes& tm = prefill.times();
        std::printf("    %.1f s: embed %.0f  engram io %.0f / gpu %.0f  mhc %.0f  attention %.0f  gate %.0f  "
                    "shared %.0f  expert io %.0f / gpu %.0f  head %.0f  other %.0f ms; %u experts, %.2f GB, "
                    "%u dispatches, %u submits\n",
                    tm.total / 1e3, tm.embed, tm.engram_io, tm.engram, tm.mhc, tm.attention, tm.gate,
                    tm.shared_expert, tm.expert_io, tm.expert_gpu, tm.head, tm.host, tm.experts_read,
                    tm.expert_bytes / 1e9, tm.dispatches, tm.submits);
        const runtime::RefLogits& ref0 = st->logits(0);
        double maxd = 0;
        for (size_t i = 0; i < ref0.top_ids.size(); ++i)
            maxd = std::max(maxd, std::fabs(double(out->logits[ref0.top_ids[i]]) - ref0.top_logits[i]));
        std::printf("    first token %u (reference %u) %s, margin ours %.4f ref %.4f, rho %.4f, max|dlogit| %.3f\n",
                    out->first_token, ref0.argmax, out->first_token == ref0.argmax ? "MATCH" : "DIFFER",
                    out->top1 - out->top2, ref0.margin(), rank_rho(ref0.top_ids, out->logits.data()), maxd);
        first = out->first_token;
        if (!truncated) {
            CHECK_EQ(out->first_token, ref0.argmax);
            const HandoffAgreement h = compare_handoff(*out, *st, c, N,
                                                       std::filesystem::exists(small + "/index.json") ? small : "");
            CHECK(h.win > 0.8);
        }
        if (mode == "none") return;
        std::filesystem::create_directories(tmp);
        REQUIRE_OK(gpu::Prefill::write_l3_dir(*out, c, dir, tmp.string()));
        if (truncated) {
            // our prompt, not the export's: prompt_ids / prefill_len / decode_pos
            const std::string path = (tmp / "index.json").string();
            std::FILE* f = std::fopen(path.c_str(), "rb");
            REQUIRE(f != nullptr);
            std::string text;
            char buf[65536];
            for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, n);
            std::fclose(f);
            auto set_int = [&](const std::string& key) {
                const size_t k = text.find("\"" + key + "\"");
                if (k == std::string::npos) return;
                const size_t a = text.find_first_of("0123456789", k + key.size() + 2);
                const size_t b = text.find_first_not_of("0123456789", a);
                text = text.substr(0, a) + std::to_string(N) + text.substr(b);
            };
            set_int("prefill_len");
            set_int("decode_pos");
            const size_t k = text.find("\"prompt_ids\"");
            const size_t a = text.find('[', k), b = text.find(']', a);
            std::string ids;
            for (uint32_t i = 0; i < N; ++i) ids += (i ? ", " : "") + std::to_string(prompt[i]);
            text = text.substr(0, a + 1) + ids + text.substr(b);
            f = std::fopen(path.c_str(), "wb");
            REQUIRE(f != nullptr);
            std::fwrite(text.data(), 1, text.size(), f);
            std::fclose(f);
        }
    }
    st = {};

    const std::string state_dir = from_ref ? dir : tmp.string();
    if (auto r = engine.load_decode_state(state_dir); !r) {
        std::printf("    decode: the engine refused the state: %s\n", r.error().str().c_str());
        if (!from_ref) std::filesystem::remove_all(tmp);
        return;
    }
    const runtime::DecodeState* ds = engine.decode_state();
    REQUIRE(ds != nullptr);
    const std::vector<uint32_t>& ref = ds->greedy_tokens();
    const uint32_t base = ds->decode_pos();
    std::printf("    decode from %s state, %s, %u steps\n", from_ref ? "the EXPORT's" : "OUR",
                forced ? "teacher-forced" : "free-running", ds->steps());
    uint32_t token = forced ? ref[0] : first, matched = 0, before_div = 0;
    bool diverged = false;
    for (uint32_t s = 0; s < ds->steps(); ++s) {
        const uint32_t in = forced ? ref[s] : token;
        auto r = engine.decode_step(in, base + s, static_cast<int32_t>(s));
        REQUIRE_OK(r);
        const runtime::RefLogits& rl = ds->logits(s + 1);
        const bool ok = r->token == rl.argmax;
        matched += ok;
        if (!ok) diverged = true;
        if (!diverged) ++before_div;
        std::printf("      step %u pos %u in %6u -> %6u (reference %6u) %s  margin ours %.4f ref %.4f rho %.3f\n",
                    s, base + s, in, r->token, rl.argmax, ok ? "match" : "DIFFER", r->margin(), rl.margin(),
                    rank_rho(rl.top_ids, engine.last_logits().data()));
        token = r->token;
    }
    if (forced)
        std::printf("    teacher-forced: %u/%u\n", matched, ds->steps());
    else
        std::printf("    free-running: %u/%u before divergence (%u/%u steps agree)\n", before_div, ds->steps(),
                    matched, ds->steps());
    if (!from_ref) std::filesystem::remove_all(tmp);
}

// gpu_prefill.engram_repeat
// -------------------------
// Determinism of the engram row reads at long context: the rows of layers 1
// and 14 for a Track M prompt (DEEPMOE_PF_LONGCTX), read three times, must be
// byte-identical. Guards the host staging and the IoEngine under load.
DEEPMOE_TEST(gpu_prefill, engram_repeat) {
    const char* dir_env = std::getenv("DEEPMOE_PF_LONGCTX");
    if (!dir_env) {
        std::printf("      SKIP gpu_prefill.engram_repeat: set DEEPMOE_PF_LONGCTX=traces/longctx/ctx4k\n");
        return;
    }
    if (skip_without_model("gpu_prefill")) return;
    const std::vector<uint32_t> prompt = prompt_ids(dir_env);
    const uint32_t N = static_cast<uint32_t>(prompt.size());
    REQUIRE(N > 0);
    gpu::PrefillConfig pc;
    pc.max_tokens = N;
    pc.transit_slots = 1;
    Rig rig;
    if (!rig.up({1, 14}, pc)) {
        std::printf("      SKIP gpu_prefill: %s\n", rig.why.c_str());
        return;
    }
    gpu::Prefill& P = rig.prefill;
    const size_t bytes = size_t(N) * 6144 * sizeof(float);
    for (uint32_t L : {1u, 14u}) {
        std::vector<float> first;
        uint32_t differing_runs = 0;
        for (uint32_t rep = 0; rep < 3; ++rep) {
            REQUIRE_OK(P.op_engram_rows(L, prompt));
            auto view = P.engram_x_host();
            std::vector<float> cur(view, view + bytes / sizeof(float));
            if (rep == 0) { first = std::move(cur); continue; }
            uint64_t diff_rows = 0;
            for (uint32_t p = 0; p < N; ++p)
                diff_rows += std::memcmp(first.data() + size_t(p) * 6144, cur.data() + size_t(p) * 6144,
                                         6144 * sizeof(float)) != 0;
            std::printf("    L%u read %u vs read 0: %llu / %u positions differ\n", L, rep,
                        (unsigned long long)diff_rows, N);
            differing_runs += diff_rows != 0;
        }
        CHECK_EQ(differing_runs, 0u);
    }
}

// gpu_prefill.repeat
// ------------------
// Run-to-run determinism of a whole prefill: the same prompt twice in one
// process (DEEPMOE_PF_LONGCTX's, first DEEPMOE_PF_TRUNCATE tokens if set,
// DEEPMOE_PF_REPLAY rows), every layer's stage outputs hashed; prints the first
// (layer, stage) whose bytes differ.
DEEPMOE_TEST(gpu_prefill, repeat) {
    const char* dir_env = std::getenv("DEEPMOE_PF_LONGCTX");
    if (!dir_env || !std::getenv("DEEPMOE_PF_REPEAT")) {
        std::printf("      SKIP gpu_prefill.repeat: set DEEPMOE_PF_LONGCTX and DEEPMOE_PF_REPEAT=1\n");
        return;
    }
    if (skip_without_model("gpu_prefill")) return;
    std::vector<uint32_t> prompt = prompt_ids(dir_env);
    if (const char* t = std::getenv("DEEPMOE_PF_TRUNCATE"))
        if (uint32_t n = static_cast<uint32_t>(std::atoi(t)); n && n < prompt.size()) prompt.resize(n);
    const uint32_t N = static_cast<uint32_t>(prompt.size());
    REQUIRE(N > 0);
    gpu::PrefillConfig pc;
    pc.max_tokens = N;
    pc.replay = std::getenv("DEEPMOE_PF_REPLAY") ? static_cast<uint32_t>(std::atoi(std::getenv("DEEPMOE_PF_REPLAY"))) : N;
    if (pc.replay == 0) pc.replay = N;
    pc.probe_layers = true;
    apply_kernel_env(pc);
    std::vector<uint32_t> layers(40);
    for (uint32_t L = 0; L < 40; ++L) layers[L] = L;
    Rig rig;
    if (!rig.up(layers, pc)) {
        std::printf("      SKIP gpu_prefill: %s\n", rig.why.c_str());
        return;
    }
    // four independent 64-bit lanes over whole words (the tensors are float /
    // uint32 arrays, so `bytes` is a multiple of 4); fast enough for 1 GB a layer
    auto fnv = [](const void* p, size_t bytes) {
        uint64_t h[4] = {1469598103934665603ull, 7ull, 11ull, 13ull};
        const auto* b = static_cast<const uint8_t*>(p);
        size_t i = 0;
        for (; i + 32 <= bytes; i += 32)
            for (uint32_t l = 0; l < 4; ++l) {
                uint64_t w;
                std::memcpy(&w, b + i + l * 8, 8);
                h[l] = (h[l] ^ w) * 1099511628211ull;
            }
        for (; i < bytes; ++i) h[0] = (h[0] ^ b[i]) * 1099511628211ull;
        return h[0] ^ (h[1] << 1) ^ (h[2] << 2) ^ (h[3] << 3);
    };
    static const char* const kStages[] = {"block_in", "attn_norm_out", "kv", "cmp_cache", "attn_out",
                                          "attn_block_out", "ffn_norm_out", "gate_ids", "moe_out", "block_out"};
    std::vector<std::vector<uint64_t>> runs[2];
    for (uint32_t run = 0; run < 2; ++run) {
        runs[run].assign(40, std::vector<uint64_t>(10, 0));
        rig.prefill.probe = [&](const gpu::PrefillProbe& pv) {
            auto& h = runs[run][pv.layer];
            h[0] = fnv(pv.block_in, size_t(pv.rows) * kHc * kDim * 4);
            h[1] = fnv(pv.attn_norm_out, size_t(pv.front_rows) * kDim * 4);
            h[2] = fnv(pv.kv, size_t(pv.attn_rows) * kHd * 4);
            h[3] = pv.cmp_cache ? fnv(pv.cmp_cache, size_t(pv.n_cmp) * kHd * 4) : 0;
            h[4] = fnv(pv.attn_out, size_t(pv.attn_rows) * kDim * 4);
            h[5] = fnv(pv.attn_block_out, size_t(pv.attn_rows) * kHc * kDim * 4);
            h[6] = fnv(pv.ffn_norm_out, size_t(pv.attn_rows) * kDim * 4);
            h[7] = fnv(pv.gate_ids, size_t(pv.attn_rows) * 6 * 4);
            h[8] = fnv(pv.moe_out, size_t(pv.attn_rows) * kDim * 4);
            h[9] = fnv(pv.block_out, size_t(pv.attn_rows) * kHc * kDim * 4);
        };
        auto out = rig.prefill.run(prompt);
        REQUIRE_OK(out);
        std::printf("    run %u: %.1f s, first token %u, margin %.4f, %u experts\n", run,
                    rig.prefill.times().total / 1e3, out->first_token, out->top1 - out->top2,
                    rig.prefill.times().experts_read);
    }
    bool same = true;
    for (uint32_t L = 0; L < 40 && same; ++L)
        for (uint32_t s = 0; s < 10; ++s)
            if (runs[0][L][s] != runs[1][L][s]) {
                std::printf("    first difference: layer %u, %s\n", L, kStages[s]);
                same = false;
                break;
            }
    if (same) std::printf("    the two runs are byte-identical at every probed stage of every layer\n");
    CHECK(same);
}
