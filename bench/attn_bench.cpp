// Effective GB/s of the design §7.2-§7.8 kernels, on the real checkpoint.
//
// design §7.1 rule 2: a GEMV is scored by effective bandwidth against the
// 216-218 GB/s streaming ceiling kernel_p1.md §2.2 measured, and nothing else.
// This is the attention-path counterpart of bench/kernel_bench, which did the
// same for the two MoE dispatches (§7.9.1).
//
// Two things make the numbers honest:
//
//   * **Several layers, cycled.** One layer's attention weights are 126 MB and
//     the MALL is 32 MB, but wq_a (6.6 MB) and wkv (2.6 MB) would sit in it
//     entirely and report a fantasy. `--layers` pins N layers and iteration i
//     uses layer `i % N`, which is what a decode token actually does.
//   * **Bytes touched, not bytes allocated.** Each kernel's byte count is the
//     weight plane plus its UE8M0 scale plane, computed from the manifest's own
//     shapes. `sparse_attn` is reported separately because it is not a
//     bandwidth kernel at all: 320 KiB of KV read once per head, so what it
//     reports is a latency, and the interesting number is the microseconds.
//
// Usage:
//   attn_bench --model D:/models/DeepSeek-V4.1-Flash [--layers 4] [--iters 64]
//              [--lanes 32] [--csv bench/results/attn_p2.csv] [--note "..."]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <memory>
#include <string>
#include <vector>

#include "core/config.h"
#include "gpu/vulkan/attn_kernels.h"
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/device.h"
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

struct Opts {
    std::string model;
    std::string csv;
    std::string note;
    uint32_t    layers = 4;
    uint32_t    iters  = 64;
    uint32_t    lanes  = 32;
    uint32_t    subgroup = 32;
    uint32_t    rows   = 4;
    uint32_t    heads  = 1;   // sparse_attn heads per workgroup (§7.5); AttnSpec default
};

// design §2.3's per-layer byte budget, recomputed here from the shapes so the
// two cannot drift.
struct Dims {
    uint32_t dim = layout::kHiddenSize;         // 5120
    uint32_t hc = 4, n_heads = 64, head_dim = 512, rope_dim = 64;
    uint32_t q_lora = 1280, o_lora = 1024, o_groups = 8, window = 128;
    uint32_t n_experts = layout::kRoutedExperts;
    uint32_t vocab = 129280;
    uint32_t qrows() const { return n_heads * head_dim; }
    uint32_t ocols() const { return qrows() / o_groups; }
    uint32_t orows() const { return o_groups * o_lora; }
};

// fp8: one byte an element plus a [rows/32][K/32] UE8M0 plane.
uint64_t fp8_bytes(uint64_t rows, uint64_t k) { return rows * k + (rows / 32) * (k / 32); }
uint64_t bf16_bytes(uint64_t rows, uint64_t k) { return rows * k * 2; }

struct Row {
    const char* name;
    double      ms;
    uint64_t    bytes;
    double gbps() const { return ms > 0 ? bytes / (ms * 1e-3) / 1e9 : 0.0; }
};

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
    o.layers = static_cast<uint32_t>(std::atoi(arg_after(argc, argv, "--layers", "4").c_str()));
    o.iters  = static_cast<uint32_t>(std::atoi(arg_after(argc, argv, "--iters", "64").c_str()));
    o.lanes  = static_cast<uint32_t>(std::atoi(arg_after(argc, argv, "--lanes", "32").c_str()));
    o.rows   = static_cast<uint32_t>(std::atoi(arg_after(argc, argv, "--rows", "4").c_str()));
    o.heads  = static_cast<uint32_t>(std::atoi(arg_after(argc, argv, "--heads", "1").c_str()));
    o.subgroup = static_cast<uint32_t>(std::atoi(arg_after(argc, argv, "--subgroup", "32").c_str()));
    if (o.model.empty()) {
        std::printf("usage: attn_bench --model <checkpoint dir> [--layers N] [--iters N]\n"
                    "                  [--lanes 16|32|64] [--csv PATH] [--note TEXT]\n");
        return 2;
    }

    Dims d;
    gpu::Device device;
    if (auto r = device.create({}); !r) { std::printf("device: %s\n", r.error().str().c_str()); return 1; }
    if (auto r = device.caps().check_required(); !r) {
        std::printf("device: %s\n", r.error().str().c_str());
        return 1;
    }
    gpu::MemoryAllocator alloc;
    if (auto r = alloc.init(device, MemoryPath::DeviceLocalHostVisible); !r) {
        std::printf("alloc: %s\n", r.error().str().c_str());
        return 1;
    }
    auto mf = Manifest::load(store::ShardSet::join(o.model, layout::kManifestFile));
    if (!mf) { std::printf("manifest: %s\n", mf.error().str().c_str()); return 1; }
    store::ShardSet shards;
    if (auto r = shards.open_all(o.model, *mf, true); !r) {
        std::printf("shards: %s\n", r.error().str().c_str());
        return 1;
    }
    IoConfig iocfg;
    auto backend = storage::make_default_backend(iocfg);
    if (!backend) { std::printf("backend: %s\n", backend.error().str().c_str()); return 1; }
    storage::IoEngine io;
    if (auto r = io.start(std::move(*backend), iocfg); !r) {
        std::printf("io: %s\n", r.error().str().c_str());
        return 1;
    }

    store::PinnedStore pinned;
    auto backing = alloc.make_slab_backing();
    if (!backing) { std::printf("backing: %s\n", backing.error().str().c_str()); io.stop(); return 1; }
    store::PinnedConfig pc;
    pc.region_bytes = 1ull << 30;
    if (auto r = pinned.init(std::move(*backing), pc); !r) {
        std::printf("pinned: %s\n", r.error().str().c_str());
        io.stop();
        return 1;
    }
    // design §7.4's compressor and indexer live only on a source layer, so the
    // benchmark pins those four as well and cycles the Cmp*/Idx* rows over them
    // -- otherwise every iteration would read layer 2's 10 MB out of the MALL.
    static const uint32_t kSrcLayers[] = {2, 8, 14, 20};
    static const uint32_t kNumSrc = 4;
    std::vector<std::string> names;
    for (uint32_t L = 0; L < o.layers; ++L) {
        auto n = store::pinned_layer_tensors(*mf, L);
        names.insert(names.end(), n.begin(), n.end());
    }
    for (uint32_t L : kSrcLayers) {
        if (L < o.layers) continue;
        auto n = store::pinned_layer_tensors(*mf, L);
        names.insert(names.end(), n.begin(), n.end());
    }
    names.push_back("head.weight");
    std::printf("loading %u layers + the head (%.2f GB)...\n", o.layers,
                store::pinned_bytes(*mf, names) / 1e9);
    if (auto r = pinned.load(*mf, shards, io, names); !r) {
        std::printf("pinned load: %s\n", r.error().str().c_str());
        io.stop();
        return 1;
    }

    gpu::AttnSpec spec;
    spec.lanes_per_row = o.lanes;
    spec.subgroup_size = o.subgroup;
    spec.rows_per_lane = o.rows;
    spec.heads_per_wg  = o.heads;
    // ONE RUNNER PER LAYER. A runner owns one address table, so with a single
    // runner every iteration inside a command buffer reads the same layer's
    // weights and anything under the 32 MB MALL reports a fantasy -- the first
    // version of this benchmark had wq_a at 374 GB/s and wo_a at 397 against a
    // 217 GB/s memory system. Iteration i uses runner i % layers, which is what
    // a decode token does.
    std::vector<std::unique_ptr<gpu::AttnRunner>> runners(o.layers);
    for (uint32_t i = 0; i < o.layers; ++i) {
        runners[i] = std::make_unique<gpu::AttnRunner>();
        if (auto r = runners[i]->create(device, alloc, gpu::default_shader_dir(), spec); !r) {
            std::printf("runner %u: %s\n", i, r.error().str().c_str());
            io.stop();
            return 1;
        }
    }
    gpu::AttnRunner& runner = *runners[0];
    gpu::GpuScratch scratch;
    if (auto r = scratch.create(alloc, 128ull << 20); !r) {
        std::printf("scratch: %s\n", r.error().str().c_str());
        io.stop();
        return 1;
    }

    auto take = [&](uint64_t bytes) {
        auto v = scratch.alloc(bytes);
        return v ? *v : gpu::GpuScratch::View{};
    };
    const uint32_t hcdim = d.hc * d.dim;
    auto X = take(hcdim * 4), A = take(d.dim * 4), Post = take(64), Comb = take(256);
    auto Pre = take(64), Xout = take(hcdim * 4), Utmp = take(d.dim * 4), Scr = take(2048);
    auto MixR = take(128), MixO = take(128), U = take(d.dim * 4);
    auto Qr = take(d.q_lora * 4), Q = take(d.qrows() * 2), Rope = take(256);
    auto KvRaw = take(d.head_dim * 4), KvScr = take(256);
    auto WinV = take(uint64_t(d.window) * d.head_dim);
    auto WinS = take(uint64_t(d.window) * (d.head_dim / 32));
    auto Kv = take(d.head_dim * 4), Cmp = take(1024ull * d.head_dim * 2);
    auto Top = take(1024 * 4), Score = take(uint64_t(d.n_heads) * 1024 * 4);
    auto O = take(uint64_t(d.qrows()) * 4), Woa = take(d.orows() * 4), Wob = take(d.dim * 4);
    auto Gs = take(d.n_experts * 4), Gid = take(64), Gw = take(64), Done = take(64);
    auto Logits = take(uint64_t(d.vocab) * 4);
    // design §7.4: the compressor's two projections and its carried state, the
    // indexer's queries, its key cache and its scores.
    const uint32_t idx_heads = 32, idx_dim = 128, n_cmp_max = 1024;
    auto CmpY = take(d.head_dim * 4), CmpG = take(d.head_dim * 4);
    auto CmpKvSt = take(uint64_t(8) * d.head_dim * 4);
    auto CmpScSt = take(uint64_t(8) * d.head_dim * 4);
    auto CmpLat = take(d.head_dim * 4), CmpLatQ = take(d.head_dim * 4);
    auto CmpFp4 = take(d.head_dim / 2), CmpScB = take(d.head_dim / 16);
    auto IdxQRaw = take(uint64_t(idx_heads) * idx_dim * 4);
    auto IdxQv  = take(uint64_t(idx_heads) * idx_dim * 2);
    auto IdxQFp4 = take(uint64_t(idx_heads) * idx_dim / 2);
    auto IdxQSc = take(uint64_t(idx_heads) * idx_dim / 32);
    auto IdxKRaw = take(idx_dim * 4);
    auto IdxKC  = take(uint64_t(n_cmp_max) * idx_dim * 2);
    auto IdxKFp4 = take(idx_dim / 2), IdxKSc = take(idx_dim / 32 + 4);
    auto IdxWt  = take(idx_heads * 4), IdxSc = take(n_cmp_max * 4);
    if (!Logits.valid()) { std::printf("scratch exhausted\n"); io.stop(); return 1; }

    // A plausible activation: the kernels are bandwidth-bound and the values
    // only matter for denormal behaviour, which fp8 does not have.
    for (uint32_t i = 0; i < hcdim; ++i)
        static_cast<float*>(X.host)[i] = 0.3f * std::sin(0.01f * float(i));
    for (uint32_t i = 0; i < d.dim; ++i) {
        static_cast<float*>(U.host)[i] = 0.3f * std::sin(0.013f * float(i));
        static_cast<float*>(A.host)[i] = 0.1f * std::cos(0.017f * float(i));
    }
    for (uint32_t i = 0; i < d.q_lora; ++i)
        static_cast<float*>(Qr.host)[i] = 0.2f * std::sin(0.021f * float(i));
    for (uint32_t i = 0; i < d.orows(); ++i)
        static_cast<float*>(Woa.host)[i] = 0.2f * std::sin(0.007f * float(i));
    for (uint32_t i = 0; i < d.qrows(); ++i)
        static_cast<float*>(O.host)[i] = 0.2f * std::sin(0.003f * float(i));
    const std::vector<float> rope = runtime::rope_table(runtime::rope_for_layer(1, d.rope_dim), 64);
    std::memcpy(Rope.host, rope.data(), rope.size() * 4);
    // Every window slot live, and 65 compressed positions, as at decode
    // position 64 of a 64-token prefill (the L2 export's geometry).
    const uint32_t n_cmp = 65, n_kv = d.window + n_cmp;
    for (uint32_t i = 0; i < n_kv; ++i) static_cast<uint32_t*>(Top.host)[i] = i;
    for (uint32_t i = 0; i < uint64_t(n_cmp) * d.head_dim; ++i)
        static_cast<uint16_t*>(Cmp.host)[i] = 0x3C00;      // bf16 1.0

    gpu::CommandPool pool;
    if (auto r = pool.create(device); !r) { std::printf("pool: %s\n", r.error().str().c_str()); io.stop(); return 1; }
    gpu::QueryPool queries;
    const bool timed = queries.create(device, 2) && device.caps().timestamp_valid_bits > 0;

    // Runs `iters` dispatches of one stage back to back in one command buffer,
    // re-pointing the weight slots at layer `i % layers` each time. Recording
    // the push constants per iteration is exactly what design §7.1's
    // "pre-recorded command buffer" does, minus the pre.
    auto run = [&](const char* label, gpu::AttnStage stage, uint64_t bytes,
                   auto set_slots, const void* push, uint32_t push_bytes,
                   uint32_t groups) -> Row {
        for (uint32_t i = 0; i < o.layers; ++i) set_slots(*runners[i], i);
        // Warm the caches and the pipelines before timing.
        (void)runners[0]->dispatch_now(stage, push, push_bytes, groups);

        auto cb = pool.acquire();
        if (!cb) return Row{label, 0, bytes};
        gpu::CommandBuffer cmd = *cb;
        (void)cmd.begin();
        if (timed) {
            (void)cmd.reset_queries(queries, 0, 2);
            (void)cmd.write_timestamp(queries, 0, false);
        }
        for (uint32_t i = 0; i < o.iters; ++i) {
            (void)runners[i % o.layers]->record(cmd, stage, push, push_bytes, groups);
            (void)cmd.barrier();
        }
        if (timed) (void)cmd.write_timestamp(queries, 1, true);
        (void)cmd.end();

        double best = 1e9;
        for (uint32_t rep = 0; rep < 3; ++rep) {
            if (auto r = gpu::submit_and_wait(device, cmd); !r) return Row{label, 0, bytes};
            if (timed) {
                if (auto s = queries.elapsed_seconds(0, 1); s)
                    best = std::min(best, *s * 1e3 / o.iters);
            }
        }
        return Row{label, timed ? best : 0.0, bytes};
    };

    auto addr = [&](uint32_t L, const char* suffix) {
        const store::PinnedTensor* t = pinned.find(std::format("layers.{}.{}", L, suffix));
        return t ? t->data : 0ull;
    };
    auto sc_addr = [&](uint32_t L, const char* suffix) {
        const store::PinnedTensor* t = pinned.find(std::format("layers.{}.{}", L, suffix));
        return t ? t->scale : 0ull;
    };

    std::vector<Row> rows;

    // --- mega_mhc -------------------------------------------------------
    gpu::MhcPush mp{d.dim, d.hc, (2 + d.hc) * d.hc, d.dim / 256, 20, gpu::kMhcFlagPost,
                    1e-20f, 1e-6f};
    auto mhc_slots = [&](gpu::AttnRunner& runner, uint32_t L) {
        uint64_t* s = runner.slots(gpu::AttnStage::MhcPost);
        s[gpu::slot::kX] = X.addr;         s[gpu::slot::kA] = A.addr;
        s[gpu::slot::kPostIn] = Post.addr; s[gpu::slot::kCombIn] = Comb.addr;
        s[gpu::slot::kPreMix] = Pre.addr;
        s[gpu::slot::kHcFn] = addr(L, "hc_attn_fn");
        s[gpu::slot::kHcBase] = addr(L, "hc_attn_base");
        s[gpu::slot::kHcScale] = addr(L, "hc_attn_scale");
        s[gpu::slot::kNormW] = addr(L, "attn_norm.weight");
        s[gpu::slot::kXout] = Xout.addr;   s[gpu::slot::kUtmp] = Utmp.addr;
        s[gpu::slot::kScratch] = Scr.addr; s[gpu::slot::kMixRaw] = MixR.addr;
        s[gpu::slot::kMixOut] = MixO.addr; s[gpu::slot::kU] = U.addr;
        std::memcpy(runner.slots(gpu::AttnStage::MhcMix), s, gpu::kAttnStageStride);
        std::memcpy(runner.slots(gpu::AttnStage::MhcFinal), s, gpu::kAttnStageStride);
    };
    const uint64_t hcfn_bytes = uint64_t((2 + d.hc) * d.hc) * hcdim * 4;
    rows.push_back(run("mega_mhc.post", gpu::AttnStage::MhcPost, uint64_t(hcdim) * 4 * 2,
                       mhc_slots, &mp, sizeof mp, mp.n_wg0));
    rows.push_back(run("mega_mhc.mix", gpu::AttnStage::MhcMix, hcfn_bytes,
                       mhc_slots, &mp, sizeof mp, mp.mix_rows));
    rows.push_back(run("mega_mhc.final", gpu::AttnStage::MhcFinal, uint64_t(d.dim) * 4 * 2,
                       mhc_slots, &mp, sizeof mp, mp.n_wg0));

    // --- wq_a -----------------------------------------------------------
    gpu::GemvPush qa{d.q_lora, d.dim, d.dim / 32, 0};
    rows.push_back(run("wq_a", gpu::AttnStage::WqA, fp8_bytes(d.q_lora, d.dim), [&](gpu::AttnRunner& runner, uint32_t L) {
        uint64_t* s = runner.slots(gpu::AttnStage::WqA);
        s[gpu::slot::kGemvW] = addr(L, "attn.wq_a.weight");
        s[gpu::slot::kGemvS] = sc_addr(L, "attn.wq_a.weight");
        s[gpu::slot::kGemvX] = U.addr; s[gpu::slot::kGemvY] = Qr.addr;
    }, &qa, sizeof qa, runner.gemv_groups(gpu::AttnStage::WqA, d.q_lora)));

    // --- wq_b -----------------------------------------------------------
    gpu::WqbPush qb{d.qrows(), d.q_lora, d.q_lora / 32, d.head_dim, d.rope_dim, 1e-20f};
    rows.push_back(run("wq_b", gpu::AttnStage::WqB, fp8_bytes(d.qrows(), d.q_lora),
                       [&](gpu::AttnRunner& runner, uint32_t L) {
        uint64_t* s = runner.slots(gpu::AttnStage::WqB);
        s[gpu::slot::kWqbW] = addr(L, "attn.wq_b.weight");
        s[gpu::slot::kWqbS] = sc_addr(L, "attn.wq_b.weight");
        s[gpu::slot::kWqbQr] = Qr.addr;
        s[gpu::slot::kWqbNormW] = addr(L, "attn.q_norm.weight");
        s[gpu::slot::kWqbRope] = Rope.addr; s[gpu::slot::kWqbQ] = Q.addr;
    }, &qb, sizeof qb, runner.gemv_groups(gpu::AttnStage::WqB, d.qrows())));

    // --- wkv ------------------------------------------------------------
    const uint32_t kvg = runner.gemv_groups(gpu::AttnStage::WkvGemv, d.head_dim);
    gpu::WkvPush kp{d.head_dim, d.dim, d.dim / 32, d.rope_dim, 64 % d.window, kvg, 1e-20f};
    auto kv_slots = [&](gpu::AttnRunner& runner, uint32_t L) {
        uint64_t* s = runner.slots(gpu::AttnStage::WkvGemv);
        s[gpu::slot::kWkvW] = addr(L, "attn.wkv.weight");
        s[gpu::slot::kWkvS] = sc_addr(L, "attn.wkv.weight");
        s[gpu::slot::kWkvX] = U.addr;
        s[gpu::slot::kWkvNormW] = addr(L, "attn.kv_norm.weight");
        s[gpu::slot::kWkvRope] = Rope.addr; s[gpu::slot::kWkvRaw] = KvRaw.addr;
        s[gpu::slot::kWkvScratch] = KvScr.addr; s[gpu::slot::kWkvVal] = WinV.addr;
        s[gpu::slot::kWkvScale] = WinS.addr; s[gpu::slot::kWkvKv] = Kv.addr;
        std::memcpy(runner.slots(gpu::AttnStage::WkvFinish), s, gpu::kAttnStageStride);
    };
    rows.push_back(run("wkv.gemv", gpu::AttnStage::WkvGemv, fp8_bytes(d.head_dim, d.dim),
                       kv_slots, &kp, sizeof kp, kvg));
    rows.push_back(run("wkv.finish", gpu::AttnStage::WkvFinish, uint64_t(d.head_dim) * 5,
                       kv_slots, &kp, sizeof kp, 1));

    // --- sparse_attn ----------------------------------------------------
    gpu::AttnPush ap{n_kv, d.window, d.head_dim, d.rope_dim, 1024,
                     1.0f / std::sqrt(float(d.head_dim)), d.n_heads};
    const uint32_t at_groups = runner.attn_groups(d.n_heads);
    auto at_slots = [&](gpu::AttnRunner& runner, uint32_t L) {
        uint64_t* s = runner.slots(gpu::AttnStage::AttnScore);
        s[gpu::slot::kAttnQ] = Q.addr; s[gpu::slot::kAttnWinVal] = WinV.addr;
        s[gpu::slot::kAttnWinScale] = WinS.addr; s[gpu::slot::kAttnCmpKv] = Cmp.addr;
        s[gpu::slot::kAttnTopIdx] = Top.addr;
        s[gpu::slot::kAttnSink] = addr(L, "attn.attn_sink");
        s[gpu::slot::kAttnRope] = Rope.addr; s[gpu::slot::kAttnScore] = Score.addr;
        s[gpu::slot::kAttnO] = O.addr;
        std::memcpy(runner.slots(gpu::AttnStage::AttnCombine), s, gpu::kAttnStageStride);
    };
    // The unique KV, read once: the window ring's fp8 bytes plus its scales,
    // plus the compressed rows in bf16. What the two stages actually pull
    // through L2 is 64x this, once per head; see docs/p2_attention.md.
    const uint64_t kv_unique = uint64_t(d.window) * (d.head_dim + d.head_dim / 32)
                             + uint64_t(n_cmp) * d.head_dim * 2;
    rows.push_back(run("sparse_attn.score", gpu::AttnStage::AttnScore, kv_unique,
                       at_slots, &ap, sizeof ap, at_groups));
    rows.push_back(run("sparse_attn.combine", gpu::AttnStage::AttnCombine, kv_unique,
                       at_slots, &ap, sizeof ap, at_groups));

    // --- wo_a / wo_b ----------------------------------------------------
    gpu::WoaPush wa{d.orows(), d.ocols(), d.ocols() / 32, d.o_lora};
    rows.push_back(run("wo_a", gpu::AttnStage::WoA, fp8_bytes(d.orows(), d.ocols()),
                       [&](gpu::AttnRunner& runner, uint32_t L) {
        uint64_t* s = runner.slots(gpu::AttnStage::WoA);
        s[gpu::slot::kWoaW] = addr(L, "attn.wo_a.weight");
        s[gpu::slot::kWoaS] = sc_addr(L, "attn.wo_a.weight");
        s[gpu::slot::kWoaO] = O.addr; s[gpu::slot::kWoaY] = Woa.addr;
    }, &wa, sizeof wa, runner.gemv_groups(gpu::AttnStage::WoA, d.orows())));

    gpu::GemvPush wb{d.dim, d.orows(), d.orows() / 32, 0};
    rows.push_back(run("wo_b", gpu::AttnStage::WoB, fp8_bytes(d.dim, d.orows()),
                       [&](gpu::AttnRunner& runner, uint32_t L) {
        uint64_t* s = runner.slots(gpu::AttnStage::WoB);
        s[gpu::slot::kGemvW] = addr(L, "attn.wo_b.weight");
        s[gpu::slot::kGemvS] = sc_addr(L, "attn.wo_b.weight");
        s[gpu::slot::kGemvX] = Woa.addr; s[gpu::slot::kGemvY] = Wob.addr;
    }, &wb, sizeof wb, runner.gemv_groups(gpu::AttnStage::WoB, d.dim)));

    // --- gate -----------------------------------------------------------
    gpu::GatePush gp{d.n_experts, d.dim, 6, 16, 1.0f, 1.5f};
    auto gate_slots = [&](gpu::AttnRunner& runner, uint32_t L) {
        uint64_t* s = runner.slots(gpu::AttnStage::GateScore);
        s[gpu::slot::kGateW] = addr(L, "ffn.gate.weight");
        s[gpu::slot::kGateBias] = addr(L, "ffn.gate.bias");
        s[gpu::slot::kGateX] = U.addr; s[gpu::slot::kGateScores] = Gs.addr;
        s[gpu::slot::kGateIds] = Gid.addr; s[gpu::slot::kGateWeights] = Gw.addr;
        s[gpu::slot::kGateLayerDone] = Done.addr;
        std::memcpy(runner.slots(gpu::AttnStage::GateTopK), s, gpu::kAttnStageStride);
    };
    rows.push_back(run("gate.score", gpu::AttnStage::GateScore, bf16_bytes(d.n_experts, d.dim),
                       gate_slots, &gp, sizeof gp, runner.gemv_groups(gpu::AttnStage::GateScore, d.n_experts)));
    rows.push_back(run("gate.topk", gpu::AttnStage::GateTopK, uint64_t(d.n_experts) * 8,
                       gate_slots, &gp, sizeof gp, 1));

    // --- head -----------------------------------------------------------
    const store::PinnedTensor* head = pinned.find("head.weight");
    if (head && head->data) {
        gpu::HeadPush hp{d.vocab, d.dim, 0};
        rows.push_back(run("head", gpu::AttnStage::Head, bf16_bytes(d.vocab, d.dim),
                           [&](gpu::AttnRunner& runner, uint32_t) {
            uint64_t* s = runner.slots(gpu::AttnStage::Head);
            s[gpu::slot::kHeadW] = head->data;
            s[gpu::slot::kHeadX] = U.addr; s[gpu::slot::kHeadLogits] = Logits.addr;
        }, &hp, sizeof hp, runner.gemv_groups(gpu::AttnStage::Head, d.vocab)));
    }

    // --- compressor / indexer (design §7.4) ------------------------------
    // Only four layers have a compressor and eight an indexer, so these rows
    // are cycled over the pinned SOURCE layers, not over 0..layers-1.
    auto src = [&](uint32_t i) { return kSrcLayers[i % kNumSrc]; };

    gpu::CmpPush cp{d.head_dim, d.dim, 2, 0, 1, d.rope_dim, 32, 1e-20f};
    auto cmp_slots = [&](gpu::AttnRunner& r, uint32_t i) {
        const uint32_t L = src(i);
        uint64_t* s = r.slots(gpu::AttnStage::CmpKvGemv);
        s[gpu::slot::kCmpW] = addr(L, "attn.compressor.wkv.weight");
        s[gpu::slot::kCmpX] = U.addr;  s[gpu::slot::kCmpY] = CmpY.addr;
        s[gpu::slot::kCmpG] = CmpG.addr;
        s[gpu::slot::kCmpKvState] = CmpKvSt.addr;
        s[gpu::slot::kCmpScoreState] = CmpScSt.addr;
        s[gpu::slot::kCmpNormW] = addr(L, "attn.compressor.norm.weight");
        s[gpu::slot::kCmpLatent] = CmpLat.addr;
        s[gpu::slot::kCmpRope] = Rope.addr;
        s[gpu::slot::kCmpVal] = Cmp.addr;
        s[gpu::slot::kCmpFp4] = CmpFp4.addr;
        s[gpu::slot::kCmpScaleB] = CmpScB.addr;
        s[gpu::slot::kCmpLatentQ] = CmpLatQ.addr;
        std::memcpy(r.slots(gpu::AttnStage::CmpNorm), s, gpu::kAttnStageStride);
        std::memcpy(r.slots(gpu::AttnStage::CmpStore), s, gpu::kAttnStageStride);
        uint64_t* g = r.slots(gpu::AttnStage::CmpGateGemv);
        std::memcpy(g, s, gpu::kAttnStageStride);
        // layer 20 is ratio 1 and has no wgate; fall back to wkv so the
        // dispatch still reads 5.2 MB of real weights.
        const uint64_t wg = addr(L, "attn.compressor.wgate.weight");
        g[gpu::slot::kCmpW] = wg ? wg : addr(L, "attn.compressor.wkv.weight");
        g[gpu::slot::kCmpY] = CmpG.addr;
    };
    const uint32_t cmp_groups = runner.gemv_groups(gpu::AttnStage::CmpKvGemv, d.head_dim);
    rows.push_back(run("compressor.wkv", gpu::AttnStage::CmpKvGemv,
                       bf16_bytes(d.head_dim, d.dim), cmp_slots, &cp, sizeof cp, cmp_groups));
    rows.push_back(run("compressor.wgate", gpu::AttnStage::CmpGateGemv,
                       bf16_bytes(d.head_dim, d.dim), cmp_slots, &cp, sizeof cp, cmp_groups));
    rows.push_back(run("compressor.norm", gpu::AttnStage::CmpNorm,
                       uint64_t(d.head_dim) * 4 * 4, cmp_slots, &cp, sizeof cp, 1));
    rows.push_back(run("compressor.store", gpu::AttnStage::CmpStore,
                       uint64_t(d.head_dim) * 6, cmp_slots, &cp, sizeof cp, 1));

    const float idx_wscale = 1.0f / std::sqrt(float(idx_dim)) / std::sqrt(float(idx_heads));
    auto idx_slots = [&](gpu::AttnRunner& r, uint32_t i) {
        const uint32_t L = src(i);
        uint64_t* s = r.slots(gpu::AttnStage::IdxQGemv);
        s[gpu::slot::kIdxW] = addr(L, "attn.indexer.wq_b.weight");
        s[gpu::slot::kIdxS] = sc_addr(L, "attn.indexer.wq_b.weight");
        s[gpu::slot::kIdxQr] = Qr.addr;
        s[gpu::slot::kIdxQNormW] = addr(L, "attn.q_norm.weight");
        s[gpu::slot::kIdxRope] = Rope.addr;
        s[gpu::slot::kIdxQRaw] = IdxQRaw.addr;  s[gpu::slot::kIdxQ] = IdxQv.addr;
        s[gpu::slot::kIdxWk] = addr(L, "attn.indexer.wk.weight");
        s[gpu::slot::kIdxKNormW] = addr(L, "attn.indexer.k_norm.weight");
        s[gpu::slot::kIdxLatent] = CmpLat.addr;
        s[gpu::slot::kIdxKRaw] = IdxKRaw.addr;  s[gpu::slot::kIdxKCache] = IdxKC.addr;
        s[gpu::slot::kIdxKFp4] = IdxKFp4.addr;  s[gpu::slot::kIdxKScale] = IdxKSc.addr;
        s[gpu::slot::kIdxWProjW] = addr(L, "attn.indexer.weights_proj.weight");
        s[gpu::slot::kIdxX] = U.addr;           s[gpu::slot::kIdxWeights] = IdxWt.addr;
        s[gpu::slot::kIdxScore] = IdxSc.addr;   s[gpu::slot::kIdxOut] = Top.addr;
        s[gpu::slot::kIdxQFp4] = IdxQFp4.addr;  s[gpu::slot::kIdxQScale] = IdxQSc.addr;
        for (gpu::AttnStage st : {gpu::AttnStage::IdxQFinish, gpu::AttnStage::IdxKey,
                                  gpu::AttnStage::IdxWeights, gpu::AttnStage::IdxScore,
                                  gpu::AttnStage::IdxTopK})
            std::memcpy(r.slots(st), s, gpu::kAttnStageStride);
    };
    const uint32_t idx_rows = idx_heads * idx_dim;
    gpu::IdxPush iq{idx_rows, d.q_lora, d.q_lora / 32, idx_heads, idx_dim, d.rope_dim,
                    n_cmp, n_cmp, d.window, 32, 1e-20f, idx_wscale};
    rows.push_back(run("indexer.wq_b", gpu::AttnStage::IdxQGemv,
                       fp8_bytes(idx_rows, d.q_lora), idx_slots, &iq, sizeof iq,
                       runner.gemv_groups(gpu::AttnStage::IdxQGemv, idx_rows)));
    rows.push_back(run("indexer.q_finish", gpu::AttnStage::IdxQFinish,
                       uint64_t(idx_rows) * 6, idx_slots, &iq, sizeof iq, 1));
    gpu::IdxPush ik = iq; ik.k = d.head_dim;
    rows.push_back(run("indexer.key", gpu::AttnStage::IdxKey,
                       bf16_bytes(idx_dim, d.head_dim), idx_slots, &ik, sizeof ik, 1));
    gpu::IdxPush iw = iq; iw.rows = idx_heads; iw.k = d.dim;
    rows.push_back(run("indexer.weights", gpu::AttnStage::IdxWeights,
                       bf16_bytes(idx_heads, d.dim), idx_slots, &iw, sizeof iw, 1));
    rows.push_back(run("indexer.score", gpu::AttnStage::IdxScore,
                       uint64_t(n_cmp) * idx_dim * 2, idx_slots, &iq, sizeof iq,
                       (n_cmp + gpu::kIdxScoreTile - 1) / gpu::kIdxScoreTile));
    rows.push_back(run("indexer.topk", gpu::AttnStage::IdxTopK,
                       uint64_t(n_cmp) * 4, idx_slots, &iq, sizeof iq, 1));

    // --- report ---------------------------------------------------------
    std::printf("\ndesign 7.14 decode path, lanes=%u subgroup=%u rows=%u heads/wg=%u, "
                "%u layers cycled, %u iterations/submit\n", o.lanes, o.subgroup, o.rows,
                o.heads, o.layers, o.iters);
    if (!o.note.empty()) std::printf("note: %s\n", o.note.c_str());
    std::printf("%-22s %10s %12s %10s\n", "kernel", "us", "bytes", "GB/s");
    // The 7.14 dispatches 1-9 run on every layer; the 7.4 compressor runs on
    // four layers of forty and the indexer on eight, so they are totalled
    // separately and the per-token line weights them by that.
    double layer_us = 0, layer_bytes = 0, head_us = 0, cmp_us = 0, idx_us = 0;
    for (const Row& r : rows) {
        std::printf("%-22s %10.2f %12llu %10.1f\n", r.name, r.ms * 1e3,
                    static_cast<unsigned long long>(r.bytes), r.gbps());
        if (std::strcmp(r.name, "head") == 0)                  head_us += r.ms * 1e3;
        else if (std::strncmp(r.name, "compressor.", 11) == 0) cmp_us  += r.ms * 1e3;
        else if (std::strncmp(r.name, "indexer.", 8) == 0)     idx_us  += r.ms * 1e3;
        else { layer_us += r.ms * 1e3; layer_bytes += double(r.bytes); }
    }
    std::printf("%-22s %10.2f %12.0f %10.1f\n", "  (one layer, 1-9)", layer_us, layer_bytes,
                layer_us > 0 ? layer_bytes / (layer_us * 1e-6) / 1e9 : 0.0);
    std::printf("%-22s %10.2f   (4 of 40 layers)\n", "  (compressor, 7.4)", cmp_us);
    std::printf("%-22s %10.2f   (8 of 40 layers)\n", "  (indexer, 7.4)", idx_us);
    std::printf("  40 layers + head: %.2f ms/token, + %.3f ms for the 4 compressor and "
                "8 indexer layers\n", (layer_us * 40 + head_us) / 1e3,
                (cmp_us * 4 + idx_us * 8) / 1e3);

    if (!o.csv.empty()) {
        std::FILE* f = std::fopen(o.csv.c_str(), "w");
        if (f) {
            std::fprintf(f, "kernel,lanes,subgroup,rows,heads_per_wg,layers,iters,us,bytes,gbps,note\n");
            for (const Row& r : rows)
                std::fprintf(f, "%s,%u,%u,%u,%u,%u,%u,%.4f,%llu,%.2f,\"%s\"\n", r.name, o.lanes,
                             o.subgroup, o.rows, o.heads, o.layers, o.iters, r.ms * 1e3,
                             static_cast<unsigned long long>(r.bytes), r.gbps(), o.note.c_str());
            std::fclose(f);
            std::printf("-> %s\n", o.csv.c_str());
        }
    }

    scratch.destroy();
    runner.destroy();
    queries.destroy();
    pool.destroy();
    pinned.reset();
    io.stop();
    return 0;
}
