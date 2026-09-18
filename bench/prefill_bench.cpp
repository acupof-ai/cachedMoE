// prefill_bench -- the prefill GEMM choice of docs/p3_prefill.md §5, and the
// wall time of a whole prefill (§10), on the real checkpoint.
//
// Section `gemm` (default)
// ------------------------
// Option (b), gpu/shaders/prefill_gemm.slang: the GEMV of design §7.1 rule 7
// with TileM token columns per lane, weight read n / TileM times.
// Option (a), gpu/shaders/prefill_coopmat.slang: `VK_KHR_cooperative_matrix`
// 16x16 tiles, 32 tokens a wave, weights decoded to fp16 first.
//
// Both on the same real tensors -- layer 0's first routed expert (FP4 w1/w3 =
// dispatch A's pair, and w2) and layer 0's `wq_b` (FP8 [32768 x 1280]) -- for
// n in {16, 64, 256, 1024} rows. Reported per point:
//
//   ms        GPU time from timestamp queries around the dispatch(es), best of
//             --reps (the machine is shared; see --load)
//   rows/s    n / ms
//   eff GB/s  weight + scale bytes x n / time: the bandwidth a token-at-a-time
//             GEMV would need to match it, i.e. design §7.1 rule 2's metric
//             scaled to n tokens
//   relL2     against cpu/gemv_avx512.h's scalar reference on the same inputs
//
// Section `prefill`
// -----------------
// A whole prefill through gpu/vulkan/prefill_kernels.h: wall time and the
// per-stage breakdown (PrefillTimes), experts and bytes streamed, first token.
// N equal to the L3 prompt length uses that prompt; larger N take the first N
// ids of --ids (a whitespace-separated token-id file). --replay 0 = oracle
// mode (every layer-20+ row replayed). --coop-min sweeps
// PrefillConfig::coopmat_min_rows (-1 = tiled MoE only, 0 = coopmat only),
// --coop-dense PrefillConfig::coopmat_dense_min_rows the same way.
//
// Usage:
//   prefill_bench --model-dir D:\models\DeepSeek-V4.1-Flash --csv bench/results/prefill_p3.csv
//                 [--reps 5] [--load "shared: tracks I/J/K on the GPU"]
//   prefill_bench --section prefill --n 64,512,4096 --ids ids.txt [--replay 128,0]
//                 [--coop-min -1,16,0] [--coop-dense -1,64] [--transit 32] [--handoff-dir dir]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <future>
#include <random>
#include <string>
#include <vector>

#include "core/align.h"
#include "core/json.h"
#include "core/config.h"
#include "core/log.h"
#include "core/status.h"
#include "cpu/dequant.h"
#include "cpu/gemv_avx512.h"
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/moe_kernels.h"   // default_shader_dir
#include "gpu/vulkan/prefill_kernels.h"
#include "model/layout.h"
#include "model/manifest.h"
#include "model/v41_config.h"
#include "runtime/engram.h"
#include "storage/backend.h"
#include "storage/io_engine.h"
#include "store/pinned.h"
#include "store/shard_set.h"

using namespace deepmoe;

namespace {

struct Options {
    std::string model_dir;
    std::string csv;
    std::string section = "gemm";
    std::string load = "unlabelled";
    uint32_t    reps = 5;
    std::vector<uint32_t> ns = {16, 64, 256, 1024};
    std::vector<uint32_t> tiles = {4, 8, 16};
    bool        coopmat = true;
    std::string ids;                       // section prefill: token ids for N > the L3 prompt
    std::string l3 = "tests/data/l3";      // engram tables and the 64-token prompt
    std::vector<uint32_t> replays = {128};  // 0 = oracle mode
    uint32_t    transit = 32;
    std::vector<uint32_t> coop_min = {16};    // section prefill: PrefillConfig::coopmat_min_rows (-1 = never)
    std::vector<uint32_t> coop_dense = {64};  // PrefillConfig::coopmat_dense_min_rows (-1 = never)
    std::string handoff_dir;
    std::string only;                      // section coopgeo: shape names to run
};

const char* env(const char* name) {
#if defined(_MSC_VER)
    static std::string v; char* b = nullptr; size_t n = 0;
    if (_dupenv_s(&b, &n, name) == 0 && b) { v = b; free(b); return v.c_str(); }
    return nullptr;
#else
    return std::getenv(name);
#endif
}

struct Csv {
    std::FILE* f = nullptr;
    void open(const std::string& path) {
        if (path.empty()) return;
        const bool fresh = !std::fopen(path.c_str(), "rb");
        f = std::fopen(path.c_str(), "ab");
        if (f && fresh)
            std::fputs("section,kernel,variant,n,rows,k,weight_bytes,ms,rows_per_s,eff_gbps,"
                       "rel_l2,cos,load\n", f);
    }
    ~Csv() { if (f) std::fclose(f); }
};

double ms_of(gpu::QueryPool& q, const gpu::Device& d) {
    auto v = q.read_range(0, 2);
    if (!v) return -1.0;
    return double((*v)[1] - (*v)[0]) * d.caps().timestamp_period_ns / 1e6;
}

struct Rig {
    gpu::Device          device;
    gpu::MemoryAllocator alloc;
    Manifest             manifest;
    store::ShardSet      shards;
    storage::IoEngine    io;
    store::PinnedStore   pinned;
    gpu::PrefillRunner   runner;
    gpu::QueryPool       q;
    gpu::CommandBuffer   cmd{};
    bool                 io_started = false;

    ~Rig() {
        runner.destroy();
        q.destroy();
        pinned.reset();
        if (io_started) io.stop();
    }
    Result<void> up(const std::string& dir) {
        if (auto r = device.create(); !r) return r;
        if (auto r = alloc.init(device, MemoryPath::DeviceLocalHostVisible); !r) return r;
        auto mf = Manifest::load(store::ShardSet::join(dir, layout::kManifestFile));
        if (!mf) return std::unexpected(mf.error());
        manifest = std::move(*mf);
        if (auto r = shards.open_all(dir, manifest, true); !r) return r;
        IoConfig cfg;
        auto be = storage::make_default_backend(cfg);
        if (!be) return std::unexpected(be.error());
        if (auto r = io.start(std::move(*be), cfg); !r) return r;
        io_started = true;
        auto backing = alloc.make_slab_backing();
        if (!backing) return std::unexpected(backing.error());
        store::PinnedConfig pc;
        pc.region_bytes = 512ull << 20;
        if (auto r = pinned.init(std::move(*backing), pc); !r) return r;
        if (auto r = runner.create(device, alloc, gpu::default_shader_dir()); !r) return r;
        if (auto r = q.create(device, 2); !r) return r;
        auto cb = runner.pool().acquire();
        if (!cb) return std::unexpected(cb.error());
        cmd = *cb;
        return {};
    }
    // Dispatches `fn` between two timestamps and waits. Returns GPU ms.
    template <class F> Result<double> timed(F&& fn) {
        if (auto r = cmd.begin(); !r) return std::unexpected(r.error());
        if (auto r = cmd.reset_queries(q, 0, 2); !r) return std::unexpected(r.error());
        if (auto r = cmd.write_timestamp(q, 0, false); !r) return std::unexpected(r.error());
        if (auto r = fn(cmd); !r) return std::unexpected(r.error());
        if (auto r = cmd.write_timestamp(q, 1, true); !r) return std::unexpected(r.error());
        if (auto r = cmd.end(); !r) return std::unexpected(r.error());
        if (auto r = gpu::submit_and_wait(device, cmd); !r) return std::unexpected(r.error());
        return ms_of(q, device);
    }
};

gpu::GpuBuffer must_alloc(gpu::MemoryAllocator& a, uint64_t bytes) {
    auto b = a.allocate(align_up(bytes, kPageSize), true, true);
    if (!b) { std::fprintf(stderr, "alloc %llu: %s\n", (unsigned long long)bytes,
                           b.error().str().c_str()); std::exit(1); }
    std::memset(b->host_ptr, 0, static_cast<size_t>(b->bytes));
    return *b;
}

// Activations that look like an RMSNorm output: unit RMS, a few outliers.
std::vector<float> make_x(uint32_t n, uint32_t k, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> x(size_t(n) * k);
    for (float& v : x) v = cpu::bf16_to_float(cpu::float_to_bf16(d(g)));
    for (uint32_t m = 0; m < n; ++m) x[size_t(m) * k + (m * 131) % k] *= 20.0f;
    return x;
}

double rel_l2(const float* a, const float* b, size_t n, double* cos_out) {
    double sd = 0, nb = 0, na = 0, dot = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = double(a[i]) - double(b[i]);
        sd += d * d; nb += double(b[i]) * b[i]; na += double(a[i]) * a[i];
        dot += double(a[i]) * b[i];
    }
    if (cos_out) *cos_out = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 1.0;
    return nb > 0 ? std::sqrt(sd / nb) : std::sqrt(sd);
}

int run_gemm(const Options& o) {
    Rig rig;
    if (auto r = rig.up(o.model_dir); !r) {
        std::fprintf(stderr, "bring-up: %s\n", r.error().str().c_str());
        return 1;
    }
    Csv csv;
    csv.open(o.csv);
    std::printf("prefill_bench gemm: %s, load = %s\n", rig.device.caps().device_name.c_str(),
                o.load.c_str());

    // --- the tensors ---------------------------------------------------------
    const uint32_t dim = 5120, inter = 2304;
    auto e0 = gpu::pf_load_expert(rig.alloc, rig.manifest, rig.shards, rig.io, {0, 0});
    if (!e0) { std::fprintf(stderr, "expert: %s\n", e0.error().str().c_str()); return 1; }
    if (auto r = rig.pinned.load(rig.manifest, rig.shards, rig.io,
                                 {"layers.0.attn.wq_b.weight"}); !r) {
        std::fprintf(stderr, "wq_b: %s\n", r.error().str().c_str());
        return 1;
    }
    const store::PinnedTensor* wqb = rig.pinned.find("layers.0.attn.wq_b.weight");
    const uint32_t qrows = 32768, qk = 1280;
    const uint64_t fp4_bytes = uint64_t(inter) * (dim / 2) + uint64_t(inter) * (dim / 32);
    const uint64_t wqb_bytes = wqb->data_bytes + wqb->scale_bytes;

    const uint32_t nmax = *std::max_element(o.ns.begin(), o.ns.end());
    const uint32_t npad = (nmax + 31) / 32 * 32;
    // x planes (fp16 values + fp32 block scales) for dim and q_lora widths
    gpu::GpuBuffer xq   = must_alloc(rig.alloc, uint64_t(npad) * dim * 2);
    gpu::GpuBuffer xs   = must_alloc(rig.alloc, uint64_t(npad) * (dim / 32) * 4);
    gpu::GpuBuffer qq   = must_alloc(rig.alloc, uint64_t(npad) * qk * 2);
    gpu::GpuBuffer qs   = must_alloc(rig.alloc, uint64_t(npad) * (qk / 32) * 4);
    gpu::GpuBuffer idx  = must_alloc(rig.alloc, uint64_t(npad) * 4);
    gpu::GpuBuffer rw   = must_alloc(rig.alloc, uint64_t(npad) * 4);
    gpu::GpuBuffer job  = must_alloc(rig.alloc, 64);
    gpu::GpuBuffer h    = must_alloc(rig.alloc, uint64_t(npad) * inter * 4);
    gpu::GpuBuffer y    = must_alloc(rig.alloc, uint64_t(npad) * qrows * 4);
    gpu::GpuBuffer x16  = must_alloc(rig.alloc, uint64_t(npad) * dim * 2);
    gpu::GpuBuffer x16q = must_alloc(rig.alloc, uint64_t(npad) * qk * 2);
    gpu::GpuBuffer w16  = must_alloc(rig.alloc, uint64_t(qrows) * qk * 2);   // >= 2304 x 5120 x 2

    const std::vector<float> xd = make_x(npad, dim, 1);
    const std::vector<float> xq_host = make_x(npad, qk, 2);
    gpu::pf_act_quant_host(xd.data(), npad, dim, static_cast<uint16_t*>(xq.host_ptr),
                           static_cast<float*>(xs.host_ptr));
    gpu::pf_act_quant_host(xq_host.data(), npad, qk, static_cast<uint16_t*>(qq.host_ptr),
                           static_cast<float*>(qs.host_ptr));
    for (uint32_t i = 0; i < npad; ++i) {
        static_cast<uint32_t*>(idx.host_ptr)[i] = i;
        static_cast<float*>(rw.host_ptr)[i] = 1.0f;
    }
    gpu::PfJob j{};
    j.w1 = e0->addr[0]; j.s1 = e0->addr[1]; j.w2 = e0->addr[2]; j.s2 = e0->addr[3];
    j.w3 = e0->addr[4]; j.s3 = e0->addr[5];
    j.fmt = gpu::kPfFp4;
    std::memcpy(job.host_ptr, &j, sizeof(j));

    // CPU reference for w1 and wq_b on the dequantised x, rows of the first token.
    auto dequant_x = [](const gpu::GpuBuffer& q16, const gpu::GpuBuffer& sc, uint32_t m,
                        uint32_t k) {
        std::vector<float> v(k);
        const auto* q = static_cast<const uint16_t*>(q16.host_ptr) + size_t(m) * k;
        const auto* s = static_cast<const float*>(sc.host_ptr) + size_t(m) * (k / 32);
        for (uint32_t i = 0; i < k; ++i) v[i] = cpu::fp16_to_float(q[i]) * s[i / 32];
        return v;
    };
    std::vector<float> ref_w1(inter), ref_q(qrows);
    {
        const std::vector<float> x0 = dequant_x(xq, xs, 0, dim);
        std::span<const uint8_t> w(static_cast<const uint8_t*>(e0->host(0)), inter * (dim / 2));
        std::span<const uint8_t> s(static_cast<const uint8_t*>(e0->host(1)), inter * (dim / 32));
        (void)cpu::gemv_fp4_ref(w, s, x0, {inter, dim, 1}, ref_w1);
        const std::vector<float> q0 = dequant_x(qq, qs, 0, qk);
        std::span<const uint8_t> ww(static_cast<const uint8_t*>(wqb->data_host), wqb->data_bytes);
        std::span<const uint8_t> ss(static_cast<const uint8_t*>(wqb->scale_host), wqb->scale_bytes);
        (void)cpu::gemv_fp8_ref(ww, ss, q0, {qrows, qk, 1}, ref_q);
    }

    auto emit = [&](const char* kernel, const std::string& variant, uint32_t n, uint32_t rows,
                    uint32_t k, uint64_t wbytes, double ms, double rl2, double cs) {
        const double rps = n / (ms / 1e3);
        const double gbps = double(wbytes) * n / (ms / 1e3) / 1e9;
        std::printf("  %-14s %-18s n=%5u  %9.3f ms  %9.0f rows/s  %8.1f eff GB/s  relL2 %.2e cos %.7f\n",
                    kernel, variant.c_str(), n, ms, rps, gbps, rl2, cs);
        if (csv.f)
            std::fprintf(csv.f, "gemm,%s,%s,%u,%u,%u,%llu,%.4f,%.1f,%.2f,%.3e,%.9f,\"%s\"\n",
                         kernel, variant.c_str(), n, rows, k, (unsigned long long)wbytes, ms,
                         rps, gbps, rl2, cs, o.load.c_str());
    };

    for (uint32_t n : o.ns) {
        // --- (b) tiled GEMV ----------------------------------------------------
        for (uint32_t t : o.tiles) {
            // expert gate/up (dispatch A: w1 and w3 both)
            auto kh = rig.runner.kernel({"prefill_gemm", 1, gpu::kPfFp4, gpu::kPfActQ, t});
            if (!kh) { std::fprintf(stderr, "%s\n", kh.error().str().c_str()); return 1; }
            std::memcpy(&static_cast<gpu::PfJob*>(job.host_ptr)->rows_off, "\0\0\0\0", 4);
            static_cast<gpu::PfJob*>(job.host_ptr)->n = n;
            uint64_t* sl = rig.runner.slots(*kh);
            sl[gpu::kPgX] = xq.dev_addr; sl[gpu::kPgXS] = xs.dev_addr; sl[gpu::kPgY] = h.dev_addr;
            sl[gpu::kPgIdx] = idx.dev_addr; sl[gpu::kPgRW] = rw.dev_addr; sl[gpu::kPgJob] = job.dev_addr;
            gpu::PfGemmPush p;
            p.rows = inter; p.k = dim; p.scale_cols = dim / 32; p.n = n; p.x_stride = dim;
            p.y_stride = inter; p.job = 0; p.flags = 0; p.swiglu_limit = 1e30f;   // plain w1: check below
            double best = 1e30;
            for (uint32_t r = 0; r < o.reps; ++r) {
                auto ms = rig.timed([&](gpu::CommandBuffer& c) {
                    return rig.runner.record(c, *kh, &p, sizeof(p), gpu::PrefillRunner::gemm_gx(inter),
                                             gpu::PrefillRunner::gemm_gy(n, t));
                });
                if (!ms) { std::fprintf(stderr, "%s\n", ms.error().str().c_str()); return 1; }
                best = std::min(best, *ms);
            }
            // h = swiglu(w1 x, w3 x) with an effectively infinite limit; compare
            // the first token's w1 x through the SwiGLU inverse is not possible,
            // so check it separately with the dense stage on w1 alone.
            emit("expert.gateup", std::format("tiled M{}", t), n, inter, dim, 2 * fp4_bytes, best,
                 0.0, 1.0);
        }
        for (uint32_t t : o.tiles) {
            // dense FP4 w1 alone -- the correctness anchor for the expert path
            auto kh = rig.runner.kernel({"prefill_gemm", 0, gpu::kPfFp4, gpu::kPfActQ, t});
            if (!kh) { std::fprintf(stderr, "%s\n", kh.error().str().c_str()); return 1; }
            uint64_t* sl = rig.runner.slots(*kh);
            sl[gpu::kPgW] = e0->addr[0]; sl[gpu::kPgS] = e0->addr[1];
            sl[gpu::kPgX] = xq.dev_addr; sl[gpu::kPgXS] = xs.dev_addr; sl[gpu::kPgY] = h.dev_addr;
            gpu::PfGemmPush p;
            p.rows = inter; p.k = dim; p.scale_cols = dim / 32; p.n = n; p.x_stride = dim;
            p.y_stride = inter;
            double best = 1e30;
            for (uint32_t r = 0; r < o.reps; ++r) {
                auto ms = rig.timed([&](gpu::CommandBuffer& c) {
                    return rig.runner.record(c, *kh, &p, sizeof(p), gpu::PrefillRunner::gemm_gx(inter),
                                             gpu::PrefillRunner::gemm_gy(n, t));
                });
                if (!ms) { std::fprintf(stderr, "%s\n", ms.error().str().c_str()); return 1; }
                best = std::min(best, *ms);
            }
            double cs = 0;
            const double rl = rel_l2(static_cast<const float*>(h.host_ptr), ref_w1.data(), inter, &cs);
            emit("expert.w1", std::format("tiled M{}", t), n, inter, dim, fp4_bytes, best, rl, cs);
        }
        for (uint32_t t : o.tiles) {
            auto kh = rig.runner.kernel({"prefill_gemm", 0, gpu::kPfFp8, gpu::kPfActQ, t});
            if (!kh) { std::fprintf(stderr, "%s\n", kh.error().str().c_str()); return 1; }
            uint64_t* sl = rig.runner.slots(*kh);
            sl[gpu::kPgW] = wqb->data; sl[gpu::kPgS] = wqb->scale;
            sl[gpu::kPgX] = qq.dev_addr; sl[gpu::kPgXS] = qs.dev_addr; sl[gpu::kPgY] = y.dev_addr;
            gpu::PfGemmPush p;
            p.rows = qrows; p.k = qk; p.scale_cols = qk / 32; p.n = n; p.x_stride = qk;
            p.y_stride = qrows;
            double best = 1e30;
            for (uint32_t r = 0; r < o.reps; ++r) {
                auto ms = rig.timed([&](gpu::CommandBuffer& c) {
                    return rig.runner.record(c, *kh, &p, sizeof(p), gpu::PrefillRunner::gemm_gx(qrows),
                                             gpu::PrefillRunner::gemm_gy(n, t));
                });
                if (!ms) { std::fprintf(stderr, "%s\n", ms.error().str().c_str()); return 1; }
                best = std::min(best, *ms);
            }
            double cs = 0;
            const double rl = rel_l2(static_cast<const float*>(y.host_ptr), ref_q.data(), qrows, &cs);
            emit("wq_b", std::format("tiled M{}", t), n, qrows, qk, wqb_bytes, best, rl, cs);
        }

        // --- (a) cooperative matrix ------------------------------------------
        if (!o.coopmat) continue;
        const uint32_t nc = (n + 31) / 32 * 32;
        // decode the weight to fp16 (the transit write), stage x as fp16, then
        // the GEMM -- once with one workgroup per 32-token wave walking every
        // output tile row, once with (wave, output tile row) workgroups.
        auto coop = [&](const char* kernel, uint64_t wa, uint64_t sa, uint32_t wfmt,
                        uint32_t R, uint32_t K, const gpu::GpuBuffer& q16,
                        const gpu::GpuBuffer& qsc, const gpu::GpuBuffer& x16b,
                        const gpu::GpuBuffer& out, const std::vector<float>& ref,
                        uint64_t wbytes) -> int {
            auto kd = rig.runner.kernel({"prefill_gemm", 3, wfmt, 0, 8});
            auto kx = rig.runner.kernel({"prefill_coopmat", 1, 0, 0, 8, R, K});
            auto km = rig.runner.kernel({"prefill_coopmat", 0, 0, 0, 8, R, K});
            if (!kd || !kx || !km) {
                std::fprintf(stderr, "coopmat pipelines: %s\n",
                             (!kd ? kd.error() : !kx ? kx.error() : km.error()).str().c_str());
                return 1;
            }
            uint64_t* sd = rig.runner.slots(*kd);
            sd[gpu::kPgW] = wa; sd[gpu::kPgS] = sa; sd[gpu::kPgY] = w16.dev_addr;
            gpu::PfGemmPush pd;
            pd.rows = R; pd.k = K; pd.scale_cols = K / 32;
            uint64_t* sx = rig.runner.slots(*kx);
            sx[gpu::kPcQ] = q16.dev_addr; sx[gpu::kPcQS] = qsc.dev_addr; sx[gpu::kPcX] = x16b.dev_addr;
            gpu::PfCoopPush px; px.n = nc; px.k = K;
            uint64_t* sm = rig.runner.slots(*km);
            sm[gpu::kPcW] = w16.dev_addr; sm[gpu::kPcX] = x16b.dev_addr; sm[gpu::kPcY] = out.dev_addr;
            for (uint32_t mode = 0; mode < 2; ++mode) {
                gpu::PfCoopPush pm; pm.n = nc; pm.flags = mode ? 64u : 0u;
                double best_dec = 1e30, best_x = 1e30, best_m = 1e30;
                for (uint32_t r = 0; r < o.reps; ++r) {
                    auto a = rig.timed([&](gpu::CommandBuffer& c) {
                        return rig.runner.record(c, *kd, &pd, sizeof(pd),
                                                 gpu::PrefillRunner::per_block_groups(R, K));
                    });
                    auto b = rig.timed([&](gpu::CommandBuffer& c) {
                        return rig.runner.record(c, *kx, &px, sizeof(px),
                                                 static_cast<uint32_t>((uint64_t(nc) * (K / 32) + 31) / 32));
                    });
                    auto m = rig.timed([&](gpu::CommandBuffer& c) {
                        return rig.runner.record(c, *km, &pm, sizeof(pm), nc / 32, mode ? R / 16 : 1);
                    });
                    if (!a || !b || !m) {
                        std::fprintf(stderr, "coopmat run: %s\n",
                                     (!a ? a.error() : !b ? b.error() : m.error()).str().c_str());
                        return 1;
                    }
                    best_dec = std::min(best_dec, *a); best_x = std::min(best_x, *b);
                    best_m = std::min(best_m, *m);
                }
                double cs = 0;
                const double rl = rel_l2(static_cast<const float*>(out.host_ptr), ref.data(), R, &cs);
                const std::string v = mode ? "coopmat rowtile" : "coopmat wave";
                emit(kernel, v + " gemm", n, R, K, wbytes, best_m, rl, cs);
                emit(kernel, v + " +decode+x", n, R, K, wbytes, best_m + best_dec + best_x, rl, cs);
            }
            return 0;
        };
        if (coop("expert.w1", e0->addr[0], e0->addr[1], gpu::kPfFp4, inter, dim, xq, xs, x16, h,
                 ref_w1, fp4_bytes)) return 1;
        if (coop("wq_b", wqb->data, wqb->scale, gpu::kPfFp8, qrows, qk, qq, qs, x16q, y, ref_q,
                 wqb_bytes)) return 1;
        if (csv.f) std::fflush(csv.f);
    }
    for (gpu::GpuBuffer* b : {&xq, &xs, &qq, &qs, &idx, &rw, &job, &h, &y, &x16, &x16q, &w16})
        rig.alloc.free(*b);
    e0->release(rig.alloc);
    return 0;
}


// --- section `coopgeo`: the cooperative-matrix GEMM's dispatch geometry ---------------
// Random fp16 W [R][K] and X [n][K] (the kernel's cost does not depend on the
// values), every variant writing the same fp32 [n][R]; each variant is one
// command buffer holding all of its block dispatches between two timestamps.
// tt = 16-token tiles per workgroup, rt = 16-row output tiles per workgroup,
// tb / rb = tokens / rows per dispatch (0 = all). Outputs are compared with
// the first variant's bit for bit (the tile arithmetic is the same).
int run_coopgeo(const Options& o) {
    Rig rig;
    if (auto r = rig.up(o.model_dir); !r) {
        std::fprintf(stderr, "bring-up: %s\n", r.error().str().c_str());
        return 1;
    }
    Csv csv;
    csv.open(o.csv);
    struct Shape { const char* name; uint32_t R, K; };
    const Shape shapes[] = {{"w1", 2304, 5120}, {"w2", 5120, 2304}, {"wq_b", 32768, 1280},
                            {"wo_b", 5120, 8192}, {"wo_a.g", 1024, 4096}, {"eng.wkv", 25600, 6144}};
    struct Geo { uint32_t tt, rt, tb, rb; };
    std::vector<Geo> geos;
    for (uint32_t tt : {1u, 2u, 4u})
        for (uint32_t rt : {1u, 4u, 16u})
            for (uint32_t tb : {0u, 256u})
                for (uint32_t rb : {0u, 2048u})
                    geos.push_back({tt, rt, tb, rb});
    if (const char* g = env("DEEPMOE_PF_GEO")) {
        // "tt,rt,tb,rb;tt,rt,tb,rb;..."
        geos.clear();
        std::string s = g;
        size_t p = 0;
        while (p < s.size()) {
            size_t q = s.find(';', p);
            if (q == std::string::npos) q = s.size();
            Geo x{};
            if (std::sscanf(s.substr(p, q - p).c_str(), "%u,%u,%u,%u", &x.tt, &x.rt, &x.tb, &x.rb) == 4)
                geos.push_back(x);
            p = q + 1;
        }
    }
    std::mt19937 gen(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (const Shape& sh : shapes) {
        if (!o.only.empty() && o.only.find(sh.name) == std::string::npos) continue;
        const uint32_t R = sh.R, K = sh.K;
        const uint32_t nmax = *std::max_element(o.ns.begin(), o.ns.end());
        const uint32_t npad = (nmax + 127) / 128 * 128;
        gpu::GpuBuffer W = must_alloc(rig.alloc, uint64_t(R) * K * 2);
        gpu::GpuBuffer X = must_alloc(rig.alloc, uint64_t(npad) * K * 2);
        gpu::GpuBuffer Y = must_alloc(rig.alloc, uint64_t(npad) * R * 4);
        {
            std::vector<uint16_t> w(size_t(R) * K), x(size_t(npad) * K);
            for (auto& v : w) v = cpu::float_to_fp16(dist(gen) * 0.02f);
            for (auto& v : x) v = cpu::float_to_fp16(dist(gen));
            std::memcpy(W.host_ptr, w.data(), w.size() * 2);
            std::memcpy(X.host_ptr, x.data(), x.size() * 2);
        }
        for (uint32_t n : o.ns) {
            // Round robin: every rep visits every variant, starting one further
            // along each time, so a burst of another process's GPU work lands
            // on different variants in different reps; the best rep is kept.
            struct Var { Geo g; uint32_t kh, tok, ntok, tb, rtile, rb, dispatches = 0; double best = 1e30; size_t diff = 0; };
            std::vector<Var> vars;
            for (const Geo& g : geos) {
                Var v;
                v.g = g;
                v.tok = 16 * g.tt;
                v.ntok = (n + v.tok - 1) / v.tok * v.tok;
                if (v.ntok > npad) continue;
                v.tb = g.tb ? std::max(v.tok, g.tb / v.tok * v.tok) : v.ntok;
                v.rtile = 16 * g.rt;
                v.rb = g.rb ? std::max(v.rtile, g.rb / v.rtile * v.rtile) : (R + v.rtile - 1) / v.rtile * v.rtile;
                auto k = rig.runner.kernel({"prefill_coopmat", 0, 0, 0, 8, R, K, g.tt, g.rt});
                if (!k) { std::fprintf(stderr, "%s\n", k.error().str().c_str()); return 1; }
                v.kh = *k;
                vars.push_back(v);
            }
            std::vector<float> ref;
            for (uint32_t rep = 0; rep < o.reps; ++rep)
                for (size_t vi = 0; vi < vars.size(); ++vi) {
                    Var& v = vars[(vi + rep) % vars.size()];
                    uint64_t* sl = rig.runner.slots(v.kh);
                    sl[gpu::kPcW] = W.dev_addr; sl[gpu::kPcX] = X.dev_addr; sl[gpu::kPcY] = Y.dev_addr;
                    v.dispatches = 0;
                    auto ms = rig.timed([&](gpu::CommandBuffer& c) -> Result<void> {
                        for (uint32_t x0 = 0; x0 < v.ntok; x0 += v.tb)
                            for (uint32_t r0 = 0; r0 < R; r0 += v.rb) {
                                gpu::PfCoopPush p;
                                p.n = v.ntok; p.flags = 64; p.x_off = x0; p.row0 = r0;
                                const uint32_t gx = (std::min(v.tb, v.ntok - x0) + v.tok - 1) / v.tok;
                                const uint32_t gy = (std::min(v.rb, R - r0) + v.rtile - 1) / v.rtile;
                                if (auto r = rig.runner.record(c, v.kh, &p, sizeof(p), gx, gy); !r) return r;
                                ++v.dispatches;
                            }
                        return {};
                    });
                    if (!ms) { std::fprintf(stderr, "%s\n", ms.error().str().c_str()); return 1; }
                    v.best = std::min(v.best, *ms);
                    if (rep == 0) {
                        const float* y = static_cast<const float*>(Y.host_ptr);
                        if (ref.empty()) ref.assign(y, y + size_t(n) * R);
                        else for (size_t i = 0; i < ref.size(); ++i) v.diff += (std::memcmp(&ref[i], &y[i], 4) != 0);
                    }
                }
            for (const Var& v : vars) {
                const double tps = double(n) * R * K / (v.best / 1e3) / 1e12;
                const std::string name = std::format("tt{} rt{} tb{} rb{}", v.g.tt, v.g.rt, v.g.tb, v.g.rb);
                std::printf("  %-8s n=%5u %-22s %9.3f ms  %6.2f T flop/s  %5u dispatches  %s\n", sh.name, n,
                            name.c_str(), v.best, tps, v.dispatches, v.diff ? std::format("DIFF {}", v.diff).c_str() : "same");
                if (csv.f)
                    std::fprintf(csv.f, "coopgeo,%s,%s,%u,%u,%u,%llu,%.4f,%.1f,%.3f,%zu,0,\"%s\"\n", sh.name, name.c_str(),
                                 n, R, K, (unsigned long long)(uint64_t(R) * K * 2), v.best, n / (v.best / 1e3), tps,
                                 v.diff, o.load.c_str());
            }
            std::fflush(stdout);
            if (csv.f) std::fflush(csv.f);
        }
        for (gpu::GpuBuffer* b : {&W, &X, &Y}) rig.alloc.free(*b);
    }
    return 0;
}

// --- section `prefill`: a whole prefill, per stage -----------------------------------

std::vector<uint32_t> read_ids(const std::string& path) {
    std::vector<uint32_t> ids;
    if (path.empty()) return ids;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return ids;
    unsigned long v;
    while (std::fscanf(f, "%lu", &v) == 1) ids.push_back(static_cast<uint32_t>(v));
    std::fclose(f);
    return ids;
}

std::vector<uint32_t> l3_prompt(const std::string& dir) {
    std::vector<uint32_t> out;
    auto doc = json_parse_file(dir + "/index.json");
    if (!doc) return out;
    if (const JsonValue* a = doc->find("prompt_ids"))
        if (auto arr = a->as_array())
            for (const JsonValue& e : **arr) out.push_back(static_cast<uint32_t>(e.as_int().value_or(0)));
    return out;
}

int run_prefill(const Options& o) {
    Rig rig;
    if (auto r = rig.up(o.model_dir); !r) {
        std::fprintf(stderr, "bring-up: %s\n", r.error().str().c_str());
        return 1;
    }
    auto cfgj = V41Config::load(store::ShardSet::join(o.model_dir, "config.json"));
    if (!cfgj) { std::fprintf(stderr, "config: %s\n", cfgj.error().str().c_str()); return 1; }
    const TextConfig& c = cfgj->text;
    auto tables = runtime::EngramTables::load(o.l3);
    if (!tables) { std::fprintf(stderr, "engram tables (%s): %s\n", o.l3.c_str(), tables.error().str().c_str()); return 1; }
    {
        std::vector<std::string> names = store::pinned_global_tensors(rig.manifest);
        for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
            auto n = store::pinned_layer_tensors(rig.manifest, L);
            names.insert(names.end(), n.begin(), n.end());
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (auto r = rig.pinned.load(rig.manifest, rig.shards, rig.io, names); !r) {
            std::fprintf(stderr, "pinned: %s\n", r.error().str().c_str());
            return 1;
        }
        std::printf("pinned set %.2f GiB in %.1f s\n", rig.pinned.bytes_loaded() / double(1ull << 30),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    const std::vector<uint32_t> text_ids = read_ids(o.ids);
    const std::vector<uint32_t> l3_ids = l3_prompt(o.l3);
    Csv csv;
    csv.open(o.csv);

    for (uint32_t n : o.ns) {
        std::vector<uint32_t> prompt;
        std::string source;
        if (n == l3_ids.size()) { prompt = l3_ids; source = "L2/L3 prompt"; }
        else if (text_ids.size() >= n) { prompt.assign(text_ids.begin(), text_ids.begin() + n); source = o.ids; }
        else { std::fprintf(stderr, "N=%u: no %u token ids (--ids)\n", n, n); return 1; }

        for (uint32_t replay : o.replays)
        for (uint32_t cmin : o.coop_min)
        for (uint32_t cden : o.coop_dense) {
            gpu::PrefillConfig pc;
            pc.max_tokens = n;
            pc.replay = replay ? replay : n;
            pc.tile = o.tiles.empty() ? 8 : o.tiles[0];
            pc.transit_slots = o.transit;
            pc.coopmat_min_rows = cmin;
            pc.coopmat_dense_min_rows = cden;
            if (const char* e = env("DEEPMOE_PF_ATTN")) pc.attn_coop = std::string(e) != "legacy";
            if (const char* e = env("DEEPMOE_PF_ATTN_HT")) pc.attn_head_tiles = static_cast<uint32_t>(std::atoi(e));
            gpu::Prefill pf;
            if (auto r = pf.create(rig.device, rig.alloc, rig.runner, rig.manifest, rig.shards, rig.io,
                                   rig.pinned, c, &*tables, pc); !r) {
                std::fprintf(stderr, "prefill create: %s\n", r.error().str().c_str());
                return 1;
            }
            const std::string moe = cmin == UINT32_MAX ? "MoE tiled"
                                    : cmin == 0 ? "MoE coopmat" : std::format("MoE coopmat n>={}", cmin);
            const std::string dense = cden == UINT32_MAX ? "dense tiled" : std::format("dense coopmat n>={}", cden);
            const std::string mode = (pc.replay >= n ? std::string("oracle mode") : std::format("replay {}", pc.replay)) +
                                     " " + moe + " " + dense;
            std::printf("\nN=%u (%s), %s, load = %s\n", n, source.c_str(), mode.c_str(), o.load.c_str());
            auto out = pf.run(prompt);
            if (!out) { std::fprintf(stderr, "prefill: %s\n", out.error().str().c_str()); return 1; }
            const gpu::PrefillTimes& t = pf.times();
            bool finite = true;
            for (float v : out->logits) finite = finite && std::isfinite(v);
            std::printf("  total %.2f s = %.0f tok/s | embed %.0f  engram io %.0f gpu %.0f  mHC %.0f  attention %.0f  "
                        "gate+route %.0f  shared %.0f  expert io %.0f  expert gpu %.0f  head %.0f  host/other %.0f ms\n",
                        t.total / 1e3, n / (t.total / 1e3), t.embed, t.engram_io, t.engram, t.mhc, t.attention,
                        t.gate, t.shared_expert, t.expert_io, t.expert_gpu, t.head, t.host);
            std::printf("  %u experts streamed = %.2f GB in %.1f s of waiting (%.2f GB/s against that wait), "
                        "%llu engram reads, %u dispatches in %u submits\n",
                        t.experts_read, t.expert_bytes / 1e9, t.expert_io / 1e3,
                        t.expert_io > 0 ? t.expert_bytes / 1e9 / (t.expert_io / 1e3) : 0.0,
                        (unsigned long long)t.engram_reads, t.dispatches, t.submits);
            std::printf("  first token %u, margin %.3f, logits finite: %s\n", out->first_token,
                        out->top1 - out->top2, finite ? "yes" : "NO");
            {
                // single-dispatch ops by wall time (the batched MoE submits are not in here)
                std::vector<std::pair<std::string, std::pair<double, uint32_t>>> ops(t.per_op.begin(),
                                                                                   t.per_op.end());
                std::sort(ops.begin(), ops.end(),
                          [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
                std::printf("  ops by wall time:");
                for (size_t i = 0; i < ops.size() && i < 14; ++i)
                    std::printf("%s %s %.0f ms/%u", i % 3 ? "," : "\n   ", ops[i].first.c_str(),
                                ops[i].second.first, ops[i].second.second);
                std::printf("\n");
            }
            if (csv.f) {
                const std::string v = std::format("N={} {}", n, mode);
                const std::pair<const char*, double> buckets[] = {
                    {"total", t.total}, {"embed", t.embed}, {"engram_io", t.engram_io},
                    {"engram_gpu", t.engram}, {"mhc", t.mhc}, {"attention", t.attention},
                    {"gate_route", t.gate}, {"shared_expert", t.shared_expert},
                    {"expert_io", t.expert_io}, {"expert_gpu", t.expert_gpu}, {"head", t.head},
                    {"host_other", t.host}};
                for (const auto& [name, ms] : buckets)
                    std::fprintf(csv.f, "prefill,%s,%s,%u,%u,0,%llu,%.1f,%.1f,0,0,0,\"%s\"\n", v.c_str(), name, n,
                                 t.experts_read, (unsigned long long)t.expert_bytes, ms,
                                 name == std::string("total") ? n / (ms / 1e3) : 0.0, o.load.c_str());
                std::fflush(csv.f);
            }
            if (!o.handoff_dir.empty()) {
                const std::string d = std::format("{}/N{}_{}_c{}_d{}", o.handoff_dir, n, pc.replay >= n ? "oracle" : "replay",
                                                  static_cast<int32_t>(cmin), static_cast<int32_t>(cden));
                std::filesystem::create_directories(d);
                // just the window rings, for a production-vs-oracle comparison
                std::FILE* f = std::fopen((d + "/win_kv.f32").c_str(), "wb");
                if (f) {
                    for (const auto& l : out->layers) std::fwrite(l.win_kv.data(), sizeof(float), l.win_kv.size(), f);
                    std::fwrite(out->logits.data(), sizeof(float), out->logits.size(), f);
                    std::fclose(f);
                }
            }
            pf.destroy();
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    set_log_level(LogLevel::Warn);
    Options o;
    if (const char* e = env("DEEPMOE_MODEL_DIR")) o.model_dir = e;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fputs("missing value\n", stderr); std::exit(2); }
            return argv[++i];
        };
        auto list = [](const std::string& s) {
            std::vector<uint32_t> v;
            size_t p = 0;
            while (p < s.size()) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) q = s.size();
                v.push_back(static_cast<uint32_t>(std::atoi(s.substr(p, q - p).c_str())));
                p = q + 1;
            }
            return v;
        };
        if (a == "--model-dir")    o.model_dir = next();
        else if (a == "--csv")     o.csv = next();
        else if (a == "--section") o.section = next();
        else if (a == "--load")    o.load = next();
        else if (a == "--reps")    o.reps = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (a == "--n")       o.ns = list(next());
        else if (a == "--tiles")   o.tiles = list(next());
        else if (a == "--no-coopmat") o.coopmat = false;
        else if (a == "--ids")     o.ids = next();
        else if (a == "--l3")      o.l3 = next();
        else if (a == "--replay")  o.replays = list(next());
        else if (a == "--coop-min") o.coop_min = list(next());
        else if (a == "--coop-dense") o.coop_dense = list(next());
        else if (a == "--transit") o.transit = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (a == "--handoff-dir") o.handoff_dir = next();
        else if (a == "--only")    o.only = next();
        else { std::fprintf(stderr, "unknown option %.*s\n", int(a.size()), a.data()); return 2; }
    }
    if (o.model_dir.empty()) { std::fputs("set --model-dir or DEEPMOE_MODEL_DIR\n", stderr); return 2; }
    if (o.section == "gemm") return run_gemm(o);
    if (o.section == "prefill") return run_prefill(o);
    if (o.section == "coopgeo") return run_coopgeo(o);
    std::fprintf(stderr, "unknown section '%s'\n", o.section.c_str());
    return 2;
}
