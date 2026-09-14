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
// Filled by the driver in gpu/vulkan/prefill_kernels.h once it exists.
//
// Usage:
//   prefill_bench --model-dir D:\models\DeepSeek-V4.1-Flash --csv bench/results/prefill_p3.csv
//                 [--reps 5] [--load "shared: tracks I/J/K on the GPU"]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <future>
#include <random>
#include <string>
#include <vector>

#include "core/align.h"
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
    if (const char* dbg = std::getenv("PF_DEBUG_ONES")) {
        // every activation 1.0 with scale 1 (or a single 1.0 at element atoi(dbg))
        const int only = std::atoi(dbg);
        for (size_t i = 0; i < size_t(npad) * dim; ++i)
            static_cast<uint16_t*>(xq.host_ptr)[i] =
                cpu::float_to_fp16((only <= 0 || int(i % dim) == only) ? 1.0f : 0.0f);
        for (size_t i = 0; i < size_t(npad) * (dim / 32); ++i)
            static_cast<float*>(xs.host_ptr)[i] = 1.0f;
    }
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
            static gpu::GpuBuffer dbg = must_alloc(rig.alloc, 4096);
            if (std::getenv("PF_DEBUG")) sl[gpu::kPgRowScale] = dbg.dev_addr;
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
            if (std::getenv("PF_DEBUG")) {
                const auto* g = static_cast<const float*>(h.host_ptr);
                for (uint32_t r : {0u, 1u, 7u, 8u, 100u, 2303u})
                    std::printf("    row %4u gpu %12.6f cpu %12.6f  (token1 %12.6f)\n", r, g[r],
                                ref_w1[r], g[inter + r]);
                const auto* d = static_cast<const float*>(dbg.host_ptr);
                for (uint32_t l = 0; l < 32; ++l)
                    std::printf("    lane %2u acc %10.6f lane %4.0f tid %4.0f e %5.0f w %8.3f dot %10.5f s8 %4.0f host s8 %u\n",
                                l, d[l], d[32 + l], d[64 + l], d[96 + l], d[128 + l], d[160 + l],
                                d[192 + l], static_cast<const uint8_t*>(e0->host(1))[l]);
            }
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
        else { std::fprintf(stderr, "unknown option %.*s\n", int(a.size()), a.data()); return 2; }
    }
    if (o.model_dir.empty()) { std::fputs("set --model-dir or DEEPMOE_MODEL_DIR\n", stderr); return 2; }
    if (o.section == "gemm") return run_gemm(o);
    std::fprintf(stderr, "unknown section '%s'\n", o.section.c_str());
    return 2;
}
