// model_probe -- measures the constants of the per-dispatch GPU model
// (tools/gpu_model.py): the model says what each decode dispatch SHOULD cost
// from its bytes, its FLOPs and its workgroup count, and the hot-step trace
// says what it DID cost; the gap, ranked, is the optimisation list, and a
// change is priced by the model before it is written.
//
//   dispatch   a chain of empty dispatches with and without a barrier between
//              them: the fixed cost every stage pays before it does anything
//   stream     t(B, W): B bytes streamed by W disjoint workgroups, one dispatch
//              timed on its own -- launch and ramp included, because that is
//              what a small kernel pays
//   barrier    one workgroup's LDS write + barrier + read + barrier round
//   wave       one dependent WaveActiveSum
//   chase      one dependent global load, at four working-set sizes
//   fma        fp32 FMA throughput over the whole GPU
//   mma        fp16 cooperative-matrix throughput (probe_mma.slang, Wave32):
//              the ceiling of every prefill tile multiply (tools/prefill_model.py)
//
// Output: a table on stdout and, with --json, the constants as JSON (what
// tools/gpu_model.py reads). One GPU job at a time (CLAUDE.md).
//
// Ownership/threading: one instance, one thread, RAII teardown in reverse.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "bench/probes/probe_common.h"

using namespace cachedmoe;
using namespace cachedmoe::probe;

namespace {

struct Push {
    uint32_t mode, n, per_group, base;
};

struct Ctx {
    Gpu&                 g;
    gpu::Pipeline&       pipe;
    gpu::DescriptorPool& desc;
    std::vector<gpu::BufferBinding> binds;
    uint32_t             reps = 5;

    // Records `count` dispatches of `groups` workgroups with `p`, optionally a
    // barrier after each, and returns the best GPU-timed span in seconds.
    // `cold`: first stream 128 MB from the far end of Src (4x the 32 MB MALL)
    // so the timed dispatch finds nothing of its own in any cache -- what a
    // weight read is on every decode token.
    Result<double> time(const Push& p, uint32_t groups, uint32_t count, bool barriers,
                        bool cold = false) {
        desc.reset();
        auto set = desc.allocate(pipe, binds);
        if (!set) return std::unexpected(set.error());
        if (auto r = g.cmd.begin(); !r) return std::unexpected(r.error());
        (void)g.cmd.bind(pipe, *set);
        if (cold) {
            constexpr uint32_t kFlushGroups = 640;
            const Push f{1, 0, uint32_t((128ull << 20) / 16 / kFlushGroups), uint32_t((384ull << 20) / 16)};
            (void)g.cmd.push(pipe, &f, sizeof f);
            (void)g.cmd.dispatch(kFlushGroups);
            (void)g.cmd.barrier();
        }
        if (g.timed) {
            (void)g.cmd.reset_queries(g.queries, 0, 2);
            (void)g.cmd.write_timestamp(g.queries, 0, false);
        }
        (void)g.cmd.push(pipe, &p, sizeof p);
        for (uint32_t i = 0; i < count; ++i) {
            (void)g.cmd.dispatch(groups);
            if (barriers && i + 1 < count) (void)g.cmd.barrier();
        }
        if (g.timed) (void)g.cmd.write_timestamp(g.queries, 1, true);
        if (auto r = g.cmd.end(); !r) return std::unexpected(r.error());
        return best_seconds(g, g.cmd, reps);
    }
};

std::string jnum(double v) { char b[64]; std::snprintf(b, sizeof b, "%.6g", v); return b; }

}  // namespace

int main(int argc, char** argv) {
    if (has_flag(argc, argv, "--help")) {
        std::puts("model_probe [--json PATH] [--reps N]\n"
                  "  the GPU constants tools/gpu_model.py models a decode dispatch with");
        return 0;
    }
    const std::string json_path = arg_after(argc, argv, "--json", "");
    Gpu g;
    if (auto r = g.create(); !r) {
        std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
        return 1;
    }
    if (!g.timed) {
        std::fprintf(stderr, "probe: the model needs GPU timestamps\n");
        return 1;
    }
    std::printf("device: %s\n", g.device.caps().to_string().c_str());

    gpu::Pipeline pipe;
    gpu::PipelineLayoutSpec lspec;
    lspec.storage_buffers    = 3;
    lspec.push_constant_size = sizeof(Push);
    if (auto r = pipe.create(g.device, g.shader_dir + "/probe_model.spv", lspec); !r) {
        std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
        return 1;
    }
    gpu::DescriptorPool desc;
    if (auto r = desc.create(g.device, 1, 3); !r) {
        std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
        return 1;
    }

    constexpr uint64_t kSrcBytes   = 512ull << 20;
    constexpr uint32_t kMaxGroups  = 2560;
    constexpr uint64_t kChaseBytes = 256ull << 20;
    auto src   = g.alloc.allocate(kSrcBytes, false, false);
    auto sink  = g.alloc.allocate(uint64_t(kMaxGroups) * 256 * 16, false, false);
    auto chase = g.alloc.allocate(kChaseBytes, true, false);
    if (!src || !sink || !chase || !chase->host_ptr) {
        std::fprintf(stderr, "probe: allocation failed\n");
        return 1;
    }
    Ctx c{g, pipe, desc, std::vector<gpu::BufferBinding>(3),
          uint32_t(arg_u64(argc, argv, "--reps", 5))};
    c.binds[0].binding = 0; c.binds[0].buffer = src->buffer;
    c.binds[1].binding = 1; c.binds[1].buffer = sink->buffer;
    c.binds[2].binding = 2; c.binds[2].buffer = chase->buffer;

    std::string js = "{\n";
    auto jkv = [&](const std::string& k, const std::string& v, bool last = false) {
        js += "  \"" + k + "\": " + v + (last ? "\n" : ",\n");
    };

    // --- dispatch overhead ------------------------------------------------
    {
        const uint32_t k = 400;
        auto t1 = c.time({0, 0, 0, 0}, 1, 1, false);
        auto tb = c.time({0, 0, 0, 0}, 1, k, true);
        auto tn = c.time({0, 0, 0, 0}, 1, k, false);
        auto tw = c.time({0, 0, 0, 0}, 640, k, true);
        if (!t1 || !tb || !tn || !tw) { std::fprintf(stderr, "probe: dispatch timing failed\n"); return 1; }
        const double b = *tb / k * 1e6, n = *tn / k * 1e6, w = *tw / k * 1e6;
        std::printf("\ndispatch  one alone %.2f us; in a chain of %u: %.2f us each with a barrier, "
                    "%.2f without; 640 empty workgroups + barrier %.2f us\n", *t1 * 1e6, k, b, n, w);
        jkv("dispatch_alone_us", jnum(*t1 * 1e6));
        jkv("dispatch_barrier_us", jnum(b));
        jkv("dispatch_nobarrier_us", jnum(n));
        jkv("dispatch_640wg_barrier_us", jnum(w));
    }

    // --- stream t(B, W) -----------------------------------------------------
    const uint64_t bytes[] = {64ull << 10, 256ull << 10, 1ull << 20, 4ull << 20,
                              16ull << 20, 64ull << 20, 256ull << 20};   // the flush reads 384..512 MB
    const uint32_t groups[] = {1, 2, 4, 8, 16, 20, 40, 64, 80, 128, 160, 256, 320, 640, 1280, 2560};
    double peak = 0;
    for (bool cold : {true, false}) {
    std::printf("\nstream, %s: us for one dispatch reading B bytes over W workgroups (GB/s in brackets)\n",
                cold ? "COLD (caches flushed first: a weight read)" : "HOT (the same bytes again: an activation)");
    std::printf("  %9s", "B \\ W");
    for (uint32_t w : groups) std::printf(" %13u", w);
    std::printf("\n");
    js += cold ? "  \"stream_cold\": [\n" : "  \"stream_hot\": [\n";
    bool first = true;
    for (uint64_t B : bytes) {
        std::printf("  %7llu K", static_cast<unsigned long long>(B >> 10));
        for (uint32_t w : groups) {
            const uint64_t per = B / w / 16 / 1024 * 1024;     // uint4 elements, whole 1024 blocks
            if (per == 0) { std::printf(" %13s", "-"); continue; }
            auto t = c.time({1, 0, uint32_t(per), 0}, w, 1, false, cold);
            if (!t) { std::printf(" %13s", "err"); continue; }
            const double real = double(per) * 16 * w;
            const double gbs = real / 1e9 / *t;
            if (cold) peak = std::max(peak, gbs);
            std::printf(" %7.1f(%4.0f)", *t * 1e6, gbs);
            js += std::string(first ? "" : ",\n") + "    {\"bytes\": " + std::to_string(uint64_t(real)) +
                  ", \"groups\": " + std::to_string(w) + ", \"us\": " + jnum(*t * 1e6) + "}";
            first = false;
        }
        std::printf("\n");
    }
    js += "\n  ],\n";
    }
    jkv("stream_cold_peak_gbs", jnum(peak));

    // --- access shape: moe_down's row block of every expert vs contiguous ----
    // Same bytes three ways, cold: (a) W workgroups each read a 36 KB row
    // block from each of 7 matrices (dispatch B today), (b) W workgroups each
    // read one contiguous 7 x 36 KB run, (c) 7W workgroups each read one 36 KB
    // block, expert-major -- what an expert-split dispatch would issue.
    {
        const uint32_t seg = 36 * 1024 / 16 / 1024 * 1024;       // uint4s, whole 1024 blocks
        std::printf("\naccess shape, cold, 7 segments of %u KB per row block:\n", seg * 16 / 1024);
        for (uint32_t w : {40u, 80u, 160u, 320u}) {
            const uint32_t stride = w * seg;
            auto ta = c.time({6, 7, seg, stride}, w, 1, false, true);
            auto tb = c.time({1, 0, 7 * seg, 0}, w, 1, false, true);
            auto tc = c.time({1, 0, seg, 0}, 7 * w, 1, false, true);
            if (!ta || !tb || !tc) continue;
            const double mb = double(7) * seg * 16 * w / 1e6;
            std::printf("  W %4u (%6.1f MB): 7 segments each %7.1f us (%3.0f GB/s) | contiguous %7.1f us "
                        "(%3.0f) | %u workgroups x 1 segment %7.1f us (%3.0f)\n",
                        w, mb, *ta * 1e6, mb / 1e3 / *ta, *tb * 1e6, mb / 1e3 / *tb, 7 * w, *tc * 1e6,
                        mb / 1e3 / *tc);
            js += "  \"shape_w" + std::to_string(w) + "\": {\"segments_us\": " + jnum(*ta * 1e6) +
                  ", \"contiguous_us\": " + jnum(*tb * 1e6) + ", \"split_us\": " + jnum(*tc * 1e6) +
                  ", \"mb\": " + jnum(mb) + "},\n";
        }
    }

    // --- one workgroup: barrier and wave reduction ---------------------------
    {
        const uint32_t n = 4000;
        auto t0 = c.time({2, 0, 0, 0}, 1, 1, false);
        auto tb = c.time({2, n, 0, 0}, 1, 1, false);
        auto tw = c.time({5, n, 0, 0}, 1, 1, false);
        if (tb && tw && t0) {
            const double bar = (*tb - *t0) / (2.0 * n) * 1e9, wav = (*tw - *t0) / n * 1e9;
            std::printf("\none workgroup: barrier %.1f ns, dependent WaveActiveSum %.1f ns\n", bar, wav);
            jkv("barrier_ns", jnum(bar));
            jkv("wave_reduce_ns", jnum(wav));
        }
    }

    // --- dependent loads -----------------------------------------------------
    {
        // A random single cycle through slots 64 B apart, so every step is a
        // new cache line and the prefetchers have nothing to follow.
        const uint64_t sets[] = {32ull << 10, 2ull << 20, 24ull << 20, 256ull << 20};
        const char* label[] = {"32 KB", "2 MB", "24 MB", "256 MB"};
        std::printf("\ndependent load latency:");
        auto* words = static_cast<uint32_t*>(chase->host_ptr);
        std::mt19937 rng(7);
        for (int si = 0; si < 4; ++si) {
            const uint32_t slots = uint32_t(sets[si] / 64);
            std::vector<uint32_t> order(slots);
            std::iota(order.begin(), order.end(), 0u);
            std::shuffle(order.begin() + 1, order.end(), rng);
            for (uint32_t i = 0; i < slots; ++i)
                words[order[i] * 16] = order[(i + 1) % slots] * 16;
            const uint32_t n = 20000;
            auto t0 = c.time({3, 0, 0, 0}, 1, 1, false);
            auto t = c.time({3, n, 0, 0}, 1, 1, false);
            if (!t || !t0) continue;
            const double ns = (*t - *t0) / n * 1e9;
            std::printf("  %s %.0f ns", label[si], ns);
            jkv(std::string("load_latency_ns_") + std::to_string(sets[si] >> 10) + "k", jnum(ns));
        }
        std::printf("\n");
    }

    // --- fp32 FMA ------------------------------------------------------------
    {
        const uint32_t n = 20000, w = 2560;
        auto t = c.time({4, n, 0, 0}, w, 1, false);
        if (t) {
            const double tflops = double(w) * 256 * n * 8 * 2 / *t / 1e12;
            std::printf("\nfp32 FMA: %.2f TFLOP/s over %u workgroups\n", tflops, w);
            jkv("fma_tflops", jnum(tflops));
        }
    }

    // --- fp16 cooperative matrix ----------------------------------------------
    gpu::Pipeline mma;
    gpu::PipelineSpec wave32;
    wave32.subgroup_size = 32;
    if (auto r = mma.create(g.device, g.shader_dir + "/probe_mma.spv", lspec, wave32); !r) {
        std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
    } else {
        Ctx cm{g, mma, desc, c.binds, c.reps};
        double best = 0;
        std::printf("\nfp16 coopmat 16x16x16 (8 waves a workgroup, 8 accumulators a wave):");
        for (uint32_t w : {40u, 80u, 160u, 320u, 640u, 1280u, 2560u}) {
            const uint32_t n = 4000;
            auto t = cm.time({0, n, 1000, 1000}, w, 1, false);
            if (!t) continue;
            const double tflops = double(w) * 8 * 8 * n * 16 * 16 * 16 * 2 / *t / 1e12;
            best = std::max(best, tflops);
            std::printf("  W %u %.1f", w, tflops);
        }
        std::printf(" TFLOP/s\n");
        jkv("mma_f16_tflops", jnum(best));
    }

    // --- what separates the real GEMM's 14.2 TFLOP/s from the probe's peak ---
    // probe_occ adds the real kernel's LDS footprint, then its LDS tile reads,
    // to the same WMMA loop. See gpu/shaders/probe_occ.slang.
    gpu::Pipeline occ;
    if (auto r = occ.create(g.device, g.shader_dir + "/probe_occ.spv", lspec, wave32); !r) {
        std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
    } else {
        Ctx co{g, occ, desc, c.binds, c.reps};
        static const char* names[6] = {"no LDS (control)", "+27,648 B LDS live",
                                       "+A/B read from LDS", "+wm4wn2 tile ratio",
                                       "+791 VALU ops/iter",
                                       "791 VALU, 8 chains"};
        static const char* keys[6] = {"occ_tflops_nolds", "occ_tflops_lds_alloc",
                                      "occ_tflops_lds_read", "occ_tflops_tile_ratio",
                                      "occ_tflops_valu",
                                      "occ_tflops_valu_ilp"};
        std::printf("\nfp16 coopmat at the real kernel's occupancy:\n");
        for (uint32_t mode = 0; mode < 6; ++mode) {
            double best_m = 0;
            std::printf("  mode %u %-20s", mode, names[mode]);
            for (uint32_t w : {40u, 80u, 160u, 320u, 640u}) {
                const uint32_t n = 4000;
                auto t = co.time({mode, n, 1000, 1000}, w, 1, false);
                if (!t) continue;
                const double tf = double(w) * 8 * 8 * n * 16 * 16 * 16 * 2 / *t / 1e12;
                best_m = std::max(best_m, tf);
                std::printf("  W %u %.1f", w, tf);
            }
            std::printf("  | best %.1f TFLOP/s\n", best_m);
            jkv(keys[mode], jnum(best_m));
        }

        // --- the two levers 0bx left: the wn4 ratio, and the dot2 pattern ----
        std::printf("\nthe wn4 ratio and the dot2 pattern:\n");
        static const char* names2[3] = {"wm4wn4 tile ratio", "96 fma_mix / iter",
                                        "48 dot2-pattern / iter"};
        static const char* keys2[3] = {"occ_tflops_wn4_ratio", "occ_gflops_fma_mix",
                                       "occ_gflops_dot2"};
        for (uint32_t m2 = 0; m2 < 3; ++m2) {
            const uint32_t mode = 6 + m2;
            double best_m = 0;
            std::printf("  mode %u %-24s", mode, names2[m2]);
            for (uint32_t w : {40u, 80u, 160u, 320u, 640u}) {
                const uint32_t n = mode == 6 ? 2000 : 40000;
                auto t = co.time({mode, n, 1000, 1000}, w, 1, false);
                if (!t) continue;
                // mode 6: 8 waves x 16 MACs; modes 7/8: 256 threads x 1,584 FLOP.
                const double fl = mode == 6
                    ? double(w) * 8 * 16 * n * 16 * 16 * 16 * 2
                    : double(w) * 256 * n * 12 * 8 * 2;
                const double tf = fl / *t / 1e12;
                best_m = std::max(best_m, tf);
                std::printf("  W %u %.1f", w, tf);
            }
            std::printf("  | best %.1f TFLOP/s\n", best_m);
            jkv(keys2[m2], jnum(best_m));
        }
        occ.destroy();
    }

    jkv("device", "\"" + g.device.caps().device_name + "\"", true);
    js += "}\n";
    if (!json_path.empty()) {
        if (FILE* f = std::fopen(json_path.c_str(), "wb")) {
            std::fputs(js.c_str(), f);
            std::fclose(f);
            std::printf("-> %s\n", json_path.c_str());
        }
    }
    g.alloc.free(*src);
    g.alloc.free(*sink);
    g.alloc.free(*chase);
    desc.destroy();
    mma.destroy();
    pipe.destroy();
    return 0;
}
