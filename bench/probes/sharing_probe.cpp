// sharing_probe -- is sharing a read free? (Track W, docs/plan_p5.md §5,
// after fleet-mi300x's microbench (g).)
//
// A whole family of kernel ideas rests on one unexamined premise: that if two
// consumers need the same bytes, making ONE workgroup read them and share the
// result is cheaper than letting both read. docs/STATUS.md has two of those
// measured the expensive way and both lost:
//
//   item 10  `heads_per_wg` ("read the KV once"): 1/2/4/8 heads a workgroup
//            measured 69 / 100 / 169 / 288 us while L2 traffic fell 41 -> 5 MB.
//   item 11  `pv_heads_per_wg = 8`: 120 us against 25-27 us at 1.
//
// Each cost a kernel. One probe answers the class: stream the SAME buffer from
// 1, 2, ... N workgroups at once and report the per-workgroup rate. If the
// rows are flat, sharing is free, redundant reads cost nothing, and every
// "read it once and publish" idea is dead before it is written. If the rate
// collapses with readers, the opposite, and the L2/MALL is the thing to design
// around.
//
// The same sweep, read along the other axis, locates the capacity cliff: each
// column is one working-set size, so a rate that falls once between two sizes
// and then stays flat is a cache-capacity boundary, while one that keeps
// falling is a TLB/paging effect. (capacity_probe does that axis at finer
// granularity.)
//
//   build/sharing_probe --csv bench/results/sharing_probe.csv
//
// Reading it: divide the 8060S's ~217 GB/s aggregate ceiling
// (docs/STATUS.md §1) by the per-workgroup rate to see how many readers the
// memory system can actually feed; compare that with the 40 CUs.

#include <algorithm>
#include <cstdio>
#include <vector>

#include "bench/probes/probe_common.h"

using namespace deepmoe;
using namespace deepmoe::probe;

namespace {

struct Push {
    uint32_t per_group;   // uint4 elements one workgroup walks
    uint32_t share;       // 1 = every workgroup reads from offset 0
    uint32_t repeats;
    uint32_t reserved;
};

// A pass has to be long enough that the launch is noise. `repeats` is chosen
// per working set so every cell moves roughly the same number of bytes.
uint32_t repeats_for(uint64_t working_set) {
    const uint64_t target = 256ull << 20;   // ~256 MB a workgroup a cell
    return uint32_t(std::clamp<uint64_t>(target / std::max<uint64_t>(working_set, 1), 1, 4096));
}

}  // namespace

int main(int argc, char** argv) {
    if (has_flag(argc, argv, "--help")) {
        std::puts(
            "sharing_probe [--buffer-mb N] [--max-groups N] [--reps N] [--csv PATH] [--disjoint]\n"
            "  --buffer-mb   the source buffer (default 256)\n"
            "  --max-groups  largest concurrent reader count (default 320)\n"
            "  --disjoint    also run the rawread-style disjoint-slice sweep for comparison");
        return 0;
    }
    const uint64_t buf_mb   = arg_u64(argc, argv, "--buffer-mb", 256);
    const uint32_t max_grp  = uint32_t(arg_u64(argc, argv, "--max-groups", 320));
    const uint32_t reps     = uint32_t(arg_u64(argc, argv, "--reps", 3));
    const bool     disjoint = has_flag(argc, argv, "--disjoint");
    const std::string csv_path = arg_after(argc, argv, "--csv", "");

    Gpu g;
    if (auto r = g.create(); !r) {
        std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
        return 1;
    }
    std::printf("device: %s\n", g.device.caps().to_string().c_str());

    gpu::Pipeline pipe;
    gpu::PipelineLayoutSpec lspec;
    lspec.storage_buffers    = 2;
    lspec.push_constant_size = sizeof(Push);
    if (auto r = pipe.create(g.device, g.shader_dir + "/probe_stream.spv", lspec); !r) {
        std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
        return 1;
    }
    gpu::DescriptorPool desc;
    if (auto r = desc.create(g.device, 1, 2); !r) {
        std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
        return 1;
    }

    auto src  = g.alloc.allocate(buf_mb << 20, /*host_visible=*/false, /*device_address=*/false);
    auto sink = g.alloc.allocate(uint64_t(max_grp) * 256 * 16, false, false);
    if (!src || !sink) {
        std::fprintf(stderr, "probe: cannot allocate %llu MB + sink\n",
                     static_cast<unsigned long long>(buf_mb));
        return 1;
    }

    // The working sets to sweep. 128 KB is inside any L1/L2; 32 MB is Strix
    // Halo's MALL; 128 MB is past it and into LPDDR5X.
    const uint64_t sets[] = {128ull << 10, 1ull << 20, 4ull << 20, 8ull << 20,
                             16ull << 20, 32ull << 20, 64ull << 20, 128ull << 20};
    const uint32_t readers[] = {1, 2, 4, 8, 16, 32, 40, 80, 160, 320};

    Csv csv;
    csv.open(csv_path, "mode,working_set_bytes,readers,per_reader_gbps,aggregate_gbps,seconds");

    auto sweep = [&](bool share) {
        std::printf("\n%s: per-workgroup GB/s (rows: working set, columns: concurrent readers)\n",
                    share ? "SHARED -- every workgroup reads the same bytes"
                          : "DISJOINT -- rawread's slices, for comparison");
        std::printf("  %12s", "working set");
        for (uint32_t n : readers) if (n <= max_grp) std::printf(" %8u", n);
        std::printf("\n");
        for (uint64_t ws : sets) {
            if (ws > src->bytes) continue;
            std::printf("  %9llu KB", static_cast<unsigned long long>(ws >> 10));
            for (uint32_t n : readers) {
                if (n > max_grp) continue;
                // Shared: each workgroup walks the whole working set.
                // Disjoint: the working set is split n ways, so each walks 1/n.
                const uint64_t per_wg_bytes = share ? ws : ws / n;
                const uint64_t per_group = per_wg_bytes / 16;           // uint4 elements
                if (per_group < 1024) { std::printf(" %8s", "-"); continue; }
                Push p{uint32_t(per_group / 1024 * 1024), share ? 1u : 0u,
                       repeats_for(per_wg_bytes), 0};

                std::vector<gpu::BufferBinding> binds(2);
                binds[0].binding = 0; binds[0].buffer = src->buffer;
                binds[1].binding = 1; binds[1].buffer = sink->buffer;
                desc.reset();
                auto set = desc.allocate(pipe, binds);
                if (!set) { std::printf(" %8s", "err"); continue; }

                if (!g.cmd.begin()) { std::printf(" %8s", "err"); continue; }
                if (g.timed) {
                    (void)g.cmd.reset_queries(g.queries, 0, 2);
                    (void)g.cmd.write_timestamp(g.queries, 0, false);
                }
                (void)g.cmd.bind(pipe, *set);
                (void)g.cmd.push(pipe, &p, sizeof p);
                (void)g.cmd.dispatch(n);
                if (g.timed) (void)g.cmd.write_timestamp(g.queries, 1, true);
                (void)g.cmd.end();

                auto s = best_seconds(g, g.cmd, reps);
                if (!s || *s <= 0.0) { std::printf(" %8s", "err"); continue; }
                const uint64_t bytes_per_reader = uint64_t(p.per_group) * 16 * p.repeats;
                const double per   = double(bytes_per_reader) / 1e9 / *s;
                const double aggr  = per * n;
                std::printf(" %8.1f", per);
                csv.row("%s,%llu,%u,%.3f,%.3f,%.6f", share ? "shared" : "disjoint",
                        static_cast<unsigned long long>(ws), n, per, aggr, *s);
            }
            std::printf("\n");
        }
    };

    sweep(/*share=*/true);
    if (disjoint) sweep(/*share=*/false);

    std::printf("\nHow to read it (docs/plan_p5.md §5):\n"
                "  * a FLAT row  -> sharing is free; 'read it once and publish' cannot pay, and\n"
                "                   STATUS.md items 10/11 are explained rather than merely known.\n"
                "  * a FALLING row -> the L2/MALL is the bottleneck under concurrency and a\n"
                "                   shared-read design is worth writing.\n"
                "  * a column that falls ONCE between two working sets and then stays flat is a\n"
                "    cache-capacity boundary; one that keeps falling is paging. capacity_probe\n"
                "    resolves it at finer granularity.\n");

    csv.close(csv_path);
    g.alloc.free(*src);
    g.alloc.free(*sink);
    desc.destroy();
    pipe.destroy();
    return 0;
}
