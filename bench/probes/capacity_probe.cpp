// capacity_probe -- where does the read rate fall off, and is it capacity or
// paging? (Track W, docs/plan_p5.md §5.)
//
// Two decisions hang on this and today neither has a number:
//
//   * Is a tile size worth tuning at all? docs/STATUS.md §3 item 17 found that
//     a TIGHTER LDS budget was SLOWER (wq_a 152 GB/s at 5120 vs 166 at 8192)
//     because more resident workgroups each re-staged a wide activation. That
//     is a capacity argument made without knowing where the capacity is.
//   * Is the expert GEMV's shortfall a capacity/locality effect or a paging
//     one? Item 30 attributes path B's 12% MoE deficit to a GART 4 KiB
//     page walk under the 7-expert x 2304-row pattern -- a paging claim,
//     never separated from capacity.
//
// The separation is the same one fleet-mi300x's microbench (g) makes: sweep
// the working set at a fixed reader count and look at the SHAPE. A rate that
// falls once between two sizes and is then flat is a cache boundary. One that
// keeps degrading all the way out is a TLB effect, because the page count
// keeps growing while the cache miss rate has already saturated.
//
// Strix Halo's prior: a 32 MB MALL between the CUs and LPDDR5X, and a measured
// 215-218 GB/s raw-read ceiling (docs/STATUS.md §1, four runs within 0.5% on
// both memory paths). A working set inside the MALL should beat that ceiling;
// one outside it should sit at it. Where the two meet is the cliff.
//
//   build/capacity_probe --csv bench/results/capacity_probe.csv
//   build/capacity_probe --path b        # the same sweep on path B memory
//
// `--path b` is the one that decides the paging question: if the cliff is at
// the same size on both paths but path B's plateau is lower, that is the
// page-walk cost isolated from capacity, which is what item 30 asserts.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bench/probes/probe_common.h"

using namespace deepmoe;
using namespace deepmoe::probe;

namespace {

struct Push {
    uint32_t per_group, share, repeats, reserved;
};

uint32_t repeats_for(uint64_t per_wg_bytes) {
    const uint64_t target = 512ull << 20;
    return uint32_t(std::clamp<uint64_t>(target / std::max<uint64_t>(per_wg_bytes, 1), 1, 8192));
}

}  // namespace

int main(int argc, char** argv) {
    if (has_flag(argc, argv, "--help")) {
        std::puts(
            "capacity_probe [--buffer-mb N] [--readers N] [--reps N] [--path a|b] [--csv PATH]\n"
            "  --readers   concurrent workgroups, all reading the same bytes (default 40 = one\n"
            "              per CU on the 8060S)\n"
            "  --path      a = DEVICE_LOCAL|HOST_VISIBLE, b = imported host memory\n"
            "Sweeps the working set from 64 KB to the buffer size in fine steps and prints the\n"
            "per-step change, so the cliff is visible rather than inferred.");
        return 0;
    }
    const uint64_t buf_mb  = arg_u64(argc, argv, "--buffer-mb", 512);
    const uint32_t readers = uint32_t(arg_u64(argc, argv, "--readers", 40));
    const uint32_t reps    = uint32_t(arg_u64(argc, argv, "--reps", 3));
    const std::string path_s   = arg_after(argc, argv, "--path", "a");
    const std::string csv_path = arg_after(argc, argv, "--csv", "");

    const MemoryPath path = (path_s == "b" || path_s == "B") ? MemoryPath::ExternalMemoryHost
                                                            : MemoryPath::DeviceLocalHostVisible;
    Gpu g;
    if (auto r = g.create(path); !r) {
        std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
        return 1;
    }
    std::printf("device: %s\n", g.device.caps().to_string().c_str());
    std::printf("path: %s, %u concurrent readers\n", path_s.c_str(), readers);

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

    auto src  = g.alloc.allocate(buf_mb << 20, false, false);
    auto sink = g.alloc.allocate(uint64_t(readers) * 256 * 16, false, false);
    if (!src || !sink) {
        std::fprintf(stderr, "probe: cannot allocate %llu MB\n",
                     static_cast<unsigned long long>(buf_mb));
        return 1;
    }

    Csv csv;
    csv.open(csv_path, "path,readers,working_set_bytes,per_reader_gbps,aggregate_gbps,delta_pct");

    std::printf("\n  %12s %14s %14s %10s\n", "working set", "per reader", "aggregate", "d vs prev");
    double prev = 0.0;
    double first_plateau = 0.0;
    uint64_t cliff_lo = 0, cliff_hi = 0;

    // Fine steps: every power of two, plus the 1.5x point, so a cliff is
    // bracketed rather than straddled.
    std::vector<uint64_t> sets;
    for (uint64_t ws = 64ull << 10; ws <= (buf_mb << 20); ws *= 2) {
        sets.push_back(ws);
        if (ws * 3 / 2 <= (buf_mb << 20)) sets.push_back(ws * 3 / 2);
    }
    std::sort(sets.begin(), sets.end());

    for (uint64_t ws : sets) {
        const uint64_t per_group_u4 = (ws / 16) / 1024 * 1024;
        if (per_group_u4 < 1024) continue;
        const uint64_t real_ws = per_group_u4 * 16;
        Push p{uint32_t(per_group_u4), 1u, repeats_for(real_ws), 0};

        std::vector<gpu::BufferBinding> binds(2);
        binds[0].binding = 0; binds[0].buffer = src->buffer;
        binds[1].binding = 1; binds[1].buffer = sink->buffer;
        desc.reset();
        auto set = desc.allocate(pipe, binds);
        if (!set) continue;

        if (!g.cmd.begin()) continue;
        if (g.timed) {
            (void)g.cmd.reset_queries(g.queries, 0, 2);
            (void)g.cmd.write_timestamp(g.queries, 0, false);
        }
        (void)g.cmd.bind(pipe, *set);
        (void)g.cmd.push(pipe, &p, sizeof p);
        (void)g.cmd.dispatch(readers);
        if (g.timed) (void)g.cmd.write_timestamp(g.queries, 1, true);
        (void)g.cmd.end();

        auto s = best_seconds(g, g.cmd, reps);
        if (!s || *s <= 0.0) continue;
        const double per  = double(real_ws) * p.repeats / 1e9 / *s;
        const double aggr = per * readers;
        const double d    = prev > 0 ? 100.0 * (per - prev) / prev : 0.0;
        std::printf("  %9llu KB %11.1f GB/s %11.1f GB/s %9.1f%%\n",
                    static_cast<unsigned long long>(real_ws >> 10), per, aggr, d);
        csv.row("%s,%u,%llu,%.3f,%.3f,%.2f", path_s.c_str(), readers,
                static_cast<unsigned long long>(real_ws), per, aggr, d);
        // The first drop of more than 15% is the candidate cliff.
        if (prev > 0 && d < -15.0 && cliff_lo == 0) {
            cliff_lo = real_ws / 2;
            cliff_hi = real_ws;
            first_plateau = per;
        }
        prev = per;
    }

    std::printf("\n");
    if (cliff_lo) {
        std::printf("cliff between %llu KB and %llu KB.\n",
                    static_cast<unsigned long long>(cliff_lo >> 10),
                    static_cast<unsigned long long>(cliff_hi >> 10));
        const double last = prev;
        const double drift = first_plateau > 0 ? 100.0 * (last - first_plateau) / first_plateau : 0.0;
        if (drift > -10.0)
            std::printf("After the cliff the rate is flat (%.1f%% from the first post-cliff point\n"
                        "to the last): a CAPACITY boundary. A working set that fits below it is\n"
                        "worth arranging; beyond it, tile size does not matter.\n", drift);
        else
            std::printf("After the cliff the rate keeps falling (%.1f%%): that is a PAGING/TLB\n"
                        "effect, not capacity, and item 30's GART page-walk attribution has\n"
                        "support. Large pages would then be worth the privilege fight\n"
                        "(STATUS.md item 9).\n", drift);
    } else {
        std::printf("no drop over 15%% anywhere in the sweep: the memory system serves every\n"
                    "working set here at the same rate, so neither tile size nor page size has\n"
                    "anything to win. That closes two open questions at once.\n");
    }

    csv.close(csv_path);
    g.alloc.free(*src);
    g.alloc.free(*sink);
    desc.destroy();
    pipe.destroy();
    return 0;
}
