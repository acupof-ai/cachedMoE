// residency_probe -- can a persistent-dispatch decode exist on this machine?
// (Track W, docs/plan_p5.md §3(a) and §5.)
//
// The prize is large and exactly known: 41 submits x 0.156 ms of host round
// trip per token (docs/STATUS.md §2.1), plus every barrier tail the
// per-dispatch trace attributes. On an MI300X, fleet-mi300x collapses ~800
// launches a token into ONE cooperative launch of 304 resident workgroups that
// hand work to each other through a host-built task graph.
//
// Vulkan has no cooperative launch and no forward-progress guarantee between
// workgroups. So before anyone writes a megakernel here, three numbers:
//
//   1. RESIDENCY   how many workgroups of 256 threads actually run at once?
//                  A megakernel whose workgroup N waits on workgroup M
//                  deadlocks if M was never scheduled. This is a legality
//                  question, not a performance one.
//   2. LATENCY     what does one cross-workgroup handshake through device
//                  memory cost? fleet measures 1.36 us idle / 5.88 us under
//                  load. 40 layers x a few events each has to stay well under
//                  the 6.4 ms of submits it replaces.
//   3. PROGRESS    does a spinning workgroup starve the one it waits for?
//
// Every wait in gpu/shaders/probe_resident.slang is bounded by `--spin`; a
// loop that reaches the cap reports a timeout instead of hanging. There is no
// unbounded wait anywhere in this probe. That matters: this runs on the
// owner's desktop and a TDR is a reboot.
//
//   build/residency_probe --max-groups 2048 --spin 200000 \
//       --csv bench/results/residency_probe.csv
//
// Reading it:
//   * `peak` stops tracking `groups` at the residency limit. If peak keeps
//     rising to 2048, this machine will host a megakernel of any size we
//     want and the limit is somewhere else.
//   * A mode-2 timeout at a group count BELOW the mode-0 peak means the
//     scheduler does not guarantee progress even among co-resident groups:
//     that is a hard NO-GO for spin-waits, and the answer is a graph of
//     dispatches, not one dispatch.
//   * The ping-pong number divided into 0.156 ms says how many device-side
//     events one saved submit buys.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "bench/probes/probe_common.h"

using namespace deepmoe;
using namespace deepmoe::probe;

namespace {

struct Push {
    uint32_t mode, groups, spin_limit, rounds;
};

constexpr uint32_t kCtlWords = 8;
constexpr uint32_t kOutWordsPerGroup = 4;

// Control-block words, mirroring gpu/shaders/probe_resident.slang.
constexpr uint32_t kLive = 0, kPeak = 1, kFlagA = 2, kFlagB = 3, kDone = 4;
// Per-group result words.
constexpr uint32_t kOutSeen = 0, kOutTimeout = 1, kOutRounds = 2;

struct Run {
    uint32_t peak = 0;
    uint32_t timeouts = 0;
    uint32_t rounds = 0;
    double   seconds = 0.0;
};

Result<Run> dispatch(Gpu& g, gpu::Pipeline& pipe, gpu::DescriptorPool& desc,
                     gpu::GpuBuffer& ctl, gpu::GpuBuffer& out, const Push& push,
                     uint32_t reps) {
    // Zero the control block and the results between runs; both are host
    // visible so this is a memset, not a dispatch.
    std::memset(ctl.host_ptr, 0, size_t(kCtlWords) * 4);
    std::memset(out.host_ptr, 0, size_t(push.groups) * kOutWordsPerGroup * 4);

    std::vector<gpu::BufferBinding> binds(2);
    binds[0].binding = 0; binds[0].buffer = ctl.buffer;
    binds[1].binding = 1; binds[1].buffer = out.buffer;
    desc.reset();
    auto set = desc.allocate(pipe, binds);
    if (!set) return std::unexpected(set.error());

    if (auto r = g.cmd.begin(); !r) return std::unexpected(r.error());
    if (g.timed) {
        (void)g.cmd.reset_queries(g.queries, 0, 2);
        (void)g.cmd.write_timestamp(g.queries, 0, /*bottom=*/false);
    }
    if (auto r = g.cmd.bind(pipe, *set); !r) return std::unexpected(r.error());
    if (auto r = g.cmd.push(pipe, &push, sizeof push); !r) return std::unexpected(r.error());
    if (auto r = g.cmd.dispatch(push.groups); !r) return std::unexpected(r.error());
    if (g.timed) (void)g.cmd.write_timestamp(g.queries, 1, /*bottom=*/true);
    if (auto r = g.cmd.end(); !r) return std::unexpected(r.error());

    // Only mode 1 is timed across repeats; the census and the progress test
    // mutate the control block, so they run exactly once.
    auto s = best_seconds(g, g.cmd, push.mode == 1 ? reps : 1);
    if (!s) return std::unexpected(s.error());

    Run out_run;
    out_run.seconds = *s;
    const uint32_t* c = static_cast<const uint32_t*>(ctl.host_ptr);
    const uint32_t* o = static_cast<const uint32_t*>(out.host_ptr);
    out_run.peak = c[kPeak];
    for (uint32_t i = 0; i < push.groups; ++i) {
        const uint32_t* row = o + size_t(i) * kOutWordsPerGroup;
        out_run.timeouts += row[kOutTimeout];
        out_run.rounds = std::max(out_run.rounds, row[kOutRounds]);
    }
    return out_run;
}

}  // namespace

int main(int argc, char** argv) {
    if (has_flag(argc, argv, "--help")) {
        std::puts(
            "residency_probe [--max-groups N] [--spin N] [--rounds N] [--reps N] [--csv PATH]\n"
            "  --max-groups  largest workgroup count for the census sweep (default 2048)\n"
            "  --spin        hard cap on every device-side loop (default 200000)\n"
            "  --rounds      ping-pong round trips (default 1000)\n"
            "  --reps        timed repeats of the ping-pong (default 3)\n"
            "Every wait is bounded; a loop that hits --spin reports a timeout.");
        return 0;
    }
    const uint32_t max_groups = uint32_t(arg_u64(argc, argv, "--max-groups", 2048));
    const uint32_t spin       = uint32_t(arg_u64(argc, argv, "--spin", 200000));
    const uint32_t rounds     = uint32_t(arg_u64(argc, argv, "--rounds", 1000));
    const uint32_t reps       = uint32_t(arg_u64(argc, argv, "--reps", 3));
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
    if (auto r = pipe.create(g.device, g.shader_dir + "/probe_resident.spv", lspec); !r) {
        std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
        return 1;
    }
    gpu::DescriptorPool desc;
    if (auto r = desc.create(g.device, 1, 2); !r) {
        std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
        return 1;
    }

    auto ctl = g.alloc.allocate_host_coherent(kCtlWords * 4);
    auto out = g.alloc.allocate_host_coherent(uint64_t(max_groups) * kOutWordsPerGroup * 4);
    if (!ctl || !out) {
        std::fprintf(stderr, "probe: cannot allocate the control block\n");
        return 1;
    }

    Csv csv;
    csv.open(csv_path, "probe,groups,peak,timeouts,rounds,us");

    // --- 1. residency census ------------------------------------------------
    std::printf("\nresidency census (256 threads a workgroup, spin %u)\n", spin);
    std::printf("  %8s %8s %10s\n", "groups", "peak", "verdict");
    uint32_t residency = 0;
    for (uint32_t n = 32; n <= max_groups; n *= 2) {
        Push p{0, n, spin, 0};
        auto r = dispatch(g, pipe, desc, *ctl, *out, p, 1);
        if (!r) { std::fprintf(stderr, "  dispatch failed: %s\n", r.error().str().c_str()); break; }
        const bool saturated = r->peak < n;
        if (saturated && residency == 0) residency = r->peak;
        std::printf("  %8u %8u %10s\n", n, r->peak,
                    saturated ? "saturated" : "all live");
        csv.row("census,%u,%u,%u,%u,%.3f", n, r->peak, r->timeouts, r->rounds,
                r->seconds * 1e6);
    }
    if (residency)
        std::printf("  -> at most ~%u workgroups are co-resident. A persistent kernel may not\n"
                    "     exceed that, and a task graph built for more deadlocks.\n", residency);
    else
        std::printf("  -> the peak tracked the dispatch up to %u: residency was not reached here.\n",
                    max_groups);

    // --- 2. cross-workgroup ping-pong --------------------------------------
    std::printf("\ncross-workgroup handshake (2 workgroups, %u round trips)\n", rounds);
    {
        Push p{1, 2, spin, rounds};
        auto r = dispatch(g, pipe, desc, *ctl, *out, p, reps);
        if (!r) {
            std::fprintf(stderr, "  dispatch failed: %s\n", r.error().str().c_str());
        } else if (r->timeouts) {
            std::printf("  TIMEOUT after %u of %u rounds: the two workgroups did not both stay\n"
                        "  resident, or one starved the other. Spin-waits are not usable here.\n",
                        r->rounds, rounds);
            csv.row("pingpong,2,0,%u,%u,%.3f", r->timeouts, r->rounds, r->seconds * 1e6);
        } else {
            const double us = r->seconds * 1e6 / double(rounds);
            std::printf("  %.3f us a round trip (%.3f ms for %u rounds, %s)\n", us,
                        r->seconds * 1e3, rounds, g.timed ? "GPU timed" : "wall clock");
            std::printf("  -> one saved submit (0.156 ms) buys about %.0f device-side events.\n",
                        156.0 / (us > 0 ? us : 1e9));
            csv.row("pingpong,2,0,0,%u,%.3f", rounds, us);
        }
    }

    // --- 3. forward progress ------------------------------------------------
    std::printf("\nforward progress (workgroup 0 spins for the last one)\n");
    std::printf("  %8s %10s %12s\n", "groups", "result", "spins");
    for (uint32_t n = 2; n <= max_groups; n *= 4) {
        Push p{2, n, spin, 0};
        auto r = dispatch(g, pipe, desc, *ctl, *out, p, 1);
        if (!r) { std::fprintf(stderr, "  dispatch failed: %s\n", r.error().str().c_str()); break; }
        const uint32_t* o = static_cast<const uint32_t*>(out->host_ptr);
        std::printf("  %8u %10s %12u\n", n, r->timeouts ? "TIMEOUT" : "progressed", o[kOutSeen]);
        csv.row("progress,%u,0,%u,%u,%.3f", n, r->timeouts, r->rounds, r->seconds * 1e6);
        if (r->timeouts && residency && n <= residency) {
            std::printf("  -> a timeout at %u groups, BELOW the %u-group residency limit, means\n"
                        "     co-residency does not imply progress. NO-GO for spin-waits.\n",
                        n, residency);
        }
    }

    csv.close(csv_path);
    g.alloc.free(*ctl);
    g.alloc.free(*out);
    desc.destroy();
    pipe.destroy();
    return 0;
}
