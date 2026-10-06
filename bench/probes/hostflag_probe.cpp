// hostflag_probe -- host->device signalling inside ONE submitted command
// buffer (Track HG, docs/p4_hostflag_gate.md).
//
// The question this asks is NOT the one residency_probe asked and closed.
//
// STATUS section 3 item 48: two workgroups, both far below the ~406-group
// residency limit, failed the 3rd of 1,000 round trips. Co-residency does not
// imply forward progress, so a workgroup may not wait on a PEER workgroup, and
// the device-side task queue of plan_p5 section 3(a) is illegal on this
// machine.
//
// STATUS section 3 item 53: the gate round trip costs 0.40 ms/layer = 16 ms
// a token, and 265 us of the 400 is the driver -- vkQueueSubmit2 99.5 us,
// submit to GPU start ~150 us, fence wake ~15 us. It is paid forty times a
// token for one reason: the command buffer is CUT at the gate, because the
// host must read the top-16 before it knows which experts to fetch.
//
// The mechanism under test removes the cut without removing the host. One
// workgroup spins on a uint the HOST writes. The dependency now points OUT of
// the device's scheduling domain at a CPU thread that is always running, so
// item 48's failure mode -- waiting on a peer the scheduler may never run --
// does not apply by construction. Four things could still kill it, and each
// is a column below:
//
//   LATENCY     device write -> host poll -> host write -> device load. If
//               this is not well under 400 us there is nothing to win.
//   PROGRESS    does the spinning workgroup still get its loads serviced
//               while other workgroups of the same command buffer are
//               resident? That is item 48's question in the shape we need.
//   SAFETY      does the driver reset? A command buffer that sits in a spin
//               is exactly what the Windows TDR timer is for.
//   CORRECTNESS does the payload the host wrote become visible to the
//               dispatch after the spin? kOutBad is that bit.
//
// BOUNDING, which is the part that has to be right before anything runs:
//   * the shader's spin is capped by `spin_limit` iterations and nothing else;
//   * the host CALIBRATES iterations->nanoseconds with the same loop, so the
//     cap lands on --max-spin-us of wall clock rather than on a guess;
//   * the SUM over the rounds of one command buffer is clamped below
//     --tdr-budget-ms (default 1500, against a 2000 ms default TDR window),
//     because TDR counts the packet, not the dispatch;
//   * the host's own wait on completion has a deadline, so a wedged queue is
//     a printed failure and an exit, not a hung process.
//
//   build/hostflag_probe --rounds 40 --bufs 1000 --max-spin-us 20000 \
//       --csv bench/results/hostflag_probe.csv

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "bench/probes/probe_common.h"
#include "gpu/vulkan/timeline.h"

using namespace cachedmoe;
using namespace cachedmoe::probe;

namespace {

struct Push {
    uint32_t mode, round, spin_limit, burn_iters, load_atomic;
};

// Mirrors gpu/shaders/probe_hostflag.slang.
constexpr uint32_t kReq = 0, kFlag = 1, kVal = 2;
constexpr uint32_t kCtlWords = 16;
constexpr uint32_t kOutWordsPerRound = 4;
constexpr uint32_t kOutTimeout = 0, kOutSpins = 1, kOutBad = 3;

constexpr uint32_t kModeRequest = 0, kModeSpin = 1, kModeConsume = 2, kModeBurn = 3;

inline uint32_t payload_for(uint32_t round) { return round * 1000003u; }

using Clock = std::chrono::steady_clock;
inline double us_since(Clock::time_point t) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t).count();
}

// The host half of the gate: poll the request word, "fetch the experts" (a
// configurable sleep), publish the payload, then the flag. The order matters
// and is the whole contract: the payload store must be visible before the
// flag store, or the dispatch after the spin reads garbage.
struct HostServer {
    volatile uint32_t*    ctl = nullptr;
    uint32_t              rounds = 0;
    double                service_ms = 0.0;
    std::atomic<bool>     stop{false};
    std::atomic<uint32_t> served{0};
    double                deadline_us = 0.0;

    // `hot` is the difference between two real designs, and the probe found it
    // matters more than anything else it measures. hot = false is one thread
    // that both polls and fetches: while it is fetching it is NOT reading the
    // request word, and the poll loop goes cold. hot = true is a dedicated
    // poller that never stops reading the request word, with the fetch
    // overlapped -- what a real implementation would have to build, and the
    // only shape in which the steady-state microseconds survive a miss.
    bool hot = false;

    void run() {
        uint32_t   last = 0;
        const auto t0 = Clock::now();
        while (!stop.load(std::memory_order_relaxed) && last < rounds) {
            if (us_since(t0) > deadline_us) return;
            const uint32_t req = ctl[kReq];
            if (req <= last) continue;
            if (service_ms > 0.0) {
                // A miss costs NVMe time.
                const auto until = Clock::now() +
                                   std::chrono::microseconds(int64_t(service_ms * 1000.0));
                if (hot)
                    while (Clock::now() < until) (void)ctl[kReq];   // keep the loop hot
                else
                    while (Clock::now() < until) std::this_thread::yield();
            }
            ctl[kVal] = payload_for(req);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            ctl[kFlag] = req;
            // The SECOND fence is not symmetry. This memory is DEVICE_LOCAL |
            // HOST_VISIBLE, which on this part is write-combined on the CPU
            // side: without a store fence AFTER the flag, the write can sit in
            // a WC buffer until something else evicts it, and "the device
            // never saw the flag" would be a bug in this file rather than a
            // property of the hardware. The read-back is belt and braces --
            // it also proves the store retired.
            std::atomic_thread_fence(std::memory_order_seq_cst);
            (void)ctl[kFlag];
            last = req;
            served.store(last, std::memory_order_relaxed);
        }
    }
};

struct Stats {
    double mean = 0, p50 = 0, p99 = 0, max = 0, min = 0;
};
Stats summarize(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / double(v.size());
    s.p50  = v[v.size() / 2];
    s.p99  = v[size_t(double(v.size()) * 0.99)];
    s.min  = v.front();
    s.max  = v.back();
    return s;
}

struct Harness {
    Gpu&                 g;
    gpu::Pipeline&       pipe;
    gpu::DescriptorPool& desc;
#if defined(CACHEDMOE_ENABLE_VULKAN)
    VkDescriptorSet set = VK_NULL_HANDLE;
#endif
    gpu::GpuBuffer ctl{}, out{}, scratch{};
    gpu::QueryPool stamps;
    gpu::Timeline  done;
    uint64_t       tick = 0;
    double         service_ms = 0.0;
    bool           hot = false;

    // Records: for each round, [burn] -> request -> stamp -> spin -> stamp ->
    // consume. One command buffer, N rounds, exactly the shape of one token's
    // forty layers with the MoE dispatch after each gate.
    Result<void> record(uint32_t rounds, uint32_t spin_limit, uint32_t backoff,
                        uint32_t bg_groups, uint32_t burn_iters, bool timed,
                        uint32_t load_atomic) {
        if (auto r = g.cmd.begin(); !r) return r;
        if (timed)
            if (auto r = g.cmd.reset_queries(stamps, 0, 2 * rounds); !r) return r;
        if (auto r = g.cmd.bind(pipe, set); !r) return r;
        for (uint32_t i = 0; i < rounds; ++i) {
            const uint32_t round = i + 1;
            if (bg_groups) {
                Push b{kModeBurn, round, spin_limit, burn_iters, load_atomic};
                if (auto r = g.cmd.push(pipe, &b, sizeof b); !r) return r;
                if (auto r = g.cmd.dispatch(bg_groups); !r) return r;
                if (auto r = g.cmd.barrier(); !r) return r;
            }
            Push p{kModeRequest, round, spin_limit, backoff, load_atomic};
            if (auto r = g.cmd.push(pipe, &p, sizeof p); !r) return r;
            if (auto r = g.cmd.dispatch(1); !r) return r;
            if (auto r = g.cmd.barrier(); !r) return r;
            if (timed)
                if (auto r = g.cmd.write_timestamp(stamps, 2 * i, false); !r) return r;
            p.mode = kModeSpin;
            if (auto r = g.cmd.push(pipe, &p, sizeof p); !r) return r;
            if (auto r = g.cmd.dispatch(1); !r) return r;
            if (auto r = g.cmd.barrier(); !r) return r;
            if (timed)
                if (auto r = g.cmd.write_timestamp(stamps, 2 * i + 1, true); !r) return r;
            p.mode = kModeConsume;
            if (auto r = g.cmd.push(pipe, &p, sizeof p); !r) return r;
            if (auto r = g.cmd.dispatch(1); !r) return r;
            if (auto r = g.cmd.barrier(); !r) return r;
        }
        return g.cmd.end();
    }

    // Submits, runs the host server alongside, and waits with a deadline.
    // `false` means the queue did not retire inside the deadline, which is the
    // only outcome this probe cannot recover from.
    uint32_t last_served = 0;   // how far the HOST got, for attribution

    Result<bool> run_once(uint32_t rounds, double deadline_ms) {
        std::memset(ctl.host_ptr, 0, kCtlWords * 4);
        std::memset(out.host_ptr, 0, size_t(rounds + 2) * kOutWordsPerRound * 4);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        HostServer srv;
        srv.ctl         = static_cast<volatile uint32_t*>(ctl.host_ptr);
        srv.rounds      = rounds;
        srv.deadline_us = deadline_ms * 1000.0;
        srv.service_ms  = service_ms;
        srv.hot         = hot;
        std::thread th(&HostServer::run, &srv);

        gpu::Submission s;
        s.cmd             = &g.cmd;
        s.signal_timeline = &done;
        s.signal_value    = ++tick;
        auto sub = gpu::submit(g.device, s);
        if (!sub) {
            srv.stop.store(true);
            th.join();
            return std::unexpected(sub.error());
        }

        const auto w = done.wait(tick, std::chrono::milliseconds(int64_t(deadline_ms)));
        srv.stop.store(true);
        th.join();
        last_served = srv.served.load(std::memory_order_relaxed);
        return w.has_value();
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (has_flag(argc, argv, "--help")) {
        std::puts(
            "hostflag_probe [--rounds N] [--bufs N] [--max-spin-us N] [--bg-groups N]\n"
            "               [--burn-iters N] [--tdr-budget-ms N] [--csv PATH]\n"
            "  --rounds         host round trips inside ONE command buffer (default 40 = layers)\n"
            "  --bufs           command buffers in the soak (default 1000)\n"
            "  --max-spin-us    wall-clock cap on ONE device spin (default 20000)\n"
            "  --bg-groups      background workgroups before each spin (default 256)\n"
            "  --burn-iters     arithmetic iterations per background workgroup (default 2000)\n"
            "  --tdr-budget-ms  cap on rounds x max-spin-us (default 1500)\n"
            "The spin cap is calibrated against the real loop, and the SUM over a command\n"
            "buffer is clamped below the TDR window. No wait here is unbounded.");
        return 0;
    }
    const uint32_t rounds    = uint32_t(arg_u64(argc, argv, "--rounds", 40));
    const uint32_t bufs      = uint32_t(arg_u64(argc, argv, "--bufs", 1000));
    const uint32_t bg_groups = uint32_t(arg_u64(argc, argv, "--bg-groups", 256));
    const uint32_t burn      = uint32_t(arg_u64(argc, argv, "--burn-iters", 2000));
    uint64_t       max_spin_us = arg_u64(argc, argv, "--max-spin-us", 20000);
    const uint64_t tdr_ms      = arg_u64(argc, argv, "--tdr-budget-ms", 1500);
    const std::string csv_path = arg_after(argc, argv, "--csv", "");

    // TDR counts the PACKET, not the dispatch: rounds x the per-spin cap is
    // what the driver's timer sees, so that product is the thing to clamp.
    // This is why the cap here is tens of milliseconds and not the half second
    // a single isolated spin could afford.
    if (rounds && max_spin_us * rounds > tdr_ms * 1000) {
        const uint64_t clamped = (tdr_ms * 1000) / rounds;
        std::printf("note: --max-spin-us %llu x %u rounds = %.0f ms exceeds the %llu ms TDR\n"
                    "      budget; clamping to %llu us a spin.\n",
                    (unsigned long long)max_spin_us, rounds,
                    double(max_spin_us * rounds) / 1000.0, (unsigned long long)tdr_ms,
                    (unsigned long long)clamped);
        max_spin_us = std::max<uint64_t>(clamped, 1);
    }

    Gpu g;
    if (auto r = g.create(); !r) {
        std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
        return 1;
    }
    std::printf("device: %s\n", g.device.caps().to_string().c_str());

    gpu::Pipeline           pipe;
    gpu::PipelineLayoutSpec lspec;
    lspec.storage_buffers    = 3;
    lspec.push_constant_size = sizeof(Push);
    if (auto r = pipe.create(g.device, g.shader_dir + "/probe_hostflag.spv", lspec); !r) {
        std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
        return 1;
    }
    gpu::DescriptorPool desc;
    if (auto r = desc.create(g.device, 1, 3); !r) {
        std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
        return 1;
    }

    Harness h{g, pipe, desc};
    auto    ctl = g.alloc.allocate_host_coherent(kCtlWords * 4);
    auto    out = g.alloc.allocate_host_coherent(uint64_t(rounds + 2) * kOutWordsPerRound * 4);
    auto    scr = g.alloc.allocate_host_coherent(4096);
    if (!ctl || !out || !scr) {
        std::fprintf(stderr, "probe: cannot allocate the control block\n");
        return 1;
    }
    h.ctl = *ctl;
    h.out = *out;
    h.scratch = *scr;

    std::vector<gpu::BufferBinding> binds(3);
    binds[0].binding = 0; binds[0].buffer = h.ctl.buffer;
    binds[1].binding = 1; binds[1].buffer = h.out.buffer;
    binds[2].binding = 2; binds[2].buffer = h.scratch.buffer;
    auto set = desc.allocate(pipe, binds);
    if (!set) {
        std::fprintf(stderr, "probe: %s\n", set.error().str().c_str());
        return 1;
    }
    h.set = *set;
    if (auto r = h.done.create(g.device, 0); !r) {
        std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
        return 1;
    }
    const bool timed = g.timed;
    if (auto r = h.stamps.create(g.device, 2 * rounds); !r) {
        std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
        return 1;
    }
    const double tick_ns = g.device.caps().timestamp_period_ns;

    Csv csv;
    csv.open(csv_path,
             "probe,backoff,bg_groups,service_ms,hot,bufs,rounds,ok,timeouts,bad,served,late,early,"
             "mean_us,p50_us,p99_us,max_us,wall_ms_per_buf");

    // --- 0. calibrate the spin loop ----------------------------------------
    // Price ONE iteration of the loop under test, per backoff, by spinning on
    // a round the host never serves. Start small and scale only if the
    // measurement was too short to trust: every step is bounded by the
    // previous measurement, so this cannot run away.
    const uint32_t backoffs[2] = {0, 64};
    double         ns_per_iter[2] = {0, 0};
    uint32_t       spin_cap[2] = {1, 1};
    std::printf("\ncalibration (spin cap %llu us a round, %llu ms of TDR budget for %u rounds)\n",
                (unsigned long long)max_spin_us, (unsigned long long)tdr_ms, rounds);
    for (int b = 0; b < 2; ++b) {
        uint32_t iters = 20000;
        double   secs  = 0;
        for (int attempt = 0; attempt < 4; ++attempt) {
            if (auto r = h.record(1, iters, backoffs[b], 0, burn, false, 0); !r) {
                std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
                return 1;
            }
            std::memset(h.ctl.host_ptr, 0, kCtlWords * 4);
            const auto t0 = Clock::now();
            if (auto r = gpu::submit_and_wait(g.device, g.cmd); !r) {
                std::fprintf(stderr, "probe: %s\n", r.error().str().c_str());
                return 1;
            }
            secs = std::chrono::duration<double>(Clock::now() - t0).count();
            if (secs > 2e-3) break;   // 2 ms is enough signal; stop scaling
            iters *= 8;
        }
        ns_per_iter[b] = secs * 1e9 / double(iters);
        const double want = double(max_spin_us) * 1000.0 / ns_per_iter[b];
        spin_cap[b] = uint32_t(std::clamp(want, 1.0, 2.0e9));
        std::printf("  backoff %3u: %7.1f ns an iteration -> spin cap %u iterations (%llu us)\n",
                    backoffs[b], ns_per_iter[b], spin_cap[b],
                    (unsigned long long)max_spin_us);
    }

    struct Cell {
        const char* name;
        uint32_t    backoff;
        uint32_t    bg;
        double      service;
        uint32_t    bufs;
        uint32_t    atomic;
        bool        hot;
    };
    const std::vector<Cell> cells = {
        {"busy/idle",       0,  0,         0.0, bufs,                      0, false},
        {"busy/idle atomic",0,  0,         0.0, std::max(bufs / 200, 1u),  1, false},
        {"backoff/idle",    64, 0,         0.0, bufs,                      0, false},
        {"busy/loaded",     0,  bg_groups, 0.0, std::max(bufs / 4, 1u),    0, false},
        {"backoff/loaded",  64, bg_groups, 0.0, std::max(bufs / 4, 1u),    0, false},
        {"busy 1ms cold",   0,  0,         1.0, std::max(bufs / 20, 1u),   0, false},
        {"busy 5ms cold",   0,  0,         5.0, std::max(bufs / 50, 1u),   0, false},
        {"busy 1ms HOT",    0,  0,         1.0, std::max(bufs / 20, 1u),   0, true},
        {"busy 5ms HOT",    0,  0,         5.0, std::max(bufs / 50, 1u),   0, true},
        {"busy/idle HOT",   0,  0,         0.0, std::max(bufs / 4, 1u),    0, true},
        {"busy/loaded HOT", 0,  bg_groups, 0.0, std::max(bufs / 4, 1u),    0, true},
    };

    std::printf("\nhost-flag round trip: %u rounds inside ONE command buffer\n", rounds);
    // `served` separates the two directions. A timeout with served == rounds
    // means the HOST saw every request and the DEVICE never saw the flag;
    // served == 0 means the device's store never reached the host at all.
    // They are different failures and only one of them has a fix.
    // `late` is the whole experiment in one column: a round that ended with
    // spins > 0 OBSERVED a store the host made while the kernel was already
    // running. A round that ends with spins == 0 saw the flag on its very
    // first load -- the host won a race against the dispatch launch, and no
    // mid-kernel visibility was demonstrated at all. If `late` is 0 across
    // the table, the mechanism does not exist here, whatever the latency says.
    std::printf("  %-16s %6s %5s %8s %7s %10s %8s %8s %8s %8s\n", "cell", "bufs", "ok", "to/bad",
                "served", "late/ok", "mean us", "p50 us", "p99 us", "ms/buf");
    bool fatal = false;
    for (const Cell& c : cells) {
        const int bi = (c.backoff == 0) ? 0 : 1;
        h.service_ms = c.service;
        h.hot        = c.hot;
        // Deadline: every spin at its cap, plus the host's own service time,
        // plus slack. A queue that misses this is reported, never waited on.
        const double deadline_ms =
            double(rounds) * (double(max_spin_us) / 1000.0 + c.service) + 2000.0;
        if (auto r = h.record(rounds, spin_cap[bi], c.backoff, c.bg, burn, timed, c.atomic); !r) {
            std::fprintf(stderr, "  record failed: %s\n", r.error().str().c_str());
            return 1;
        }
        // `late` is the whole experiment in one counter: a round that ended
        // with spins > 0 is a round where the spin loop OBSERVED a store the
        // host made while the kernel was already running. A round that ends
        // with spins == 0 saw the flag on its very first load -- the host won
        // a race against the dispatch launch, and no mid-kernel visibility was
        // demonstrated at all. If `late` is zero everywhere, the mechanism
        // does not exist on this hardware whatever the latency column says.
        uint32_t ok = 0, timeouts = 0, bad = 0, ran = 0, served = 0, late = 0, early = 0;
        std::vector<double> lat;
        lat.reserve(size_t(c.bufs) * rounds);
        std::vector<std::vector<double>> by_round(rounds);
        double wall_total = 0;
        for (uint32_t i = 0; i < c.bufs; ++i) {
            const auto t0 = Clock::now();
            auto       r  = h.run_once(rounds, deadline_ms);
            if (!r) {
                std::fprintf(stderr, "  submit/wait failed: %s\n", r.error().str().c_str());
                fatal = true;
                break;
            }
            wall_total += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            ++ran;
            served += h.last_served;
            if (!*r) {
                std::fprintf(stderr,
                             "  DEADLINE: the queue did not retire in %.0f ms. Stopping.\n",
                             deadline_ms);
                fatal = true;
                break;
            }
            const uint32_t* o = static_cast<const uint32_t*>(h.out.host_ptr);
            bool            clean = true;
            for (uint32_t rr = 1; rr <= rounds; ++rr) {
                const uint32_t* row = o + size_t(rr) * kOutWordsPerRound;
                if (row[kOutTimeout]) { ++timeouts; clean = false; }
                else if (row[kOutSpins]) ++late;
                else ++early;
                if (row[kOutBad])     { ++bad; clean = false; }
            }
            if (clean) ++ok;
            if (timed && i > 0) {   // skip the first buffer: cold pipeline
                auto ticks = h.stamps.read_range(0, 2 * rounds);
                if (ticks)
                    for (uint32_t rr = 0; rr < rounds; ++rr) {
                        const uint64_t a = (*ticks)[2 * rr], b2 = (*ticks)[2 * rr + 1];
                        if (b2 > a) {
                            const double us = double(b2 - a) * tick_ns / 1000.0;
                            lat.push_back(us);
                            by_round[rr].push_back(us);
                        }
                    }
            }
        }
        // Per-round-index breakdown. The aggregate above is bimodal, and the
        // two readings of that are very different: a tail concentrated in the
        // first rounds is a warm-up artefact and diagnosable, a tail spread
        // evenly over the forty is a property of the signalling path and is
        // what a token would actually pay.
        if (!by_round.empty()) {
            const size_t nb = by_round[0].size();
            if (nb) {
                double head = 0, tail = 0;
                uint32_t slow_head = 0, slow_tail = 0;
                for (size_t k = 0; k < nb; ++k) {
                    head += by_round[0][k];
                    if (by_round[0][k] > 1000.0) ++slow_head;
                }
                const size_t li = rounds - 1;
                for (size_t k = 0; k < by_round[li].size(); ++k) {
                    tail += by_round[li][k];
                    if (by_round[li][k] > 1000.0) ++slow_tail;
                }
                std::printf("      round 1: %.1f us mean, %u/%zu over 1 ms   "
                            "round %u: %.1f us mean, %u/%zu over 1 ms\n",
                            head / double(nb), slow_head, nb, rounds,
                            by_round[li].empty() ? 0.0 : tail / double(by_round[li].size()),
                            slow_tail, by_round[li].size());
            }
        }
        const Stats  s = summarize(lat);
        const double per_buf = ran ? wall_total / double(ran) : 0.0;
        char lateb[32];
        std::snprintf(lateb, sizeof lateb, "%u/%u", late, late + early);
        std::printf("  %-16s %6u %5u %4u/%-3u %7.2f %10s %8.1f %8.1f %8.1f %8.2f\n", c.name, ran,
                    ok, timeouts, bad, ran ? double(served) / double(ran) : 0.0, lateb, s.mean,
                    s.p50, s.p99, per_buf);
        csv.row("roundtrip,%u,%u,%.1f,%d,%u,%u,%u,%u,%u,%.2f,%u,%u,%.3f,%.3f,%.3f,%.3f,%.4f",
                c.backoff, c.bg, c.service, c.hot ? 1 : 0, ran, rounds, ok, timeouts, bad,
                ran ? double(served) / double(ran) : 0.0, late, early, s.mean, s.p50, s.p99,
                s.max, per_buf);
        if (fatal) break;
    }

    std::printf("\nverdict inputs\n");
    std::printf("  today's gate round trip (STATUS section 3 item 53): 400 us a layer,\n"
                "  40 layers = 16.0 ms a token out of a 102 ms hot step.\n");
    std::printf("  the saving is 40 x (400 - measured) us a token; 100 us off the round trip\n"
                "  is 4.0 ms a token = 3.9%% of the hot step, and the halving rule takes half.\n");
    if (fatal)
        std::printf("  A DEADLINE OR A SUBMIT FAILURE WAS OBSERVED -> NO-GO whatever the latency.\n");

    csv.close(csv_path);
    h.stamps.destroy();
    h.done.destroy();
    g.alloc.free(h.ctl);
    g.alloc.free(h.out);
    g.alloc.free(h.scratch);
    desc.destroy();
    pipe.destroy();
    return fatal ? 2 : 0;
}
