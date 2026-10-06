// Choosing WHICH drive a read goes to (Track D2, docs/p4_dual_source.md).
//
// STATUS.md §4 says the stall is the drive and nothing else, so the only lever
// left that is not "make the drive faster" is "use a second drive". The
// checkpoint is read-only and immutable, so a byte-identical copy on another
// volume is a second, independent source for exactly the same bytes: a P0 that
// would have queued behind three other P0s on D: can be served by E: instead,
// and the two drives' bandwidths add.
//
// The drives are NOT symmetric here -- D: is a 4.6 GB/s internal NVMe, E: is a
// 1.0 GB/s USB enclosure -- so round-robin would be strictly worse than not
// mirroring at all (every other request would take 4.6x as long). The rule is
// WEIGHTED LEAST-OUTSTANDING-BYTES: send the request to the source that will
// finish it soonest given what it is already carrying,
//
//     argmin_s  (outstanding_bytes[s] + bytes) / weight[s]
//
// with `weight` proportional to each source's measured 4 MiB random-read rate.
// The quantity being minimised is time, not bytes: `(queue + me) / rate` is an
// estimate of when this request would land. In steady state that splits the
// byte stream in proportion to the weights -- 4.6 : 1.0 gives E: ~18% -- which
// is the split that keeps both drives busy and neither one the tail.
//
// Ignoring the weights (comparing raw outstanding bytes) turns this into
// least-outstanding-bytes, which converges to a 50/50 split and is ~2x slower
// end to end than the primary alone. `io.source_router_respects_weights` is the
// test that fails when that happens.
#pragma once

#include "core/namespace.h"
#include "core/runtime_environment.h"

#include <cstdint>
#include <span>
#include <string>

namespace cachedmoe::storage {

// More than this is not useful: the machine has two drives, and the third and
// fourth entries exist only so a future NAS/second-USB experiment does not need
// a format change.
inline constexpr uint32_t kMaxIoSources = 4;

// Weighted least-outstanding-bytes. `candidate_mask` bit s means source s holds
// this file; sources outside the mask are never chosen. Ties -- including the
// ordinary single-source case -- go to the lowest index, so a run with no
// mirror picks source 0 every time and behaves exactly as it did before.
//
// Returns kMaxIoSources when the mask is empty (the caller must fall back to
// the primary).
inline uint32_t pick_source(std::span<const double> weights,
                            std::span<const uint64_t> outstanding,
                            uint32_t candidate_mask, uint64_t bytes) {
    uint32_t best = kMaxIoSources;
    double best_eta = 0.0;
    const uint32_t n = static_cast<uint32_t>(weights.size() < outstanding.size()
                                                 ? weights.size() : outstanding.size());
    for (uint32_t s = 0; s < n; ++s) {
        if (!(candidate_mask & (1u << s))) continue;
        const double w = weights[s] > 0.0 ? weights[s] : 1e-9;
        const double eta = (static_cast<double>(outstanding[s]) + static_cast<double>(bytes)) / w;
        if (best == kMaxIoSources || eta < best_eta) { best = s; best_eta = eta; }
    }
    return best;
}

// --- runtime health: dropping a source that stops answering ------------------
//
// docs/p4_e_drive_diag.md §5.2. `open_mirror` succeeding says the files are
// there, not that the drive can serve them: Track DX watched E: open all 48
// handles and then stop answering 17 seconds in, and because the pinned load
// runs through the router, that killed the engine. A mirror is an optimisation
// (docs/p4_dual_source.md §2), so the right answer to "this source is failing"
// is to stop using it, not to fail the run.
//
// The router already chooses by candidate mask, so "drop it" is one AND on the
// submit path and nothing else -- and taking it back, once the drive answers
// again (IoEngine::readmit_source), is clearing one bit.
//
// CONSECUTIVE, not cumulative. One bad read is a retry; a drive that has fallen
// off the bus fails every read after it. A cumulative counter would eventually
// drop a perfectly healthy source on a long enough run, which is why
// `io.source_health_drops_a_mirror_that_keeps_failing` checks that a success in
// between puts the budget back.
//
// Source 0 is never dropped: it is the only copy the run is guaranteed to have,
// and its failures are the run's failures -- reported by the read that failed,
// not swallowed here.
inline constexpr uint32_t kDefaultSourceErrorBudget = configuration::kSourceErrorBudget;

class SourceHealth {
public:
    explicit SourceHealth(uint32_t budget = kDefaultSourceErrorBudget)
        : budget_(budget ? budget : 1) {}

    // Returns true when THIS error is the one that takes the source out, so the
    // caller logs once rather than once per failed request.
    bool note_error(uint32_t s) {
        if (s == 0 || s >= kMaxIoSources) return false;
        if (dropped_ & (1u << s)) return false;
        if (++consecutive_[s] >= budget_) { dropped_ |= 1u << s; return true; }
        return false;
    }
    void note_success(uint32_t s) {
        if (s > 0 && s < kMaxIoSources) consecutive_[s] = 0;
    }
    // The startup gate (a failed health probe) uses the same switch.
    void drop(uint32_t s) { if (s > 0 && s < kMaxIoSources) dropped_ |= 1u << s; }
    // Back in, with the full budget: the drive answers again (IoEngine's
    // readmit_source), so its old failures say nothing about the next read.
    void readmit(uint32_t s) {
        if (s > 0 && s < kMaxIoSources) { dropped_ &= ~(1u << s); consecutive_[s] = 0; }
    }

    bool dropped(uint32_t s) const {
        return s < kMaxIoSources && (dropped_ & (1u << s)) != 0;
    }
    uint32_t consecutive_errors(uint32_t s) const {
        return s < kMaxIoSources ? consecutive_[s] : 0;
    }
    uint32_t budget() const { return budget_; }

    // `candidates` with the dropped sources removed. Bit 0 always survives.
    uint32_t live_mask(uint32_t candidates) const { return candidates & ~dropped_; }

    void reset() {
        dropped_ = 0;
        for (uint32_t& c : consecutive_) c = 0;
    }

private:
    uint32_t budget_;
    uint32_t consecutive_[kMaxIoSources] = {};
    uint32_t dropped_ = 0;
};

// --- resting a mirror that runs hot ------------------------------------------
//
// The USB4 enclosure idles at ~63 C, reaches ~74 C over an 8-turn striped chat,
// and has fallen off the bus when hot (its warning threshold is 90 C). A mirror
// at or above `hot_c` gets no new reads until it is back at or below `cool_c`.
// The router already chooses by candidate mask, so resting is the same one AND
// as a drop, only it comes back. Source 0 is never rested: it is the copy the
// run cannot do without.
inline constexpr int kDefaultMirrorHotC = configuration::kMirrorHotCelsius;
inline constexpr int kMirrorCoolDropC   = 8;

struct ThermalGate {
    int      hot_c   = kDefaultMirrorHotC;
    int      cool_c  = kDefaultMirrorHotC - kMirrorCoolDropC;
    uint32_t resting = 0;              // bit s: source s is cooling off

    // Folds one reading in; true when source s changed state, so the caller
    // logs each transition once.
    bool update(uint32_t s, int temp_c) {
        if (s == 0 || s >= kMaxIoSources) return false;
        const uint32_t bit = 1u << s;
        const bool was = (resting & bit) != 0;
        if (!was && temp_c >= hot_c) resting |= bit;
        else if (was && temp_c <= cool_c) resting &= ~bit;
        return was != ((resting & bit) != 0);
    }
    uint32_t live_mask(uint32_t candidates) const { return candidates & ~resting; }
};

// --- Track D6: keeping an idle mirror awake ---------------------------------
//
// docs/p4_dual_source.md §10. The D5 hypothesis was that the mirror's 5-6x mean
// latency is a POWER state: it gets few bytes, so it goes idle between bursts,
// so every burst pays a USB4/NVMe wake-up (measured standalone at up to
// 1,093 ms on a first read after a few hundred ms of idle), so the router gives
// it even less. If that were true, a 4 KiB read every `idle_ms` would break the
// loop.
//
// The scheduler below is the whole policy, pulled out of IoEngine so it can be
// tested on the CPU with no drive, no backend and no model:
//
//   * only a non-primary source is ever poked -- the primary is the drive the
//     run is already hammering;
//   * at most ONE keep-alive read per source is outstanding, so a stuck one
//     cannot fan out into a queue of its own;
//   * NEVER while a real request to that source is in flight. A real read keeps
//     the link awake by itself, and a 4 KiB read squeezed in beside a P0 burst
//     is pure contention on the thing we are trying to make faster. That is the
//     line `io.keepalive_never_races_a_real_request` mutates.
inline constexpr uint32_t kDefaultKeepAliveMs = 15;

struct KeepAliveState {
    bool     has_file       = false;   // this source has an open shard to poke
    bool     dropped        = false;   // SourceHealth took it out of the router
    bool     probe_inflight = false;   // our own 4 KiB read is still outstanding
    uint32_t real_inflight  = 0;       // routed requests submitted and not finished
    int64_t  last_activity_ns = 0;     // last completion on this source, steady clock
};

// True when source `s` should be poked right now. `idle_ns <= 0` means the
// feature is off and the answer is always false, so the default build path is
// one comparison.
inline bool keepalive_due(const KeepAliveState& s, int64_t now_ns, int64_t idle_ns) {
    if (idle_ns <= 0) return false;
    if (!s.has_file || s.dropped) return false;
    if (s.probe_inflight) return false;
    if (s.real_inflight != 0) return false;
    return (now_ns - s.last_activity_ns) >= idle_ns;
}

// Per-source accounting, reported through IoStats and so through the serve
// endpoint's status.json.
struct SourceStats {
    std::string root;              // the model directory this source reads from
    double   weight   = 1.0;       // GB/s, measured or CACHEDMOE_MIRROR_WEIGHTS
    uint64_t requests = 0;         // routed requests that have completed
    uint64_t bytes    = 0;         // bytes those requests moved
    uint64_t lat_ns_sum = 0;       // submit -> last chunk, summed
    uint64_t outstanding_bytes = 0;  // routed but not yet finished
    uint32_t inflight_requests = 0;
    uint64_t errors   = 0;         // routed requests that came back failed
    bool     dropped  = false;     // taken out of the router (SourceHealth)
    uint32_t readmits = 0;         // times it answered again and came back
    uint64_t failovers = 0;        // of those errors, re-read from the primary
    // Track ST: chunks this source served as one share of a STRIPED P0 request
    // (CACHEDMOE_MIRROR_STRIPE). For a striped request `requests` counts each
    // source that carried at least one chunk, `bytes` is that source's share
    // and the latencies run from submit to that source's LAST chunk -- so the
    // slower drive's tail is visible per source instead of being averaged away.
    uint64_t stripe_chunks = 0;

    // --- Track D6 ----------------------------------------------------------
    // `requests`/`lat_ns_sum` above mix the classes that are routed (P0 and
    // P3), and P3 backfill latency is queue-dominated -- it waits for every P0
    // to clear. D5 read src[1]'s 149 ms mean as "the drive is slow" when most
    // of that mass was backfill sitting in the engine's own queue, so the P0
    // path now carries its own counters and the two can never be confused
    // again (docs/p4_dual_source.md §10.1).
    uint64_t p0_requests = 0, p0_bytes = 0, p0_lat_ns_sum = 0;
    uint64_t p0_queue_wait_ns_sum = 0, p0_service_ns_sum = 0, p0_copy_ns_sum = 0;
    // How long this source was carrying NOTHING before a request arrived,
    // counted only for gaps at or above the keep-alive window. This is the
    // quantity the power-state hypothesis is about, and it is measured whether
    // or not keep-alive is enabled -- the off arm is the evidence.
    uint64_t idle_gaps = 0, idle_gap_ns_sum = 0, idle_gap_ns_max = 0;
    // Keep-alive reads issued / that came back / that the backend refused.
    uint64_t keepalive_reads = 0, keepalive_done = 0, keepalive_refused = 0;
    // ThermalGate: the drive's last and highest reading (-1 = no sensor), how
    // often it was rested and whether it is resting now.
    int32_t  temp_c = -1, temp_max_c = -1;
    uint32_t rests = 0;
    bool     resting = false;

    double mean_latency_ms() const {
        return requests ? lat_ns_sum / 1e6 / static_cast<double>(requests) : 0.0;
    }
    double p0_mean_latency_ms() const {
        return p0_requests ? p0_lat_ns_sum / 1e6 / static_cast<double>(p0_requests) : 0.0;
    }
    double idle_gap_mean_ms() const {
        return idle_gaps ? idle_gap_ns_sum / 1e6 / static_cast<double>(idle_gaps) : 0.0;
    }
};

}  // namespace cachedmoe::storage
