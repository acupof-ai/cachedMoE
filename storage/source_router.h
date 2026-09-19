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

#include <cstdint>
#include <span>
#include <string>

namespace deepmoe::storage {

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

// Per-source accounting, reported through IoStats and so through the serve
// endpoint's status.json.
struct SourceStats {
    std::string root;              // the model directory this source reads from
    double   weight   = 1.0;       // GB/s, measured or DEEPMOE_MIRROR_WEIGHTS
    uint64_t requests = 0;         // routed requests that have completed
    uint64_t bytes    = 0;         // bytes those requests moved
    uint64_t lat_ns_sum = 0;       // submit -> last chunk, summed
    uint64_t outstanding_bytes = 0;  // routed but not yet finished
    uint32_t inflight_requests = 0;

    double mean_latency_ms() const {
        return requests ? lat_ns_sum / 1e6 / static_cast<double>(requests) : 0.0;
    }
};

}  // namespace deepmoe::storage
