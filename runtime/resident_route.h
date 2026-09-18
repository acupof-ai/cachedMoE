// Track Y -- resident-only routing (docs/p4_resident_routing.md).
//
// The decode step's MoE dispatch normally waits for the gate's six experts to
// be resident: a miss is an NVMe read the GPU stalls on. This turns that stall
// into a quality loss instead -- the experts that are NOT resident are dropped,
// the gate weights of the ones that are left are renormalised over the kept
// subset, and the dropped ones are handed to the background fetcher so the next
// token may have them. Nothing waits on the drive.
//
// Why renormalising is just a smaller gather
// ------------------------------------------
// `inference/model.py`'s `Gate.forward` gathers the top-k raw scores, divides by
// their sum (`norm_topk_prob = true`) and multiplies by `routed_scaling_factor`.
// gpu/shaders/gate.slang writes exactly that into the weight buffer, so the
// weights this function is handed are `score_s / sum(score) * 1.5`. Renormalising
// over the kept subset is therefore one scalar: multiply every kept weight by
// `sum(all w) / sum(kept w)`. The scaling factor divides out, which is why it
// never appears here, and the shared expert -- which the model adds outside the
// gate -- is untouched.
//
// Why the skipped slots are not removed
// ----------------------------------------
// gpu::MoeRunner has a fixed seven slots (six routed + the fp8 shared one) and
// runtime/moe_bridge.cpp asserts `topk + 1 == slots`, so a short list is not
// expressible. A skipped slot instead keeps a VALID expert id -- the first kept
// slot's, whose pointer-table row is resident -- and a routing weight of exactly
// zero. gpu/shaders/moe_gateup.slang applies the route weight in dispatch A
// (`RouteW[m * num_slots + slot]`), so `h` for that slot is an exact zero and
// dispatch B adds nothing. The duplicate costs the arithmetic of one expert, and
// this track is not trying to save arithmetic: it is trying not to wait on a
// drive.
#pragma once

#include <cstdint>
#include <span>

namespace deepmoe::runtime {

struct ResidentRoute {
    uint32_t kept      = 0;      // how many of the top-k were resident
    float    mass_lost = 0.0f;   // 1 - sum(kept w) / sum(all w), in [0, 1]
    bool     shared_only = false;   // kept == 0: only the shared expert contributes
};

// `resident[s]` says whether top-k slot `s`'s expert is in the cache.
// `ids_out`/`w_out` are `topk` long and are what the MoE dispatch should use.
//
// When nothing is resident there is no valid id to borrow, so every weight is
// zero and `ids_out` is filled with `fill_id`, which the caller must have
// checked is resident in this layer; `shared_only` is set either way so the
// caller can count it.
inline ResidentRoute resident_route(const uint32_t* ids_in, const float* w_in, uint32_t topk,
                                    std::span<const uint8_t> resident, uint32_t fill_id,
                                    uint32_t* ids_out, float* w_out) {
    ResidentRoute r;
    double sum_all = 0.0, sum_kept = 0.0;
    uint32_t first_kept = topk;
    for (uint32_t s = 0; s < topk; ++s) {
        sum_all += w_in[s];
        if (s < resident.size() && resident[s]) {
            if (first_kept == topk) first_kept = s;
            sum_kept += w_in[s];
            ++r.kept;
        }
    }
    if (r.kept == 0) {
        r.shared_only = true;
        r.mass_lost   = 1.0f;
        for (uint32_t s = 0; s < topk; ++s) { ids_out[s] = fill_id; w_out[s] = 0.0f; }
        return r;
    }
    // sum_all is a sum of non-negative gate weights and sum_kept is a non-empty
    // subset of it, so this is finite and >= 1.
    const double scale = sum_all > 0.0 ? sum_all / sum_kept : 1.0;
    r.mass_lost = sum_all > 0.0 ? static_cast<float>(1.0 - sum_kept / sum_all) : 0.0f;
    const uint32_t borrow = ids_in[first_kept];
    for (uint32_t s = 0; s < topk; ++s) {
        const bool keep = s < resident.size() && resident[s];
        ids_out[s] = keep ? ids_in[s] : borrow;
        w_out[s]   = keep ? static_cast<float>(w_in[s] * scale) : 0.0f;
    }
    return r;
}

// The background miss queue's staleness cutoff (docs/p4_resident_routing.md
// section 8). The first measurement's P3 queue was unbounded: the drive saw
// 19,356 experts at a mean latency of 5.6 s against a 100 ms step, so what it
// was fetching was 56 steps out of date by the time it landed. Bounding the
// queue to the most recent `keep_steps` steps and draining it newest-first is
// what makes the drive spend its 4 GB/s on demand that is still current.
// Returns the oldest step id worth keeping; anything below it is dropped.
inline uint64_t resident_queue_cutoff(uint64_t now_step, uint32_t keep_steps) {
    if (keep_steps == 0) keep_steps = 1;
    return now_step >= keep_steps ? now_step - keep_steps + 1 : 0;
}

// What a run spent in resident-only mode. Printed at the end of a run.
struct ResidentRouteStats {
    uint64_t layers        = 0;   // layer-steps routed in resident-only mode
    uint64_t requested     = 0;   // routed experts asked for (topk * layers)
    uint64_t served        = 0;   // ... of which resident, so actually computed
    uint64_t skipped       = 0;   // ... of which dropped
    uint64_t shared_only   = 0;   // layer-steps with 0 of topk resident
    double   mass_lost_sum = 0.0; // sum over layer-steps of the dropped gate mass
    uint64_t bg_enqueued   = 0;   // dropped experts handed to the P3 fetcher
    uint64_t bg_refused    = 0;   // ... that the cache would not admit right now
    uint64_t bg_stale      = 0;   // ... dropped unissued: older than the queue window
    uint64_t bg_depth_sum  = 0;   // queue depth sampled once per layer
    uint64_t bg_depth_n    = 0;
    uint32_t bg_depth_peak = 0;
    uint64_t stall1_p0     = 0;   // stall1: single highest-weight experts fetched at P0
    double   stall1_ms     = 0.0; // ... and the wall time that cost
    double bg_depth_mean() const { return bg_depth_n ? double(bg_depth_sum) / double(bg_depth_n) : 0.0; }
    double mass_lost() const { return layers ? mass_lost_sum / double(layers) : 0.0; }
    double served_frac() const { return requested ? double(served) / double(requested) : 0.0; }
};

}  // namespace deepmoe::runtime
