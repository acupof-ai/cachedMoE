// Track Y: the resident-only routing arithmetic (runtime/resident_route.h,
// docs/p4_resident_routing.md §6 step 1).
//
// These cases are meant to be mutation-worthy: a renormalisation that divides
// by the wrong sum, forgets the scaling factor, keeps the dropped weights, or
// renormalises to 1 instead of to the gate's own total must fail here.
#include <cmath>
#include <vector>

#include "runtime/resident_route.h"
#include "tests/test_framework.h"

using namespace deepmoe;
using namespace deepmoe::runtime;

namespace {

// The gate's weights as gpu/shaders/gate.slang writes them: raw scores
// normalised over the top-k and multiplied by routed_scaling_factor, which for
// V4.1-Flash is 1.5 (inference/model.py Gate, norm_topk_prob = true).
constexpr float kRouteScale = 1.5f;

std::vector<float> gate_weights(const std::vector<float>& scores) {
    double sum = 0.0;
    for (float s : scores) sum += s;
    std::vector<float> w;
    for (float s : scores) w.push_back(static_cast<float>(s / sum * kRouteScale));
    return w;
}

}  // namespace

DEEPMOE_TEST(resident_route, all_resident_is_the_identity) {
    const std::vector<float> sc{4.0f, 3.0f, 2.0f, 1.5f, 1.0f, 0.5f};
    const std::vector<float> w = gate_weights(sc);
    const uint32_t ids[6]{10, 11, 12, 13, 14, 15};
    const uint8_t  res[6]{1, 1, 1, 1, 1, 1};
    uint32_t out_ids[6];
    float    out_w[6];
    const ResidentRoute r = resident_route(ids, w.data(), 6,
                                           std::span<const uint8_t>(res, 6), 0, out_ids, out_w);
    CHECK_EQ(r.kept, 6u);
    CHECK(!r.shared_only);
    CHECK_CLOSE(r.mass_lost, 0.0f, 1e-6f);
    for (uint32_t i = 0; i < 6; ++i) {
        CHECK_EQ(out_ids[i], ids[i]);
        CHECK_CLOSE(out_w[i], w[i], 1e-6f);
    }
}

DEEPMOE_TEST(resident_route, renormalises_over_the_kept_subset) {
    // Ranks 2 and 5 (0-based 1 and 4) are missing. What must come out is the
    // gate applied to a four-entry gather: score / sum(kept scores) * 1.5.
    const std::vector<float> sc{4.0f, 3.0f, 2.0f, 1.5f, 1.0f, 0.5f};
    const std::vector<float> w = gate_weights(sc);
    const uint32_t ids[6]{10, 11, 12, 13, 14, 15};
    const uint8_t  res[6]{1, 0, 1, 1, 0, 1};
    uint32_t out_ids[6];
    float    out_w[6];
    const ResidentRoute r = resident_route(ids, w.data(), 6,
                                           std::span<const uint8_t>(res, 6), 0, out_ids, out_w);
    REQUIRE_EQ(r.kept, 4u);
    CHECK(!r.shared_only);

    const double kept_sc = 4.0 + 2.0 + 1.5 + 0.5;
    const double all_sc  = 4.0 + 3.0 + 2.0 + 1.5 + 1.0 + 0.5;
    // mass_lost is defined on the gate's own weights: the scores of the absent
    // experts over the total. docs/p4_resident_routing.md: independent of 1.5.
    CHECK_CLOSE(r.mass_lost, static_cast<float>(1.0 - kept_sc / all_sc), 1e-5f);

    // The kept weights are the four-way gate, and they sum to the scaling
    // factor -- not to 1, which is the mutation this checks for.
    double sum_out = 0.0;
    for (uint32_t i = 0; i < 6; ++i) sum_out += out_w[i];
    CHECK_CLOSE(static_cast<float>(sum_out), kRouteScale, 1e-5f);

    const double expect[6]{4.0 / kept_sc * kRouteScale, 0.0, 2.0 / kept_sc * kRouteScale,
                           1.5 / kept_sc * kRouteScale, 0.0, 0.5 / kept_sc * kRouteScale};
    for (uint32_t i = 0; i < 6; ++i)
        CHECK_CLOSE(out_w[i], static_cast<float>(expect[i]), 1e-5f);

    // Ratios between kept experts are untouched by the renormalisation -- a
    // renormalisation that scaled each weight by a different factor would pass
    // the sum check above and fail here.
    CHECK_CLOSE(out_w[0] / out_w[2], w[0] / w[2], 1e-5f);
    CHECK_CLOSE(out_w[3] / out_w[5], w[3] / w[5], 1e-5f);
}

DEEPMOE_TEST(resident_route, skipped_slots_are_zero_weighted_and_point_at_a_live_row) {
    // The MoE runner has a fixed seven slots, so a skipped slot keeps a valid
    // (resident) expert id and a weight of exactly zero; gpu/shaders/
    // moe_gateup.slang multiplies h by that weight, so its contribution is an
    // exact zero and the pointer table is never indexed at a dead row.
    const std::vector<float> w = gate_weights({4.0f, 3.0f, 2.0f, 1.5f, 1.0f, 0.5f});
    const uint32_t ids[6]{10, 11, 12, 13, 14, 15};
    const uint8_t  res[6]{0, 0, 1, 0, 1, 0};
    uint32_t out_ids[6];
    float    out_w[6];
    const ResidentRoute r = resident_route(ids, w.data(), 6,
                                           std::span<const uint8_t>(res, 6), 999, out_ids, out_w);
    REQUIRE_EQ(r.kept, 2u);
    // The first kept slot is 2, so every skipped slot borrows expert 12.
    for (uint32_t i = 0; i < 6; ++i) {
        if (res[i]) { CHECK_EQ(out_ids[i], ids[i]); CHECK(out_w[i] > 0.0f); }
        else        { CHECK_EQ(out_ids[i], 12u);    CHECK_EQ(out_w[i], 0.0f); }
    }
    // `fill_id` is only for the nothing-resident case, so it must NOT appear.
    for (uint32_t i = 0; i < 6; ++i) CHECK(out_ids[i] != 999u);
}

DEEPMOE_TEST(resident_route, nothing_resident_is_shared_expert_only) {
    const std::vector<float> w = gate_weights({4.0f, 3.0f, 2.0f, 1.5f, 1.0f, 0.5f});
    const uint32_t ids[6]{10, 11, 12, 13, 14, 15};
    const uint8_t  res[6]{0, 0, 0, 0, 0, 0};
    uint32_t out_ids[6];
    float    out_w[6];
    const ResidentRoute r = resident_route(ids, w.data(), 6,
                                           std::span<const uint8_t>(res, 6), 77, out_ids, out_w);
    CHECK_EQ(r.kept, 0u);
    CHECK(r.shared_only);
    CHECK_CLOSE(r.mass_lost, 1.0f, 1e-6f);
    // Every routed slot contributes nothing and every row the kernel will read
    // is the caller's known-resident borrow.
    for (uint32_t i = 0; i < 6; ++i) { CHECK_EQ(out_ids[i], 77u); CHECK_EQ(out_w[i], 0.0f); }
}

DEEPMOE_TEST(resident_route, one_resident_carries_the_whole_gate) {
    // The degenerate renormalisation: a single survivor takes the entire routed
    // mass, i.e. exactly routed_scaling_factor.
    const std::vector<float> w = gate_weights({4.0f, 3.0f, 2.0f, 1.5f, 1.0f, 0.5f});
    const uint32_t ids[6]{10, 11, 12, 13, 14, 15};
    const uint8_t  res[6]{0, 0, 0, 0, 0, 1};
    uint32_t out_ids[6];
    float    out_w[6];
    const ResidentRoute r = resident_route(ids, w.data(), 6,
                                           std::span<const uint8_t>(res, 6), 0, out_ids, out_w);
    REQUIRE_EQ(r.kept, 1u);
    CHECK_CLOSE(out_w[5], kRouteScale, 1e-5f);
    CHECK_CLOSE(r.mass_lost, static_cast<float>(1.0 - 0.5 / 12.0), 1e-5f);
}

DEEPMOE_TEST(resident_route, stats_average_over_layer_steps) {
    ResidentRouteStats s;
    s.layers = 4; s.requested = 24; s.served = 21; s.skipped = 3;
    s.mass_lost_sum = 0.04 + 0.00 + 0.12 + 0.08;
    CHECK_CLOSE(static_cast<float>(s.mass_lost()), 0.06f, 1e-6f);
    CHECK_CLOSE(static_cast<float>(s.served_frac()), 21.0f / 24.0f, 1e-6f);
    ResidentRouteStats z;
    CHECK_EQ(z.mass_lost(), 0.0);
    CHECK_EQ(z.served_frac(), 0.0);
}

// --- Track Y step 3: the bounded background miss queue -----------------------
//
// docs/p4_resident_routing.md section 8. The window is "the most recent N
// steps", inclusive of the current one, so at N = 2 and step 9 the queue keeps
// steps 8 and 9 and drops everything at 7 or below.
DEEPMOE_TEST(resident_route, queue_cutoff_keeps_the_most_recent_n_steps) {
    CHECK_EQ(resident_queue_cutoff(9, 2), 8u);
    CHECK_EQ(resident_queue_cutoff(9, 1), 9u);      // only this step's misses
    CHECK_EQ(resident_queue_cutoff(9, 10), 0u);     // the whole run so far
    // A window of 0 would keep nothing at all and starve the drive; it is
    // clamped to 1, which is what the env parser also refuses to set.
    CHECK_EQ(resident_queue_cutoff(9, 0), 9u);
    // No underflow in the first steps of a run.
    CHECK_EQ(resident_queue_cutoff(0, 2), 0u);
    CHECK_EQ(resident_queue_cutoff(1, 4), 0u);
    // Monotone in the step: the window slides, it does not grow.
    for (uint64_t t = 1; t < 50; ++t)
        CHECK_EQ(resident_queue_cutoff(t, 3) - resident_queue_cutoff(t - 1, 3),
                 t >= 3 ? 1u : 0u);
}

DEEPMOE_TEST(resident_route, queue_depth_and_stall1_counters) {
    ResidentRouteStats s;
    s.bg_depth_sum = 10 + 6 + 2; s.bg_depth_n = 3; s.bg_depth_peak = 10;
    CHECK_CLOSE(static_cast<float>(s.bg_depth_mean()), 6.0f, 1e-6f);
    s.bg_stale = 7; s.stall1_p0 = 40; s.stall1_ms = 160.0;
    CHECK_CLOSE(static_cast<float>(s.stall1_ms / double(s.stall1_p0)), 4.0f, 1e-6f);
    ResidentRouteStats z;
    CHECK_EQ(z.bg_depth_mean(), 0.0);
}
