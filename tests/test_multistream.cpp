// Track MS (docs/p4_multistream.md): two decode streams in one engine process.
//
// The whole design rests on one claim -- that the streams are INDEPENDENT: the
// expert cache, the pinned set and the planner are shared, but the KV store,
// the activations and the address tables the shaders read are per stream, so a
// token decoded while another sequence is halfway through its own layer L is
// the same token, bit for bit, as one decoded alone.
//
//   multistream.no_cross_contamination
//       Stream 0 decodes a prompt alone and its trajectory is recorded --
//       token ids, and top1/top2 compared as RAW BIT PATTERNS, which is a
//       stricter test than any tolerance. Both streams are then reset and the
//       SAME prompt is decoded on stream 0 while a DIFFERENT prompt runs on
//       stream 1, layer-interleaved. Stream 0 must produce the identical
//       trajectory and the identical window-KV ring bytes.
//
//       It is a real test rather than a tautology because everything that
//       could leak is exercised: the two streams' misses are outstanding at
//       the drive together (so the eviction guard has to hold a slot one
//       stream is still reading while the other completes a buffer), they
//       share the LRU clock, and the interleave rewrites the MoE staging
//       buffers and the attention address tables between stream 0's bind and
//       stream 0's dispatch.
//
//   multistream.interleave_matches_pipeline
//   multistream.pingpong_matches_interleave
//       The same trajectory again under the other two schedules, which are
//       different orders of the same work.
//
// Needs DEEPMOE_MODEL_DIR and tests/data/l3 (the engram tables).
#include <cstdio>
#include <cstring>
#include <array>
#include <format>
#include <string>
#include <vector>

#include "core/config.h"
#include "runtime/engine.h"
#include "tests/l1_golden.h"
#include "tests/test_framework.h"

#ifndef DEEPMOE_TEST_DATA_DIR
#define DEEPMOE_TEST_DATA_DIR "tests/data"
#endif

using namespace deepmoe;
using namespace deepmoe::testing;

namespace {

// Two prompts that share no prefix, so the two streams route differently and a
// leak shows up as a changed token rather than as a coincidence.
const std::vector<uint32_t> kPromptA{0, 2054, 3891, 617, 10234, 45, 7781, 1290};
const std::vector<uint32_t> kPromptB{0, 8811, 122, 60312, 907, 4410, 33, 25607};

struct Step {
    uint32_t token = 0;
    uint32_t top1_bits = 0, top2_bits = 0;
};

// Every layer's window ring, raw, for the slots holding positions [0, n).
std::vector<uint8_t> ring_bytes(const runtime::Engine& e, uint32_t n) {
    const TextConfig& c = e.model().text;
    std::vector<uint8_t> out;
    for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
        auto v = e.kv().layer(L);
        if (!v) return out;
        for (uint32_t p = 0; p < n; ++p) {
            const uint32_t s = p % c.sliding_window;
            out.insert(out.end(), v->win_val_host + size_t(s) * c.head_dim,
                       v->win_val_host + size_t(s + 1) * c.head_dim);
            out.insert(out.end(), v->win_scale_host + size_t(s) * (c.head_dim / 32),
                       v->win_scale_host + size_t(s + 1) * (c.head_dim / 32));
        }
    }
    return out;
}

Step to_step(const runtime::DecodeStepResult& r) {
    Step s;
    s.token = r.token;
    std::memcpy(&s.top1_bits, &r.top1, 4);
    std::memcpy(&s.top2_bits, &r.top2, 4);
    return s;
}

bool bring_up(runtime::Engine& e, uint32_t streams, std::string& why) {
    RuntimeConfig rc;
    rc.model_dir = model_dir();
    rc.cache.budget_bytes = 8ull << 30;
    rc.cache.slots_per_slab = 100;
    if (auto r = e.init(rc); !r) { why = r.error().str(); return false; }
    if (auto r = e.init_gpu(); !r) { why = r.error().str(); return false; }
    if (auto r = e.set_streams(streams); !r) { why = r.error().str(); return false; }
    runtime::SessionConfig sc;
    sc.engram_tables_dir = std::string(DEEPMOE_TEST_DATA_DIR) + "/l3";
    sc.max_context = 256;
    if (auto r = e.begin_session(sc); !r) { why = r.error().str(); return false; }
    runtime::SamplingParams sp;
    sp.temperature = 0.0f;          // greedy: the trajectory is a function of the weights
    e.set_sampling(sp);
    return true;
}

// Feeds `prompt` on the selected stream and then takes `n` greedy steps.
bool run_alone(runtime::Engine& e, const std::vector<uint32_t>& prompt, uint32_t n,
               std::vector<Step>& out, std::string& why) {
    e.reset_context();
    auto f = e.feed(prompt);
    if (!f) { why = f.error().str(); return false; }
    out.push_back(to_step(*f));
    runtime::DecodeStepResult cur = *f;
    for (uint32_t i = 1; i < n; ++i) {
        const std::array<uint32_t, 1> one{cur.token};
        auto r = e.feed(one);
        if (!r) { why = r.error().str(); return false; }
        cur = *r;
        out.push_back(to_step(cur));
    }
    return true;
}

// Both prompts, one per stream, `n` steps each, decoded together.
bool run_together(runtime::Engine& e, uint32_t n, std::vector<Step>& a, std::vector<Step>& b,
                  std::string& why) {
    for (uint32_t s = 0; s < 2; ++s) {
        if (auto r = e.select_stream(s); !r) { why = r.error().str(); return false; }
        e.reset_context();
        auto f = e.feed(s == 0 ? kPromptA : kPromptB);
        if (!f) { why = f.error().str(); return false; }
        (s == 0 ? a : b).push_back(to_step(*f));
    }
    std::array<runtime::Engine::MultiStep, 2> steps{};
    std::array<runtime::DecodeStepResult, 2> out{};
    steps[0].stream = 0;
    steps[1].stream = 1;
    steps[0].in_token = a.back().token;
    steps[1].in_token = b.back().token;
    for (uint32_t i = 1; i < n; ++i) {
        if (auto r = e.feed_multi(steps, out); !r) { why = r.error().str(); return false; }
        a.push_back(to_step(out[0]));
        b.push_back(to_step(out[1]));
        steps[0].in_token = out[0].token;
        steps[1].in_token = out[1].token;
    }
    return true;
}

bool same(const std::vector<Step>& x, const std::vector<Step>& y, std::string& where) {
    if (x.size() != y.size()) {
        where = std::format("{} steps vs {}", x.size(), y.size());
        return false;
    }
    for (size_t i = 0; i < x.size(); ++i)
        if (x[i].token != y[i].token || x[i].top1_bits != y[i].top1_bits ||
            x[i].top2_bits != y[i].top2_bits) {
            where = std::format("step {}: token {} vs {}, top1 {:08x} vs {:08x}, "
                                "top2 {:08x} vs {:08x}", i, x[i].token, y[i].token,
                                x[i].top1_bits, y[i].top1_bits, x[i].top2_bits, y[i].top2_bits);
            return false;
        }
    return true;
}

void run_case(::deepmoe::test::Context& _ctx, runtime::Engine::MsSched sched,
              const char* name) {
    if (skip_without_model("multistream")) return;
    constexpr uint32_t kSteps = 6;
    runtime::Engine e;
    std::string why;
    if (!bring_up(e, 2, why)) {
        std::printf("SKIP multistream: %s\n", why.c_str());
        return;
    }
    e.set_ms_sched(sched);

    // (1) Stream 0's prompt, alone, on stream 0.
    std::vector<Step> solo_a, solo_b;
    CHECK(e.select_stream(0).has_value());
    if (!run_alone(e, kPromptA, kSteps, solo_a, why)) { _ctx.fail(__FILE__, __LINE__, why); return; }
    const std::vector<uint8_t> solo_ring =
        ring_bytes(e, static_cast<uint32_t>(kPromptA.size()) + kSteps - 1);
    // (2) Stream 1's prompt, alone, on stream 1 -- so both halves of the
    //     comparison are per-stream and neither is the other's leftovers.
    CHECK(e.select_stream(1).has_value());
    if (!run_alone(e, kPromptB, kSteps, solo_b, why)) { _ctx.fail(__FILE__, __LINE__, why); return; }

    // (3) Both together.
    std::vector<Step> both_a, both_b;
    if (!run_together(e, kSteps, both_a, both_b, why)) { _ctx.fail(__FILE__, __LINE__, why); return; }

    std::string where;
    if (!same(solo_a, both_a, where))
        _ctx.fail(__FILE__, __LINE__,
                  std::format("{}: stream 0 differs when stream 1 runs beside it -- {}",
                              name, where));
    if (!same(solo_b, both_b, where))
        _ctx.fail(__FILE__, __LINE__,
                  std::format("{}: stream 1 differs when stream 0 runs beside it -- {}",
                              name, where));
    CHECK(e.select_stream(0).has_value());
    const std::vector<uint8_t> both_ring =
        ring_bytes(e, static_cast<uint32_t>(kPromptA.size()) + kSteps - 1);
    if (!(solo_ring.size() == both_ring.size() && !solo_ring.empty() &&
          std::memcmp(solo_ring.data(), both_ring.data(), solo_ring.size()) == 0))
        _ctx.fail(__FILE__, __LINE__,
                  std::format("{}: stream 0's window ring is not the same {} bytes", name,
                              solo_ring.size()));
    std::printf("  %s: %u steps x 2 streams, trajectories and ring bit-identical\n", name, kSteps);
    e.shutdown();
}

}  // namespace

DEEPMOE_TEST(multistream, no_cross_contamination) {
    run_case(_ctx, runtime::Engine::MsSched::Pipeline, "pipeline");
}

DEEPMOE_TEST(multistream, interleave_matches_pipeline) {
    run_case(_ctx, runtime::Engine::MsSched::Interleave, "interleave");
}

DEEPMOE_TEST(multistream, pingpong_matches_interleave) {
    run_case(_ctx, runtime::Engine::MsSched::PingPong, "pingpong");
}
