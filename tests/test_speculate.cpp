// The DSpark cycle, without a GPU or a checkpoint (runtime/speculate.h).
//
// What this pins is the part of speculative decoding that is arithmetic rather
// than kernels: the position alignment, the acceptance, which ring slots a
// rejection has to undo, and design §10.2's hard invariant -- at temperature 0,
// the emitted stream must be the stream the M = 1 path would have emitted,
// whatever the drafts were.
//
// The `SpecModel` here replays a fixed "true" stream: the model's answer at
// position p is `truth[p]`, always, whether it is reached as row j of a verify
// batch or one token at a time. That is exactly the assumption the invariant
// rests on and the one docs/p3_dspark.md §14 measures separately (batch-boundary
// effects on the logits); with it granted, the loop either reproduces the stream
// or it has a bookkeeping bug, and this catches the bookkeeping bug.
//
// The drafts are deliberately adversarial: a draft quality knob from "always
// right" to "always wrong", so acceptance sweeps 0..k and every rollback shape
// is exercised.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <format>
#include <string>
#include <vector>

#include "cpu/dspark_tree.h"
#include "runtime/speculate.h"
#include "tests/test_framework.h"

using namespace deepmoe;
using cpu::dspark::kPositions;

namespace {

// A stand-in model over a known stream.
//
//   truth[p]     the token that LIVES at position p, so a model whose input ends
//                at position p produces truth[p + 1]
//   hit_every    a draft is correct when (position % hit_every) != 0; 1 makes
//                every draft wrong, a large value makes them all right
struct ReplayModel final : runtime::SpecModel {
    std::vector<uint32_t> truth;
    uint32_t hit_every = 3;
    uint32_t window = 128;
    uint32_t layers = 40;

    // What the cycle asked for, so the test can check the rollback shape.
    struct Call { uint32_t p0 = 0, m = 0, accepted = 0; };
    std::vector<Call> snapshots, restores, commits;
    std::vector<uint32_t> committed;          // every token, in order
    std::vector<uint32_t> ring;               // ring slot -> the position living in it
    std::vector<std::vector<uint32_t>> saved; // the snapshot stack
    uint32_t forwards = 0, rows_run = 0;

    explicit ReplayModel(std::vector<uint32_t> t) : truth(std::move(t)) {
        ring.assign(window, 0xFFFFFFFFu);
    }

    uint32_t draft_for(uint32_t pos, uint32_t j) const {
        // The token a draft proposes for position `pos`. Right unless the knob
        // says otherwise, and when wrong it is wrong by a fixed offset so the
        // test can tell a wrong draft from a right one.
        const uint32_t want = truth[pos];
        return ((pos + j) % hit_every) == 0 ? want + 100000u : want;
    }

    Result<void> draft_forward(uint32_t p0, uint32_t last_token, uint32_t lattice_k,
                               cpu::dspark::Lattice& lat,
                               std::span<float, kPositions> conf) override {
        (void)last_token;
        // A lattice whose candidate 0 at position i is this model's draft for
        // p0 + 1 + i, so `Objective::Chain` and `Eal` both pick the drafts.
        // Everything else is filler far below it.
        const uint32_t K = lattice_k;
        std::vector<int32_t> ids(size_t(kPositions) * K);
        std::vector<float>   logit(size_t(kPositions) * K);
        std::vector<float>   e_in(cpu::dspark::kRank, 0.0f);
        std::vector<float>   e_cand(size_t(kPositions - 1) * K * cpu::dspark::kRank, 0.0f);
        std::vector<float>   h_cand(size_t(kPositions) * K * cpu::dspark::kRank, 0.0f);
        for (uint32_t i = 0; i < kPositions; ++i) {
            const uint32_t pos = p0 + 1 + i;
            for (uint32_t c = 0; c < K; ++c) {
                ids[i * K + c] = int32_t(c == 0 ? draft_for(pos < truth.size() ? pos : 0, i)
                                                : 900000u + i * 100 + c);
                logit[i * K + c] = c == 0 ? 8.0f : -4.0f - float(c);
            }
            conf[i] = 3.0f - 0.9f * float(i);   // sigmoid ~0.95 down to ~0.5
        }
        cpu::dspark::LatticeInput in;
        in.K = K;
        in.cand_ids = ids;
        in.cand_logit = logit;
        in.e_in = e_in;
        in.e_cand = e_cand;
        in.h_cand = h_cand;
        return lat.build(in);
    }

    Result<void> verify_forward(uint32_t p0, std::span<const uint32_t> tokens,
                                std::span<const int32_t> cand, uint32_t cand_stride,
                                bool want_exact, std::span<runtime::VerifyRow> rows) override {
        (void)cand; (void)cand_stride; (void)want_exact;
        ++forwards;
        rows_run += uint32_t(rows.size());
        // Row j's answer is the model's token for position p0 + j + 1. It is
        // only meaningful while every row before it held the right token, which
        // is exactly what greedy acceptance consumes -- the first row whose
        // input was wrong is the row that rejects.
        for (uint32_t j = 0; j < rows.size(); ++j) {
            // Row j's input token sits at position p0 + j, so its answer is the
            // token at p0 + j + 1.
            const uint32_t pos = p0 + j + 1;
            rows[j].argmax = pos < truth.size() ? truth[pos] : 0u;
            rows[j].top1 = 10.0f;
            rows[j].top2 = 1.0f;
        }
        // The forward writes the ring for every row it ran.
        for (uint32_t j = 0; j < tokens.size(); ++j) ring[(p0 + j) % window] = p0 + j;
        return {};
    }

    Result<void> snapshot_ring(uint32_t p0, uint32_t m) override {
        snapshots.push_back({p0, m, 0});
        std::vector<uint32_t> s;
        for (uint32_t j = 0; j < m; ++j) s.push_back(ring[(p0 + j) % window]);
        saved.push_back(std::move(s));
        return {};
    }

    Result<void> restore_ring(uint32_t p0, uint32_t accepted, uint32_t m) override {
        restores.push_back({p0, m, accepted});
        if (saved.empty()) return fail(Err::FailedPrecondition, "restore without a snapshot");
        const auto& s = saved.back();
        // Only the rejected positions: p0 + accepted + 1 .. p0 + m - 1.
        for (uint32_t j = accepted + 1; j < m; ++j) ring[(p0 + j) % window] = s[j];
        return {};
    }

    Result<void> commit(uint32_t p0, std::span<const uint32_t> tokens) override {
        commits.push_back({p0, uint32_t(tokens.size()), 0});
        committed.insert(committed.end(), tokens.begin(), tokens.end());
        if (!saved.empty()) saved.pop_back();
        return {};
    }
};

// The stream the M = 1 path would emit from `p0`, for `n` tokens.
std::vector<uint32_t> reference_stream(const std::vector<uint32_t>& truth, uint32_t p0,
                                       uint32_t n) {
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < n && p0 + i < truth.size(); ++i) out.push_back(truth[p0 + i]);
    return out;
}

std::vector<uint32_t> make_truth(uint32_t n) {
    std::vector<uint32_t> t(n);
    for (uint32_t i = 0; i < n; ++i) t[i] = 1000u + (i * 7919u) % 4001u;
    return t;
}

}  // namespace

// The invariant: greedy speculation emits the non-speculative stream, for every
// fixed k and every draft quality.
DEEPMOE_TEST(speculate, greedy_reproduces_the_unspeculated_stream) {
    const auto truth = make_truth(400);
    const uint32_t p_start = 64;

    for (uint32_t hit_every : {1u, 2u, 3u, 5u, 97u}) {
        for (uint32_t k : {1u, 2u, 3u, 5u}) {
            ReplayModel model(truth);
            model.hit_every = hit_every;
            runtime::SpecConfig cfg;
            cfg.mode = runtime::SpecMode::Greedy;
            cfg.k = k;
            cfg.lattice_k = 16;
            runtime::Speculator spec(model, cfg);

            std::vector<uint32_t> out;
            uint32_t p = p_start;
            uint32_t last = truth[p];
            while (out.size() < 64) {
                auto c = spec.cycle(p, last, out);
                REQUIRE_OK(c);
                CHECK(c->emitted == c->accepted + 1);
                CHECK(c->accepted <= c->k);
                p += c->emitted;
                last = out.back();
            }
            const auto want = reference_stream(truth, p_start + 1, uint32_t(out.size()));
            bool same = out.size() == want.size();
            uint32_t first_bad = 0;
            for (uint32_t i = 0; same && i < out.size(); ++i)
                if (out[i] != want[i]) { same = false; first_bad = i; }
            const auto& st = spec.stats();
            std::printf("      hit_every %-3u k=%u: %zu tokens in %llu cycles, accepted "
                        "%.2f/verify, %.2f tokens a cycle -- %s\n",
                        hit_every, k, out.size(), (unsigned long long)st.cycles,
                        st.accepted_per_verify(), st.tokens_per_cycle(),
                        same ? "identical to the unspeculated stream"
                             : std::format("DIFFERS at {}", first_bad).c_str());
            CHECK(same);
            // A perfect draft accepts every position; a always-wrong one accepts
            // none. Anything in between is the knob.
            if (hit_every == 97) CHECK(st.accepted_per_verify() > double(k) - 0.5);
            if (hit_every == 1)  CHECK(st.accepted_per_verify() == 0.0);
        }
    }
}

// The rollback shape: a snapshot every cycle, a restore only when something was
// rejected, and the restored slots are exactly the rejected positions'.
DEEPMOE_TEST(speculate, a_rejection_restores_exactly_the_rejected_slots) {
    const auto truth = make_truth(400);
    ReplayModel model(truth);
    model.hit_every = 3;
    runtime::SpecConfig cfg;
    cfg.mode = runtime::SpecMode::Greedy;
    cfg.k = 5;
    runtime::Speculator spec(model, cfg);

    std::vector<uint32_t> out;
    uint32_t p = 64, last = truth[64];
    uint32_t rejecting_cycles = 0;
    while (out.size() < 48) {
        auto c = spec.cycle(p, last, out);
        REQUIRE_OK(c);
        if (c->accepted < c->k) ++rejecting_cycles;
        p += c->emitted;
        last = out.back();
    }
    std::printf("      %zu cycles: %zu snapshots, %zu restores (%u cycles rejected "
                "something), %zu commits\n",
                model.commits.size(), model.snapshots.size(), model.restores.size(),
                rejecting_cycles, model.commits.size());
    CHECK(model.snapshots.size() == model.commits.size());
    CHECK(model.restores.size() == rejecting_cycles);
    CHECK(rejecting_cycles > 0);
    for (const auto& r : model.restores) {
        CHECK(r.accepted < r.m);
        CHECK(r.m >= 2);
    }
    // Every ring slot the accepted positions own holds the position it should.
    for (uint32_t pos = 64; pos < p; ++pos)
        CHECK(model.ring[pos % model.window] == pos || pos + model.window <= p);
}

// k from the confidence head, and that it never asks for nothing.
DEEPMOE_TEST(speculate, confidence_chooses_k_by_the_prefix_rule) {
    const auto truth = make_truth(200);
    for (double theta : {0.3, 0.5, 0.7, 0.95}) {
        ReplayModel model(truth);
        model.hit_every = 97;                 // every draft right, so k is the only variable
        runtime::SpecConfig cfg;
        cfg.mode = runtime::SpecMode::Greedy;
        cfg.k = 0;                            // confidence-chosen
        cfg.theta = theta;
        runtime::Speculator spec(model, cfg);
        std::vector<uint32_t> out;
        uint32_t p = 64, last = truth[64];
        auto c = spec.cycle(p, last, out);
        REQUIRE_OK(c);
        // The draft's confidences are 3.0, 2.1, 1.2, 0.3, -0.6, so the running
        // product of sigmoids falls monotonically and theta picks a prefix.
        std::array<float, kPositions> conf{};
        for (uint32_t i = 0; i < kPositions; ++i) conf[i] = 3.0f - 0.9f * float(i);
        uint32_t want = cpu::dspark::k_from_confidence(
            std::span<const float, kPositions>(conf), theta);
        if (want == 0) want = 1;              // the loop's floor
        std::printf("      theta %.2f -> k %u (rule says %u), accepted %u\n",
                    theta, c->k, want, c->accepted);
        CHECK(c->k == want);
        CHECK(c->k >= 1);
    }
}

// The cycle refuses rather than silently doing nothing when it is misconfigured.
DEEPMOE_TEST(speculate, a_misconfigured_cycle_refuses) {
    const auto truth = make_truth(100);
    ReplayModel model(truth);
    std::vector<uint32_t> out;
    {
        runtime::SpecConfig cfg;             // mode Off
        runtime::Speculator spec(model, cfg);
        CHECK(!spec.cycle(64, truth[64], out).has_value());
    }
    {
        runtime::SpecConfig cfg;
        cfg.mode = runtime::SpecMode::Greedy;
        cfg.lattice_k = gpu::kDsVerifyMaxCand + 1;
        runtime::Speculator spec(model, cfg);
        auto r = spec.cycle(64, truth[64], out);
        CHECK(!r.has_value());
        if (!r) std::printf("      refused: %s\n", r.error().str().c_str());
    }
    CHECK(runtime::parse_spec_mode("greedy").has_value());
    CHECK(!runtime::parse_spec_mode("yes").has_value());
    CHECK(std::string(runtime::spec_mode_name(runtime::SpecMode::Sample)) == "sample");
}
