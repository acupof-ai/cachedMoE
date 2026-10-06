// Same-engine benchmark controls must change only request-boundary policy.
// The real draft is compared bit for bit on a common golden KV state, then
// every arm runs a cycle and proves that verification calls the target once.
#include "tests/env_guard.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <format>
#include <span>
#include <string>
#include <vector>

#include "core/config.h"
#include "model/layout.h"
#include "runtime/engine.h"
#include "tests/l1_golden.h"
#include "tests/l2_golden.h"
#include "tests/test_framework.h"

#ifndef CACHEDMOE_TEST_DATA_DIR
#define CACHEDMOE_TEST_DATA_DIR "tests/data"
#endif

using namespace cachedmoe;
using namespace cachedmoe::testing;

namespace {

using ScopedFlag = test::ScopedEnvironment;

struct Arm {
    uint32_t k;
    bool onecb, gpu_route;
};

} // namespace

CACHEDMOE_TEST(gpu_dspark, benchmark_switch_keeps_draft_bits_and_one_target_call) {
    if (skip_without_model("gpu_dspark.benchmark_switch_keeps_draft_bits_and_one_target_call"))
        return;
    ScopedFlag onecb("CACHEDMOE_DSPARK_ONECB", "1"), route("CACHEDMOE_BATCH_GPU_ROUTE", "1"),
               mega("CACHEDMOE_DSPARK_MEGA", "0"), dynamic("CACHEDMOE_MASK_DYNAMIC_LRU", "1"),
               wait("CACHEDMOE_MASK_WAIT_TAU", nullptr);
    auto golden = load_l2(std::string(CACHEDMOE_TEST_DATA_DIR) + "/dspark");
    REQUIRE_OK(golden);
    const L2Step* step = nullptr;
    for (const auto& candidate : golden->steps)
        if (candidate.step == "golden_pos64") step = &candidate;
    REQUIRE(step);
    auto state = runtime::DecodeState::load(std::string(CACHEDMOE_TEST_DATA_DIR) + "/l3");
    REQUIRE_OK(state);
    REQUIRE(state->prompt_ids().size() >= 8);

    RuntimeConfig cfg;
    cfg.model_dir = model_dir();
    cfg.cache.budget_bytes = 512ull * layout::kExpertSlotBytes;
    cfg.cache.slots_per_slab = 64;
    cfg.speculation.enabled = true;
    cfg.speculation.max_draft = layout::kDsparkBlockSize;
    runtime::Engine engine;
    REQUIRE_OK(engine.init(cfg));
    REQUIRE_OK(engine.init_gpu());
    engine.set_mask_cache_fixed(false);
    engine.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    runtime::SessionConfig sc;
    sc.max_context = 256;
    sc.engram_tables_dir = std::string(CACHEDMOE_TEST_DATA_DIR) + "/l3";
    REQUIRE_OK(engine.begin_session(sc));

    const std::array arms{Arm{5, true, true}, Arm{2, true, true}, Arm{5, true, false},
                          Arm{2, false, false}};
    std::array<std::span<const float>, layout::kMtpBlocks> rings;
    for (uint32_t stage = 0; stage < rings.size(); ++stage)
        rings[stage] = std::span(step->f(std::format("s{}.sparse_kv", stage)))
                           .first(layout::kSlidingWindow * engine.model().text.head_dim);
    std::vector<float> reference;
    std::array<uint32_t, layout::kDsparkBlockSize> reference_tokens{};
    for (const auto& arm : arms) {
        engine.reset_context();
        REQUIRE_OK(engine.set_spec_config(arm.k, arm.onecb, arm.gpu_route));
        CHECK_EQ(engine.config().speculation.max_draft, arm.k);
        CHECK_EQ(engine.config().gpu.draft_onecb, arm.onecb);
        CHECK_EQ(engine.config().gpu.batch_gpu_route, arm.gpu_route);
        auto& draft = *engine.dspark_runtime();
        REQUIRE_OK(draft.seed_window(64, rings));
        REQUIRE_OK(draft.append(64, step->f("main_hidden")));
        auto proposal = draft.draft(64, uint32_t(step->f("draft_ids")[0]), arm.k);
        REQUIRE_OK(proposal);
        if (reference.empty()) {
            reference = proposal->logits;
            reference_tokens = proposal->tokens;
        }
        CHECK(std::equal(proposal->logits.begin(), proposal->logits.end(), reference.begin()));
        CHECK(std::equal(proposal->tokens.begin(), proposal->tokens.begin() + arm.k,
                         reference_tokens.begin()));

        engine.reset_context();
        auto first = engine.feed(std::span(state->prompt_ids()).first(8));
        REQUIRE_OK(first);
        CHECK_ERR(engine.set_spec_config(arm.k, arm.onecb, arm.gpu_route),
                  Err::FailedPrecondition);
        const auto calls = engine.batch_forward_calls();
        auto cycle = engine.speculative_step(first->token, arm.k + 1);
        REQUIRE_OK(cycle);
        CHECK_EQ(cycle->cycle.k, arm.k);
        CHECK_EQ(engine.batch_forward_calls() - calls, 1ull);
        CHECK_EQ(engine.last_batch_layers(), engine.model().text.num_hidden_layers);
        CHECK_EQ(cycle->rows.size(), size_t(cycle->cycle.accepted + 1));
        if (arm.gpu_route) CHECK_EQ(engine.last_batch_submits(), 1u);
        std::printf("switch k=%u ONECB=%u GPU route=%u: draft prefix bit-identical; target calls=1\n",
                    arm.k, arm.onecb, arm.gpu_route);
    }
    engine.reset_context();
    engine.set_resident_only(runtime::Engine::ResidentOnly::Off);
    CHECK_ERR(engine.set_spec_config(2, true, true), Err::FailedPrecondition);
    engine.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    engine.set_mask_cache_fixed(true);
    CHECK_ERR(engine.set_spec_config(2, true, true), Err::FailedPrecondition);
}
