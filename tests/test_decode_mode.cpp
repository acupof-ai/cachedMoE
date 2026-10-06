// Request policies share startup resources and may change only at a completed
// request boundary. CPU cases cover parsing/metadata; the model fixture covers
// actual routing, retained KV, cancellation and named-session replay.
#include "tests/env_guard.h"
#include <array>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <string>
#include <utility>
#include <vector>

#include "core/json.h"
#include "model/layout.h"
#include "runtime/decode_boundary.h"
#include "runtime/session.h"
#include "tests/l1_golden.h"
#include "tests/test_framework.h"

#ifndef CACHEDMOE_TEST_DATA_DIR
#define CACHEDMOE_TEST_DATA_DIR "tests/data"
#endif

using namespace deepmoe;
using namespace deepmoe::testing;
using runtime::DecodeMode;

namespace {

using ScopedFlag = test::ScopedEnvironment;

// This comparison is deliberately short (< window), so position and ring slot
// coincide. Includes every layer's FP8 values and scales, not just token IDs.
std::vector<uint8_t> short_window_bytes(const runtime::Engine& engine) {
    std::vector<uint8_t> bytes;
    const auto& text = engine.model().text;
    for (uint32_t layer = 0; layer < text.num_hidden_layers; ++layer) {
        auto view = engine.kv().layer(layer);
        if (!view) return {};
        const size_t n = engine.context_length();
        bytes.insert(bytes.end(), view->win_val_host, view->win_val_host + n * text.head_dim);
        bytes.insert(bytes.end(), view->win_scale_host,
                     view->win_scale_host + n * (text.head_dim / 32));
    }
    return bytes;
}

} // namespace

CACHEDMOE_TEST(decode_mode, strict_names_and_uninitialized_rejection) {
    for (const auto mode : {DecodeMode::MaskSpec, DecodeMode::MaskPlain, DecodeMode::OffPlain}) {
        auto parsed = runtime::parse_decode_mode(runtime::decode_mode_name(mode));
        REQUIRE_OK(parsed);
        CHECK(*parsed == mode);
    }
    for (const char* invalid : {"", "startup", "mask", "off", "MASK-SPEC", "off-plain "})
        CHECK_ERR(runtime::parse_decode_mode(invalid), Err::InvalidArgument);

    runtime::Engine engine;
    for (const auto mode : {DecodeMode::MaskSpec, DecodeMode::MaskPlain, DecodeMode::OffPlain}) {
        CHECK_ERR(engine.request_decode_policy(mode), Err::FailedPrecondition);
        CHECK(engine.resident_only() == runtime::Engine::ResidentOnly::Off);
    }
    CHECK(engine.available_decode_modes().empty());
}

CACHEDMOE_TEST(decode_mode, inherited_policy_is_a_move_only_noop) {
    runtime::Engine engine;
    engine.set_resident_only(runtime::Engine::ResidentOnly::Verify);
    auto inherited = engine.request_decode_policy(DecodeMode::Startup);
    REQUIRE_OK(inherited);
    CHECK_EQ(inherited->mode(), std::string("verify-plain"));
    CHECK(!inherited->speculative());
    {
        auto moved = std::move(*inherited);
        CHECK_EQ(moved.mode(), std::string("verify-plain"));
    }
    CHECK(engine.resident_only() == runtime::Engine::ResidentOnly::Verify);
}

CACHEDMOE_TEST(decode_mode, stats_report_actual_policy_and_unset_metadata) {
    runtime::GenerateStats stats;
    auto unset = json_parse("{" + stats.json_fields() + "}");
    REQUIRE_OK(unset);
    CHECK(unset->find("decode_mode")->is_null());
    CHECK(!unset->bool_or("speculation_enabled", true));
    stats.decode_mode = "off-plain";
    stats.speculation_enabled = false;
    auto plain = json_parse("{" + stats.json_fields() + "}");
    REQUIRE_OK(plain);
    CHECK_EQ(plain->string_or("decode_mode", ""), std::string("off-plain"));
    CHECK(!plain->bool_or("speculation_enabled", true));
    CHECK_EQ(stats.speculation.cycles, 0ull);
    stats.decode_mode = "mask-spec";
    stats.speculation_enabled = true;
    auto spec = json_parse("{" + stats.json_fields() + "}");
    REQUIRE_OK(spec);
    CHECK_EQ(spec->string_or("decode_mode", ""), std::string("mask-spec"));
    CHECK(spec->bool_or("speculation_enabled", false));
}

CACHEDMOE_TEST(decode_mode, exact_boundary_waits_for_capacity_not_all_fills) {
    store::ExpertStore store;
    CacheConfig cache;
    cache.slots_per_slab = 4;
    cache.budget_bytes = 4ull * layout::kExpertSlotBytes;
    REQUIRE_OK(store.init(std::make_unique<store::HostSlabBacking>(), cache, 1, 4));
    std::array<uint32_t, 4> slots{};
    for (uint16_t expert = 0; expert < slots.size(); ++expert) {
        auto fill = store.begin_fill({0, expert});
        REQUIRE_OK(fill);
        slots[expert] = fill->slot;
    }
    std::thread completion([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        store.finish_fill(slots[2], true);
        store.finish_fill(slots[3], true);
    });
    auto boundary = runtime::settle_decode_boundary(store, 2, std::chrono::seconds(1));
    completion.join();
    REQUIRE_OK(boundary);
    CHECK_EQ(boundary->before.filling, 4u);
    CHECK(boundary->waits > 0);
    CHECK_EQ(boundary->after.available(), 2u);
    CHECK_EQ(boundary->after.filling, 2u);
    CHECK_EQ(store.stats().filling, 2u);
    CHECK(store.slot_info(slots[0])->state == SlotState::Filling);
    CHECK(store.slot_info(slots[1])->state == SlotState::Filling);
    // Broad slack around the 5 ms completion. Waiting the full 1 s budget
    // on the unfinished first slot must fail even if capacity is found later.
    CHECK(boundary->elapsed_ms < 500);
    CHECK_EQ(store.completed_timeline(), 0ull);
}

CACHEDMOE_TEST(decode_mode, exact_boundary_rechecks_completion_after_initial_snapshot) {
    store::ExpertStore store;
    CacheConfig cache;
    cache.slots_per_slab = 1;
    cache.budget_bytes = layout::kExpertSlotBytes;
    REQUIRE_OK(store.init(std::make_unique<store::HostSlabBacking>(), cache, 1, 1));
    auto fill = store.begin_fill({0, 0});
    REQUIRE_OK(fill);
    const auto before = runtime::decode_boundary_capacity(store);
    CHECK_EQ(before.filling, 1u);
    REQUIRE_OK(store.finish_fill(fill->slot, true));
    // The last fill completes between capacity inspection and the search for
    // a key to wait on. Even a zero remaining budget must accept the capacity.
    auto ready = runtime::settle_decode_boundary(store, 1, std::chrono::milliseconds(0), before);
    REQUIRE_OK(ready);
    CHECK_EQ(ready->before.filling, 1u);
    CHECK_EQ(ready->after.evictable, 1u);
    CHECK_EQ(ready->waits, 0u);
}

CACHEDMOE_TEST(decode_mode, exact_boundary_rejects_guards_without_clearing_them) {
    store::ExpertStore store;
    CacheConfig cache;
    cache.slots_per_slab = 1;
    cache.budget_bytes = layout::kExpertSlotBytes;
    REQUIRE_OK(store.init(std::make_unique<store::HostSlabBacking>(), cache, 1, 1));
    auto fill = store.begin_fill({0, 0});
    REQUIRE_OK(fill);
    REQUIRE_OK(store.finish_fill(fill->slot, true));
    REQUIRE_OK(store.set_guard(fill->slot, 7));
    store.set_completed_timeline(6);
    CHECK_ERR(runtime::settle_decode_boundary(store, 1, std::chrono::seconds(1)),
              Err::ResourceExhausted);
    auto capacity = runtime::decode_boundary_capacity(store);
    CHECK_EQ(capacity.guarded, 1u);
    CHECK_EQ(capacity.filling, 0u);
    CHECK_EQ(capacity.max_guard, 7ull);
    CHECK_EQ(store.completed_timeline(), 6ull);
    store.set_completed_timeline(7);
    auto ready = runtime::settle_decode_boundary(store, 1, std::chrono::milliseconds(0));
    REQUIRE_OK(ready);
    CHECK_EQ(ready->waits, 0u);
}

CACHEDMOE_TEST(decode_mode, exact_boundary_timeout_and_failed_fill) {
    store::ExpertStore store;
    CacheConfig cache;
    cache.slots_per_slab = 1;
    cache.budget_bytes = layout::kExpertSlotBytes;
    REQUIRE_OK(store.init(std::make_unique<store::HostSlabBacking>(), cache, 1, 1));
    auto fill = store.begin_fill({0, 0});
    REQUIRE_OK(fill);
    CHECK_ERR(runtime::settle_decode_boundary(store, 1, std::chrono::milliseconds(1)),
              Err::ResourceExhausted);
    CHECK_EQ(store.stats().filling, 1u);
    std::thread completion([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        store.finish_fill(fill->slot, false);
    });
    auto ready = runtime::settle_decode_boundary(store, 1, std::chrono::seconds(1));
    completion.join();
    REQUIRE_OK(ready);
    CHECK_EQ(ready->after.free, 1u);
    CHECK_EQ(store.stats().fills_failed, 1ull);
}

CACHEDMOE_TEST(gpu_request_policy, retained_window_sessions_and_exact_plain) {
    if (skip_without_model("gpu_request_policy.retained_window_sessions_and_exact_plain")) return;
    ScopedFlag dynamic("CACHEDMOE_MASK_DYNAMIC_LRU", "1"), onecb("CACHEDMOE_DSPARK_ONECB", "1"),
               route("CACHEDMOE_BATCH_GPU_ROUTE", "0"), mega("CACHEDMOE_DSPARK_MEGA", "0"),
               wait("CACHEDMOE_MASK_WAIT_TAU", nullptr);
    auto state = runtime::DecodeState::load(std::string(CACHEDMOE_TEST_DATA_DIR) + "/l3");
    REQUIRE_OK(state);
    REQUIRE(state->prompt_ids().size() >= 8);
    auto tokenizer = text::Tokenizer::load(std::string(model_dir()) + "/tokenizer.json");
    REQUIRE_OK(tokenizer);

    RuntimeConfig config;
    config.model_dir = model_dir();
    // Tiny fixture cache, not a quality or production-loading measurement.
    config.cache.budget_bytes = 512ull * layout::kExpertSlotBytes;
    config.cache.slots_per_slab = 64;
    config.speculation.enabled = true;
    config.speculation.max_draft = 2;
    runtime::Engine engine;
    REQUIRE_OK(engine.init(config));
    REQUIRE_OK(engine.init_gpu());
    engine.set_mask_cache_fixed(false);
    engine.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    runtime::SessionConfig session_config;
    session_config.max_context = 512;
    session_config.engram_tables_dir = std::string(CACHEDMOE_TEST_DATA_DIR) + "/l3";
    REQUIRE_OK(engine.begin_session(session_config));
    engine.set_mask_cache_fixed(true);
    CHECK(engine.available_decode_modes().empty());
    CHECK_ERR(engine.request_decode_policy(DecodeMode::OffPlain), Err::FailedPrecondition);
    CHECK_ERR(engine.request_decode_policy(DecodeMode::MaskSpec), Err::FailedPrecondition);
    engine.set_mask_cache_fixed(false);
    // DSpark and weighted waits are already mutually exclusive at the setter;
    // do not manufacture an unsupported combination just to enter the guard.
    CHECK_ERR(engine.set_mask_wait(0.2), Err::FailedPrecondition);
    CHECK_EQ(engine.available_decode_modes().size(), size_t(3));
    runtime::SessionOptions options;
    options.gpu_prefill_min = 0;
    options.progress_every = layout::kSlidingWindow;
    runtime::SessionPool pool(engine, *tokenizer, options);
    runtime::GenerateRequest request;
    request.prompt_ids.assign(state->prompt_ids().begin(), state->prompt_ids().begin() + 8);
    request.max_tokens = 3;
    request.stop_ids.clear();
    request.sampling.temperature = 0;

    // Exact routing under a plain request must match the existing feed chain,
    // despite DSpark being loaded. Compare margins and every window byte too.
    engine.set_resident_only(runtime::Engine::ResidentOnly::Off);
    engine.set_sampling(request.sampling);
    auto first = engine.feed(request.prompt_ids);
    REQUIRE_OK(first);
    std::vector<uint32_t> reference_tokens{first->token};
    std::vector<uint32_t> reference_margins{std::bit_cast<uint32_t>(first->margin())};
    for (uint32_t i = 1; i < request.max_tokens; ++i) {
        const std::array one{reference_tokens.back()};
        auto next = engine.feed(one);
        REQUIRE_OK(next);
        reference_tokens.push_back(next->token);
        reference_margins.push_back(std::bit_cast<uint32_t>(next->margin()));
    }
    const auto reference_window = short_window_bytes(engine);
    engine.reset_context();
    engine.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    request.decode_mode = DecodeMode::OffPlain;
    std::vector<uint32_t> tokens, margins;
    auto calls = engine.batch_forward_calls();
    auto exact = pool.live().generate(request, [&](const runtime::TokenEvent& event) {
        tokens.push_back(event.id);
        margins.push_back(std::bit_cast<uint32_t>(event.margin));
    });
    REQUIRE_OK(exact);
    CHECK(tokens == reference_tokens);
    CHECK(margins == reference_margins);
    CHECK(short_window_bytes(engine) == reference_window);
    CHECK_EQ(exact->speculation.cycles, 0ull);
    CHECK_EQ(engine.batch_forward_calls(), calls);
    CHECK_EQ(exact->decode_mode, std::string("off-plain"));
    CHECK(!exact->speculation_enabled);
    CHECK(engine.resident_only() == runtime::Engine::ResidentOnly::Mask);

    // Exercise the actual Session guard's early error and cancellation paths.
    auto bad = request;
    bad.prompt_ids.clear();
    CHECK_ERR(pool.live().generate(bad, {}), Err::InvalidArgument);
    CHECK(engine.resident_only() == runtime::Engine::ResidentOnly::Mask);
    auto cancelled = request;
    cancelled.cancel = [] { return true; };
    auto cancelled_stats = pool.live().generate(cancelled, {});
    REQUIRE_OK(cancelled_stats);
    CHECK_EQ(cancelled_stats->finish, std::string("cancel"));
    CHECK_EQ(cancelled_stats->decode_mode, std::string("off-plain"));
    CHECK(engine.resident_only() == runtime::Engine::ResidentOnly::Mask);

    // More than one full window of ordinary feed steps leaves the draft KV
    // stale. The next nonempty-context speculative turn must rebuild it.
    engine.reset_context();
    request.decode_mode = DecodeMode::MaskPlain;
    request.max_tokens = layout::kSlidingWindow + 2;
    std::vector<uint32_t> completion;
    calls = engine.batch_forward_calls();
    auto plain = pool.live().generate(request, [&](const runtime::TokenEvent& event) {
        completion.push_back(event.id);
    });
    REQUIRE_OK(plain);
    CHECK(plain->decode_steps > layout::kSlidingWindow);
    CHECK_EQ(plain->speculation.cycles, 0ull);
    CHECK_EQ(engine.batch_forward_calls(), calls);
    CHECK(!plain->speculation_enabled);
    const auto retained_history = engine.history();
    auto continued = request;
    continued.decode_mode = DecodeMode::MaskSpec;
    continued.prompt_ids.insert(continued.prompt_ids.end(), completion.begin(), completion.end());
    continued.max_tokens = 3;
    calls = engine.batch_forward_calls();
    auto spec = pool.live().generate(continued, {});
    REQUIRE_OK(spec);
    CHECK_EQ(spec->reused_tokens, uint32_t(retained_history.size()));
    CHECK(spec->speculation.cycles > 0);
    CHECK_EQ(engine.batch_forward_calls() - calls, spec->speculation.cycles);
    CHECK(spec->speculation_enabled);
    CHECK_EQ(spec->decode_mode, std::string("mask-spec"));

    // Match the CLI ordering: establish policy before session activation.
    const auto default_history = engine.history();
    {
        const auto before = runtime::decode_boundary_capacity(engine.store());
        std::printf("named Off boundary before: %s\n", before.describe().c_str());
        auto policy = engine.request_decode_policy(DecodeMode::OffPlain);
        REQUIRE_OK(policy);
        const auto after = runtime::decode_boundary_capacity(engine.store());
        std::printf("named Off boundary after: %s\n", after.describe().c_str());
        CHECK(after.available() >= engine.model().text.num_experts_per_tok);
        REQUIRE_OK(pool.activate("other"));
        CHECK(engine.resident_only() == runtime::Engine::ResidentOnly::Off);
        auto other = request;
        other.decode_mode = DecodeMode::OffPlain;
        other.max_tokens = 2;
        auto other_stats = pool.live().generate(other, {});
        REQUIRE_OK(other_stats);
        CHECK_EQ(other_stats->speculation.cycles, 0ull);
    }
    CHECK(engine.resident_only() == runtime::Engine::ResidentOnly::Mask);
    {
        auto policy = engine.request_decode_policy(DecodeMode::MaskSpec);
        REQUIRE_OK(policy);
        auto restored = pool.activate("default");
        REQUIRE_OK(restored);
        CHECK(engine.history() == default_history);
        auto returned = request;
        returned.decode_mode = DecodeMode::MaskSpec;
        returned.prompt_ids = default_history;
        returned.prompt_ids.push_back(completion.back());
        returned.max_tokens = 3;
        calls = engine.batch_forward_calls();
        auto returned_stats = pool.live().generate(returned, {});
        REQUIRE_OK(returned_stats);
        CHECK(returned_stats->speculation.cycles > 0);
        CHECK_EQ(engine.batch_forward_calls() - calls, returned_stats->speculation.cycles);
    }
    CHECK(engine.config().speculation.enabled);
    CHECK_EQ(engine.config().speculation.max_draft, 2u);
    CHECK(engine.config().gpu.draft_onecb);
    CHECK(!engine.config().gpu.batch_gpu_route);
    CHECK(engine.resident_only() == runtime::Engine::ResidentOnly::Mask);
    std::printf("request modes: exact feed tokens/margins/window bit-identical; >128 plain steps; "
                "named-session restore; one target per speculative cycle; error/cancel restored\n");
}
