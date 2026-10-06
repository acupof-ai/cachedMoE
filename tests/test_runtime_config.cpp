// Startup configuration must remain stable when a benchmark changes the
// environment before creating its next engine. No GPU or checkpoint needed.
#include "tests/env_guard.h"
#include <array>
#include <cstdlib>
#include <string>
#include <string_view>

#include "core/config.h"
#include "runtime/engine.h"
#include "tests/test_framework.h"

using namespace deepmoe;

namespace {

using ScopedEnvironment = test::ScopedEnvironment;

struct DecodeEnvironment {
    std::array<ScopedEnvironment, 8> values{{
        ScopedEnvironment("CACHEDMOE_GPU_WAIT_S"),
        ScopedEnvironment("CACHEDMOE_FENCE_SPIN_US"),
        ScopedEnvironment("CACHEDMOE_SHARED_EARLY"),
        ScopedEnvironment("CACHEDMOE_MS_EAGER_MOE"),
        ScopedEnvironment("CACHEDMOE_SHARED_EARLY_MS"),
        ScopedEnvironment("CACHEDMOE_SE_CHECK"),
        ScopedEnvironment("CACHEDMOE_MASK_DYNAMIC_LRU"),
        ScopedEnvironment("CACHEDMOE_IO_ENGRAM_DEADLINE"),
    }};

    void set(std::string_view name, const char* value) {
        for (auto& entry : values)
            if (entry.key == name) entry.set(value);
    }
};

} // namespace

CACHEDMOE_TEST(runtime_config, decode_defaults_and_api_overrides) {
    DecodeEnvironment environment;
    DecodeExecutionConfig defaults;
    defaults.apply_environment();
    CHECK_EQ(defaults.gpu_wait_budget_seconds, 900.0);
    CHECK_EQ(defaults.fence_spin_microseconds, 0.0);
    CHECK(!defaults.shared_early && !defaults.eager_moe && !defaults.engram_deadline);
    CHECK(!defaults.shared_early_multistream && !defaults.shared_early_check);
    CHECK(defaults.dynamic_mask_lru);

    DecodeExecutionConfig explicit_config;
    explicit_config.gpu_wait_budget_seconds = 45;
    explicit_config.fence_spin_microseconds = 25;
    explicit_config.shared_early = false;
    explicit_config.eager_moe = true;
    explicit_config.engram_deadline = false;
    explicit_config.dynamic_mask_lru = false;
    explicit_config.apply_environment();
    CHECK_EQ(explicit_config.gpu_wait_budget_seconds, 45.0);
    CHECK_EQ(explicit_config.fence_spin_microseconds, 25.0);
    CHECK(explicit_config.shared_early == false);
    CHECK(explicit_config.eager_moe == true);
    CHECK(explicit_config.engram_deadline == false);
    CHECK(!explicit_config.dynamic_mask_lru);
}

CACHEDMOE_TEST(runtime_config, legacy_environment_semantics) {
    DecodeEnvironment environment;
    environment.set("CACHEDMOE_GPU_WAIT_S", "12.5");
    environment.set("CACHEDMOE_FENCE_SPIN_US", "3.25");
    environment.set("CACHEDMOE_SHARED_EARLY", "0x");
    environment.set("CACHEDMOE_MS_EAGER_MOE", "true");
    environment.set("CACHEDMOE_SHARED_EARLY_MS", "yes");
    environment.set("CACHEDMOE_SE_CHECK", "0"); // Presence, not Boolean parsing.
    environment.set("CACHEDMOE_MASK_DYNAMIC_LRU", "0x"); // Only exact "0" freezes.
    environment.set("CACHEDMOE_IO_ENGRAM_DEADLINE", "1x"); // Only exact "1" enables.
    DecodeExecutionConfig resolved;
    resolved.apply_environment();
    CHECK_EQ(resolved.gpu_wait_budget_seconds, 12.5);
    CHECK_EQ(resolved.fence_spin_microseconds, 3.25);
    CHECK(resolved.shared_early == false && resolved.eager_moe == true);
    CHECK(resolved.shared_early_multistream && resolved.shared_early_check);
    CHECK(resolved.dynamic_mask_lru && resolved.engram_deadline == false);

    for (const char* invalid : {"invalid", "nan", "-1", "0"}) {
        environment.set("CACHEDMOE_GPU_WAIT_S", invalid);
        environment.set("CACHEDMOE_FENCE_SPIN_US", invalid);
        DecodeExecutionConfig fallback;
        fallback.apply_environment();
        CHECK_EQ(fallback.gpu_wait_budget_seconds, 900.0);
        CHECK_EQ(fallback.fence_spin_microseconds, 0.0);
    }
#ifndef _WIN32
    // Windows _putenv_s removes an empty variable; Linux can preserve it.
    environment.set("CACHEDMOE_SHARED_EARLY", "");
    environment.set("CACHEDMOE_MS_EAGER_MOE", "");
    DecodeExecutionConfig platform_defaults;
    platform_defaults.shared_early = true;
    platform_defaults.eager_moe = false;
    platform_defaults.apply_environment();
    CHECK(!platform_defaults.shared_early && !platform_defaults.eager_moe);
#endif
}

CACHEDMOE_TEST(runtime_config, mask_policy_is_resolved_for_each_engine) {
    DecodeEnvironment environment;
    environment.set("CACHEDMOE_MASK_DYNAMIC_LRU", "1");
    runtime::Engine dynamic;
    environment.set("CACHEDMOE_MASK_DYNAMIC_LRU", "0");
    dynamic.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    CHECK(!dynamic.store().fixed_cache());

    runtime::Engine frozen;
    environment.set("CACHEDMOE_MASK_DYNAMIC_LRU", "1");
    frozen.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    CHECK(frozen.store().fixed_cache());
    frozen.set_mask_cache_fixed(false); // Explicit runtime control still wins.
    frozen.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    CHECK(!frozen.store().fixed_cache());
}

CACHEDMOE_TEST(runtime_config, spec_switch_rejects_invalid_or_uninitialized_engine) {
    DecodeEnvironment environment;
    runtime::Engine engine;
    const auto before = engine.config();
    CHECK_ERR(engine.set_spec_config(0, true, true), Err::InvalidArgument);
    CHECK_ERR(engine.set_spec_config(layout::kDsparkBlockSize + 1, false, false),
              Err::InvalidArgument);
    CHECK_ERR(engine.set_spec_config(2, true, true), Err::FailedPrecondition);
    CHECK_EQ(engine.config().speculation.max_draft, before.speculation.max_draft);
    CHECK_EQ(engine.config().gpu.draft_onecb, before.gpu.draft_onecb);
    CHECK_EQ(engine.config().gpu.batch_gpu_route, before.gpu.batch_gpu_route);
}
