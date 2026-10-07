// Configuration fixtures need neither Vulkan nor a checkpoint. They exercise
// the real parser and consumers rather than duplicating their implementations.
#include "cli/generate_options.h"
#include "core/runtime_environment.h"
#include "runtime/engine.h"
#include "storage/io_engine.h"
#include "tests/env_guard.h"
#include "tests/fake_backend.h"
#include "tests/test_framework.h"

#include <algorithm>
#include <memory>
#include <set>
#include <type_traits>

using namespace cachedmoe;
using configuration::EnvironmentSnapshot;
using configuration::Key;
using configuration::RuntimeEnvironment;

namespace {

std::shared_ptr<const RuntimeEnvironment> resolved(EnvironmentSnapshot raw = {}) {
    // Model-less lifecycle probes must not change the test process affinity.
    raw.set(Key::CPU_AFFINITY, "off");
    return std::make_shared<const RuntimeEnvironment>(std::move(raw));
}

static_assert(std::is_same_v<runtime::Engine::ResidentOnly, configuration::ResidentPolicy>);
static_assert(static_cast<uint8_t>(configuration::ResidentPolicy::Off) == 0);
static_assert(static_cast<uint8_t>(configuration::ResidentPolicy::Mask) == 4);

} // namespace

CACHEDMOE_TEST(runtime_environment, all_production_keys_have_one_registry_entry) {
    CHECK_EQ(configuration::kKeyNames.size(), 88u);
    std::set<std::string_view> unique;
    for (const auto name : configuration::kKeyNames) {
        CHECK(name.starts_with("CACHEDMOE_"));
        CHECK(unique.insert(name).second);
    }
}

CACHEDMOE_TEST(runtime_environment, snapshot_owns_alias_bytes_and_empty_presence) {
    test::ScopedEnvironment guard("CACHEDMOE_MOE_L");
    test::ScopedEnvironment::write("DEEPMOE_MOE_L", "16suffix");
    const auto legacy = EnvironmentSnapshot::capture();
    CHECK_EQ(legacy[Key::MOE_L].source, environment::Source::Legacy);
    guard.set("32suffix");
    const auto canonical = EnvironmentSnapshot::capture();
    CHECK_EQ(canonical[Key::MOE_L].source, environment::Source::Canonical);
    guard.set(nullptr);
    CHECK_EQ(RuntimeEnvironment(legacy).moe.lanes.value(), 16u);
    CHECK_EQ(RuntimeEnvironment(canonical).moe.lanes.value(), 32u);
#ifndef _WIN32
    test::ScopedEnvironment::write("DEEPMOE_MOE_L", "64");
    test::ScopedEnvironment::write("CACHEDMOE_MOE_L", "");
    const auto empty = EnvironmentSnapshot::capture();
    CHECK(empty[Key::MOE_L].present());
    CHECK(!empty[Key::MOE_L].nonempty());
    CHECK_EQ(empty[Key::MOE_L].source, environment::Source::Canonical);
    CHECK(!RuntimeEnvironment(empty).moe.lanes);
#endif
}

CACHEDMOE_TEST(runtime_environment, distinct_flag_parsers_preserve_legacy_semantics) {
    EnvironmentSnapshot raw;
    raw.set(Key::BATCH_GPU_ROUTE, "1x");
    raw.set(Key::DSPARK_ONECB, "10");
    raw.set(Key::MGT_PAIR_DOT, "1x");
    raw.set(Key::MGT_FOLD_SCALE, "");
    raw.set(Key::SE_CHECK, "0");
    raw.set(Key::DSPARK_MEGA_DIAG, "");
    raw.set(Key::MASK_DYNAMIC_LRU, "0x");
    raw.set(Key::IO_ENGRAM_DEADLINE, "1x");
    raw.set(Key::PERF_COUNTERS, "");
    RuntimeEnvironment env(raw);
    GpuExecutionConfig gpu;
    DecodeExecutionConfig decode;
    gpu.apply_environment(env);
    decode.apply_environment(env);
    CHECK(!gpu.batch_gpu_route && !gpu.draft_onecb);
    CHECK(gpu.mgt_pair_dot && gpu.mgt_fold_scale && gpu.draft_diagnostics);
    CHECK(decode.shared_early_check && decode.dynamic_mask_lru);
    CHECK(decode.engram_deadline == false);
    CHECK(!env.perf_counters);
    raw.set(Key::BATCH_GPU_ROUTE, "1");
    CHECK(RuntimeEnvironment(raw).gpu.batch_gpu_route == true);
}

CACHEDMOE_TEST(runtime_environment, api_values_survive_absence_but_empty_clears_optional_flags) {
    EnvironmentSnapshot raw;
    DecodeExecutionConfig config;
    config.shared_early = true;
    config.eager_moe = false;
    config.gpu_wait_budget_seconds = 47;
    config.apply_environment(RuntimeEnvironment(raw));
    CHECK(config.shared_early == true && config.eager_moe == false);
    CHECK_EQ(config.gpu_wait_budget_seconds, 47.0);
    raw.set(Key::SHARED_EARLY, "");
    raw.set(Key::MS_EAGER_MOE, "");
    raw.set(Key::GPU_WAIT_S, "bogus");
    config.apply_environment(RuntimeEnvironment(raw));
    CHECK(!config.shared_early && !config.eager_moe);
    CHECK_EQ(config.gpu_wait_budget_seconds, configuration::kGpuWaitSeconds);
}

CACHEDMOE_TEST(runtime_environment, numeric_prefix_zero_and_shape_inheritance_are_not_booleanized) {
    EnvironmentSnapshot raw;
    raw.set(Key::MOE_L, "16tail");
    raw.set(Key::MOE_R, "not-a-number");
    raw.set(Key::MOE_LB, "0");
    raw.set(Key::MOE_HQUANT, "-1tail");
    raw.set(Key::IO_P0_QD, "0");
    raw.set(Key::IO_SUBMIT_THREADS, "100tail");
    const RuntimeEnvironment env(raw);
    CHECK_EQ(env.moe.lanes.value(), 16u);
    CHECK(!env.moe.rows);
    CHECK_EQ(env.moe.lanes_b.value(), 0u);
    CHECK_EQ(env.moe.hquant.value(), UINT32_MAX);
    CHECK(!env.moe.union_lanes_b && !env.moe.union_rows_b);
    CHECK_EQ(env.io.p0_qd.value(), 0u);
    CHECK_EQ(env.io.submit_threads.value(), 100u); // scheduler clamps after cfg/caps
}

CACHEDMOE_TEST(runtime_environment, slot_cap_presence_and_value_are_one_owned_source) {
    EnvironmentSnapshot raw;
    const RuntimeEnvironment absent(raw);
    CHECK_EQ(absent.cache_slot_cap, configuration::kAutoSlotCap);
    CHECK(!absent.cache_slot_cap_nonempty);
    raw.set(Key::CACHE_SLOT_CAP, "");
    CHECK(!RuntimeEnvironment(raw).cache_slot_cap_nonempty);
    raw.set(Key::CACHE_SLOT_CAP, "0");
    const RuntimeEnvironment zero(raw);
    CHECK(zero.cache_slot_cap_nonempty && zero.cache_slot_cap == 0);
    raw.set(Key::CACHE_SLOT_CAP, "-10");
    CHECK_EQ(RuntimeEnvironment(raw).cache_slot_cap, configuration::kAutoSlotCap);
    raw.set(Key::CACHE_SLOT_CAP, "5500suffix");
    CHECK_EQ(RuntimeEnvironment(raw).cache_slot_cap, 5500u);
}

CACHEDMOE_TEST(runtime_environment, strict_budget_is_irrelevant_without_tau) {
    EnvironmentSnapshot raw;
    raw.set(Key::MASK_WAIT_BUDGET, "garbage");
    const RuntimeEnvironment off(raw);
    CHECK(!off.mask_wait.enabled && !off.mask_wait.error);
    raw.set(Key::MASK_WAIT_TAU, "0.2");
    CHECK(RuntimeEnvironment(raw).mask_wait.error.has_value());
    raw.set(Key::MASK_WAIT_BUDGET, "0,0");
    const RuntimeEnvironment unlimited(raw);
    CHECK(unlimited.mask_wait.enabled && !unlimited.mask_wait.error);
    CHECK_EQ(unlimited.mask_wait.tau, 0.2);
    CHECK_EQ(unlimited.mask_wait.experts, 0u);
    CHECK_EQ(unlimited.mask_wait.milliseconds, 0.0);
    for (const auto invalid : {"", "nan", "inf", "-1", "1.01", "0.2junk"}) {
        raw.set(Key::MASK_WAIT_TAU, invalid);
        CHECK(RuntimeEnvironment(raw).mask_wait.error.has_value());
    }
}

CACHEDMOE_TEST(runtime_environment, paths_preserve_raw_empty_and_list_precedence) {
    EnvironmentSnapshot raw;
    raw.set(Key::SHADER_DIR, "");
    raw.set(Key::KV_DIR, "");
    raw.set(Key::MODEL_MIRRORS, " ; \"disk-a\" ; disk-b; ");
    raw.set(Key::MIRROR_WEIGHTS, "4.6;bad; -1");
    const RuntimeEnvironment env(raw);
    CHECK(env.shader_dir.empty());
    CHECK(env.kv_dir.has_value() && env.kv_dir->empty());
    REQUIRE(env.model_mirrors.has_value());
    CHECK_EQ(env.model_mirrors->size(), 2u);
    CHECK_EQ((*env.model_mirrors)[0], "disk-a");
    CHECK_EQ((*env.model_mirrors)[1], "disk-b");
    REQUIRE(env.mirror_weights.has_value());
    CHECK_EQ(env.mirror_weights->size(), 3u);
    CHECK_EQ((*env.mirror_weights)[1], 0.0); // normalization remains at source sizing
}

CACHEDMOE_TEST(runtime_environment, io_widening_and_capability_clamp_use_same_snapshot) {
    EnvironmentSnapshot raw;
    raw.set(Key::IO_P0_QD, "64");
    raw.set(Key::IO_P0_INFLIGHT_MB, "128");
    raw.set(Key::IO_ENGRAM_QD, "128");
    raw.set(Key::IO_SUBMIT_THREADS, "3");
    IoConfig cfg;
    cfg.environment = resolved(raw);
    storage::IoEngine::runtime_shape(cfg);
    CHECK_EQ(cfg.max_inflight_ops, 128u);
    CHECK_EQ(cfg.max_inflight_bytes, 128u << 20);
    // An ambient mutation after backend sizing must not create a different
    // runtime policy or request a queue shape that the backend never received.
    test::ScopedEnvironment changed("CACHEDMOE_IO_P0_QD", "1");
    storage::IoEngine io;
    REQUIRE_OK(io.start(std::make_unique<test::FakeBackend>(std::vector<std::byte>{}, 32), cfg));
    CHECK_EQ(io.tuning().p0_qd, 32u);
    CHECK_EQ(io.tuning().engram_qd, 32u);
    CHECK_EQ(io.tuning().p0_inflight_bytes, 128ull << 20);
    CHECK_EQ(io.tuning().bg_qd, IoConfig{}.max_inflight_ops);
    CHECK_EQ(io.tuning().bg_inflight_bytes, IoConfig{}.max_inflight_bytes);
    CHECK_EQ(io.tuning().submit_threads, 3u);
    io.stop();
}

CACHEDMOE_TEST(runtime_environment, standalone_io_reinit_captures_new_epoch) {
    test::ScopedEnvironment depth("CACHEDMOE_IO_P0_QD", "2");
    IoConfig cfg;
    storage::IoEngine io;
    REQUIRE_OK(io.start(std::make_unique<test::FakeBackend>(std::vector<std::byte>{}), cfg));
    CHECK_EQ(io.tuning().p0_qd, 2u);
    depth.set("4");
    CHECK_EQ(io.tuning().p0_qd, 2u);
    io.stop();
    REQUIRE_OK(io.start(std::make_unique<test::FakeBackend>(std::vector<std::byte>{}), cfg));
    CHECK_EQ(io.tuning().p0_qd, 4u);
    io.stop();
}

CACHEDMOE_TEST(runtime_environment, engine_failed_reinit_still_resolves_new_startup_policy) {
    EnvironmentSnapshot first;
    first.set(Key::ROUTE_RESIDENT_ONLY, "mask");
    first.set(Key::GATE_PROBE, "1");
    first.set(Key::MOE_OVERLAP, "0");
    first.set(Key::PREFILL_HANDOFF, "0");
    first.set(Key::HEAT_FILE, "first-epoch.txt");
    RuntimeConfig cfg;
    cfg.environment = resolved(first);
    runtime::Engine engine(cfg.environment);
    // No model supplied: this deliberately stops before files, IO or Vulkan.
    CHECK_ERR(engine.init(cfg), Err::InvalidArgument);
    CHECK(engine.resident_only() == configuration::ResidentPolicy::Mask);
    CHECK(engine.gate_probe_on());
    const auto epoch = engine.config().environment;
    test::ScopedEnvironment changed("CACHEDMOE_HEAT_FILE", "late-epoch.txt");
    CHECK_EQ(epoch->heat_file, "first-epoch.txt");
    cfg.environment = resolved();
    CHECK_ERR(engine.init(cfg), Err::InvalidArgument);
    CHECK(engine.resident_only() == runtime::Engine::ResidentOnly::Off);
    CHECK(!engine.gate_probe_on());
    CHECK(engine.config().environment->overlap && engine.config().environment->prefill_handoff);
    CHECK(engine.config().environment->heat_file.empty());
}

CACHEDMOE_TEST(runtime_environment, explicit_mask_cache_control_survives_reinit) {
    EnvironmentSnapshot raw;
    raw.set(Key::MASK_DYNAMIC_LRU, "1");
    RuntimeConfig cfg;
    cfg.environment = resolved(raw);
    runtime::Engine engine(cfg.environment);
    engine.set_mask_cache_fixed(true);
    CHECK_ERR(engine.init(cfg), Err::InvalidArgument);
    engine.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    CHECK(engine.store().fixed_cache());
    CHECK_ERR(engine.init(cfg), Err::InvalidArgument);
    engine.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    CHECK(engine.store().fixed_cache());
}

CACHEDMOE_TEST(runtime_environment,
               common_generation_options_use_typed_defaults_and_explicit_values) {
    auto empty = json_parse("{}");
    REQUIRE(empty.has_value());
    runtime::GenerateRequest request;
    const auto defaults = request;
    cli::apply_generate_options(*empty, request);
    CHECK_EQ(request.max_tokens, defaults.max_tokens);
    CHECK_EQ(request.sampling.temperature, defaults.sampling.temperature);
    CHECK_EQ(request.sampling.top_p, defaults.sampling.top_p);
    CHECK_EQ(request.sampling.seed, defaults.sampling.seed);
    CHECK_EQ(request.reuse, defaults.reuse);
    CHECK(request.stop_ids == defaults.stop_ids);
    auto override = json_parse(
        R"({"max_tokens":7,"temperature":0,"top_p":0.8,"seed":19,"reuse":false,"stop_ids":[2,3]})");
    REQUIRE(override.has_value());
    cli::apply_generate_options(*override, request);
    CHECK_EQ(request.max_tokens, 7u);
    CHECK_EQ(request.sampling.temperature, 0.0f);
    CHECK_EQ(request.sampling.top_p, 0.8f);
    CHECK_EQ(request.sampling.seed, 19u);
    CHECK(!request.reuse);
    CHECK(request.stop_ids == std::vector<uint32_t>({2, 3}));
    // Null fields use this request's typed value, the same in both adapters.
    auto nulls = json_parse(
        R"({"max_tokens":null,"temperature":null,"top_p":null,"seed":null,"reuse":null})");
    REQUIRE(nulls.has_value());
    cli::apply_generate_options(*nulls, request);
    CHECK_EQ(request.max_tokens, 7u);
    CHECK_EQ(request.sampling.temperature, 0.0f);
    CHECK_EQ(request.sampling.seed, 19u);
    CHECK(!request.reuse);
}

CACHEDMOE_TEST(runtime_environment, draft_fp8_is_startup_only_and_rejects_unsupported_modes) {
    EnvironmentSnapshot raw;
    GpuExecutionConfig defaults;
    defaults.apply_environment(RuntimeEnvironment(raw));
    CHECK(!defaults.draft_head_fp8);
    raw.set(Key::DSPARK_HEAD_FP8, "1");
    auto environment = std::make_shared<RuntimeEnvironment>(raw);
    GpuExecutionConfig captured;
    captured.apply_environment(*environment);
    CHECK(captured.draft_head_fp8);
    raw.set(Key::DSPARK_HEAD_FP8, "0");
    CHECK(captured.draft_head_fp8);
    GpuExecutionConfig later;
    later.apply_environment(RuntimeEnvironment(raw));
    CHECK(!later.draft_head_fp8);
    RuntimeConfig cfg;
    cfg.environment = environment;
    runtime::Engine no_spec;
    auto rejected = no_spec.init(cfg);
    CHECK(!rejected);
    CHECK(rejected.error().message.find("requires speculation") != std::string::npos);
    cfg.speculation.enabled = true;
    cfg.gpu.draft_mega = true;
    runtime::Engine mega;
    auto unsupported = mega.init(cfg);
    CHECK(!unsupported);
    CHECK(unsupported.error().message.find("non-mega") != std::string::npos);
}
