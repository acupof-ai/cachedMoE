#include "cli/kv_options.h"
#include "tests/test_framework.h"

using namespace cachedmoe;

CACHEDMOE_TEST(serve_options, disabled_disk_wins_in_both_argument_orders) {
    configuration::EnvironmentSnapshot raw;
    raw.set(configuration::Key::KV_DIR, "environment/cache");
    const configuration::RuntimeEnvironment env(raw);
    for (const bool directory_first : {true, false}) {
        runtime::KvDiskOptions disk;
        cli::KvCommandLineOptions args;
        if (directory_first)
            args.set_directory(disk, "literal/../cache");
        args.disable();
        if (!directory_first)
            args.set_directory(disk, "literal/../cache");
        bool inspected_unrelated_root = false;
        auto root = cli::resolve_kv_options(
            disk, args.disabled, args.explicit_directory, env, "checkpoint",
            [&]() -> Result<state_paths::StateRoot> {
                inspected_unrelated_root = true;
                return fail(Err::Io, "unrelated default root must not be inspected");
            });
        REQUIRE(root.has_value());
        CHECK(!root->has_value());
        CHECK(disk.dir.empty() && disk.model_tag.empty());
        CHECK(!inspected_unrelated_root);
    }
}

CACHEDMOE_TEST(serve_options, explicit_literal_wins_without_default_root_inspection) {
    configuration::EnvironmentSnapshot raw;
    raw.set(configuration::Key::KV_DIR, "environment/cache");
    const configuration::RuntimeEnvironment env(raw);
    runtime::KvDiskOptions disk;
    cli::KvCommandLineOptions args;
    args.set_directory(disk, "literal/../cache");
    auto root = cli::resolve_kv_options(
        disk, args.disabled, args.explicit_directory, env, "checkpoint",
        []() -> Result<state_paths::StateRoot> { return fail(Err::Io, "unrelated root"); });
    REQUIRE(root.has_value());
    CHECK(!root->has_value());
    CHECK_EQ(disk.dir, "literal/../cache");
    CHECK_EQ(disk.model_tag, "checkpoint");
}

CACHEDMOE_TEST(serve_options, empty_environment_disables_and_absent_environment_selects_default) {
    configuration::EnvironmentSnapshot raw;
    raw.set(configuration::Key::KV_DIR, "");
    runtime::KvDiskOptions disk;
    bool called = false;
    auto select = [&]() -> Result<state_paths::StateRoot> {
        called = true;
        return state_paths::StateRoot{std::filesystem::path("root"), "canonical-new", false};
    };
    auto empty = cli::resolve_kv_options(disk, false, false, configuration::RuntimeEnvironment(raw),
                                         "checkpoint", select);
    REQUIRE(empty.has_value());
    CHECK(!empty->has_value() && disk.dir.empty() && !called);
    raw.set(configuration::Key::KV_DIR, std::nullopt);
    auto default_root = cli::resolve_kv_options(
        disk, false, false, configuration::RuntimeEnvironment(raw), "checkpoint", select);
    REQUIRE(default_root.has_value());
    CHECK(default_root->has_value() && called);
    CHECK_EQ(disk.dir, (std::filesystem::path("root") / "kv" / "checkpoint").string());
    CHECK_EQ(disk.model_tag, "checkpoint");
}

CACHEDMOE_TEST(serve_options, gpu_route_stream_guard_matches_the_effective_exact_one_parser) {
    for (const char *text : {"0", "1x", "10", "1"}) {
        configuration::EnvironmentSnapshot raw;
        raw.set(configuration::Key::BATCH_GPU_ROUTE, text);
        const auto env = std::make_shared<const configuration::RuntimeEnvironment>(raw);
        GpuExecutionConfig early_cli;
        early_cli.apply_environment(*env);
        runtime::Engine engine(env); // constructor's pre-init guard uses same epoch
        CHECK_EQ(early_cli.batch_gpu_route, engine.config().gpu.batch_gpu_route);
        CHECK_EQ(early_cli.batch_gpu_route, std::string_view(text) == "1");
        if (early_cli.batch_gpu_route)
            CHECK_ERR(engine.set_streams(2), Err::FailedPrecondition);
    }
}
