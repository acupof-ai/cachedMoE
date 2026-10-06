// CPU-only root selection and real parked-KV compatibility. All fixtures live
// in temporary directories; selecting a root itself never creates any paths.
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "core/state_paths.h"
#include "runtime/session.h"
#include "tests/env_guard.h"
#include "tests/test_framework.h"

using namespace cachedmoe;
namespace fs = std::filesystem;
namespace sp = state_paths;

namespace {

struct Fixture {
    fs::path base;
    bool valid = false;

    Fixture() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        base = fs::temp_directory_path() / ("cachedmoe_state_paths_" + std::to_string(stamp));
        std::error_code ec;
        valid = fs::create_directory(base, ec) && !ec;
    }
    ~Fixture() {
        std::error_code ec;
        fs::remove_all(base, ec);
    }
    fs::path canonical() const { return base / sp::kAppDirectory; }
    fs::path legacy() const { return base / sp::kLegacyDirectory; }
};

std::string read_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

} // namespace

CACHEDMOE_TEST(state_paths, platform_cache_home_priority) {
    sp::Environment env;
    env.xdg_cache_home = "/xdg";
    env.home = "/home/user";
    env.local_appdata = "/local";
    env.temp = "/temp";
    env.tmpdir = "/tmpdir";
    auto posix = sp::cache_home(env, sp::Platform::Posix);
    REQUIRE_OK(posix);
    CHECK_EQ(posix->generic_string(), "/xdg");
    env.xdg_cache_home.clear();
    posix = sp::cache_home(env, sp::Platform::Posix);
    REQUIRE_OK(posix);
    CHECK_EQ(posix->generic_string(), "/home/user/.cache");

    auto windows = sp::cache_home(env, sp::Platform::Windows);
    REQUIRE_OK(windows);
    CHECK_EQ(windows->generic_string(), "/local");
    env.local_appdata.clear();
    windows = sp::cache_home(env, sp::Platform::Windows);
    REQUIRE_OK(windows);
    CHECK_EQ(windows->generic_string(), "/temp");
}

CACHEDMOE_TEST(state_paths, shared_temporary_fallback_order) {
    sp::Environment env;
    env.tmpdir = "/tmpdir";
    env.temp = "/temp";
    env.tmp = "/tmp-variable";
    env.tempdir = "/tempdir";
    for (const std::string_view expected : {"/tmpdir", "/temp", "/tmp-variable", "/tempdir"}) {
        auto base = sp::cache_home(env, sp::Platform::Posix);
        REQUIRE_OK(base);
        CHECK_EQ(base->generic_string(), expected);
        if (!env.tmpdir.empty()) env.tmpdir.clear();
        else if (!env.temp.empty()) env.temp.clear();
        else if (!env.tmp.empty()) env.tmp.clear();
        else env.tempdir.clear();
    }
    auto posix = sp::cache_home(env, sp::Platform::Posix);
    REQUIRE_OK(posix);
    CHECK_EQ(posix->generic_string(), "/tmp");
    auto windows = sp::cache_home(env, sp::Platform::Windows, "/system-temp");
    REQUIRE_OK(windows);
    CHECK_EQ(windows->generic_string(), "/system-temp");
}

CACHEDMOE_TEST(state_paths, native_environment_snapshot_owns_its_values) {
    test::ScopedEnvironment xdg("XDG_CACHE_HOME", "/first-xdg");
    test::ScopedEnvironment home("HOME", "/first-home");
    const auto captured = sp::read_environment();
    xdg.set("/second-xdg");
    home.set("/second-home");
    auto selected = sp::cache_home(captured, sp::Platform::Posix);
    REQUIRE_OK(selected);
    CHECK_EQ(selected->generic_string(), "/first-xdg");
    CHECK_EQ(captured.home, "/first-home");
}

CACHEDMOE_TEST(state_paths, missing_roots_select_new_without_creation) {
    Fixture f;
    REQUIRE(f.valid);
    auto selected = sp::choose_directory(f.base);
    REQUIRE_OK(selected);
    CHECK_EQ(selected->path, f.canonical());
    CHECK_EQ(selected->source, "canonical-new");
    CHECK(!selected->both_exist);
    CHECK(!fs::exists(f.canonical()));
    CHECK(!fs::exists(f.legacy()));
}

CACHEDMOE_TEST(state_paths, existing_canonical_root_is_selected) {
    Fixture f;
    REQUIRE(f.valid);
    fs::create_directory(f.canonical());
    auto selected = sp::choose_directory(f.base);
    REQUIRE_OK(selected);
    CHECK_EQ(selected->path, f.canonical());
    CHECK_EQ(selected->source, "canonical-existing");
    CHECK(!selected->both_exist);
    CHECK(!fs::exists(f.legacy()));
}

CACHEDMOE_TEST(state_paths, existing_legacy_root_loads_kv_without_migration) {
    Fixture f;
    REQUIRE(f.valid);
    fs::create_directory(f.legacy());
    const std::string model = "/models/legacy-model";
    sp::StateRoot legacy{f.legacy(), "legacy-existing"};
    runtime::KvDiskOptions options;
    options.dir = sp::model_kv_directory(legacy, model).string();
    options.model_tag = model;
    runtime::ParkedContext parked;
    parked.tokens = {11, 12, 13};
    parked.kv.positions = 2;
    REQUIRE_OK(runtime::save_parked_context(parked, options, "old-session"));
    const auto file = fs::path(options.dir) / "old-session.pkv";
    const auto original = read_file(file);
    REQUIRE(!original.empty());

    auto selected = sp::choose_directory(f.base);
    REQUIRE_OK(selected);
    CHECK_EQ(selected->source, "legacy-existing");
    CHECK(!selected->both_exist);
    options.dir = sp::model_kv_directory(*selected, model).string();
    auto loaded = runtime::load_parked_context(options, "old-session");
    REQUIRE_OK(loaded);
    CHECK_EQ(loaded->tokens, parked.tokens);
    CHECK_EQ(loaded->kv.positions, parked.kv.positions);
    CHECK_EQ(read_file(file), original);
    CHECK(!fs::exists(f.canonical()));
}

CACHEDMOE_TEST(state_paths, both_roots_prefer_canonical_and_retain_legacy_payloads) {
    Fixture f;
    REQUIRE(f.valid);
    fs::create_directory(f.canonical());
    fs::create_directory(f.legacy());
    const fs::path old_file = f.legacy() / "transcript.json";
    std::ofstream(old_file, std::ios::binary) << "{\"old\":true}\n";
    const auto original = read_file(old_file);
    auto selected = sp::choose_directory(f.base);
    REQUIRE_OK(selected);
    CHECK_EQ(selected->path, f.canonical());
    CHECK_EQ(selected->source, "canonical-existing");
    CHECK(selected->both_exist);
    CHECK_EQ(read_file(old_file), original);
}

CACHEDMOE_TEST(state_paths, canonical_file_collision_is_rejected) {
    Fixture f;
    REQUIRE(f.valid);
    std::ofstream(f.canonical(), std::ios::binary) << "retained";
    fs::create_directory(f.legacy());
    const auto original = read_file(f.canonical());
    CHECK_ERR(sp::choose_directory(f.base), Err::AlreadyExists);
    CHECK_EQ(read_file(f.canonical()), original);
}

CACHEDMOE_TEST(state_paths, unused_legacy_file_collision_is_also_rejected) {
    Fixture f;
    REQUIRE(f.valid);
    fs::create_directory(f.canonical());
    std::ofstream(f.legacy(), std::ios::binary) << "retained";
    const auto original = read_file(f.legacy());
    CHECK_ERR(sp::choose_directory(f.base), Err::AlreadyExists);
    CHECK_EQ(read_file(f.legacy()), original);
}

CACHEDMOE_TEST(state_paths, metadata_error_is_not_treated_as_absence) {
    Fixture f;
    REQUIRE(f.valid);
    const fs::path parent_file = f.base / "not-a-directory";
    std::ofstream(parent_file) << "retained";
    CHECK_ERR(sp::choose_directory(parent_file), Err::Io);
    CHECK_EQ(read_file(parent_file), "retained");
}

CACHEDMOE_TEST(state_paths, model_tag_keeps_previous_path_encoding) {
    sp::StateRoot root{fs::path("/state/cachedmoe"), "canonical-new"};
    CHECK_EQ(sp::model_kv_directory(root, "C:\\models/model:v4").generic_string(),
             "/state/cachedmoe/kv/C__models_model_v4");
}

#if !defined(_WIN32)
CACHEDMOE_TEST(state_paths, directory_symlink_is_valid_but_dangling_root_is_rejected) {
    Fixture f;
    REQUIRE(f.valid);
    const auto target = f.base / "directory-target";
    fs::create_directory(target);
    fs::create_directory_symlink(target, f.canonical());
    auto selected = sp::choose_directory(f.base);
    REQUIRE_OK(selected);
    CHECK_EQ(selected->path, f.canonical());
    fs::remove(f.canonical());
    fs::create_directory_symlink(f.base / "missing-target", f.canonical());
    auto broken = sp::choose_directory(f.base);
    CHECK(!broken);
    CHECK(fs::is_symlink(fs::symlink_status(f.canonical())));
    CHECK(!fs::exists(f.legacy()));
}
#endif
