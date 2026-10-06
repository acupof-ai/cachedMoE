#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "core/config.h"
#include "core/env.h"
#include "tests/env_guard.h"
#include "tests/test_framework.h"

using namespace deepmoe;
const char* env_alias_other_tu(const char* key);

namespace {

class StderrCapture {
public:
    StderrCapture() {
        std::fflush(stderr);
        file_ = std::tmpfile();
#ifdef _WIN32
        saved_ = _dup(_fileno(stderr));
        if (file_ && saved_ >= 0) _dup2(_fileno(file_), _fileno(stderr));
#else
        saved_ = dup(fileno(stderr));
        if (file_ && saved_ >= 0) dup2(fileno(file_), fileno(stderr));
#endif
    }
    ~StderrCapture() {
        std::fflush(stderr);
#ifdef _WIN32
        if (saved_ >= 0) { _dup2(saved_, _fileno(stderr)); _close(saved_); }
#else
        if (saved_ >= 0) { dup2(saved_, fileno(stderr)); close(saved_); }
#endif
        if (file_) std::fclose(file_);
    }
    bool valid() const { return file_ && saved_ >= 0; }
    std::string text() {
        std::fflush(stderr);
        if (!file_) return {};
        std::rewind(file_);
        std::string result;
        std::array<char, 512> bytes;
        while (const size_t n = std::fread(bytes.data(), 1, bytes.size(), file_))
            result.append(bytes.data(), n);
        return result;
    }
private:
    std::FILE* file_ = nullptr;
    int saved_ = -1;
};

} // namespace

DEEPMOE_TEST(env_alias, raw_presence_and_source_vectors) {
    struct Case {
        const char* canonical;
        const char* legacy;
        const char* value;
        environment::Source source;
        const char* warning;
    };
    const Case cases[] = {
        {nullptr, nullptr, nullptr, environment::Source::Absent, nullptr},
        {nullptr, "1", "1", environment::Source::Legacy, "legacy fallback"},
        {"1", nullptr, "1", environment::Source::Canonical, nullptr},
        {"1", "1", "1", environment::Source::Canonical, nullptr},
        {"0", "1", "0", environment::Source::Canonical, "conflict: canonical wins"},
        {"", "1", "", environment::Source::Canonical, "conflict: canonical wins"},
        {nullptr, "", "", environment::Source::Legacy, "legacy fallback"},
        {"", "", "", environment::Source::Canonical, nullptr},
        {" ", "1", " ", environment::Source::Canonical, "conflict: canonical wins"},
        {"10", "1", "10", environment::Source::Canonical, "conflict: canonical wins"},
    };
    for (size_t i = 0; i < std::size(cases); ++i) {
        const auto& c = cases[i];
#ifdef _WIN32
        // The Windows CRT removes empty values. Empty-presence cases run on
        // Linux, the supported main line; they must not be faked as absence.
        if ((c.canonical && !*c.canonical) || (c.legacy && !*c.legacy)) continue;
#endif
        const std::string name = "CACHEDMOE_ENV_VECTOR_" + std::to_string(i);
        test::ScopedEnvironment scope(name.c_str());
        const auto names = environment::names(name);
        if (c.canonical) test::ScopedEnvironment::write(names.canonical.c_str(), c.canonical);
        if (c.legacy) test::ScopedEnvironment::write(names.legacy.c_str(), c.legacy);
        StderrCapture capture;
        CHECK(capture.valid());
        const auto result = environment::lookup(name.c_str());
        CHECK_EQ(result.source, c.source);
        CHECK_EQ(result.present(), c.value != nullptr);
        if (c.value) CHECK_EQ(std::string(result.value), std::string(c.value));
        CHECK_EQ(environment::get(names.legacy.c_str()), result.value);
        const std::string warning = capture.text();
        CHECK_EQ(std::count(warning.begin(), warning.end(), '\n'), c.warning ? 1 : 0);
        if (c.warning) CHECK(warning.find(c.warning) != std::string::npos);
    }
}

DEEPMOE_TEST(env_alias, warnings_once_across_tus_and_threads_without_values) {
    test::ScopedEnvironment scope("CACHEDMOE_ENV_WARNING_ONCE");
    const auto names = environment::names(scope.key);
    test::ScopedEnvironment::write(names.legacy.c_str(), "secret-legacy-payload");
    StderrCapture capture;
    CHECK(capture.valid());
    std::vector<std::thread> threads;
    for (int n = 0; n < 8; ++n) threads.emplace_back([&, n] {
        for (int i = 0; i < 16; ++i) {
            if (n % 2) (void)environment::get(scope.key);
            else (void)env_alias_other_tu(scope.key);
        }
    });
    for (auto& thread : threads) thread.join();
    (void)environment::get(scope.key);
    (void)env_alias_other_tu(scope.key);
    const std::string fallback = capture.text();
    CHECK_EQ(std::count(fallback.begin(), fallback.end(), '\n'), 1);
    CHECK(fallback.find("legacy fallback") != std::string::npos);
    CHECK(fallback.find("secret-legacy-payload") == std::string::npos);
    test::ScopedEnvironment::write(names.canonical.c_str(), "secret-canonical-payload");
    threads.clear();
    for (int n = 0; n < 8; ++n) threads.emplace_back([&, n] {
        if (n % 2) (void)environment::get(scope.key);
        else (void)env_alias_other_tu(scope.key);
    });
    for (auto& thread : threads) thread.join();
    const std::string conflict = capture.text();
    CHECK_EQ(std::count(conflict.begin(), conflict.end(), '\n'), 2);
    CHECK(conflict.find("conflict: canonical wins") != std::string::npos);
    CHECK(conflict.find("secret-canonical-payload") == std::string::npos);
    CHECK(conflict.find("secret-legacy-payload") == std::string::npos);
}

DEEPMOE_TEST(env_alias, scoped_override_isolates_and_restores_both_families) {
    test::ScopedEnvironment outer("CACHEDMOE_ENV_SCOPE");
    const auto names = environment::names(outer.key);
    test::ScopedEnvironment::write(names.canonical.c_str(), "new-original");
    test::ScopedEnvironment::write(names.legacy.c_str(), "old-original");
    {
        test::ScopedEnvironment inner(names.legacy.c_str(), "legacy-fixture");
        CHECK(environment::raw_get(names.canonical.c_str()) == nullptr);
        CHECK_EQ(std::string(environment::get(outer.key)), "legacy-fixture");
        inner.set(nullptr);
        CHECK(!environment::lookup(outer.key).present());
    }
    CHECK_EQ(std::string(environment::raw_get(names.canonical.c_str())), "new-original");
    CHECK_EQ(std::string(environment::raw_get(names.legacy.c_str())), "old-original");
}

DEEPMOE_TEST(env_alias, consumer_parsers_keep_empty_and_nonboolean_semantics) {
    test::ScopedEnvironment onecb("CACHEDMOE_DSPARK_ONECB", "10");
    test::ScopedEnvironment trim("CACHEDMOE_DSPARK_TRIM_TAIL", "0x");
    test::ScopedEnvironment pair("CACHEDMOE_MGT_PAIR_DOT", "10");
    test::ScopedEnvironment dynamic("CACHEDMOE_MASK_DYNAMIC_LRU", "0x");
    test::ScopedEnvironment present("CACHEDMOE_SE_CHECK", "0");
    GpuExecutionConfig gpu;
    gpu.apply_environment();
    CHECK(!gpu.draft_onecb); // exact "1" only
    CHECK(!gpu.draft_trim_tail); // first character differs from '0'
    CHECK(gpu.mgt_pair_dot); // first character '1'
    DecodeExecutionConfig decode;
    decode.apply_environment();
    CHECK(decode.dynamic_mask_lru && decode.shared_early_check);
#ifndef _WIN32
    trim.set("");
    test::ScopedEnvironment::write("DEEPMOE_DSPARK_TRIM_TAIL", "0");
    gpu.apply_environment();
    CHECK(gpu.draft_trim_tail); // present empty overrides old and *e != '0'
#endif
}

DEEPMOE_TEST(env_alias, external_names_are_native) {
    test::ScopedEnvironment native("ENV_ALIAS_NATIVE_FIXTURE", "native-value");
    const auto value = environment::lookup(native.key);
    CHECK_EQ(value.source, environment::Source::Native);
    CHECK_EQ(std::string(value.value), "native-value");
}
