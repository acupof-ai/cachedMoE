// A ~150-line test framework. No network fetch, no submodule: deepMoE has no
// third-party dependencies (design §14) and a unit test harness is not where to
// start making exceptions.
//
// Usage:
//   DEEPMOE_TEST(suite_name, case_name) { CHECK(...); REQUIRE_OK(...); }
//   int main(int argc, char** argv) { return deepmoe::test::run_all(argc, argv); }
//
// CHECK   records a failure and continues.
// REQUIRE records a failure and returns from the case.
// REQUIRE_OK / CHECK_ERR are for Result<T>.
//
// Ownership/threading: the registry is a function-local static, filled at
// static-init time by DEEPMOE_TEST. Tests run sequentially on one thread.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace deepmoe::test {

struct Case {
    const char* suite;
    const char* name;
    void (*fn)(struct Context&);
};

struct Context {
    int         failures = 0;
    bool        abort_case = false;
    const Case* current = nullptr;

    void fail(const char* file, int line, const std::string& what) {
        ++failures;
        // Keep the two streams in order: stdout is block-buffered when piped,
        // stderr is not, so a failure would otherwise jump ahead of its [RUN].
        std::fflush(stdout);
        std::fprintf(stderr, "  FAIL %s:%d\n       %s\n", file, line, what.c_str());
        std::fflush(stderr);
    }
};

inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

struct Registrar {
    Registrar(const char* suite, const char* name, void (*fn)(Context&)) {
        registry().push_back(Case{suite, name, fn});
    }
};

// `pattern` matches when it is a substring of "suite.name".
inline int run_all(int argc, char** argv) {
    std::string_view pattern;
    bool list_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--list") == 0) list_only = true;
        else pattern = argv[i];
    }

    int failed_cases = 0, ran = 0, total_failures = 0;
    for (const Case& c : registry()) {
        const std::string full = std::string(c.suite) + "." + c.name;
        if (!pattern.empty() && full.find(pattern) == std::string::npos) continue;
        if (list_only) { std::printf("%s\n", full.c_str()); continue; }

        Context ctx;
        ctx.current = &c;
        std::printf("[ RUN ] %s\n", full.c_str());
        c.fn(ctx);
        ++ran;
        total_failures += ctx.failures;
        if (ctx.failures) { ++failed_cases; std::printf("[FAIL ] %s (%d)\n", full.c_str(), ctx.failures); }
        else               std::printf("[  ok ] %s\n", full.c_str());
    }
    if (list_only) return 0;
    std::printf("\n%d case(s) run, %d failed, %d assertion failure(s)\n", ran, failed_cases, total_failures);
    return failed_cases ? 1 : 0;
}

// Best-effort stringifier: std::format when the type supports it, the integer
// value for an enum, a placeholder otherwise. Keeps CHECK_EQ usable on
// ExpertKey, SlotState and friends without writing formatters for them.
template <class T>
std::string show(const T& v) {
    // std::formattable, not a requires-expression around std::format: a
    // non-formattable argument is a hard consteval error inside
    // basic_format_string, not a substitution failure.
    if constexpr (std::formattable<T, char>) {
        return std::format("{}", v);
    } else if constexpr (std::is_enum_v<T>) {
        return std::to_string(static_cast<long long>(v));
    } else {
        return "<value>";
    }
}

// Approximate comparison used by the numeric oracles of design §12.
inline bool close(double a, double b, double rel = 1e-6, double abs_tol = 1e-9) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    const double d = std::fabs(a - b);
    if (d <= abs_tol) return true;
    const double m = std::fmax(std::fabs(a), std::fabs(b));
    return m > 0 && d / m <= rel;
}

}  // namespace deepmoe::test

#define DEEPMOE_TEST(suite, name)                                                      \
    static void dm_test_##suite##_##name(::deepmoe::test::Context& _ctx);               \
    static ::deepmoe::test::Registrar dm_reg_##suite##_##name(                          \
        #suite, #name, &dm_test_##suite##_##name);                                      \
    static void dm_test_##suite##_##name(::deepmoe::test::Context& _ctx)

#define DM_UNUSED_CTX() (void)_ctx

#define CHECK(expr)                                                                     \
    do {                                                                                \
        if (!(expr)) _ctx.fail(__FILE__, __LINE__, "CHECK failed: " #expr);              \
    } while (0)

#define REQUIRE(expr)                                                                   \
    do {                                                                                \
        if (!(expr)) { _ctx.fail(__FILE__, __LINE__, "REQUIRE failed: " #expr); return; }\
    } while (0)

#define DM_CMP_MSG(kind, a, b)                                                          \
    (std::string(kind " failed: " #a " == " #b " (") + ::deepmoe::test::show(_a) +       \
     " vs " + ::deepmoe::test::show(_b) + ")")

#define CHECK_EQ(a, b)                                                                  \
    do {                                                                                \
        auto _a = (a); auto _b = (b);                                                    \
        if (!(_a == _b)) _ctx.fail(__FILE__, __LINE__, DM_CMP_MSG("CHECK_EQ", a, b));    \
    } while (0)

#define REQUIRE_EQ(a, b)                                                                \
    do {                                                                                \
        auto _a = (a); auto _b = (b);                                                    \
        if (!(_a == _b)) {                                                               \
            _ctx.fail(__FILE__, __LINE__, DM_CMP_MSG("REQUIRE_EQ", a, b));               \
            return;                                                                      \
        }                                                                                \
    } while (0)

#define CHECK_CLOSE(a, b, rel)                                                          \
    do {                                                                                \
        const double _a = static_cast<double>(a), _b = static_cast<double>(b);            \
        if (!::deepmoe::test::close(_a, _b, rel))                                        \
            _ctx.fail(__FILE__, __LINE__,                                                \
                      std::format("CHECK_CLOSE failed: " #a " ~ " #b " ({} vs {}, rel {})",\
                                  _a, _b, rel));                                         \
    } while (0)

// Result<T> helpers. `expr` is evaluated once.
//
// The stringified expression is concatenated, never used as a std::format
// format string: test expressions routinely contain braces (JSON literals,
// brace-init) which would be parsed as replacement fields.
#define CHECK_OK(expr)                                                                  \
    do {                                                                                \
        auto&& _r = (expr);                                                              \
        if (!_r)                                                                         \
            _ctx.fail(__FILE__, __LINE__,                                                \
                      std::string("CHECK_OK failed: " #expr " -> ") + _r.error().str());  \
    } while (0)

#define REQUIRE_OK(expr)                                                                \
    do {                                                                                \
        auto&& _r = (expr);                                                              \
        if (!_r) {                                                                       \
            _ctx.fail(__FILE__, __LINE__,                                                \
                      std::string("REQUIRE_OK failed: " #expr " -> ") + _r.error().str());\
            return;                                                                      \
        }                                                                                \
    } while (0)

// Asserts the call failed with a specific error code. The parameter is `want`,
// not `code`, so it does not shadow Status::code inside the comparison.
#define CHECK_ERR(expr, want)                                                           \
    do {                                                                                \
        auto&& _r = (expr);                                                              \
        if (_r)                                                                          \
            _ctx.fail(__FILE__, __LINE__, "CHECK_ERR failed: " #expr " unexpectedly succeeded");\
        else if (_r.error().code != (want))                                              \
            _ctx.fail(__FILE__, __LINE__,                                                \
                      std::string("CHECK_ERR failed: " #expr " -> ") + _r.error().str()   \
                          + " (wanted " #want ")");                                      \
    } while (0)
