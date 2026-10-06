// Entry point for the whole test binary.
//
//   cachedmoe_tests            run everything
//   cachedmoe_tests io         run cases whose "suite.name" contains "io"
//   cachedmoe_tests --list     print case names
#include "core/log.h"
#include "tests/test_framework.h"

int main(int argc, char** argv) {
    // Tests assert on return values, not on log output; keep the noise down.
    cachedmoe::set_log_level(cachedmoe::LogLevel::Warn);
    return cachedmoe::test::run_all(argc, argv);
}
