// Entry point for the whole test binary.
//
//   deepmoe_tests            run everything
//   deepmoe_tests io         run cases whose "suite.name" contains "io"
//   deepmoe_tests --list     print case names
#include "core/log.h"
#include "tests/test_framework.h"

int main(int argc, char** argv) {
    // Tests assert on return values, not on log output; keep the noise down.
    deepmoe::set_log_level(deepmoe::LogLevel::Warn);
    return deepmoe::test::run_all(argc, argv);
}
