# One test binary, registered with ctest both as a whole and per suite so a
# failure names the area rather than just "tests failed".

add_executable(deepmoe_tests
    test_main.cpp
    test_core.cpp
    test_dequant.cpp
    test_store.cpp
    test_io.cpp
    test_model.cpp
    test_integration.cpp)

target_link_libraries(deepmoe_tests PRIVATE deepmoe_core)
target_compile_definitions(deepmoe_tests PRIVATE
    DEEPMOE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_test(NAME deepmoe_tests COMMAND deepmoe_tests)

foreach(_suite align layout types json profiler status bytes
                dequant gate slab expert_store planner
                io v41_config manifest block_info kvcache)
    add_test(NAME "suite.${_suite}" COMMAND deepmoe_tests "${_suite}.")
endforeach()

# The integration suite needs the real 510 GB checkpoint. It is registered
# separately so `ctest -L needs-model` runs only it and `ctest -LE needs-model`
# skips it outright; without DEEPMOE_MODEL_DIR the cases skip themselves and
# say so, which ctest reports as "Skipped" rather than a pass.
add_test(NAME "suite.integration" COMMAND deepmoe_tests "integration.")
set_tests_properties("suite.integration" PROPERTIES
    LABELS "needs-model"
    SKIP_REGULAR_EXPRESSION "set DEEPMOE_MODEL_DIR")

# Everything else must keep passing on a machine with no weights at all.
foreach(_t deepmoe_tests align layout types json profiler status bytes
           dequant gate slab expert_store planner io v41_config manifest
           block_info kvcache)
    if(_t STREQUAL "deepmoe_tests")
        set_tests_properties("${_t}" PROPERTIES LABELS "unit;needs-model")
    else()
        set_tests_properties("suite.${_t}" PROPERTIES LABELS "unit")
    endif()
endforeach()
