# One test binary, registered with ctest both as a whole and per suite so a
# failure names the area rather than just "tests failed".

add_executable(deepmoe_tests
    test_main.cpp
    test_core.cpp
    test_dequant.cpp
    test_store.cpp
    test_io.cpp
    test_model.cpp)

target_link_libraries(deepmoe_tests PRIVATE deepmoe_core)
target_compile_definitions(deepmoe_tests PRIVATE
    DEEPMOE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_test(NAME deepmoe_tests COMMAND deepmoe_tests)

foreach(_suite align layout types json profiler status bytes
                dequant gate slab expert_store planner
                io v41_config manifest block_info kvcache)
    add_test(NAME "suite.${_suite}" COMMAND deepmoe_tests "${_suite}.")
endforeach()
