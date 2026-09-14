# One test binary, registered with ctest both as a whole and per suite so a
# failure names the area rather than just "tests failed".

add_executable(deepmoe_tests
    test_main.cpp
    test_core.cpp
    test_dequant.cpp
    test_store.cpp
    test_io.cpp
    test_model.cpp
    test_integration.cpp
    test_gpu_core.cpp
    test_gpu_moe.cpp
    test_gpu_attn.cpp
    test_gpu_layer.cpp
    test_decode.cpp
    test_gpu_dspark.cpp
    test_gpu_prefill.cpp
    test_dspark_tree.cpp)

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

# The Vulkan plumbing of design §3.3 / §5.3 / §7.1. Needs a GPU but no weights.
add_test(NAME "suite.gpu" COMMAND deepmoe_tests "gpu.")
set_tests_properties("suite.gpu" PROPERTIES
    LABELS "needs-gpu"
    SKIP_REGULAR_EXPRESSION "SKIP gpu:")

# The design §7.2-§7.8 attention kernels against the oracle's L2 tensors. Same
# gate as gpu_moe, plus tests/data/l2 must exist.
add_test(NAME "suite.gpu_attn" COMMAND deepmoe_tests "gpu_attn.")
set_tests_properties("suite.gpu_attn" PROPERTIES
    LABELS "needs-model;needs-gpu"
    SKIP_REGULAR_EXPRESSION "set DEEPMOE_MODEL_DIR|SKIP gpu_attn")

# One whole decoder layer, chained, against the oracle's block output.
add_test(NAME "suite.gpu_layer" COMMAND deepmoe_tests "gpu_layer.")
set_tests_properties("suite.gpu_layer" PROPERTIES
    LABELS "needs-model;needs-gpu"
    SKIP_REGULAR_EXPRESSION "set DEEPMOE_MODEL_DIR|SKIP gpu_layer")

# The whole decode step -- forty layers, the head and the argmax -- against the
# oracle's L3 tokens. Needs the ~17.7 GB pinned set and tests/data/l3.
add_test(NAME "suite.decode" COMMAND deepmoe_tests "decode.")
set_tests_properties("suite.decode" PROPERTIES
    LABELS "needs-model;needs-gpu"
    SKIP_REGULAR_EXPRESSION "set DEEPMOE_MODEL_DIR|SKIP decode")

# The design §7.9 kernels on the real checkpoint. Same gate, and it additionally
# skips itself when there is no usable Vulkan device.
add_test(NAME "suite.gpu_moe" COMMAND deepmoe_tests "gpu_moe.")
set_tests_properties("suite.gpu_moe" PROPERTIES
    LABELS "needs-model;needs-gpu"
    SKIP_REGULAR_EXPRESSION "set DEEPMOE_MODEL_DIR|SKIP gpu_moe")

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

# The design §7.12 DSpark draft kernels against tools/oracle_dspark.py's golden
# draft cycle. Needs the ~2 GB of mtp + head weights and tests/data/dspark.
add_test(NAME "suite.gpu_dspark" COMMAND deepmoe_tests "gpu_dspark.")
set_tests_properties("suite.gpu_dspark" PROPERTIES
    LABELS "needs-model;needs-gpu"
    SKIP_REGULAR_EXPRESSION "set DEEPMOE_MODEL_DIR|SKIP gpu_dspark")

# design §7.12 / §10.1: T_draft and bandwidth of one draft cycle. Registered
# here rather than in the top-level CMakeLists.txt so the DSpark track touches
# only its own lines; it builds with the tests.
add_executable(dspark_bench "${CMAKE_SOURCE_DIR}/bench/dspark_bench.cpp")
target_link_libraries(dspark_bench PRIVATE deepmoe_core)
add_dependencies(dspark_bench shaders)

# The batched prefill kernels of docs/p3_prefill.md against tools/oracle_prefill.py's
# start_pos == 0 tensors, then a whole forty-layer GPU prefill handed to the decode
# engine (Track L). Needs tests/data/prefill and tests/data/l3.
add_test(NAME "suite.gpu_prefill" COMMAND deepmoe_tests "gpu_prefill.")
set_tests_properties("suite.gpu_prefill" PROPERTIES
    LABELS "needs-model;needs-gpu"
    SKIP_REGULAR_EXPRESSION "set DEEPMOE_MODEL_DIR|SKIP gpu_prefill")

# Track K2: the CPU half of DSpark tree sampling (cpu/dspark_tree) bit-exact against
# tools/dspark_tree.py's golden cycles. No model, no GPU.
add_test(NAME "suite.dspark_tree" COMMAND deepmoe_tests "dspark_tree.")
set_tests_properties("suite.dspark_tree" PROPERTIES LABELS "unit")
