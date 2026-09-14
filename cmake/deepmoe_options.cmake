# Build options and the per-module source lists.
#
# Platform code is confined to storage/windows and storage/linux; every other
# module is portable and is compiled for both targets (the Linux build exists so
# CI can run the CPU and storage tests -- design §14).

option(DEEPMOE_BUILD_TESTS           "Build the unit tests"              ON)
option(DEEPMOE_ENABLE_VULKAN         "Build the Vulkan GPU backend"      ON)
option(DEEPMOE_ENABLE_DIRECTSTORAGE  "Build the DirectStorage I/O backend (needs dstorage.h)" OFF)

# std::expected lives in <expected>, which libc++ guards behind C++23 even
# though the rest of the codebase is C++20. See docs/build.md.
set(DEEPMOE_CXX_STANDARD 23)

set(DEEPMOE_CORE_SOURCES
    core/json.cpp
    core/profiler.cpp)

set(DEEPMOE_MODEL_SOURCES
    model/v41_config.cpp
    model/manifest.cpp)

set(DEEPMOE_CPU_SOURCES
    cpu/dequant.cpp
    cpu/gemv_avx512.cpp
    cpu/gate.cpp)

set(DEEPMOE_STORAGE_SOURCES
    storage/file_common.cpp
    storage/io_engine.cpp)

set(DEEPMOE_STORAGE_WINDOWS_SOURCES
    storage/windows/file_win.cpp
    storage/windows/iocp.cpp
    storage/windows/directstorage.cpp)

set(DEEPMOE_STORAGE_LINUX_SOURCES
    storage/linux/file_posix.cpp
    storage/linux/io_uring.cpp)

set(DEEPMOE_STORE_SOURCES
    store/slab.cpp
    store/pinned.cpp
    store/expert_store.cpp
    store/planner.cpp
    store/predictor.cpp
    store/engram_prefetch.cpp)

set(DEEPMOE_GPU_SOURCES
    gpu/vulkan/device.cpp
    gpu/vulkan/memory.cpp
    gpu/vulkan/timeline.cpp
    gpu/vulkan/cmdbuf.cpp
    gpu/vulkan/pipeline.cpp
    gpu/vulkan/descriptor.cpp
    gpu/vulkan/rawread.cpp
    gpu/vulkan/moe_kernels.cpp
    gpu/vulkan/attn_kernels.cpp
    gpu/vulkan/decode_kernels.cpp
    gpu/vulkan/dspark_kernels.cpp)

set(DEEPMOE_RUNTIME_SOURCES
    runtime/engine.cpp
    runtime/kvstore.cpp
    runtime/decode_layer.cpp
    runtime/moe_bridge.cpp
    runtime/decode_state.cpp
    runtime/engram.cpp)

# Headers every shader in DEEPMOE_SHADERS may include; touching one rebuilds all.
set(DEEPMOE_SHADER_DEPS
    gpu/shaders/moe_common.slang
    gpu/shaders/attn_common.slang
    gpu/shaders/fp8_gemv.slang
    gpu/shaders/dspark_common.slang)

# design §7.14 dispatch list, plus the FP4 GEMV template and the raw-read
# upper bound every kernel is scored against (design §7.1 rule 2).
set(DEEPMOE_SHADERS
    gpu/shaders/moe_gemv_fp4.slang
    gpu/shaders/rawread.slang
    gpu/shaders/mega_mhc.slang
    gpu/shaders/wq_a.slang
    gpu/shaders/wq_b.slang
    gpu/shaders/wkv.slang
    gpu/shaders/sparse_attn.slang
    gpu/shaders/compressor.slang
    gpu/shaders/indexer.slang
    gpu/shaders/wo_a.slang
    gpu/shaders/wo_b.slang
    gpu/shaders/gate.slang
    gpu/shaders/moe_gateup.slang
    gpu/shaders/moe_down.slang
    gpu/shaders/moe_hquant.slang
    gpu/shaders/moe_xquant.slang
    gpu/shaders/engram.slang
    gpu/shaders/head.slang
    gpu/shaders/dspark_gemv.slang
    gpu/shaders/dspark_attn.slang
    gpu/shaders/dspark_head.slang)

function(deepmoe_report)
    message(STATUS "deepmoe: tests=${DEEPMOE_BUILD_TESTS} vulkan=${DEEPMOE_ENABLE_VULKAN} "
                   "directstorage=${DEEPMOE_ENABLE_DIRECTSTORAGE} C++${DEEPMOE_CXX_STANDARD}")
endfunction()
