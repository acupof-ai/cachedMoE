# Build options and the per-module source lists.
#
# Platform code is confined to storage/windows and storage/linux; every other
# module is portable and is compiled for both targets (the Linux build exists so
# CI can run the CPU and storage tests -- design §14).
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/cachedmoe_cache_compat.cmake")

cachedmoe_option(BUILD_TESTS "Build the unit tests" ON)
cachedmoe_option(ENABLE_VULKAN "Build the Vulkan GPU backend" ON)
cachedmoe_option(ENABLE_DIRECTSTORAGE "Build the DirectStorage I/O backend (needs dstorage.h)" OFF)

# std::expected lives in <expected>, which libc++ guards behind C++23 even
# though the rest of the codebase is C++20. See docs/build.md.
set(CACHEDMOE_CXX_STANDARD 23)

set(CACHEDMOE_CORE_SOURCES
    core/json.cpp
    core/profiler.cpp)

set(CACHEDMOE_MODEL_SOURCES
    model/v41_config.cpp
    model/manifest.cpp)

set(CACHEDMOE_CPU_SOURCES
    cpu/dequant.cpp
    cpu/gemv_avx512.cpp
    cpu/gate.cpp
    cpu/dspark_tree.cpp)

set(CACHEDMOE_STORAGE_SOURCES
    storage/file_common.cpp
    storage/io_engine.cpp)

set(CACHEDMOE_STORAGE_WINDOWS_SOURCES
    storage/windows/file_win.cpp
    storage/windows/iocp.cpp
    storage/windows/directstorage.cpp)

set(CACHEDMOE_STORAGE_LINUX_SOURCES
    storage/linux/file_posix.cpp
    storage/linux/io_uring.cpp)

set(CACHEDMOE_STORE_SOURCES
    store/slab.cpp
    store/pinned.cpp
    store/expert_store.cpp
    store/planner.cpp
    store/predictor.cpp
    store/engram_prefetch.cpp)

set(CACHEDMOE_GPU_SOURCES
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
    gpu/vulkan/dspark_onecb.cpp
    gpu/vulkan/dspark_mega.cpp
    gpu/vulkan/dspark_kernels.cpp
    gpu/vulkan/prefill_kernels.cpp)

set(CACHEDMOE_RUNTIME_SOURCES
    runtime/engine.cpp
    runtime/gpu_route_state.cpp
    runtime/kvstore.cpp
    runtime/decode_layer.cpp
    runtime/moe_bridge.cpp
    runtime/decode_state.cpp
    runtime/engram.cpp
    runtime/sampling.cpp
    runtime/session.cpp
    runtime/dspark_runtime.cpp
    runtime/speculate.cpp
    runtime/engram_tables.cpp
    runtime/trace.cpp)

# Track P: the tokenizer (docs/p3_chat.md).
set(CACHEDMOE_TEXT_SOURCES
    text/tokenizer.cpp)

# Headers every shader in CACHEDMOE_SHADERS may include; touching one rebuilds all.
set(CACHEDMOE_SHADER_DEPS
    gpu/shaders/moe_common.slang
    gpu/shaders/attn_common.slang
    gpu/shaders/fp8_gemv.slang
    gpu/shaders/dspark_common.slang
    gpu/shaders/prefill_common.slang)
list(APPEND CACHEDMOE_SHADER_DEPS gpu/shaders/decode_attn_cm.slang)

# design §7.14 dispatch list, plus the FP4 GEMV template and the raw-read
# upper bound every kernel is scored against (design §7.1 rule 2).
set(CACHEDMOE_SHADERS
    gpu/shaders/batch_route.slang
    gpu/shaders/dspark_onecb.slang
    gpu/shaders/dspark_mega.slang
    gpu/shaders/dspark_grid.slang
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
    gpu/shaders/gemv_ksplit.slang
    gpu/shaders/sparse_attn_t.slang
    gpu/shaders/decode_attn_cm.slang
    gpu/shaders/gate.slang
    gpu/shaders/moe_gateup.slang
    gpu/shaders/moe_down.slang
    gpu/shaders/moe_hquant.slang
    gpu/shaders/moe_xquant.slang
    gpu/shaders/moe_xact.slang
    gpu/shaders/engram.slang
    gpu/shaders/head.slang
    gpu/shaders/sample_topk.slang
    gpu/shaders/dspark_gemv.slang
    gpu/shaders/dspark_attn.slang
    gpu/shaders/dspark_head.slang
    gpu/shaders/dspark_verify.slang
    gpu/shaders/mgt1_gemv.slang
    gpu/shaders/mgt1_mhc.slang
    gpu/shaders/mgt1_attn.slang
    gpu/shaders/mgt1_attn_cm.slang
    gpu/shaders/mgt1_gate.slang
    gpu/shaders/mgt1_cmp.slang
    gpu/shaders/mgt1_idx.slang
    gpu/shaders/mgt1_head.slang
    gpu/shaders/mgt1_engram.slang
    gpu/shaders/prefill_gemm.slang
    gpu/shaders/prefill_coopmat.slang
    gpu/shaders/prefill_gemm_lds.slang
    gpu/shaders/prefill_elem.slang
    gpu/shaders/prefill_attn.slang
    gpu/shaders/prefill_topk.slang
    # Track W: the P5 feasibility probes (bench/probes, docs/plan_p5.md §5).
    gpu/shaders/probe_stream.slang
    gpu/shaders/probe_resident.slang
    gpu/shaders/probe_hostflag.slang
    gpu/shaders/probe_model.slang
    gpu/shaders/probe_mma.slang
    gpu/shaders/probe_occ.slang)

function(cachedmoe_report)
    message(STATUS "cachedmoe: tests=${CACHEDMOE_BUILD_TESTS} vulkan=${CACHEDMOE_ENABLE_VULKAN} "
                   "directstorage=${CACHEDMOE_ENABLE_DIRECTSTORAGE} C++${CACHEDMOE_CXX_STANDARD}")
endfunction()
