cmake_minimum_required(VERSION 3.25)
project(cachedmoe LANGUAGES C CXX)

include(cmake/cachedmoe_options.cmake)

set(CMAKE_CXX_STANDARD ${CACHEDMOE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
cachedmoe_report()

if(CACHEDMOE_ENABLE_VULKAN)
    find_package(Vulkan REQUIRED)
endif()

# --- Slang -> SPIR-V --------------------------------------------------------
# add_slang_shader(<target> <src.slang>) compiles to ${CMAKE_BINARY_DIR}/shaders/<name>.spv
# and validates it with spirv-val.
function(add_slang_shader target src)
    get_filename_component(name "${src}" NAME_WE)
    set(out "${CMAKE_BINARY_DIR}/shaders/${name}.spv")
    set(_mega_args "")
    set(_mega_memory_command "")
    if(name STREQUAL "dspark_mega" OR name STREQUAL "dspark_onecb")
        cachedmoe_resolve_cache(FUSION_PYTHON FILEPATH "Python interpreter for DSpark shader generation")
        find_program(CACHEDMOE_FUSION_PYTHON python3 REQUIRED)
        file(GLOB _mega_sources "${CMAKE_SOURCE_DIR}/gpu/shaders/*.slang")
        if(NOT TARGET dspark_ops_generated)
        add_custom_command(
            OUTPUT "${CMAKE_BINARY_DIR}/shaders/dspark_mega_ops.slang"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/shaders"
            COMMAND "${CACHEDMOE_FUSION_PYTHON}" "${CMAKE_SOURCE_DIR}/tools/fuse_dspark.py"
                    "${CMAKE_SOURCE_DIR}/gpu/shaders" "${CMAKE_BINARY_DIR}/shaders/dspark_mega_ops.slang"
            DEPENDS "${CMAKE_SOURCE_DIR}/tools/fuse_dspark.py" ${_mega_sources}
            VERBATIM)
        add_custom_target(dspark_ops_generated DEPENDS "${CMAKE_BINARY_DIR}/shaders/dspark_mega_ops.slang")
        endif()
        if(name STREQUAL "dspark_mega")
        set(_mega_memory_command COMMAND "${CACHEDMOE_FUSION_PYTHON}" "${CMAKE_SOURCE_DIR}/tools/dspark_spirv_memory.py" "${out}")
        endif()
        list(APPEND _mega_args "${CMAKE_BINARY_DIR}/shaders/dspark_mega_ops.slang"
                              "${CMAKE_SOURCE_DIR}/tools/dspark_spirv_memory.py")
    endif()
    add_custom_command(
        OUTPUT "${out}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/shaders"
        COMMAND "${SLANGC_EXECUTABLE}" "${src}" -target spirv -profile spirv_1_6 -entry main -O2
                -I "${CMAKE_SOURCE_DIR}/gpu/shaders" -I "${CMAKE_BINARY_DIR}/shaders" -o "${out}"
        ${_mega_memory_command}
        COMMAND "${SPIRV_VAL_EXECUTABLE}" --target-env vulkan1.3 "${out}"
        DEPENDS "${src}" ${_shader_dep_paths} ${_mega_args}
        COMMENT "slangc ${name}.slang"
        VERBATIM)
    add_custom_target("${target}" DEPENDS "${out}")
endfunction()

set(_shader_dep_paths "${CMAKE_SOURCE_DIR}/gpu/shaders/dspark_plan_layout.h")
foreach(_d IN LISTS CACHEDMOE_SHADER_DEPS)
    list(APPEND _shader_dep_paths "${CMAKE_SOURCE_DIR}/${_d}")
endforeach()

# Every kernel of design §7.14 is compiled and validated by the `shaders` target.
set(_shader_targets "")
foreach(_s IN LISTS CACHEDMOE_SHADERS)
    get_filename_component(_n "${_s}" NAME_WE)
    add_slang_shader("shader_${_n}" "${CMAKE_SOURCE_DIR}/${_s}")
    list(APPEND _shader_targets "shader_${_n}")
endforeach()
add_custom_target(shaders ALL DEPENDS ${_shader_targets})

# --- cachedmoe_core: everything except the CLI and the benchmarks --------------
set(_cachedmoe_sources
    ${CACHEDMOE_CORE_SOURCES}
    ${CACHEDMOE_MODEL_SOURCES}
    ${CACHEDMOE_CPU_SOURCES}
    ${CACHEDMOE_STORAGE_SOURCES}
    ${CACHEDMOE_STORE_SOURCES}
    ${CACHEDMOE_GPU_SOURCES}
    ${CACHEDMOE_RUNTIME_SOURCES}
    ${CACHEDMOE_TEXT_SOURCES})

if(WIN32)
    list(APPEND _cachedmoe_sources ${CACHEDMOE_STORAGE_WINDOWS_SOURCES})
else()
    list(APPEND _cachedmoe_sources ${CACHEDMOE_STORAGE_LINUX_SOURCES})
endif()

add_library(cachedmoe_core STATIC ${_cachedmoe_sources})
add_library(deepmoe_core ALIAS cachedmoe_core)  # Legacy source target name.
if(UNIX)
    add_custom_target(cachedmoe_core_legacy_alias ALL
        COMMAND "${CMAKE_COMMAND}" -E rm -f "$<TARGET_LINKER_FILE_DIR:cachedmoe_core>/${CMAKE_STATIC_LIBRARY_PREFIX}deepmoe_core${CMAKE_STATIC_LIBRARY_SUFFIX}"
        COMMAND "${CMAKE_COMMAND}" -E create_symlink "$<TARGET_LINKER_FILE_NAME:cachedmoe_core>" "$<TARGET_LINKER_FILE_DIR:cachedmoe_core>/${CMAKE_STATIC_LIBRARY_PREFIX}deepmoe_core${CMAKE_STATIC_LIBRARY_SUFFIX}"
        DEPENDS cachedmoe_core
        VERBATIM)
endif()
target_include_directories(cachedmoe_core PUBLIC "${CMAKE_SOURCE_DIR}")
target_compile_features(cachedmoe_core PUBLIC cxx_std_${CACHEDMOE_CXX_STANDARD})
# Where `shaders` drops the .spv files; the environment variable of the same
# name overrides it so a relocated build tree still runs.
target_compile_definitions(cachedmoe_core PUBLIC CACHEDMOE_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders")

if(CACHEDMOE_ENABLE_VULKAN)
    target_link_libraries(cachedmoe_core PUBLIC Vulkan::Vulkan)
    target_compile_definitions(cachedmoe_core PUBLIC CACHEDMOE_ENABLE_VULKAN)
endif()
if(CACHEDMOE_ENABLE_DIRECTSTORAGE)
    target_compile_definitions(cachedmoe_core PRIVATE CACHEDMOE_ENABLE_DIRECTSTORAGE)
    # Only switch the real backend on when the SDK header is actually reachable;
    # design §9.6 says do not vendor or download it.
    include(CheckIncludeFileCXX)
    cachedmoe_resolve_cache(HAVE_DSTORAGE_H INTERNAL "Have the DirectStorage SDK header")
    check_include_file_cxx("dstorage.h" CACHEDMOE_HAVE_DSTORAGE_H)
    if(CACHEDMOE_HAVE_DSTORAGE_H)
        target_compile_definitions(cachedmoe_core PRIVATE CACHEDMOE_HAVE_DSTORAGE_H)
    else()
        message(STATUS "cachedmoe: dstorage.h not found; the DirectStorage backend reports unavailable")
    endif()
endif()
if(NOT WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(cachedmoe_core PUBLIC Threads::Threads)
endif()

# --- CLI ---------------------------------------------------------------------
add_executable(cachedmoe cli/cachedmoe_main.cpp cli/serve.cpp)
target_link_libraries(cachedmoe PRIVATE cachedmoe_core)
add_dependencies(cachedmoe shaders)

# Keep old invocations on the same executable. Regenerate links on every build,
# including upgrades whose old artifact was a regular file.
function(cachedmoe_legacy_executable target legacy)
    set(alias_path "$<TARGET_FILE_DIR:${target}>/${legacy}${CMAKE_EXECUTABLE_SUFFIX}")
    if(UNIX)
        add_custom_target(${target}_legacy_alias ALL
            COMMAND "${CMAKE_COMMAND}" -E rm -f "${alias_path}"
            COMMAND "${CMAKE_COMMAND}" -E create_symlink "$<TARGET_FILE_NAME:${target}>" "${alias_path}"
            DEPENDS ${target}
            VERBATIM)
    else()
        add_custom_target(${target}_legacy_alias ALL
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:${target}>" "${alias_path}"
            DEPENDS ${target}
            VERBATIM)
    endif()
    add_custom_target(${legacy} DEPENDS ${target}_legacy_alias)
endfunction()
cachedmoe_legacy_executable(cachedmoe deepmoe)

# --- benchmarks (design §9.2, P-1) ------------------------------------------
add_executable(nvme_bench bench/nvme_bench.cpp)
target_link_libraries(nvme_bench PRIVATE cachedmoe_core)

add_executable(draft_head_capture bench/draft_head_capture.cpp)
target_link_libraries(draft_head_capture PRIVATE cachedmoe_core)
add_dependencies(draft_head_capture shaders)
add_executable(draft_head_bench bench/draft_head_bench.cpp)
target_link_libraries(draft_head_bench PRIVATE cachedmoe_core)
add_dependencies(draft_head_bench shaders)

# Track Q2 E1: the same IoEngine reads, varying only the destination memory.
add_executable(io_dst_bench bench/io_dst_bench.cpp)
target_link_libraries(io_dst_bench PRIVATE cachedmoe_core)
add_dependencies(io_dst_bench shaders)

add_executable(bw_matrix bench/bw_matrix.cpp)
target_link_libraries(bw_matrix PRIVATE cachedmoe_core)
add_dependencies(bw_matrix shaders)

# design §3.3 / §5.3: the *allocatable* ceiling on both heaps at the current
# BIOS setting, as opposed to the heap sizes the driver reports.
add_executable(heap_capacity bench/heap_capacity.cpp)
target_link_libraries(heap_capacity PRIVATE cachedmoe_core)
add_dependencies(heap_capacity shaders)

# design §7.1 rule 2 / §7.9: effective GB/s of the two MoE dispatches against
# the raw-read ceiling, over the real checkpoint's experts (P1).
add_executable(kernel_bench bench/kernel_bench.cpp)
target_link_libraries(kernel_bench PRIVATE cachedmoe_core)
add_dependencies(kernel_bench shaders)

# design 7.1 rule 2 / 7.2-7.8: effective GB/s of the attention-path kernels on
# the real checkpoint (P2).
add_executable(attn_bench bench/attn_bench.cpp)
target_link_libraries(attn_bench PRIVATE cachedmoe_core)
add_dependencies(attn_bench shaders)

# design 7.13 / docs/p3_prefill.md: the prefill GEMM choice and a whole prefill's
# wall time (P3, Track L).
add_executable(prefill_bench bench/prefill_bench.cpp)
target_link_libraries(prefill_bench PRIVATE cachedmoe_core)
add_dependencies(prefill_bench shaders)

# --- Track W: the P5 feasibility probes (docs/plan_p5.md §5) -----------------
# A probe is a question that costs minutes and can close a week of kernel work.
# They are built here, and run by the GPU owner on a quiet machine.
foreach(_probe dspark_grid_probe residency_probe sharing_probe capacity_probe hostflag_probe model_probe)
    add_executable(${_probe} bench/probes/${_probe}.cpp)
    target_link_libraries(${_probe} PRIVATE cachedmoe_core)
    add_dependencies(${_probe} shaders)
endforeach()

# --- environment self-check (design docs/build.md) ---------------------------
if(CACHEDMOE_ENABLE_VULKAN)
    add_executable(envcheck tools/envcheck/vkinfo.cpp)
    target_link_libraries(envcheck PRIVATE Vulkan::Vulkan)
    add_dependencies(envcheck shader_moe_gemv_fp4)
endif()

# --- tests -------------------------------------------------------------------
if(CACHEDMOE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
