cmake_minimum_required(VERSION 3.25)
project(deepmoe LANGUAGES C CXX)

include(cmake/deepmoe_options.cmake)

set(CMAKE_CXX_STANDARD ${DEEPMOE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
deepmoe_report()

if(DEEPMOE_ENABLE_VULKAN)
    find_package(Vulkan REQUIRED)
endif()

# --- Slang -> SPIR-V --------------------------------------------------------
# add_slang_shader(<target> <src.slang>) compiles to ${CMAKE_BINARY_DIR}/shaders/<name>.spv
# and validates it with spirv-val.
function(add_slang_shader target src)
    get_filename_component(name "${src}" NAME_WE)
    set(out "${CMAKE_BINARY_DIR}/shaders/${name}.spv")
    add_custom_command(
        OUTPUT "${out}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/shaders"
        COMMAND "${SLANGC_EXECUTABLE}" "${src}" -target spirv -profile spirv_1_6 -entry main -O2 -o "${out}"
        COMMAND "${SPIRV_VAL_EXECUTABLE}" --target-env vulkan1.3 "${out}"
        DEPENDS "${src}"
        COMMENT "slangc ${name}.slang"
        VERBATIM)
    add_custom_target("${target}" DEPENDS "${out}")
endfunction()

# Every kernel of design §7.14 is compiled and validated by the `shaders` target.
set(_shader_targets "")
foreach(_s IN LISTS DEEPMOE_SHADERS)
    get_filename_component(_n "${_s}" NAME_WE)
    add_slang_shader("shader_${_n}" "${CMAKE_SOURCE_DIR}/${_s}")
    list(APPEND _shader_targets "shader_${_n}")
endforeach()
add_custom_target(shaders ALL DEPENDS ${_shader_targets})

# --- deepmoe_core: everything except the CLI and the benchmarks --------------
set(_deepmoe_sources
    ${DEEPMOE_CORE_SOURCES}
    ${DEEPMOE_MODEL_SOURCES}
    ${DEEPMOE_CPU_SOURCES}
    ${DEEPMOE_STORAGE_SOURCES}
    ${DEEPMOE_STORE_SOURCES}
    ${DEEPMOE_GPU_SOURCES}
    ${DEEPMOE_RUNTIME_SOURCES})

if(WIN32)
    list(APPEND _deepmoe_sources ${DEEPMOE_STORAGE_WINDOWS_SOURCES})
else()
    list(APPEND _deepmoe_sources ${DEEPMOE_STORAGE_LINUX_SOURCES})
endif()

add_library(deepmoe_core STATIC ${_deepmoe_sources})
target_include_directories(deepmoe_core PUBLIC "${CMAKE_SOURCE_DIR}")
target_compile_features(deepmoe_core PUBLIC cxx_std_${DEEPMOE_CXX_STANDARD})

if(DEEPMOE_ENABLE_VULKAN)
    target_link_libraries(deepmoe_core PUBLIC Vulkan::Vulkan)
    target_compile_definitions(deepmoe_core PUBLIC DEEPMOE_ENABLE_VULKAN)
endif()
if(DEEPMOE_ENABLE_DIRECTSTORAGE)
    target_compile_definitions(deepmoe_core PRIVATE DEEPMOE_ENABLE_DIRECTSTORAGE)
    # Only switch the real backend on when the SDK header is actually reachable;
    # design §9.6 says do not vendor or download it.
    include(CheckIncludeFileCXX)
    check_include_file_cxx("dstorage.h" DEEPMOE_HAVE_DSTORAGE_H)
    if(DEEPMOE_HAVE_DSTORAGE_H)
        target_compile_definitions(deepmoe_core PRIVATE DEEPMOE_HAVE_DSTORAGE_H)
    else()
        message(STATUS "deepmoe: dstorage.h not found; the DirectStorage backend reports unavailable")
    endif()
endif()
if(NOT WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(deepmoe_core PUBLIC Threads::Threads)
endif()

# --- CLI ---------------------------------------------------------------------
add_executable(deepmoe cli/deepmoe_main.cpp)
target_link_libraries(deepmoe PRIVATE deepmoe_core)
add_dependencies(deepmoe shaders)

# --- benchmarks (design §9.2, P-1) ------------------------------------------
add_executable(nvme_bench bench/nvme_bench.cpp)
target_link_libraries(nvme_bench PRIVATE deepmoe_core)

add_executable(bw_matrix bench/bw_matrix.cpp)
target_link_libraries(bw_matrix PRIVATE deepmoe_core)

# --- environment self-check (design docs/build.md) ---------------------------
if(DEEPMOE_ENABLE_VULKAN)
    add_executable(envcheck tools/envcheck/vkinfo.cpp)
    target_link_libraries(envcheck PRIVATE Vulkan::Vulkan)
    add_dependencies(envcheck shader_moe_gemv_fp4)
endif()

# --- tests -------------------------------------------------------------------
if(DEEPMOE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
