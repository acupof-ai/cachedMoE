cmake_minimum_required(VERSION 3.25)
project(deepmoe LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)

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

# --- environment self-check --------------------------------------------------
add_executable(envcheck tools/envcheck/vkinfo.cpp)
target_link_libraries(envcheck PRIVATE Vulkan::Vulkan)
add_slang_shader(envcheck_shader "${CMAKE_SOURCE_DIR}/tools/envcheck/gemv.slang")
add_dependencies(envcheck envcheck_shader)
