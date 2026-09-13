# CMake toolchain: use Zig's bundled clang/lld as the C/C++ compiler.
#
#   cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake
#   -DZIG_TARGET=x86_64-windows-msvc   # optional A/B against the MSVC CRT
#
# CMake wants a single executable per compiler, so `zig cc` is reached through
# small .cmd wrappers. They are generated into the build tree with the absolute
# zig path and target baked in, because Ninja does not inherit environment
# variables set during configure.

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(ZIG_TARGET "x86_64-windows-gnu" CACHE STRING "Zig target triple")

find_program(ZIG_EXECUTABLE zig REQUIRED
    HINTS "$ENV{LOCALAPPDATA}/Microsoft/WinGet/Packages"
    PATH_SUFFIXES zig.zig_Microsoft.Winget.Source_8wekyb3d8bbwe/zig-x86_64-windows-0.16.0)
file(TO_NATIVE_PATH "${ZIG_EXECUTABLE}" ZIG_EXECUTABLE_NATIVE)

set(_zig_wrap_dir "${CMAKE_BINARY_DIR}/zig-wrappers")
foreach(_tool cc cxx ar ranlib rc)
    configure_file("${CMAKE_CURRENT_LIST_DIR}/zig-${_tool}.cmd.in" "${_zig_wrap_dir}/zig-${_tool}.cmd" @ONLY NEWLINE_STYLE CRLF)
endforeach()

set(CMAKE_C_COMPILER   "${_zig_wrap_dir}/zig-cc.cmd"     CACHE FILEPATH "" FORCE)
set(CMAKE_CXX_COMPILER "${_zig_wrap_dir}/zig-cxx.cmd"    CACHE FILEPATH "" FORCE)
set(CMAKE_AR           "${_zig_wrap_dir}/zig-ar.cmd"     CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB       "${_zig_wrap_dir}/zig-ranlib.cmd" CACHE FILEPATH "" FORCE)
set(CMAKE_RC_COMPILER  "${_zig_wrap_dir}/zig-rc.cmd"     CACHE FILEPATH "" FORCE)

set(CMAKE_C_COMPILER_TARGET   "${ZIG_TARGET}")
set(CMAKE_CXX_COMPILER_TARGET "${ZIG_TARGET}")

# Strix Halo is Zen 5. Zig's libc++ build spams nullability warnings on first use.
set(_common_flags "-march=znver5 -Wno-nullability-completeness")
set(CMAKE_C_FLAGS_INIT   "${_common_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_common_flags}")

# Vulkan SDK (installed to C:\VulkanSDK\<ver>; VULKAN_SDK is set machine-wide by the installer).
if(DEFINED ENV{VULKAN_SDK})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK}")
    set(SLANGC_EXECUTABLE    "$ENV{VULKAN_SDK}/Bin/slangc.exe"    CACHE FILEPATH "")
    set(SPIRV_VAL_EXECUTABLE "$ENV{VULKAN_SDK}/Bin/spirv-val.exe" CACHE FILEPATH "")
endif()
