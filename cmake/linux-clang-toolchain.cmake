# CMake toolchain for a native Linux build with the system clang.
#
#   cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/linux-clang-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
#
# Mirrors zig-toolchain.cmake: same target CPU (Strix Halo = Zen 5, the AVX2 /
# AVX-512 intrinsics in runtime/ and cpu/ rely on it), slangc / spirv-val taken
# from the distro packages (shader-slang, spirv-tools) instead of the Vulkan SDK.

set(CMAKE_C_COMPILER   clang   CACHE FILEPATH "")
set(CMAKE_CXX_COMPILER clang++ CACHE FILEPATH "")

set(_common_flags "-march=znver5")
set(CMAKE_C_FLAGS_INIT   "${_common_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_common_flags}")

find_program(SLANGC_EXECUTABLE    slangc    REQUIRED)
find_program(SPIRV_VAL_EXECUTABLE spirv-val REQUIRED)
