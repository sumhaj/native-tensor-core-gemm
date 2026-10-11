cmake_minimum_required(VERSION 3.24)
project(native_tensor_core_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Architectures)

find_package(CUDAToolkit REQUIRED)

# CUTLASS/CuTe: header-only usage. Deliberately NOT add_subdirectory(third_party/cutlass) --
# that would configure CUTLASS's own example/test binaries, which we don't need to build.
add_library(cutlass_headers INTERFACE)
target_include_directories(cutlass_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include
)
target_compile_definitions(cutlass_headers INTERFACE CUTLASS_ENABLE_CUTE=1)

add_subdirectory(common)

add_subdirectory(ampere-native-gemm/wmma)
add_subdirectory(ampere-native-gemm/mma-ptx)
add_subdirectory(ampere-native-gemm/mma-cute)

add_subdirectory(hopper-native-gemm/wgmma-ptx)
add_subdirectory(hopper-native-gemm/wgmma-cute)
