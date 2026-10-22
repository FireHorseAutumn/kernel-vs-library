cmake_minimum_required(VERSION 3.20)
# nvcc 13.2 rejects GCC 16; must be set before project() so the
# compiler-ID probe inherits it.
set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler")
project(kernel-vs-library LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

# STANDALONE: no sibling-repo includes, no archive paths.
# Library baselines come only from the CUDA toolkit itself.

add_library(kvl INTERFACE)
target_include_directories(kvl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(kvl INTERFACE
    CUDA::cudart CUDA::cublas CUDA::cusolver
)

# TODO: add targets as sources land
# add_executable(bench_joseph src/bench_joseph.cu)
# add_executable(bench_hamming src/bench_hamming.cu)
# add_executable(bench_dlt src/bench_dlt.cu)
# add_executable(bench_reproj src/bench_reproj.cu)

enable_testing()
