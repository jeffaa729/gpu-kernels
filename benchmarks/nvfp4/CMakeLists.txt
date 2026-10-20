cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 100a)
endif()
project(gpu-kernels-nvfp4 LANGUAGES CXX CUDA)
find_package(CUDAToolkit 13.1 REQUIRED)

# Isolate SM100 from the existing architecture-specific Hopper WGMMA sources.
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_library(gpu_kernels_nvfp4_bench SHARED
    "${REPO_ROOT}/benchmarks/bindings/nvfp4_bridge.cu"
    "${REPO_ROOT}/kernels/gemm/matmul/sm100.cu")
target_include_directories(gpu_kernels_nvfp4_bench PRIVATE "${REPO_ROOT}/include")
target_link_libraries(gpu_kernels_nvfp4_bench PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cublasLt)
target_compile_options(gpu_kernels_nvfp4_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xptxas=-v>)
set_target_properties(gpu_kernels_nvfp4_bench PROPERTIES
    CXX_STANDARD 17 CUDA_STANDARD 17 CUDA_RUNTIME_LIBRARY Shared
    LIBRARY_OUTPUT_DIRECTORY "${REPO_ROOT}/build")
