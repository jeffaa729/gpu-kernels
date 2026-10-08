cmake_minimum_required(VERSION 3.24)

# Set the default before CUDA compiler detection; runners detect the actual GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
project(gpu-kernels LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

find_package(CUDAToolkit REQUIRED)
enable_testing()

add_library(gpu_kernels STATIC
    kernels/common/cuda_common.cu
    kernels/attention/flash_attention/dispatch.cu
    kernels/attention/flash_attention/sm89_forward.cu
    kernels/attention/flash_attention/sm90_forward.cu
    kernels/gemm/matmul/dispatch.cu
    kernels/gemm/matmul/sm89.cu
    kernels/gemm/matmul/sm90.cu
    kernels/moe/swiglu.cu
)
target_include_directories(gpu_kernels PUBLIC include)
# Attribute Hopper memory faults to source lines without disabling optimization.
set_source_files_properties(
    kernels/gemm/matmul/sm90.cu
    kernels/attention/flash_attention/sm90_forward.cu
    PROPERTIES COMPILE_OPTIONS "-lineinfo")
target_link_libraries(gpu_kernels PUBLIC CUDA::cudart CUDA::cuda_driver)
set_target_properties(gpu_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

# Optional Nsight launchers. Normal tests/runtime benchmarks use the Python runner.
function(kernel_profilers family)
    foreach(name IN LISTS ARGN)
        add_executable(benchmark_${name} EXCLUDE_FROM_ALL benchmarks/${family}/benchmark_${name}.cu)
        target_link_libraries(benchmark_${name} PRIVATE gpu_kernels)
    endforeach()
endfunction()
kernel_profilers(attention flash_attention)
kernel_profilers(gemm matmul)
target_link_libraries(benchmark_matmul PRIVATE CUDA::cublas)

# ctypes bridges keep custom and library calls in the same Python/CUDA process.
add_library(gpu_kernels_flash_attention_bench SHARED benchmarks/bindings/flash_attention_bridge.cu)
target_link_libraries(gpu_kernels_flash_attention_bench PRIVATE gpu_kernels)
add_library(gpu_kernels_operator_bench SHARED benchmarks/bindings/operators_bridge.cu)
target_link_libraries(gpu_kernels_operator_bench PRIVATE gpu_kernels CUDA::cublas)
add_library(gpu_kernels_moe_swiglu_bench SHARED benchmarks/bindings/moe_swiglu_bridge.cu)
target_link_libraries(gpu_kernels_moe_swiglu_bench PRIVATE gpu_kernels)

# Tests and benchmarks share cases and compare numerical results with PyTorch.
set(GPU_KERNELS_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH "Python from uv sync")
if(EXISTS "${GPU_KERNELS_PYTHON}")
    add_test(NAME moe_reference COMMAND ${GPU_KERNELS_PYTHON} -m unittest discover -s tests -p test_moe_reference.py)
    set_tests_properties(moe_reference PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach(name matmul flash_attention)
        add_test(NAME ${name} COMMAND ${GPU_KERNELS_PYTHON}
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/run.py ${name} --test)
        set_tests_properties(${name} PROPERTIES RESOURCE_LOCK gpu)
    endforeach()
endif()
