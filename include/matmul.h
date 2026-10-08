#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace gpu_kernels {

// C = A * B: row-major A[M,K], column-major B[K,N] and C[M,N].
// B is physically stored as contiguous [N,K], matching fast.cu.
void gemm_fp32_cuda(
    float* output,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr);

// Same layout with BF16 inputs/output and FP32 accumulation on both GPUs.
// SM89 requires M,N multiples of 128 and K of 32.
// SM90 requires M,N multiples of 2048 and K of 64.
void gemm_bf16_cuda(
    __nv_bfloat16* output,
    const __nv_bfloat16* left,
    const __nv_bfloat16* right,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr);

}  // namespace gpu_kernels
