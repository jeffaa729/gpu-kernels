#pragma once

#include "matmul.h"

namespace gpu_kernels {

void gemm_bf16_sm90_cuda(__nv_bfloat16* output, const __nv_bfloat16* left, const __nv_bfloat16* right, int M, int N, int K, cudaStream_t stream);

void gemm_fp32_sm89_cuda(float* output, const float* left, const float* right, int M, int N, int K, cudaStream_t stream);

void gemm_bf16_sm89_cuda(__nv_bfloat16* output, const __nv_bfloat16* left, const __nv_bfloat16* right, int M, int N, int K, cudaStream_t stream);

}  // namespace gpu_kernels
