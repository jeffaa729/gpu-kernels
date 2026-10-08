#include "common.cuh"
#include "cuda_common.h"

namespace gpu_kernels {
namespace {

bool use_sm90() {
    int device = 0;
    int major = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    return major == 9;
}

}  // namespace

void gemm_fp32_cuda(float* output, const float* left, const float* right, int M, int N, int K, cudaStream_t stream) {
    gemm_fp32_sm89_cuda(output, left, right, M, N, K, stream);
}

void gemm_bf16_cuda(__nv_bfloat16* output, const __nv_bfloat16* left, const __nv_bfloat16* right, int M, int N, int K, cudaStream_t stream) {
    if (use_sm90()) {
        gemm_bf16_sm90_cuda(output, left, right, M, N, K, stream);
    } else {
        gemm_bf16_sm89_cuda(output, left, right, M, N, K, stream);
    }
}

}  // namespace gpu_kernels
