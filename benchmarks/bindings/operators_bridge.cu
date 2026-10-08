// Exposes GEMM to Python-owned CUDA buffers without a PyTorch extension build.
// cuBLAS uses the same stream and operand/output precision as the custom GEMM.
#include "matmul.h"

#include <cublas_v2.h>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;
thread_local cublasHandle_t handle = nullptr;

void cublas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS status " + std::to_string(status));
    }
}
}  // namespace

extern "C" const char* gpu_kernels_operator_last_error() {
    return last_error.c_str();
}

extern "C" int gpu_kernels_cublas_init() {
    try {
        if (!handle) cublas_check(cublasCreate(&handle));
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}

extern "C" void gpu_kernels_cublas_destroy() {
    if (handle) cublasDestroy(handle);
    handle = nullptr;
}

extern "C" int gpu_kernels_gemm(
    void* output, const void* left, const void* right,
    int M, int N, int K, int bf16, int reference, cudaStream_t stream) {
    try {
        if (!reference) {
            if (bf16) {
                gpu_kernels::gemm_bf16_cuda(
                    static_cast<__nv_bfloat16*>(output),
                    static_cast<const __nv_bfloat16*>(left),
                    static_cast<const __nv_bfloat16*>(right),
                    M, N, K, stream);
            } else {
                gpu_kernels::gemm_fp32_cuda(
                    static_cast<float*>(output),
                    static_cast<const float*>(left),
                    static_cast<const float*>(right),
                    M, N, K, stream);
            }
        } else {
            cublas_check(cublasSetStream(handle, stream));
            cublas_check(cublasSetMathMode(
                handle, bf16 ? CUBLAS_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
            const float alpha = 1.0F;
            const float beta = 0.0F;
            const cudaDataType_t type = bf16 ? CUDA_R_16BF : CUDA_R_32F;
            cublas_check(cublasGemmEx(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                M, N, K, &alpha,
                left, type, K,
                right, type, K,
                &beta, output, type, M,
                bf16 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_PEDANTIC,
                bf16 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
        }
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}
