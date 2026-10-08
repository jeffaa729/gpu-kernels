// Profiles the fast.cu layout with matching cuBLAS.

#include "cuda_common.h"
#include "matmul.h"

#include <cublas_v2.h>
#include <cuda_profiler_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

namespace {

void cublas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS call failed");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int M = argc > 1 ? std::atoi(argv[1]) : 2048;
        const int N = argc > 2 ? std::atoi(argv[2]) : 2048;
        const int K = argc > 3 ? std::atoi(argv[3]) : 2048;
        const char* backend = argc > 4 ? argv[4] : "fp32";
        const bool reference = std::strcmp(backend, "cublas_fp32") == 0 ||
                               std::strcmp(backend, "cublas_bf16") == 0;
        const bool bf16 = std::strcmp(backend, "bf16") == 0 ||
                          std::strcmp(backend, "cublas_bf16") == 0;
        if (!reference && !bf16 && std::strcmp(backend, "fp32") != 0) {
            throw std::runtime_error("backend must be fp32, bf16, cublas_fp32, or cublas_bf16");
        }
        const size_t left_elements = static_cast<size_t>(M) * K;
        const size_t right_elements = static_cast<size_t>(K) * N;
        const size_t output_elements = static_cast<size_t>(M) * N;
        const size_t element_bytes = bf16 ? sizeof(__nv_bfloat16) : sizeof(float);
        const size_t output_bytes = output_elements * element_bytes;
        void* left = gpu_kernels::device_malloc(left_elements * element_bytes);
        void* right = gpu_kernels::device_malloc(right_elements * element_bytes);
        void* output = gpu_kernels::device_malloc(output_bytes);
        if (bf16) {
            const std::vector<__nv_bfloat16> a(left_elements, __float2bfloat16(0.5F));
            const std::vector<__nv_bfloat16> b(right_elements, __float2bfloat16(0.25F));
            CUDA_CHECK(cudaMemcpy(left, a.data(), left_elements * element_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(right, b.data(), right_elements * element_bytes, cudaMemcpyHostToDevice));
        } else {
            const std::vector<float> a(left_elements, 0.5F), b(right_elements, 0.25F);
            CUDA_CHECK(cudaMemcpy(left, a.data(), left_elements * element_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(right, b.data(), right_elements * element_bytes, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemset(output, 0, output_bytes));
        cublasHandle_t handle = nullptr;
        if (reference) {
            cublas_check(cublasCreate(&handle));
            cublas_check(cublasSetMathMode(
                handle, bf16 ? CUBLAS_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
        }

        auto run = [&]() {
            if (reference) {
                const float alpha = 1.0F;
                const float beta = 0.0F;
                const cudaDataType_t type = bf16 ? CUDA_R_16BF : CUDA_R_32F;
                // GEMM uses K-contiguous A/B and column-major C, as in fast.cu.
                cublas_check(cublasGemmEx(
                    handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    M, N, K, &alpha,
                    left, type, K,
                    right, type, K,
                    &beta, output, type, M,
                    bf16 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_PEDANTIC,
                    bf16 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
            } else if (bf16) {
                gpu_kernels::gemm_bf16_cuda(
                    static_cast<__nv_bfloat16*>(output),
                    static_cast<const __nv_bfloat16*>(left),
                    static_cast<const __nv_bfloat16*>(right), M, N, K);
            } else {
                gpu_kernels::gemm_fp32_cuda(
                    static_cast<float*>(output),
                    static_cast<const float*>(left),
                    static_cast<const float*>(right), M, N, K);
            }
        };

        run();
        gpu_kernels::synchronize();
        CUDA_CHECK(cudaMemset(output, 0, output_bytes));
        gpu_kernels::synchronize();
        CUDA_CHECK(cudaProfilerStart());
        run();
        gpu_kernels::synchronize();
        CUDA_CHECK(cudaProfilerStop());

        if (handle) cublas_check(cublasDestroy(handle));
        gpu_kernels::device_free(output);
        gpu_kernels::device_free(right);
        gpu_kernels::device_free(left);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "GEMM benchmark failed: %s\n", error.what());
        return 1;
    }
}
