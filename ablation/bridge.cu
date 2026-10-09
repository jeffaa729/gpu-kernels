#include "api.cuh"
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

namespace {
thread_local cublasHandle_t handle = nullptr;
thread_local std::string error;
void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(status));
}
}

extern "C" const char* ablation_error() { return error.c_str(); }
extern "C" int ablation_init() {
    try { check(cublasCreate(&handle)); return 0; }
    catch (const std::exception& e) { error = e.what(); return 1; }
}
extern "C" void ablation_destroy() { if (handle) cublasDestroy(handle); handle = nullptr; }

extern "C" int ablation_gemm(void* C, const void* A, const void* B, int M, int N, int K, int bf16, int stage, cudaStream_t stream) {
    try {
        if (M % 128 || N % 128 || K % 32 || M <= 0 || N <= 0 || K <= 0) throw std::invalid_argument("Ablation shapes require M,N multiples of 128 and K of 32.");
        if (stage == -1) {
            check(cublasSetStream(handle, stream));
            check(cublasSetMathMode(handle, bf16 ? CUBLAS_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
            const float alpha = 1, beta = 0;
            const cudaDataType_t type = bf16 ? CUDA_R_16BF : CUDA_R_32F;
            check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, type, K, B, type, K, &beta, C, type, M,
                bf16 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_PEDANTIC, bf16 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
        } else if (bf16) {
#ifdef ABLATION_SM90
            const ablation::BF16Launch launch[] = {ablation::h0, ablation::h1, ablation::h2, ablation::h3, ablation::h4, ablation::h5, ablation::h6, ablation::h7, ablation::h8};
            if (stage < 0 || stage >= 9) throw std::invalid_argument("Hopper stage must be 0..8.");
#else
            const ablation::BF16Launch launch[] = {ablation::t0, ablation::t1, ablation::t2, ablation::t3, ablation::t4};
            if (stage < 0 || stage >= 5) throw std::invalid_argument("BF16 stage must be 0..4.");
#endif
            launch[stage](static_cast<ablation::bf16*>(C), static_cast<const ablation::bf16*>(A), static_cast<const ablation::bf16*>(B), M, N, K, stream);
        } else {
#ifdef ABLATION_SM90
            throw std::invalid_argument("Hopper ablation is BF16 only.");
#else
            const ablation::FP32Launch launch[] = {ablation::f0, ablation::f1, ablation::f2, ablation::f3, ablation::f4, ablation::f5, ablation::f6, ablation::f7};
            if (stage < 0 || stage >= 8) throw std::invalid_argument("FP32 stage must be 0..7.");
            launch[stage](static_cast<float*>(C), static_cast<const float*>(A), static_cast<const float*>(B), M, N, K, stream);
#endif
        }
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return 0;
    } catch (const std::exception& e) { error = e.what(); return 1; }
}

extern "C" int ablation_attributes(int* attributes) {
    int device;
    cudaGetDevice(&device);
    const cudaDeviceAttr names[] = {cudaDevAttrMultiProcessorCount, cudaDevAttrClockRate, cudaDevAttrMemoryClockRate, cudaDevAttrGlobalMemoryBusWidth};
    for (int i = 0; i < 4; ++i) {
        const auto status = cudaDeviceGetAttribute(attributes + i, names[i], device);
        if (status != cudaSuccess) { error = cudaGetErrorString(status); return 1; }
    }
    return 0;
}

// A large-buffer vector copy provides an empirical bandwidth reference.
// Both read and write bytes are counted; the working set exceeds the GPU L2 cache.
__global__ void bandwidth_kernel(float4* C, const float4* A, int vectors) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vectors; i += gridDim.x * blockDim.x) C[i] = A[i];
}
extern "C" int ablation_copy(void* C, const void* A, int vectors, cudaStream_t stream) {
    bandwidth_kernel<<<192, 256, 0, stream>>>(static_cast<float4*>(C), static_cast<const float4*>(A), vectors);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) error = cudaGetErrorString(status);
    return status != cudaSuccess;
}
