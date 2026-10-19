// cuBLASLt consumes the same packed FP4 values and block scales as the custom API.
// Plans, scale pointers, algorithm selection, and workspace are prepared before timing.
#include "nvfp4.h"

#include <cublasLt.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;

void lt_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLASLt status " + std::to_string(status));
    }
}

struct Plan {
    gpu_kernels::Nvfp4Params params{};
    gpu_kernels::Nvfp4Implementation implementation{};
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t A = nullptr, B = nullptr, C = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    void* workspace = nullptr;
    size_t workspace_bytes = 0;

    ~Plan() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (C) cublasLtMatrixLayoutDestroy(C);
        if (B) cublasLtMatrixLayoutDestroy(B);
        if (A) cublasLtMatrixLayoutDestroy(A);
        if (op) cublasLtMatmulDescDestroy(op);
        if (handle) cublasLtDestroy(handle);
    }
};
}  // namespace

extern "C" const char* gpu_kernels_nvfp4_last_error() {
    return last_error.c_str();
}

extern "C" int gpu_kernels_nvfp4_plan_create(void** result, int M, int N, int K, int stage,
                                            const void* A, const void* B, const void* SFA, const void* SFB,
                                            void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    try {
        auto plan = std::make_unique<Plan>();
        lt_check(cublasLtCreate(&plan->handle));
        lt_check(cublasLtMatmulDescCreate(&plan->op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t trans_A = CUBLAS_OP_T, trans_B = CUBLAS_OP_N;
        const auto scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        const int8_t fast_accum = 0;
        lt_check(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_TRANSA, &trans_A, sizeof(trans_A)));
        lt_check(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_TRANSB, &trans_B, sizeof(trans_B)));
        lt_check(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)));
        lt_check(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
        lt_check(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
        lt_check(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &SFA, sizeof(SFA)));
        lt_check(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &SFB, sizeof(SFB)));

        // Column-major [K,M]/[K,N] describe the physical row-major [M,K]/[N,K] bytes.
        lt_check(cublasLtMatrixLayoutCreate(&plan->A, CUDA_R_4F_E2M1, K, M, K));
        lt_check(cublasLtMatrixLayoutCreate(&plan->B, CUDA_R_4F_E2M1, K, N, K));
        lt_check(cublasLtMatrixLayoutCreate(&plan->C, CUDA_R_16F, M, N, M > N ? M : N));
        const cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
        const int64_t ld = N;
        lt_check(cublasLtMatrixLayoutSetAttribute(plan->C, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));
        lt_check(cublasLtMatrixLayoutSetAttribute(plan->C, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)));
        lt_check(cublasLtMatmulPreferenceCreate(&plan->preference));
        lt_check(cublasLtMatmulPreferenceSetAttribute(plan->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                    &workspace_bytes, sizeof(workspace_bytes)));
        cublasLtMatmulHeuristicResult_t heuristic{};
        int returned = 0;
        lt_check(cublasLtMatmulAlgoGetHeuristic(plan->handle, plan->op, plan->A, plan->B,
                                             plan->C, plan->C, plan->preference, 1, &heuristic, &returned));
        if (!returned) throw std::runtime_error("cuBLASLt found no NVFP4 algorithm; use a B200 and CUDA 12.8 or newer.");
        lt_check(heuristic.state);
        plan->algorithm = heuristic.algo;
        plan->workspace = workspace;
        plan->workspace_bytes = heuristic.workspaceSize;
        const gpu_kernels::Nvfp4Implementation stages[] = {
            gpu_kernels::nvfp4_r0_implementation(), gpu_kernels::nvfp4_r1_implementation(),
            gpu_kernels::nvfp4_r2_implementation(), gpu_kernels::nvfp4_r3_implementation(),
            gpu_kernels::nvfp4_r4_implementation(), gpu_kernels::nvfp4_r5_implementation(),
            gpu_kernels::nvfp4_r6_implementation()};
        plan->implementation = stage < 0 ? gpu_kernels::nvfp4_sm100_implementation() : stages[stage];
        plan->implementation.prepare(plan->params, static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(B),
            static_cast<const uint8_t*>(SFA), static_cast<const uint8_t*>(SFB), M, N, K, stream);
        *result = plan.release();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}

extern "C" void gpu_kernels_nvfp4_plan_destroy(void* plan) {
    delete static_cast<Plan*>(plan);
}

extern "C" int gpu_kernels_nvfp4_cublas(void* context, void* C, const void* A, const void* B, cudaStream_t stream) {
    try {
        auto& plan = *static_cast<Plan*>(context);
        const float alpha = 1.0F, beta = 0.0F;
        lt_check(cublasLtMatmul(plan.handle, plan.op, &alpha, A, plan.A, B, plan.B, &beta,
                              C, plan.C, C, plan.C, &plan.algorithm,
                              plan.workspace, plan.workspace_bytes, stream));
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}

extern "C" int gpu_kernels_nvfp4_custom(void* context, void* C, cudaStream_t stream) {
    try {
        auto& plan = *static_cast<Plan*>(context);
        plan.implementation.launch(plan.params, static_cast<__half*>(C), stream);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}

extern "C" int gpu_kernels_nvfp4_validate(void* context) {
    try {
        auto& plan = *static_cast<Plan*>(context);
        if (plan.implementation.validate) plan.implementation.validate();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}

extern "C" int gpu_kernels_nvfp4_l2_bytes() {
    int device = 0, bytes = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&bytes, cudaDevAttrL2CacheSize, device);
    return bytes;
}
