#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ablation {
using bf16 = __nv_bfloat16;
using FP32Launch = void (*)(float*, const float*, const float*, int, int, int, cudaStream_t);
using BF16Launch = void (*)(bf16*, const bf16*, const bf16*, int, int, int, cudaStream_t);
#define DECLARE_FP32(Stage) void f##Stage(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream)
DECLARE_FP32(0); DECLARE_FP32(1); DECLARE_FP32(2); DECLARE_FP32(3);
DECLARE_FP32(4); DECLARE_FP32(5); DECLARE_FP32(6); DECLARE_FP32(7);
#undef DECLARE_FP32
#define DECLARE_BF16(Stage) void t##Stage(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream)
DECLARE_BF16(0); DECLARE_BF16(1); DECLARE_BF16(2); DECLARE_BF16(3); DECLARE_BF16(4);
#undef DECLARE_BF16
}

extern "C" int ablation_init();
extern "C" void ablation_destroy();
extern "C" const char* ablation_error();
extern "C" int ablation_gemm(void* C, const void* A, const void* B, int M, int N, int K, int bf16, int stage, cudaStream_t stream);
extern "C" int ablation_attributes(int* attributes);
extern "C" int ablation_copy(void* C, const void* A, int vectors, cudaStream_t stream);
