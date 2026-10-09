// F5 adds float4 shared-memory fragment loads and column-major output stores.
// Each vector carries four FP32 elements without changing the arithmetic order.
#include "engine.cuh"

namespace ablation {
void f5(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream) {
    launch_fp32<5>(C, A, B, M, N, K, stream);
}
}
