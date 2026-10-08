// F1 adds cooperative input loading into 16x16x8 block tiles.
// Each thread reuses shared A/B while retaining one FP32 output accumulator.
#include "engine.cuh"

namespace ablation {
void f1(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream) {
    launch_fp32<1>(C, A, B, M, N, K, stream);
}
}
