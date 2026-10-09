// F0: scalar dot-product baseline with the same TN storage contract.
// Each thread computes one C element; no shared-memory input reuse is introduced.
#include "engine.cuh"

namespace ablation {
void f0(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream) {
    launch_fp32<0>(C, A, B, M, N, K, stream);
}
}
