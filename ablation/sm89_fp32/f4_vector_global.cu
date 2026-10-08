// F4 replaces scalar cooperative global loads with aligned float4 accesses.
// The tile shapes and warp/register computation remain the same as F3.
#include "engine.cuh"

namespace ablation {
void f4(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream) {
    launch_fp32<4>(C, A, B, M, N, K, stream);
}
}
