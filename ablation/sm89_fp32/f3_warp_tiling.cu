// F3 arranges threads into 64x32 warp tiles with an 8x4 lane grid.
// Transposed shared A/B use [K][M/N] storage for efficient fragment reads.
#include "engine.cuh"

namespace ablation {
void f3(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream) {
    launch_fp32<3>(C, A, B, M, N, K, stream);
}
}
