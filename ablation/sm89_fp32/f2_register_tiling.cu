// F2 enlarges the block to 128x128 and gives each thread an 8x8 register tile.
// Register outer products reuse eight A and eight B values for 64 FMAs.
#include "engine.cuh"

namespace ablation {
void f2(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream) {
    launch_fp32<2>(C, A, B, M, N, K, stream);
}
}
