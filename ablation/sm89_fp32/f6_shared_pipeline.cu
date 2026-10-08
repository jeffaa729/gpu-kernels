// F6 prefetches the next global input tile before computing the current tile.
// Two shared stages retain the tiles while prefetched vectors stay in registers.
#include "engine.cuh"

namespace ablation {
void f6(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream) {
    launch_fp32<6>(C, A, B, M, N, K, stream);
}
}
