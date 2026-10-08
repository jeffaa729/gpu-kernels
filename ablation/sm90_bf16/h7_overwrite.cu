// H7 overwrites accumulator registers with the first WGMMA using ScaleD=0.
// Subsequent WGMMA instructions accumulate normally, removing explicit zero initialization.
#include "engine.cuh"

namespace ablation {
void h7(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    hopper::launch<7>(C, A, B, M, N, K, stream);
}
}

