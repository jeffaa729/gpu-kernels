// H4 caps the grid at 128 persistent CTAs and reuses the ring across output tiles.
// The producer can fetch the next output tile while consumers store the previous one.
#include "engine.cuh"

namespace ablation {
void h4(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    hopper::launch<4>(C, A, B, M, N, K, stream);
}
}

