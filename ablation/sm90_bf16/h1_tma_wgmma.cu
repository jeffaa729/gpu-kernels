// H1 replaces warp MMA and vector copies with Hopper TMA and WGMMA.
// A single shared stage serializes loads and compute for a 128x128 output tile.
#include "engine.cuh"

namespace ablation {
void h1(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    hopper::launch<1>(C, A, B, M, N, K, stream);
}
}

