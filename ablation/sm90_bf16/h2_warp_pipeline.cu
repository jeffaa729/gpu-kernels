// H2 separates a TMA producer from two WGMMA consumers.
// Five shared stages overlap operand loads with Tensor Core work.
#include "engine.cuh"

namespace ablation {
void h2(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    hopper::launch<2>(C, A, B, M, N, K, stream);
}
}

