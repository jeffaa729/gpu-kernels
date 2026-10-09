// H8 converts register results into a shared C tile and stores it with TMA.
// Proxy fences and consumer barriers protect asynchronous reads before shared C is reused.
#include "engine.cuh"

namespace ablation {
void h8(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    hopper::launch<8>(C, A, B, M, N, K, stream);
}
}

