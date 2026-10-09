// T1 loads eight BF16 elements with one aligned 16-byte vector access.
// Shared-memory storage remains unswizzled and the copy remains synchronous.
#include "engine.cuh"

namespace ablation {
void t1(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    launch_bf16<1>(C, A, B, M, N, K, stream);
}
}
