// H6 widens BN from 128 to 256, increasing operand reuse and WGMMA width.
// The shared queue drops from five to three stages to fit Hopper shared memory.
#include "engine.cuh"

namespace ablation {
void h6(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    hopper::launch<6>(C, A, B, M, N, K, stream);
}
}

