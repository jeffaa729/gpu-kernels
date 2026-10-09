// T0 uses native BF16 mma.sync and ldmatrix with synchronous scalar copies.
// Block 128x128, warp 64x32 and K tile 32 stay fixed throughout this sequence.
#include "engine.cuh"

namespace ablation {
void t0(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    launch_bf16<0>(C, A, B, M, N, K, stream);
}
}
