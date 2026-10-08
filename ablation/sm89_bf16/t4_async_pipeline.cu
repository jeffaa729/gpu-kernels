// T4 introduces two shared stages and prefetches the next tile asynchronously.
// Tensor Cores consume the current stage while cp.async loads the next stage.
#include "engine.cuh"

namespace ablation {
void t4(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    launch_bf16<4>(C, A, B, M, N, K, stream);
}
}
