// T3 replaces synchronous vector copies with cp.async and an immediate wait.
// A single shared stage isolates the copy instruction from pipeline overlap.
#include "engine.cuh"

namespace ablation {
void t3(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    launch_bf16<3>(C, A, B, M, N, K, stream);
}
}
