// T2 introduces an XOR address permutation for both shared stores and ldmatrix.
// The swizzle preserves 16-byte alignment while reducing shared bank conflicts.
#include "engine.cuh"

namespace ablation {
void t2(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    launch_bf16<2>(C, A, B, M, N, K, stream);
}
}
