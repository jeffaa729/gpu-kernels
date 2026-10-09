// H3 releases producer registers with setmaxnreg and grants them to consumers.
// The tile and five-stage queue stay unchanged to isolate register redistribution.
#include "engine.cuh"

namespace ablation {
void h3(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    hopper::launch<3>(C, A, B, M, N, K, stream);
}
}

