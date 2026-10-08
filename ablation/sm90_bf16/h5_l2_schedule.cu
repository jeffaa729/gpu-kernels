// H5 groups logical tiles in 16x8 regions to encourage concurrent A/B reuse in L2.
// Smaller matrices shrink the group dimensions; no partial output tiles are used.
#include "engine.cuh"

namespace ablation {
void h5(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    hopper::launch<5>(C, A, B, M, N, K, stream);
}
}

