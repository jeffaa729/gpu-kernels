// H0 retains the existing SM89 mma.sync pipeline as the Hopper starting point.
// It uses asynchronous vector copies, XOR swizzling and FP32 accumulation.
#include "../sm89_bf16/engine.cuh"

namespace ablation {
void h0(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    launch_bf16<4>(C, A, B, M, N, K, stream);
}
}

