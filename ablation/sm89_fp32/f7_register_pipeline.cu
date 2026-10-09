// F7 alternates two register fragment buffers inside the K-tile computation.
// The next shared fragments are loaded before the current register outer product.
#include "engine.cuh"

namespace ablation {
void f7(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream) {
    launch_fp32<7>(C, A, B, M, N, K, stream);
}
}
