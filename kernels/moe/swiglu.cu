// BF16 SwiGLU and routing-weight fold between the two unfused expert GEMMs.
// Trailing padding is cleared using DeepEP's device-side valid-row count.

#include "cuda_common.h"
#include "moe_swiglu.h"

#include <cmath>

namespace gpu_kernels {
namespace {

__global__ void moe_swiglu_kernel(__nv_bfloat16* output, const __nv_bfloat16* gate_up, const float* route_weights,
                                  const int* valid_rows, int rows, int intermediate, float clamp) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * intermediate) return;
    const int row = index / intermediate;
    if (row >= *valid_rows) {
        output[index] = __float2bfloat16(0.0F);
        return;
    }
    const int column = index % intermediate;
    const float gate = fminf(__bfloat162float(gate_up[row * intermediate * 2 + column]), clamp);
    const float up = fminf(fmaxf(__bfloat162float(gate_up[row * intermediate * 2 + intermediate + column]), -clamp), clamp);
    const float silu = gate / (1.0F + expf(-gate));
    output[index] = __float2bfloat16(silu * up * route_weights[row]);
}

}  // namespace

void moe_swiglu_bf16_cuda(__nv_bfloat16* output, const __nv_bfloat16* gate_up, const float* route_weights,
                           const int* valid_rows, int rows, int intermediate, float clamp, cudaStream_t stream) {
    constexpr int THREADS = 256;
    moe_swiglu_kernel<<<(rows * intermediate + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        output, gate_up, route_weights, valid_rows, rows, intermediate, clamp);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpu_kernels
