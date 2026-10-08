#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace gpu_kernels {

// Applies gate/up SwiGLU and route weights to a packed expert tensor.
// valid_rows points to the device-side final entry of DeepEP's expert prefix sum.
void moe_swiglu_bf16_cuda(__nv_bfloat16* output, const __nv_bfloat16* gate_up, const float* route_weights,
                           const int* valid_rows, int rows, int intermediate, float clamp, cudaStream_t stream);

}  // namespace gpu_kernels
