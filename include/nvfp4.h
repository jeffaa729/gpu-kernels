#pragma once

#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace gpu_kernels {

// C[M,N] is row-major FP16; packed A[M,K] and B[N,K] are K-contiguous E2M1.
// SFA/SFB use cuBLASLt's VEC16 UE4M3 layout; fixed suites use multiples of 256.
// Tensor maps and device scheduling are prepared once, outside graph capture.
struct Nvfp4Params {
    CUtensorMap A, B, SFA, SFB;
    int M, N, K, clusters;
};

struct Nvfp4Implementation {
    void (*prepare)(Nvfp4Params&, const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*,
                    int, int, int, cudaStream_t);
    void (*launch)(const Nvfp4Params&, __half*, cudaStream_t);
    void (*validate)();
};

Nvfp4Implementation nvfp4_sm100_implementation();
Nvfp4Implementation nvfp4_r0_implementation();
Nvfp4Implementation nvfp4_r1_implementation();
Nvfp4Implementation nvfp4_r2_implementation();
Nvfp4Implementation nvfp4_r3_implementation();
Nvfp4Implementation nvfp4_r4_implementation();
Nvfp4Implementation nvfp4_r5_implementation();
Nvfp4Implementation nvfp4_r6_implementation();

}  // namespace gpu_kernels
