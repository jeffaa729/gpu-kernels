#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace gpu_kernels {

// Peer addresses are local mappings established by PyTorch symmetric memory.
// Experts share a compact, BM-aligned token pool; routes retain capacity strides.
struct MoePeers {
    const void* x[4];
    const int64_t* ids[4];
    const float* weights[4];
    void* output[4];
    void* pool;
    float* route_weights;
    int* ready;
    int64_t* offsets;  // local_experts + 1 prefix offsets, in pool/activation rows
};

struct MoeShape {
    int tokens, hidden, intermediate, experts, topk, rank, world, capacity;
    float clamp;
    int64_t pool_capacity;
};

void moe_metadata_cuda(MoePeers peers, MoeShape shape, int* routes, int* counts, cudaStream_t stream);
void moe_phase1_cuda(MoePeers peers, MoeShape shape, const void* w1, void* intermediate,
                     const int* routes, const int* counts, int sms, cudaStream_t stream);
void moe_phase2_cuda(MoePeers peers, MoeShape shape, const void* w2, const void* intermediate,
                     const int* routes, const int* counts, int sms, cudaStream_t stream);
void moe_reduce_cuda(void* output, const void* slots, MoeShape shape, cudaStream_t stream);

} // namespace gpu_kernels
