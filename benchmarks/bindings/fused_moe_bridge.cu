#include "fused_moe.h"
#include <exception>
#include <string>

namespace {
thread_local std::string last_error;
}

extern "C" const char* gpu_kernels_fused_moe_last_error() { return last_error.c_str(); }

extern "C" int gpu_kernels_moe_phase(int phase, const gpu_kernels::MoePeers* peers,
        const gpu_kernels::MoeShape* shape, const void* weights, void* activation,
        int* routes, int* counts, int sms, cudaStream_t stream) {
    try {
        switch (phase) {
            case 0: gpu_kernels::moe_metadata_cuda(*peers, *shape, routes, counts, stream); break;
            case 1: gpu_kernels::moe_phase1_cuda(*peers, *shape, weights, activation, routes, counts, sms, stream); break;
            case 2: gpu_kernels::moe_phase2_cuda(*peers, *shape, weights, activation, routes, counts, sms, stream); break;
            case 3: gpu_kernels::moe_reduce_cuda(activation, weights, *shape, stream); break;
        }
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}
