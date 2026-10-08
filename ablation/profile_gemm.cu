#include "api.cuh"
#include <cuda_profiler_api.h>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: profile_gemm BF16_FLAG STAGE SIZE\n"); return 1; }
    const int bf16 = std::atoi(argv[1]), stage = std::atoi(argv[2]), n = std::atoi(argv[3]);
    const size_t count = static_cast<size_t>(n) * n, bytes = count * (bf16 ? 2 : 4);
    void *A, *B, *C;
    cudaMalloc(&A, bytes); cudaMalloc(&B, bytes); cudaMalloc(&C, bytes);
    std::mt19937 random(123); std::normal_distribution<float> normal(0, .1f);
    auto initialize = [&](void* dst) {
        if (bf16) {
            std::vector<ablation::bf16> values(count);
            for (auto& v : values) v = __float2bfloat16(normal(random));
            cudaMemcpy(dst, values.data(), bytes, cudaMemcpyHostToDevice);
        } else {
            std::vector<float> values(count);
            for (auto& v : values) v = normal(random);
            cudaMemcpy(dst, values.data(), bytes, cudaMemcpyHostToDevice);
        }
    };
    initialize(A); initialize(B);
    if (ablation_init()) return 1;
    auto run = [&]() { return ablation_gemm(C, A, B, n, n, n, bf16, stage, nullptr); };
    if (run() || cudaDeviceSynchronize() != cudaSuccess) { std::fprintf(stderr, "%s\n", ablation_error()); return 1; }
    cudaProfilerStart();
    const int status = run();
    const auto cuda_status = cudaDeviceSynchronize();
    cudaProfilerStop();
    ablation_destroy(); cudaFree(A); cudaFree(B); cudaFree(C);
    if (status || cuda_status != cudaSuccess) { std::fprintf(stderr, "%s %s\n", ablation_error(), cudaGetErrorString(cuda_status)); return 1; }
    return 0;
}
