#pragma once
#include "common.cuh"
#include <algorithm>

namespace ablation::hopper {
template <int Stage> struct Config {
    static_assert(Stage >= 1 && Stage <= 8);
    static constexpr int BN = Stage >= 6 ? 256 : 128;
    static constexpr int STAGES = Stage == 1 ? 1 : (BN == 128 ? 5 : 3);
    static constexpr int THREADS = Stage == 1 ? 256 : 384;
};

template <int Stage> struct Storage {
    alignas(ALIGNMENT) bf16 A[Config<Stage>::STAGES][BM * BK];
    alignas(ALIGNMENT) bf16 B[Config<Stage>::STAGES][Config<Stage>::BN * BK];
    alignas(ALIGNMENT) bf16 C[Stage == 8 ? BM * Config<Stage>::BN : 1];
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
template <int Stage, int ScaleD>
__device__ __forceinline__ void consume(Storage<Stage>& shared, uint64_t* full, uint64_t* empty,
    float (&accumulator)[Config<Stage>::BN / 16][8], int consumer, int& stage, int& phase) {
    barrier_wait(&full[stage], phase);
    bf16* A = shared.A[stage] + consumer * 64 * BK;
    warpgroup_fence();
    multiply<Config<Stage>::BN, ScaleD>(accumulator, A, shared.B[stage]);
#pragma unroll
    for (int ki = 16; ki < BK; ki += 16) multiply<Config<Stage>::BN, 1>(accumulator, A + ki, shared.B[stage] + ki);
    warpgroup_commit();
    warpgroup_wait();
    if constexpr (Stage == 1) __syncthreads();
    else if (threadIdx.x % 128 == 0) barrier_arrive(&empty[stage]);
    advance<Config<Stage>::STAGES>(stage, phase);
}
#endif

// One sequence is independently traversed by every producer/consumer group.
// The small-matrix group dimensions shrink instead of producing out-of-range tiles.
template <int Stage> struct Scheduler {
    int iteration = 0;
    int tiles_m, tiles_n;
    __device__ Scheduler(int M, int N) : tiles_m(M / BM), tiles_n(N / Config<Stage>::BN) {}
    __device__ int next() {
        if constexpr (Stage < 4) return iteration++ == 0 ? int(blockIdx.x) : -1;
        int linear = blockIdx.x + iteration++ * gridDim.x;
        if (linear >= tiles_m * tiles_n) return -1;
        if constexpr (Stage >= 5) {
            const int gm = tiles_m < 16 ? tiles_m : 16, gn = tiles_n < 8 ? tiles_n : 8;
            const int group = linear / (gm * gn), position = linear % (gm * gn);
            return (group / (tiles_n / gn) * gm + position / gn) * tiles_n
                 + group % (tiles_n / gn) * gn + position % gn;
        }
        return linear;
    }
};

template <int Stage>
__global__ __launch_bounds__(Config<Stage>::THREADS) void hopper_kernel(
    bf16* C, const __grid_constant__ CUtensorMap C_map, const __grid_constant__ CUtensorMap A_map,
    const __grid_constant__ CUtensorMap B_map, int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    constexpr int BN = Config<Stage>::BN, STAGES = Config<Stage>::STAGES;
    extern __shared__ __align__(16) unsigned char storage[];
    const uint32_t base = barrier_address(reinterpret_cast<uint64_t*>(storage));
    auto& shared = *reinterpret_cast<Storage<Stage>*>(__cvta_shared_to_generic((base + ALIGNMENT - 1) & ~(ALIGNMENT - 1)));
    __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES];
    if (threadIdx.x == 0) {
        for (int i = 0; i < STAGES; ++i) {
            barrier_init(&full[i], 1);
            barrier_init(&empty[i], 2);
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    Scheduler<Stage> scheduler(M, N);
    const int tid = threadIdx.x, group_thread = tid % 128, tiles_n = N / BN;
    int stage = 0, phase = 0;

    if constexpr (Stage >= 2) {
        if (tid < 128) {
            if constexpr (Stage >= 3) warpgroup_reg_dealloc<24>();
            if (tid == 0) {
                for (int tile = scheduler.next(); tile >= 0; tile = scheduler.next()) {
                    for (int k = 0; k < K; k += BK) {
                        barrier_wait(&empty[stage], phase);
                        barrier_expect_bytes(&full[stage], sizeof(shared.A[stage]) + sizeof(shared.B[stage]));
                        tma_load(shared.A[stage], &A_map, &full[stage], k, tile / tiles_n * BM);
                        tma_load(shared.B[stage], &B_map, &full[stage], k, tile % tiles_n * BN);
                        advance<STAGES>(stage, phase);
                    }
                }
            }
            return;
        }
        if constexpr (Stage >= 3) warpgroup_reg_alloc<240>();
        for (int i = 0; i < STAGES; ++i) if (group_thread == 0) barrier_arrive(&empty[i]);
    }

    const int consumer = Stage == 1 ? tid / 128 : tid / 128 - 1;
    const int lane = group_thread & 31, warp = group_thread >> 5;
    const int row = warp * 16 + lane / 4;
    float accumulator[BN / 16][8];
    for (int tile = scheduler.next(); tile >= 0; tile = scheduler.next()) {
        const int tile_m = tile / tiles_n, tile_n = tile % tiles_n;
        if constexpr (Stage < 7) {
#pragma unroll
            for (int i = 0; i < BN / 16; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) accumulator[i][j] = 0.f;
            }
        }
        // Peel the overwrite before the ordinary accumulate loop, matching
        // production Kernel 9/10 without runtime ScaleD branches inside WGMMA.
        if constexpr (Stage >= 7) consume<Stage, 0>(shared, full, empty, accumulator, consumer, stage, phase);
        for (int k = Stage >= 7 ? BK : 0; k < K; k += BK) {
            if constexpr (Stage == 1) {
                if (tid == 0) {
                    barrier_expect_bytes(&full[0], sizeof(shared.A[0]) + sizeof(shared.B[0]));
                    tma_load(shared.A[0], &A_map, &full[0], k, tile_m * BM);
                    tma_load(shared.B[0], &B_map, &full[0], k, tile_n * BN);
                }
            }
            consume<Stage, 1>(shared, full, empty, accumulator, consumer, stage, phase);
        }

        if constexpr (Stage == 8) {
            // Only the issuing thread owns the bulk group, then all consumers
            // synchronize before either group overwrites the shared C staging tile.
            if (tid == 128) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
            asm volatile("bar.sync 10, 256;" ::: "memory");
        }
#pragma unroll
        for (int i = 0; i < BN / 16; ++i) {
            const int column = i * 16 + 2 * (lane & 3);
#define STORE(R, Col, Value) \
    if constexpr (Stage == 8) shared.C[consumer * 64 * BN + (Col) * 64 + (R)] = __float2bfloat16(Value); \
    else C[(tile_n * BN + (Col)) * M + tile_m * BM + consumer * 64 + (R)] = __float2bfloat16(Value)
            STORE(row, column, accumulator[i][0]);
            STORE(row + 8, column, accumulator[i][2]);
            STORE(row, column + 1, accumulator[i][1]);
            STORE(row + 8, column + 1, accumulator[i][3]);
            STORE(row, column + 8, accumulator[i][4]);
            STORE(row + 8, column + 8, accumulator[i][6]);
            STORE(row, column + 9, accumulator[i][5]);
            STORE(row + 8, column + 9, accumulator[i][7]);
#undef STORE
        }
        if constexpr (Stage == 8) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("bar.sync 10, 256;" ::: "memory");
            if (tid == 128) {
                tma_store(&C_map, shared.C, tile_n * BN, tile_m * BM);
                asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            }
        }
    }
    if constexpr (Stage == 8) if (tid == 128) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
#endif
}

template <int Stage>
void launch(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    constexpr int BN = Config<Stage>::BN;
    if (M <= 0 || N <= 0 || K <= 0 || M % 128 || N % 256 || K % 64)
        throw std::invalid_argument("Hopper ablation requires M%128=N%256=K%64=0.");
    const auto am = make_tensor_map<BM, BK>(A, M, K);
    const auto bm = make_tensor_map<BN, BK>(B, N, K);
    CUtensorMap cm{};
    if constexpr (Stage == 8) cm = make_tensor_map<BN, BM, false>(C, N, M);
    constexpr size_t bytes = sizeof(Storage<Stage>) + ALIGNMENT - 1;
    const auto status = cudaFuncSetAttribute(hopper_kernel<Stage>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    const int tiles = (M / BM) * (N / BN);
    const int blocks = Stage >= 4 ? std::min(128, tiles) : tiles;
    hopper_kernel<Stage><<<blocks, Config<Stage>::THREADS, bytes, stream>>>(C, cm, am, bm, M, N, K);
}
} // namespace ablation::hopper
