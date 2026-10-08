#pragma once
#include "../api.cuh"

namespace ablation {

// F0 uses scalar dot products with column-major output mapping. Input vectors
// are K-contiguous; scalar accesses across output threads are strided in this layout.
static __global__ void scalar_kernel(float* C, const float* A, const float* B, int M, int N, int K, int first_element) {
    const int index = first_element + blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= M * N) return;
    const int m = index % M, n = index / M;
    float sum = 0;
    for (int k = 0; k < K; ++k) sum = __fmaf_rn(A[m * K + k], B[n * K + k], sum);
    C[index] = sum;
}

// F1 introduces 16x16 output tiles and cooperatively loads K-contiguous inputs.
// Each thread retains one output accumulator and reuses the shared input tiles.
static __global__ void tiled_kernel(float* C, const float* A, const float* B, int M, int N, int K, int first_row_tile) {
    __shared__ float As[16][8], Bs[16][8];
    const int tid = threadIdx.x, m = tid % 16, n = tid / 16;
    const int bm = (first_row_tile + blockIdx.y) * 16, bn = blockIdx.x * 16;
    float sum = 0;
    for (int tile = 0; tile < K; tile += 8) {
        if (tid < 128) {
            As[tid / 8][tid % 8] = A[(bm + tid / 8) * K + tile + tid % 8];
            Bs[tid / 8][tid % 8] = B[(bn + tid / 8) * K + tile + tid % 8];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 8; ++k) sum = __fmaf_rn(As[m][k], Bs[n][k], sum);
        __syncthreads();
    }
    C[(bn + n) * M + bm + m] = sum;
}

// Compile-time stages progressively enable the techniques described in each
// stage file. Shared code keeps arithmetic and tile dimensions identical.
template <int Stage>
__global__ void fp32_kernel(float* __restrict__ C, const float* __restrict__ A, const float* __restrict__ B, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    constexpr int STAGES = Stage >= 6 ? 2 : 1;
    __shared__ __align__(16) float As[STAGES][BM * BK], Bs[STAGES][BN * BK];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
    const int m = Stage >= 3 ? (warp / 4) * 64 + (lane / 4) * TM : (tid / 16) * TM;
    const int n = Stage >= 3 ? (warp % 4) * 32 + (lane % 4) * TN : (tid % 16) * TN;
    float accum[TM][TN] = {};

    auto smem_index = [](int row, int k) {
        if constexpr (Stage >= 3) return k * BM + row;
        else return row * BK + k;
    };
    auto scalar_copy = [&](int stage, int tile) {
#pragma unroll
        for (int i = tid; i < BM * BK; i += 256) {
            const int row = i / BK, k = i % BK;
            As[stage][smem_index(row, k)] = A[(bm + row) * K + tile + k];
            Bs[stage][smem_index(row, k)] = B[(bn + row) * K + tile + k];
        }
    };
    auto vector_load = [&](int tile, float4& av, float4& bv) {
        const int row = tid / 2, k = (tid % 2) * 4;
        av = *reinterpret_cast<const float4*>(A + (bm + row) * K + tile + k);
        bv = *reinterpret_cast<const float4*>(B + (bn + row) * K + tile + k);
    };
    auto vector_publish = [&](int stage, const float4& av, const float4& bv) {
        const int row = tid / 2, k = (tid % 2) * 4;
        As[stage][smem_index(row, k)] = av.x; Bs[stage][smem_index(row, k)] = bv.x;
        As[stage][smem_index(row, k + 1)] = av.y; Bs[stage][smem_index(row, k + 1)] = bv.y;
        As[stage][smem_index(row, k + 2)] = av.z; Bs[stage][smem_index(row, k + 2)] = bv.z;
        As[stage][smem_index(row, k + 3)] = av.w; Bs[stage][smem_index(row, k + 3)] = bv.w;
    };
    auto fragment_load = [&](int stage, int k, float* af, float* bf) {
        if constexpr (Stage >= 5) {
#pragma unroll
            for (int i = 0; i < TM; i += 4) {
                *reinterpret_cast<float4*>(af + i) = *reinterpret_cast<const float4*>(As[stage] + smem_index(m + i, k));
                *reinterpret_cast<float4*>(bf + i) = *reinterpret_cast<const float4*>(Bs[stage] + smem_index(n + i, k));
            }
        } else {
#pragma unroll
            for (int i = 0; i < TM; ++i) { af[i] = As[stage][smem_index(m + i, k)]; bf[i] = Bs[stage][smem_index(n + i, k)]; }
        }
    };
    auto outer_product = [&](const float* af, const float* bf) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
            for (int j = 0; j < TN; ++j) accum[i][j] = __fmaf_rn(af[i], bf[j], accum[i][j]);
        }
    };

    if constexpr (Stage >= 6) {
        float4 av, bv;
        vector_load(0, av, bv);
        vector_publish(0, av, bv);
        __syncthreads();
    }
    for (int tile = 0; tile < K; tile += BK) {
        const int stage = (tile / BK) % STAGES;
        float4 next_A, next_B;
        if constexpr (Stage >= 6) {
            if (tile + BK < K) vector_load(tile + BK, next_A, next_B);
        } else {
            if constexpr (Stage >= 4) { vector_load(tile, next_A, next_B); vector_publish(stage, next_A, next_B); }
            else scalar_copy(stage, tile);
            __syncthreads();
        }
        if constexpr (Stage >= 7) {
            __align__(16) float af[2][TM], bf[2][TN];
            fragment_load(stage, 0, af[0], bf[0]);
#pragma unroll
            for (int k = 0; k < BK - 1; ++k) {
                fragment_load(stage, k + 1, af[(k + 1) & 1], bf[(k + 1) & 1]);
                outer_product(af[k & 1], bf[k & 1]);
            }
            outer_product(af[1], bf[1]);
        } else {
            __align__(16) float af[TM], bf[TN];
#pragma unroll
            for (int k = 0; k < BK; ++k) { fragment_load(stage, k, af, bf); outer_product(af, bf); }
        }
        __syncthreads();
        if constexpr (Stage >= 6) {
            if (tile + BK < K) { vector_publish(stage ^ 1, next_A, next_B); __syncthreads(); }
        }
    }
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        if constexpr (Stage >= 5) {
#pragma unroll
            for (int i = 0; i < TM; i += 4) {
                *reinterpret_cast<float4*>(C + (bn + n + j) * M + bm + m + i) = make_float4(accum[i][j], accum[i + 1][j], accum[i + 2][j], accum[i + 3][j]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < TM; ++i) C[(bn + n + j) * M + bm + m + i] = accum[i][j];
        }
    }
}

template <int Stage>
void launch_fp32(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream) {
    if constexpr (Stage == 0) {
        // Bound each slow scalar launch for display-GPU watchdog safety.
        // Chunking changes scheduling only; each dot product is unchanged.
        const int chunk = K <= 2048 ? M * N : (1 << 20);
        for (int first = 0; first < M * N; first += chunk) {
            const int count = M * N - first < chunk ? M * N - first : chunk;
            scalar_kernel<<<(count + 255) / 256, 256, 0, stream>>>(C, A, B, M, N, K, first);
        }
    }
    else if constexpr (Stage == 1) {
        const int row_tiles = M / 16, chunk = K <= 2048 ? row_tiles : 128;
        for (int first = 0; first < row_tiles; first += chunk) {
            const int count = row_tiles - first < chunk ? row_tiles - first : chunk;
            tiled_kernel<<<dim3(N / 16, count), 256, 0, stream>>>(C, A, B, M, N, K, first);
        }
    }
    else fp32_kernel<Stage><<<dim3(N / 128, M / 128), 256, 0, stream>>>(C, A, B, M, N, K);
}
}
