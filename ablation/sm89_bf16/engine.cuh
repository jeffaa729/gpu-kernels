#pragma once
#include "../api.cuh"

namespace ablation {

__device__ __forceinline__ unsigned int address(const void* p) { return static_cast<unsigned int>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void async_copy(bf16* dst, const bf16* src) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" :: "r"(address(dst)), "l"(src));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::); }
__device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group 0;" ::); }
__device__ __forceinline__ void ldmatrix_a(unsigned int (&a)[4], unsigned int p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(p));
}
__device__ __forceinline__ void ldmatrix_b(unsigned int (&b)[2], unsigned int p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b[0]), "=r"(b[1]) : "r"(p));
}
__device__ __forceinline__ void mma(float (&c)[4], const unsigned int (&a)[4], const unsigned int (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Every stage fixes block 128x128, warp 64x32, K tile 32 and native BF16 MMA.
// Only the input copy, shared address mapping and number of shared stages vary.
template <int Stage>
__global__ __launch_bounds__(256, 2) void bf16_kernel(bf16* __restrict__ C, const bf16* __restrict__ A, const bf16* __restrict__ B, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 32, STAGES = Stage == 4 ? 2 : 1;
    __shared__ __align__(16) bf16 As[STAGES][BM * BK], Bs[STAGES][BN * BK];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int wm = warp / 4, wn = warp % 4, bm = blockIdx.y * BM, bn = blockIdx.x * BN;
    float accum[4][4][4] = {};
    auto offset = [](int logical) {
        if constexpr (Stage >= 2) return logical ^ ((logical & (3 << 6)) >> 3);
        else return logical;
    };
    auto copy = [&](int stage, int tile) {
        if constexpr (Stage == 0) {
#pragma unroll
            for (int i = tid; i < BM * BK; i += 256) {
                As[stage][i] = A[(bm + i / BK) * K + tile + i % BK];
                Bs[stage][i] = B[(bn + i / BK) * K + tile + i % BK];
            }
        } else {
#pragma unroll
            for (int vector = tid; vector < BM * BK / 8; vector += 256) {
                const int logical = vector * 8, row = logical / BK, k = logical % BK;
                bf16* ad = As[stage] + offset(logical); bf16* bd = Bs[stage] + offset(logical);
                const bf16* ap = A + (bm + row) * K + tile + k; const bf16* bp = B + (bn + row) * K + tile + k;
                if constexpr (Stage >= 3) { async_copy(ad, ap); async_copy(bd, bp); }
                else { *reinterpret_cast<int4*>(ad) = *reinterpret_cast<const int4*>(ap); *reinterpret_cast<int4*>(bd) = *reinterpret_cast<const int4*>(bp); }
            }
            if constexpr (Stage >= 3) commit();
        }
    };
    if constexpr (Stage == 4) { copy(0, 0); wait(); __syncthreads(); }
    for (int tile = 0; tile < K; tile += BK) {
        const int stage = (tile / BK) % STAGES;
        if constexpr (Stage == 4) {
            if (tile + BK < K) copy(stage ^ 1, tile + BK);
        } else { copy(stage, tile); if constexpr (Stage == 3) wait(); __syncthreads(); }
#pragma unroll
        for (int k = 0; k < BK; k += 16) {
            unsigned int af[4][4], bf[4][2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int logical = (wm * 64 + i * 16 + lane % 16) * BK + k + (lane / 16) * 8;
                ldmatrix_a(af[i], address(As[stage] + offset(logical)));
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int logical = (wn * 32 + j * 8 + lane % 8) * BK + k + ((lane % 16) / 8) * 8;
                ldmatrix_b(bf[j], address(Bs[stage] + offset(logical)));
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) mma(accum[i][j], af[i], bf[j]);
            }
        }
        if constexpr (Stage == 4) {
            if (tile + BK < K) { wait(); __syncthreads(); }
        } else __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int m = bm + wm * 64 + i * 16 + lane / 4, n = bn + wn * 32 + j * 8 + (lane % 4) * 2;
            C[n * M + m] = __float2bfloat16(accum[i][j][0]);
            C[(n + 1) * M + m] = __float2bfloat16(accum[i][j][1]);
            C[n * M + m + 8] = __float2bfloat16(accum[i][j][2]);
            C[(n + 1) * M + m + 8] = __float2bfloat16(accum[i][j][3]);
        }
    }
}

template <int Stage>
void launch_bf16(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    bf16_kernel<Stage><<<dim3(N / 128, M / 128), 256, 0, stream>>>(C, A, B, M, N, K);
}
}
