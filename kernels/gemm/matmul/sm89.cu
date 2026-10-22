// Implements C = A * B with K-contiguous A/B and column-major C with FP32 CUDA Core and BF16 Tensor Core kernels.

#include "cuda_common.h"
#include "common.cuh"

#include <cuda_bf16.h>
#include <stdexcept>

namespace gpu_kernels {
namespace {

// FP32 hierarchical tiling: block 128x128, warp 64x32, thread 8x8.
// Each level reuses inputs to reduce traffic from the preceding memory level.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int WM = 64;
constexpr int WN = 32;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int VECTOR_WIDTH = 4;


static_assert(WM == 8 * TM);
static_assert(WN == 4 * TN);
static_assert(BK % VECTOR_WIDTH == 0);
static_assert(BK % 2 == 0);

// FP32 vectorized loading: read four adjacent floats with one aligned access.
// Scalar loads handle partial or unaligned vectors without reading out of bounds.
__device__ __forceinline__ float4 load_float4(const float* input, int index, bool major_valid, int start, int length) {
    int width = 0;
    if (major_valid && start < length) {
        const int remaining = length - start;
        width = remaining < VECTOR_WIDTH ? remaining : VECTOR_WIDTH;
    }
    if (width == VECTOR_WIDTH && index % VECTOR_WIDTH == 0) {
        return *reinterpret_cast<const float4*>(input + index);
    }

    float4 value = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    if (width > 0) {
        value.x = input[index];
    }
    if (width > 1) {
        value.y = input[index + 1];
    }
    if (width > 2) {
        value.z = input[index + 2];
    }
    if (width > 3) {
        value.w = input[index + 3];
    }
    return value;
}
/*
kBM : Rows of C computed by a block    (128)
kBN : Col of C computed by a block     (128)
kWM : Row of C computed by a wrap     (64)
kWN : Col of C computed by a wrap     (32)
kTM : Row of C computed by a thread     (8)
kTN : Col of C computed by a thread     (8)
*/
template <int kBM, int kBN, int kWM, int kWN, int kTM, int kTN>
__global__ void matmul_kernel(float* __restrict__ C, const float* __restrict__ A, const float* __restrict__ B, int M, int N, int K) {
    // Compile-time specialization lets the compiler fold tile indexing and
    // unroll fixed-size loops; __restrict__ also rules out pointer aliasing.

    // no of warp tiles are needed in 1 block column (128 / 32)
    constexpr int kWarpsPerRow = kBN / kWN;
    // no of threads in 1 block  (128/64) * (128/32) * 32 = 256 threads
    constexpr int kNumThreads = (kBM / kWM) * (kBN / kWN) * 32;
    // no of loads of A per thread (128 * 8) / 4 / 256 , each block load 128*8 element with 128*8 / 4 vectorized times , each thread thus loads 128*8/4/256 = 1 times
    constexpr int kALoads = (kBM * BK / VECTOR_WIDTH) / kNumThreads;
    // same as A , thus each thread loads 1 float4 from A and 1 float4 from B
    constexpr int kBLoads = (BK * kBN / VECTOR_WIDTH) / kNumThreads;
    // check the divisibility
    static_assert(kBM % kWM == 0);
    static_assert(kBN % kWN == 0);
    static_assert(kWM == 8 * kTM);
    static_assert(kWN == 4 * kTN);
    static_assert(kTM % VECTOR_WIDTH == 0);
    static_assert(kTN % VECTOR_WIDTH == 0);
    static_assert((kBM * BK / VECTOR_WIDTH) % kNumThreads == 0);
    static_assert((BK * kBN / VECTOR_WIDTH) % kNumThreads == 0);

    // FP32 shared-memory double buffering: alternate tiles to overlap global prefetch with computation.
    // stage 0 : cur/next A and B tile, stage 1: next/cur A and B tile
    // A is transposed to [K][M] for float4 reads;
    __shared__ float shared_A[2 * BK * kBM];
    // B stays [K][N]. Logical layouts include an outer [stage] dimension.
    __shared__ float shared_B[2 * BK * kBN];

    const int tid = threadIdx.x;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    // Warp tiling: 8x4 lanes cover one warp tile. Lanes sharing a row or column reuse the same A or B fragments through shared-memory reads.
    // e.g. warp id = 0 -> warp_row = 0 , warp_col = 0, warp id = 5 -> warp_row = 1 , warp_col = 1,
    const int warp_row = warp_id / kWarpsPerRow;
    const int warp_column = warp_id % kWarpsPerRow;
    // lane position inside a warp, since lanes are treated as 8 x 4 grid so /4 or %4
    const int lane_row = lane_id / 4;
    const int lane_column = lane_id % 4;
    // the beginning of a thread 8x8 output relative to the whole block output 128x128
    // e.g. warp 0 : local row = 0, local col = 0, warp 31 : local row = 56, local col = 24
    const int local_row = warp_row * kWM + lane_row * kTM;
    const int local_column = warp_column * kWN + lane_column * kTN;
    // block output matrix 128x128 relative to the whole matrix
    const int block_row = blockIdx.y * kBM;
    const int block_column = blockIdx.x * kBN;
    // Aligned interior blocks bypass all scalar edge handling. determines if the kernel use full float4 loads
    const bool full_tile = block_row + kBM <= M && block_column + kBN <= N && K % BK == 0 && K % VECTOR_WIDTH == 0;

    // Register tiling: retain every partial C across all K tiles.
    // Independent accumulators expose instruction-level parallelism (ILP).
    // init as zero, these value stay in registers throughout the complete K loop. so only written to global memory only once after the whole K tiles hv been computed.
    float accumulator[kTM][kTN] = {};
    // Register double buffering: prefetch the next K-step fragments while
    // computing with the current ones, at the cost of more live registers.
    float A_fragment[2][kTM];
    float B_fragment[2][kTN];
    int write_stage = 0;

    // Each iteration first issues the global loads for the next tile, computes the preceding tile, and then publishes the loaded values to shared memory.
    for (int tile_to_load = 0;; tile_to_load += BK) {
        const bool load_tile = tile_to_load < K;
        const bool compute_tile = tile_to_load > 0;
        float4 loaded_A[kALoads];
        float4 loaded_B[kBLoads];

        if (load_tile) {
#pragma unroll
            for (int load = 0; load < kALoads; ++load) {
                // Coalesced cooperative loading: neighboring threads own
                // neighboring float4 segments within each input tile row.

                //which float4 inside the complete A tile belongs to this thread
                const int vector_index = tid + load * kNumThreads;
                // no of vectors per A row
                const int vectors_per_row = BK / VECTOR_WIDTH;
                const int tile_row = vector_index / vectors_per_row;
                const int tile_inner = (vector_index % vectors_per_row) * VECTOR_WIDTH;
                const int global_row = block_row + tile_row;
                const int global_inner = tile_to_load + tile_inner;
                const int index = global_row * K + global_inner;
                loaded_A[load] = full_tile ? *reinterpret_cast<const float4*>(A + index) : load_float4(A, index, global_row < M, global_inner, K);
            }

#pragma unroll
            for (int load = 0; load < kBLoads; ++load) {
                const int vector_index = tid + load * kNumThreads;
                const int vectors_per_column = BK / VECTOR_WIDTH;
                const int tile_column = vector_index / vectors_per_column;
                const int tile_inner = (vector_index % vectors_per_column) * VECTOR_WIDTH;
                const int global_inner = tile_to_load + tile_inner;
                const int global_column = block_column + tile_column;
                const int index = global_column * K + global_inner;
                loaded_B[load] = full_tile ? *reinterpret_cast<const float4*>(B + index) : load_float4(B, index, global_column < N, global_inner, K);
            }
        }

        const int read_stage = write_stage ^ 1;
        if (compute_tile) {
#pragma unroll
            for (int inner = 0; inner < BK - 1; ++inner) {
                const int read_fragment = inner & 1;
                const int write_fragment = (inner + 1) & 1;

                // Vectorized shared loads prepare the next register fragments
                // before the current outer product consumes its operands.

#pragma unroll
                for (int vector = 0; vector < kTM / VECTOR_WIDTH; ++vector) {
                    const float4 value = *reinterpret_cast<const float4*>(&shared_A[(read_stage * BK + inner + 1) * kBM + local_row + vector * VECTOR_WIDTH]);
                    A_fragment[write_fragment][vector * VECTOR_WIDTH + 0] = value.x;
                    A_fragment[write_fragment][vector * VECTOR_WIDTH + 1] = value.y;
                    A_fragment[write_fragment][vector * VECTOR_WIDTH + 2] = value.z;
                    A_fragment[write_fragment][vector * VECTOR_WIDTH + 3] = value.w;
                }

#pragma unroll
                for (int vector = 0; vector < kTN / VECTOR_WIDTH; ++vector) {
                    const float4 value =
                        *reinterpret_cast<const float4*>(&shared_B[(read_stage * BK + inner + 1) * kBN + local_column + vector * VECTOR_WIDTH]);
                    B_fragment[write_fragment][vector * VECTOR_WIDTH + 0] = value.x;
                    B_fragment[write_fragment][vector * VECTOR_WIDTH + 1] = value.y;
                    B_fragment[write_fragment][vector * VECTOR_WIDTH + 2] = value.z;
                    B_fragment[write_fragment][vector * VECTOR_WIDTH + 3] = value.w;
                }

#pragma unroll
                for (int row = 0; row < kTM; ++row) {
#pragma unroll
                    for (int column = 0; column < kTN; ++column) {
                        // Register outer product: reuse each A value across
                        // columns and each B value across rows using FP32 FMA.
                        accumulator[row][column] = __fmaf_rn(A_fragment[read_fragment][row], B_fragment[read_fragment][column], accumulator[row][column]);
                    }
                }
            }
        }

        if (load_tile) {
#pragma unroll
            for (int load = 0; load < kALoads; ++load) {
                const int vector_index = tid + load * kNumThreads;
                const int vectors_per_row = BK / VECTOR_WIDTH;
                const int tile_row = vector_index / vectors_per_row;
                const int tile_inner = (vector_index % vectors_per_row) * VECTOR_WIDTH;
                // Scatter the contiguous global A vector into transposed
                // shared storage; a single contiguous cp.async cannot do this.
                shared_A[(write_stage * BK + tile_inner + 0) * kBM + tile_row] = loaded_A[load].x;
                shared_A[(write_stage * BK + tile_inner + 1) * kBM + tile_row] = loaded_A[load].y;
                shared_A[(write_stage * BK + tile_inner + 2) * kBM + tile_row] = loaded_A[load].z;
                shared_A[(write_stage * BK + tile_inner + 3) * kBM + tile_row] = loaded_A[load].w;
            }

#pragma unroll
            for (int load = 0; load < kBLoads; ++load) {
                const int vector_index = tid + load * kNumThreads;
                const int vectors_per_column = BK / VECTOR_WIDTH;
                const int tile_column = vector_index / vectors_per_column;
                const int tile_inner = (vector_index % vectors_per_column) * VECTOR_WIDTH;
                const int index = (write_stage * BK + tile_inner) * kBN + tile_column;
                shared_B[index + 0 * kBN] = loaded_B[load].x;
                shared_B[index + 1 * kBN] = loaded_B[load].y;
                shared_B[index + 2 * kBN] = loaded_B[load].z;
                shared_B[index + 3 * kBN] = loaded_B[load].w;
            }

            __syncthreads();
            const int loaded_stage = write_stage;
            write_stage ^= 1;

#pragma unroll
            for (int vector = 0; vector < kTM / VECTOR_WIDTH; ++vector) {
                const float4 value = *reinterpret_cast<const float4*>(&shared_A[loaded_stage * BK * kBM + local_row + vector * VECTOR_WIDTH]);
                A_fragment[0][vector * VECTOR_WIDTH + 0] = value.x;
                A_fragment[0][vector * VECTOR_WIDTH + 1] = value.y;
                A_fragment[0][vector * VECTOR_WIDTH + 2] = value.z;
                A_fragment[0][vector * VECTOR_WIDTH + 3] = value.w;
            }

#pragma unroll
            for (int vector = 0; vector < kTN / VECTOR_WIDTH; ++vector) {
                const float4 value = *reinterpret_cast<const float4*>(&shared_B[loaded_stage * BK * kBN + local_column + vector * VECTOR_WIDTH]);
                B_fragment[0][vector * VECTOR_WIDTH + 0] = value.x;
                B_fragment[0][vector * VECTOR_WIDTH + 1] = value.y;
                B_fragment[0][vector * VECTOR_WIDTH + 2] = value.z;
                B_fragment[0][vector * VECTOR_WIDTH + 3] = value.w;
            }
        }

        if (compute_tile) {
#pragma unroll
            for (int row = 0; row < kTM; ++row) {
#pragma unroll
                for (int column = 0; column < kTN; ++column) {
                    accumulator[row][column] = __fmaf_rn(A_fragment[1][row], B_fragment[1][column], accumulator[row][column]);
                }
            }
        }

        if (!load_tile) {
            break;
        }
    }

#pragma unroll
    for (int column = 0; column < kTN; ++column) {
        const int global_column = block_column + local_column + column;
#pragma unroll
        for (int row = 0; row < kTM; row += VECTOR_WIDTH) {
            const int global_row = block_row + local_row + row;
            const int index = global_column * M + global_row;
            const float4 value = make_float4(accumulator[row][column], accumulator[row + 1][column],
                                            accumulator[row + 2][column], accumulator[row + 3][column]);
            if (global_column < N && global_row + VECTOR_WIDTH <= M && index % VECTOR_WIDTH == 0) {
                *reinterpret_cast<float4*>(C + index) = value;
            } else if (global_column < N) {
                if (global_row < M) C[index] = value.x;
                if (global_row + 1 < M) C[index + 1] = value.y;
                if (global_row + 2 < M) C[index + 2] = value.z;
                if (global_row + 3 < M) C[index + 3] = value.w;
            }
        }
    }
}

// BF16 uses one configuration for all supported shapes: a 128x128 CTA,
// 64x64 warp tiles and K8 MMA instructions with FP32 accumulation.
__device__ __forceinline__ unsigned int shared_address(const void* pointer) { return static_cast<unsigned int>(__cvta_generic_to_shared(pointer)); }

// Load four 8x8 B matrices: one packed BF16 register per K8 operand.
__device__ __forceinline__ void load_matrix_x4(unsigned int (&fragment)[4], unsigned int address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
        : "r"(address));
}


__global__ __launch_bounds__(128, 3) void matmul_tensor_core_mma_kernel(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B, int M, int N, int K) {
    // Two K32 stages occupy 32 KiB. The register budget and shared footprint
    // allow three resident 128-thread CTAs on SM89.
    __shared__ __align__(128) __nv_bfloat16 shared[16384];
    __nv_bfloat16* shared_A = shared;
    __nv_bfloat16* shared_B = shared + 8192;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_row = warp / 2, warp_column = warp % 2;
    // The same two-row traversal is used for every matrix size. It keeps
    // neighboring CTAs close in both operands instead of sweeping a full row.
    const int linear = blockIdx.x + blockIdx.y * gridDim.x;
    const int group = linear / (2 * gridDim.x);
    const int first_row = group * 2;
    const int group_rows = min(2, static_cast<int>(gridDim.y) - first_row);
    const int within_group = linear % (2 * gridDim.x);
    const int block_row = (first_row + within_group % group_rows) * 128;
    const int block_column = (within_group / group_rows) * 128;
    auto swizzle = [](int x) { return x ^ ((x & (3 << 6)) >> 3); };
    // Hoist lane-dependent address arithmetic out of the K loop. Because
    // shared is 128-byte aligned, a K8 slice changes only address bits 4/5.
    const unsigned int A_load = shared_address(shared_A + swizzle((warp_row * 64 + (lane & 15)) * 32));
    const unsigned int B_load = shared_address(shared_B + swizzle((warp_column * 64 + lane) * 32));
    const unsigned int A_store = shared_address(shared_A + swizzle(tid * 8));
    const unsigned int B_store = shared_address(shared_B + swizzle(tid * 8));
    const __nv_bfloat16* A_source = A + (block_row + tid / 4) * K + (tid & 3) * 8;
    const __nv_bfloat16* B_source = B + (block_column + tid / 4) * K + (tid & 3) * 8;
    // Precompute one source/destination base per operand, rather than keeping
    // four source pointers live. Each thread copies four 16-byte vectors.
    auto copy = [&](unsigned int stage, int k) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                :: "r"(A_store + stage + v * 2048), "l"(A_source + v * 32 * K + k) : "memory");
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                :: "r"(B_store + stage + v * 2048), "l"(B_source + v * 32 * K + k) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    copy(0, 0);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    float accumulators[4][8][4] = {};
    for (int tile = 0; tile < K / 32; ++tile) {
        // Load the next K32 tile while MMA consumes the current shared slot.
        const unsigned int current = (tile & 1) * 8192;
        if (tile + 1 < K / 32) copy(current ^ 8192, (tile + 1) * 32);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            unsigned int B_fragments[8];
            load_matrix_x4(*reinterpret_cast<unsigned int (*)[4]>(B_fragments), (B_load ^ (k * 16)) + current);
            load_matrix_x4(*reinterpret_cast<unsigned int (*)[4]>(B_fragments + 4), (B_load ^ (k * 16)) + current + 2048);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // Retain one A fragment at a time and reuse the eight B
                // operands across all four A tiles to limit live registers.
                unsigned int A_fragment[2];
                const unsigned int addr = (A_load ^ (k * 16)) + current + i * 1024;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(A_fragment[0]), "=r"(A_fragment[1]) : "r"(addr));
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) {
                    // Serpentine traversal reuses the boundary B operand.
                    const int j = (i & 1) ? 7 - jj : jj;
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+f"(accumulators[i][j][0]), "+f"(accumulators[i][j][1]), "+f"(accumulators[i][j][2]), "+f"(accumulators[i][j][3])
                        : "r"(A_fragment[0]), "r"(A_fragment[1]), "r"(B_fragments[j]));
                }
            }
        }
        if (tile + 1 < K / 32) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
    }
    // The launcher computes C^T = B^T A^T; packed pairs are therefore
    // contiguous in the public column-major output without a transpose pass.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = warp_row * 64 + i * 16 + lane / 4;
            const int c = warp_column * 64 + j * 8 + lane % 4 * 2;
            *reinterpret_cast<__nv_bfloat162*>(C + (block_row + r) * N + block_column + c) = __floats2bfloat162_rn(accumulators[i][j][0], accumulators[i][j][1]);
            *reinterpret_cast<__nv_bfloat162*>(C + (block_row + r + 8) * N + block_column + c) = __floats2bfloat162_rn(accumulators[i][j][2], accumulators[i][j][3]);
        }
    }
}

template <int kBM, int kBN, int kWM, int kWN, int kTM, int kTN>
void launch_matmul_config(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream) {
    constexpr int kNumThreads = (kBM / kWM) * (kBN / kWN) * 32;
    const dim3 blocks((N + kBN - 1) / kBN, (M + kBM - 1) / kBM);
    matmul_kernel<kBM, kBN, kWM, kWN, kTM, kTN><<<blocks, kNumThreads, 0, stream>>>(C, A, B, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void launch_matmul(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream) {
    launch_matmul_config<BM, BN, WM, WN, TM, TN>(C, A, B, M, N, K, stream);
}

void launch_tensor_core_matmul(__nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B, int M, int N, int K, cudaStream_t stream) {
    matmul_tensor_core_mma_kernel<<<dim3(M / 128, N / 128), 128, 0, stream>>>(C, B, A, N, M, K);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void gemm_fp32_sm89_cuda(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream) {
    launch_matmul(C, A, B, M, N, K, stream);
}

void gemm_bf16_sm89_cuda(__nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B, int M, int N, int K, cudaStream_t stream) {
    launch_tensor_core_matmul(C, A, B, M, N, K, stream);
}

}  // namespace gpu_kernels
