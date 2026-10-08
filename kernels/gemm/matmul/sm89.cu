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

namespace tc_mma {

constexpr int BK = 32;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int STAGES = 2;
constexpr int VECTOR_ELEMENTS = 8;

}  // namespace tc_mma

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

// BF16 asynchronous copy: move 16 bytes (eight BF16 values) directly from
// global to shared memory, avoiding intermediate data registers.
// L2::128B is a prefetch hint, not the number of bytes copied per thread.
__device__ __forceinline__ void cp_async_bf16x8(__nv_bfloat16* destination, const __nv_bfloat16* source) {
    const unsigned int shared_address = static_cast<unsigned int>(__cvta_generic_to_shared(destination));
    // asynchronous memory copy
    /*
    .cg : cache-global policy. Operation uses L2 cache and avoids allocate the copied data in L1
    .shared.global : specifies that source and destination is shared -> global
    .L2::128B : L2 prefetch size hint. GPU fetch a 128 byte region into L2 around the requested address
    [%0]， [%1] : shared_address and source
    */
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(shared_address), "l"(source));
}
//places all uncommitted copies issued by that thread into one asynchronous group
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
//Wait until zero previously committed groups remain incomplete
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group 0;\n" ::); }

// XOR swizzle: permute shared addresses to reduce ldmatrix bank conflicts
// while preserving 16-byte vector alignment. Loads and stores use this mapping.
template <int kBits>
__device__ __forceinline__ int swizzle_bf16_offset(int offset) {
    constexpr int kMask = ((1 << kBits) - 1) << 6;
    return offset ^ ((offset & kMask) >> 3);
}

__device__ __forceinline__ unsigned int shared_address(const void* pointer) { return static_cast<unsigned int>(__cvta_generic_to_shared(pointer)); }

// Warp-cooperative ldmatrix loads distribute BF16 operands into the register
// layout required by mma.sync; these are not ordinary per-thread float arrays.
/*
ldmatrix: load matrix data into registers
sync    : all warp lanes participate together
aligned : warp execution must be uniform
m8n8    : each component matrix is 8×8
x4      : load four 8×8 matrices
shared  : source is shared memory
b16     : each element contains 16 bits
*/
__device__ __forceinline__ void load_matrix_x4(unsigned int (&fragment)[4], unsigned int address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
        : "r"(address));
}

// K-contiguous B already has the column-operand layout expected by MMA.
// Load two 8x8 matrices without transposing them.
__device__ __forceinline__ void load_matrix_b_x2(unsigned int (&fragment)[2], unsigned int address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(fragment[0]), "=r"(fragment[1])
        : "r"(address));
}

// Native Tensor Core MMA: the whole warp computes a 16x8 C update over
// K=16 using BF16 operands and FP32 accumulation (4096 FLOPs per warp).
/*
mma.sync.aligned : warp-cooperative synchronous matrix multiply-accumulate
m16n8k16 : D[16,8] = A[16,16] × B[16,8] + C[16,8]
row.col : A is row-major, B is column-major
f32.bf16.bf16.f32 : destination D: FP32, operand A: BF16, operand B: BF16, accumulator C: FP32
*/
__device__ __forceinline__ void mma_bf16_m16n8k16(float (&accumulator)[4], const unsigned int (&A)[4], const unsigned int (&B)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "     // destination D
        "{%4, %5, %6, %7}, "     // operand A
        "{%8, %9}, "             // operand B
        "{%0, %1, %2, %3};\n"    // accumulator C, accumulator register appear twice, accumulator += A * B
        : "+f"(accumulator[0]),  // +f, register is both input and output, FP32 register
          "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}

// BF16 fast path combines block/warp tiling, asynchronous copies and swizzled
// shared storage. Launch bounds guide register allocation for two-block
// residency; actual occupancy also depends on shared memory and the GPU.
template <int kBM, int kBN, int kWarpTilesM, int kWarpTilesN>
__global__ __launch_bounds__(256, 2) void matmul_tensor_core_mma_kernel(__nv_bfloat16* __restrict__ C, const __nv_bfloat16* __restrict__ A,
                                                                        const __nv_bfloat16* __restrict__ B, int M, int N, int K) {
    (void)M;
    constexpr int kWM = kWarpTilesM * tc_mma::MMA_M;
    constexpr int kWN = kWarpTilesN * tc_mma::MMA_N;
    constexpr int kWarpsN = kBN / kWN;
    constexpr int kNumThreads = (kBM / kWM) * kWarpsN * 32;
    constexpr int kAStageElements = kBM * tc_mma::BK;
    constexpr int kBStageElements = tc_mma::BK * kBN;
    static_assert(kBM % kWM == 0);
    static_assert(kBN % kWN == 0);
    static_assert(tc_mma::BK % tc_mma::MMA_K == 0);
    static_assert(kAStageElements % tc_mma::VECTOR_ELEMENTS == 0);
    static_assert(kBStageElements % tc_mma::VECTOR_ELEMENTS == 0);

    // Two aligned shared stages overlap copying tile k+1 with MMA on tile k.
    __shared__ __align__(16) __nv_bfloat16 shared_A[tc_mma::STAGES][kAStageElements];
    __shared__ __align__(16) __nv_bfloat16 shared_B[tc_mma::STAGES][kBStageElements];

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / kWarpsN;
    const int warp_column = warp_id % kWarpsN;
    const int block_row = blockIdx.y * kBM;
    const int block_column = blockIdx.x * kBN;

    // Register accumulation: reuse A/B fragments across multiple MMA tiles.
    // The current 4x4 configuration keeps 64 FP32 partial outputs per thread.
    float accumulators[kWarpTilesM][kWarpTilesN][4];
#pragma unroll
    for (int tile_row = 0; tile_row < kWarpTilesM; ++tile_row) {
#pragma unroll
        for (int tile_column = 0; tile_column < kWarpTilesN; ++tile_column) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                accumulators[tile_row][tile_column][element] = 0.0F;
            }
        }
    }

    auto copy_stage = [&](int stage, int tile_inner) {
        constexpr int kAVectors = kAStageElements / tc_mma::VECTOR_ELEMENTS;
        constexpr int kBVectors = kBStageElements / tc_mma::VECTOR_ELEMENTS;
#pragma unroll
        for (int vector = tid; vector < kAVectors; vector += kNumThreads) {
            const int local_row = vector / (tc_mma::BK / tc_mma::VECTOR_ELEMENTS);
            const int local_inner = vector % (tc_mma::BK / tc_mma::VECTOR_ELEMENTS) * tc_mma::VECTOR_ELEMENTS;
            const int logical_offset = local_row * tc_mma::BK + local_inner;
            const int shared_offset = swizzle_bf16_offset<2>(logical_offset);
            const int global_offset = (block_row + local_row) * K + tile_inner + local_inner;
            cp_async_bf16x8(shared_A[stage] + shared_offset, A + global_offset);
        }
#pragma unroll
        for (int vector = tid; vector < kBVectors; vector += kNumThreads) {
            const int local_column = vector / (tc_mma::BK / tc_mma::VECTOR_ELEMENTS);
            const int local_inner = vector % (tc_mma::BK / tc_mma::VECTOR_ELEMENTS) * tc_mma::VECTOR_ELEMENTS;
            const int logical_offset = local_column * tc_mma::BK + local_inner;
            const int shared_offset = swizzle_bf16_offset<2>(logical_offset);
            const int global_offset = (block_column + local_column) * K + tile_inner + local_inner;
            cp_async_bf16x8(shared_B[stage] + shared_offset, B + global_offset);
        }
        cp_async_commit();
    };

    const int inner_tiles = K / tc_mma::BK;
    // Pipeline prologue: make the first tile visible to all consuming warps.
    copy_stage(0, 0);
    cp_async_wait();
    __syncthreads();

    for (int tile = 0; tile < inner_tiles; ++tile) {
        const int stage = tile % tc_mma::STAGES;
        const int next_tile = tile + 1;
        if (next_tile < inner_tiles) {
            // Issue future copies before current computation to hide latency.
            copy_stage(next_tile % tc_mma::STAGES, next_tile * tc_mma::BK);
        }

#pragma unroll
        for (int tile_inner = 0; tile_inner < tc_mma::BK; tile_inner += tc_mma::MMA_K) {
            // Load once per MMA K-step and reuse across the warp's C tile.
            // Unlike FP32, this path has no explicit register ping-pong buffers.
            unsigned int A_fragments[kWarpTilesM][4];
            unsigned int B_fragments[kWarpTilesN][2];
#pragma unroll
            for (int tile_row = 0; tile_row < kWarpTilesM; ++tile_row) {
                const int tile_row_base = warp_row * kWM + tile_row * tc_mma::MMA_M;
                const int fragment_row = tile_row_base + lane % 16;
                const int fragment_inner = tile_inner + lane / 16 * 8;
                const int logical_offset = fragment_row * tc_mma::BK + fragment_inner;
                const int swizzled_offset = swizzle_bf16_offset<2>(logical_offset);
                load_matrix_x4(A_fragments[tile_row], shared_address(shared_A[stage] + swizzled_offset));
            }
#pragma unroll
            for (int tile_column = 0; tile_column < kWarpTilesN; ++tile_column) {
                const int tile_column_base = warp_column * kWN + tile_column * tc_mma::MMA_N;
                const int fragment_inner = tile_inner + ((lane % 16) / 8) * 8;
                const int fragment_column = tile_column_base + lane % 8;
                const int logical_offset = fragment_column * tc_mma::BK + fragment_inner;
                const int swizzled_offset = swizzle_bf16_offset<2>(logical_offset);
                load_matrix_b_x2(B_fragments[tile_column], shared_address(shared_B[stage] + swizzled_offset));
            }
#pragma unroll
            for (int tile_row = 0; tile_row < kWarpTilesM; ++tile_row) {
#pragma unroll
                for (int tile_column = 0; tile_column < kWarpTilesN; ++tile_column) {
                    mma_bf16_m16n8k16(accumulators[tile_row][tile_column], A_fragments[tile_row], B_fragments[tile_column]);
                }
            }
        }

        if (next_tile < inner_tiles) {
            // Complete this thread's copies, then synchronize the block before
            // consuming the next tile or reusing the previous shared stage.
            cp_async_wait();
            __syncthreads();
        }
    }

#pragma unroll
    for (int tile_row = 0; tile_row < kWarpTilesM; ++tile_row) {
#pragma unroll
        for (int tile_column = 0; tile_column < kWarpTilesN; ++tile_column) {
            const int output_row = block_row + warp_row * kWM + tile_row * tc_mma::MMA_M + lane / 4;
            const int output_column = block_column + warp_column * kWN + tile_column * tc_mma::MMA_N + (lane % 4) * 2;
            // Convert FP32 accumulators directly into column-major BF16 output.
            C[output_column * M + output_row] = __float2bfloat16(accumulators[tile_row][tile_column][0]);
            C[(output_column + 1) * M + output_row] = __float2bfloat16(accumulators[tile_row][tile_column][1]);
            C[output_column * M + output_row + 8] = __float2bfloat16(accumulators[tile_row][tile_column][2]);
            C[(output_column + 1) * M + output_row + 8] = __float2bfloat16(accumulators[tile_row][tile_column][3]);
        }
    }
}

template <int kBM, int kBN, int kWarpTilesM, int kWarpTilesN>
void launch_tensor_core_mma_config(__nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B, int M, int N, int K, cudaStream_t stream) {
    constexpr int kWM = kWarpTilesM * tc_mma::MMA_M;
    constexpr int kWN = kWarpTilesN * tc_mma::MMA_N;
    constexpr int kNumThreads = (kBM / kWM) * (kBN / kWN) * 32;
    const dim3 blocks(N / kBN, M / kBM);
    matmul_tensor_core_mma_kernel<kBM, kBN, kWarpTilesM, kWarpTilesN><<<blocks, kNumThreads, 0, stream>>>(C, A, B, M, N, K);
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
    launch_tensor_core_mma_config<128, 128, 4, 4>(C, A, B, M, N, K, stream);
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
