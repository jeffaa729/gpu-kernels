#pragma once

#include "common.cuh"
#include "cuda_common.h"

#include <stdexcept>

namespace gpu_kernels {
namespace flash_attention_sm89 {

constexpr int WARP_SIZE = 32;
constexpr int HEAD_SIZE = 128;

namespace tensor_core {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D = HEAD_SIZE;
constexpr int THREADS = 128;
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
// QK^T produces 64 score columns, while PV produces 128 output columns.
constexpr int SCORE_N_TILES = BN / MMA_N;
constexpr int OUTPUT_N_TILES = D / MMA_N;
constexpr int HEAD_K_TILES = D / MMA_K;
constexpr int TOKEN_K_TILES = BN / MMA_K;
constexpr int VECTOR_ELEMENTS = 8;

template <int STRIDE = D>
__device__ __forceinline__ int swizzle(int offset) {
    return offset ^ (((offset / STRIDE) & 7) << 3);
}

__device__ __forceinline__ unsigned int shared_address(const void* pointer) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void cp_async_bf16x8(__nv_bfloat16* destination, const __nv_bfloat16* source) {
    const unsigned int destination_address = shared_address(destination);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(destination_address), "l"(source));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void load_matrix_x4(unsigned int (&fragment)[4], unsigned int address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
        : "r"(address));
}

__device__ __forceinline__ void load_matrix_x4_transpose(unsigned int (&fragment)[4], unsigned int address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3])
        : "r"(address));
}

__device__ __forceinline__ void mma(float (&accumulator)[4], const unsigned int (&left)[4], const unsigned int* right) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
        : "r"(left[0]), "r"(left[1]), "r"(left[2]), "r"(left[3]), "r"(right[0]), "r"(right[1]));
}

__device__ __forceinline__ unsigned int pack_bf16x2(float2 values) {
    union Packed {
        __nv_bfloat162 bf16;
        unsigned int bits;
    } packed;
    packed.bf16 = __float22bfloat162_rn(values);
    return packed.bits;
}

__device__ __forceinline__ int tensor_index(int batch, int token, int head, int column, int sequence_length, int heads) {
    return ((batch * sequence_length + token) * heads + head) * D + column;
}

__device__ __forceinline__ void copy_bf16_tile(__nv_bfloat16* shared, const __nv_bfloat16* global, int batch, int first_token, int head, int sequence_length,
                                               int heads) {
    constexpr int VECTORS = BM * D / VECTOR_ELEMENTS;
    constexpr int VECTORS_PER_THREAD = VECTORS / THREADS;
    static_assert(VECTORS % THREADS == 0);
#pragma unroll
    for (int iteration = 0; iteration < VECTORS_PER_THREAD; ++iteration) {
        const int vector = threadIdx.x + iteration * THREADS;
        const int row = vector / (D / VECTOR_ELEMENTS);
        const int column = vector % (D / VECTOR_ELEMENTS) * VECTOR_ELEMENTS;
        const int logical = row * D + column;
        cp_async_bf16x8(shared + swizzle(logical), global + tensor_index(batch, first_token + row, head, column, sequence_length, heads));
    }
    cp_async_commit();
}

template <int COLUMNS = D>
__device__ __forceinline__ void load_left_fragments(unsigned int (&fragments)[COLUMNS / MMA_K][4], const __nv_bfloat16* shared, int warp_row, int lane) {
#pragma unroll
    for (int tile = 0; tile < COLUMNS / MMA_K; ++tile) {
        const int row = warp_row + lane % MMA_M;
        const int column = tile * MMA_K + lane / MMA_M * 8;
        load_matrix_x4(fragments[tile], shared_address(shared + swizzle<COLUMNS>(row * COLUMNS + column)));
    }
}

// One x4 instruction loads the B fragments for two adjacent 8-column MMA
// tiles. This halves ldmatrix issue count compared with two independent x2
// loads while preserving the same register layout consumed by mma.sync.
__device__ __forceinline__ void load_right_transposed_pair(unsigned int (&fragments)[4], const __nv_bfloat16* shared, int tile_inner, int tile_column,
                                                           int lane) {
    const int row = tile_column * MMA_N + lane / MMA_M * MMA_N + lane % MMA_N;
    const int column = tile_inner * MMA_K + (lane % MMA_M) / 8 * 8;
    load_matrix_x4(fragments, shared_address(shared + swizzle(row * D + column)));
}

__device__ __forceinline__ void load_right_pair(unsigned int (&fragments)[4], const __nv_bfloat16* shared, int tile_inner, int tile_column, int lane) {
    const int row = tile_inner * MMA_K + lane % MMA_M;
    const int column = tile_column * MMA_N + lane / MMA_M * MMA_N;
    load_matrix_x4_transpose(fragments, shared_address(shared + swizzle(row * D + column)));
}

template <int N_TILES, int K_TILES>
__device__ __forceinline__ void matrix_product_transposed_right(float (&accumulators)[N_TILES][4], const unsigned int (&left)[K_TILES][4],
                                                                const __nv_bfloat16* right, int lane) {
    static_assert(N_TILES % 2 == 0);
#pragma unroll
    for (int tile_inner = 0; tile_inner < K_TILES; ++tile_inner) {
#pragma unroll
        for (int tile_column = 0; tile_column < N_TILES; tile_column += 2) {
            unsigned int right_fragments[4];
            load_right_transposed_pair(right_fragments, right, tile_inner, tile_column, lane);
            mma(accumulators[tile_column], left[tile_inner], right_fragments);
            mma(accumulators[tile_column + 1], left[tile_inner], right_fragments + 2);
        }
    }
}

template <int N_TILES, int K_TILES>
__device__ __forceinline__ void matrix_product_right(float (&accumulators)[N_TILES][4], const unsigned int (&left)[K_TILES][4], const __nv_bfloat16* right,
                                                     int lane) {
    static_assert(N_TILES % 2 == 0);
#pragma unroll
    for (int tile_inner = 0; tile_inner < K_TILES; ++tile_inner) {
#pragma unroll
        for (int tile_column = 0; tile_column < N_TILES; tile_column += 2) {
            unsigned int right_fragments[4];
            load_right_pair(right_fragments, right, tile_inner, tile_column, lane);
            mma(accumulators[tile_column], left[tile_inner], right_fragments);
            mma(accumulators[tile_column + 1], left[tile_inner], right_fragments + 2);
        }
    }
}

__device__ __forceinline__ void store_pair(__nv_bfloat16* destination, float2 value) {
    *reinterpret_cast<__nv_bfloat162*>(destination) = __float22bfloat162_rn(value);
}

}  // namespace tensor_core

inline void validate_head_size(int head_size) {
    if (head_size != HEAD_SIZE) {
        throw std::runtime_error("flash attention requires head_size = 128");
    }
}

}  // namespace flash_attention_sm89
}  // namespace gpu_kernels
