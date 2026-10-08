// Implements the SM90 BF16 D128 causal FlashAttention-3 forward path with raw CUDA/PTX.
// A producer warp group feeds TMA while consumers overlap QK, online softmax, and P@V.

#include "common.cuh"
#include "cuda_common.h"

#include <cuda.h>
#include <cudaTypedefs.h>

#include <cfloat>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu_kernels {
namespace {

using bf16 = __nv_bfloat16;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D = 128;
constexpr int HALF_D = 64;
constexpr int WGMMA_K = 16;
constexpr int STAGES = 2;
constexpr int CONSUMER_THREADS = 128;
constexpr int NUM_THREADS = 256;
constexpr int GROUPS = 4;
constexpr int TMA_TILE_BYTES = BM * HALF_D * sizeof(bf16);
constexpr int KV_TRANSACTION_BYTES = 4 * TMA_TILE_BYTES;
constexpr int SMEM_ALIGNMENT = 1024;
constexpr float LOG2E = 1.4426950408889634F;

struct SharedStorage {
    alignas(SMEM_ALIGNMENT) bf16 Q[2][BM * HALF_D];
    alignas(SMEM_ALIGNMENT) bf16 K[STAGES][2][BN * HALF_D];
    alignas(128) bf16 V[STAGES][2][BN * HALF_D];
    alignas(SMEM_ALIGNMENT) bf16 Vt[STAGES][D * BN];
    alignas(SMEM_ALIGNMENT) bf16 P[BM * BN];
};

constexpr size_t SMEM_BYTES = sizeof(SharedStorage) + SMEM_ALIGNMENT - 1;

template <bool Swizzle>
CUtensorMap make_qkv_map(const bf16* pointer, int batch_size, int sequence_length, int heads) {
    CUtensorMap map;
    void* address = const_cast<bf16*>(pointer);
    const uint64_t global_shape[5] = {HALF_D, 2, static_cast<uint64_t>(heads), static_cast<uint64_t>(sequence_length),
                                      static_cast<uint64_t>(batch_size)};
    const uint64_t global_stride[4] = {
        HALF_D * sizeof(bf16), D * sizeof(bf16), static_cast<uint64_t>(heads) * D * sizeof(bf16),
        static_cast<uint64_t>(sequence_length) * heads * D * sizeof(bf16)};
    const uint32_t box_shape[5] = {HALF_D, 1, 1, BM, 1};
    const uint32_t element_stride[5] = {1, 1, 1, 1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, address, global_shape, global_stride, box_shape, element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, Swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(result, &name);
        throw std::runtime_error(std::string("cuTensorMapEncodeTiled failed: ") + (name ? name : "unknown driver error"));
    }
    return map;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

__device__ __forceinline__ uint32_t barrier_address(const uint64_t* barrier) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
}

__device__ __forceinline__ void barrier_init(uint64_t* barrier, uint32_t arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(barrier_address(barrier)), "r"(arrivals) : "memory");
}

__device__ __forceinline__ void barrier_arrive(uint64_t* barrier) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(barrier_address(barrier)) : "memory");
}

__device__ __forceinline__ void barrier_expect_bytes(uint64_t* barrier, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(barrier_address(barrier)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void barrier_wait(uint64_t* barrier, int phase) {
    asm volatile(
        "{ .reg .pred done;\n"
        "wait_loop_%=:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1;\n"
        "@!done bra wait_loop_%=;\n}"
        :: "r"(barrier_address(barrier)), "r"(phase) : "memory");
}

__device__ __forceinline__ void tma_load(bf16* destination, const CUtensorMap* map, uint64_t* barrier, int token, int half, int head, int batch) {
    const uint32_t shared = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {0, %4, %5, %3, %6}], [%2];"
        :: "r"(shared), "l"(map), "r"(barrier_address(barrier)), "r"(token), "r"(half), "r"(head), "r"(batch) : "memory");
}

__device__ __forceinline__ void advance_stage(int& stage, int& phase) {
    if (++stage == STAGES) {
        stage = 0;
        phase ^= 1;
    }
}

__device__ __forceinline__ uint64_t encode_descriptor(uint64_t value) {
    return (value & 0x3FFFFU) >> 4U;
}

__device__ __forceinline__ uint64_t make_smem_descriptor(bf16* pointer) {
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
    uint64_t descriptor = encode_descriptor(address);
    descriptor |= encode_descriptor(16) << 16U;
    descriptor |= encode_descriptor(1024) << 32U;
    descriptor |= 1ULL << 62U;
    return descriptor;
}

__device__ __forceinline__ int swizzle_128b_bf16(int row, int column) {
    // Match the K-major 128-byte swizzle encoded in the WGMMA descriptor.
    return row * 64 + (column ^ ((row & 7) << 3));
}

__device__ __forceinline__ void warpgroup_fence() {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

template <int Pending>
__device__ __forceinline__ void warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(Pending) : "memory");
}

__device__ __forceinline__ void consumer_sync() {
    asm volatile("bar.sync 1, 128;" ::: "memory");
}

template <int ScaleD>
__device__ __forceinline__ void wgmma_m64n64k16(float (&accumulator)[GROUPS][8], bf16* A, bf16* B) {
    const uint64_t A_descriptor = make_smem_descriptor(A);
    const uint64_t B_descriptor = make_smem_descriptor(B);
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, %34, 1, 1, 0, 0;"
        : "+f"(accumulator[0][0]), "+f"(accumulator[0][1]), "+f"(accumulator[0][2]), "+f"(accumulator[0][3]),
          "+f"(accumulator[0][4]), "+f"(accumulator[0][5]), "+f"(accumulator[0][6]), "+f"(accumulator[0][7]),
          "+f"(accumulator[1][0]), "+f"(accumulator[1][1]), "+f"(accumulator[1][2]), "+f"(accumulator[1][3]),
          "+f"(accumulator[1][4]), "+f"(accumulator[1][5]), "+f"(accumulator[1][6]), "+f"(accumulator[1][7]),
          "+f"(accumulator[2][0]), "+f"(accumulator[2][1]), "+f"(accumulator[2][2]), "+f"(accumulator[2][3]),
          "+f"(accumulator[2][4]), "+f"(accumulator[2][5]), "+f"(accumulator[2][6]), "+f"(accumulator[2][7]),
          "+f"(accumulator[3][0]), "+f"(accumulator[3][1]), "+f"(accumulator[3][2]), "+f"(accumulator[3][3]),
          "+f"(accumulator[3][4]), "+f"(accumulator[3][5]), "+f"(accumulator[3][6]), "+f"(accumulator[3][7])
        : "l"(A_descriptor), "l"(B_descriptor), "n"(ScaleD));
}

__device__ __forceinline__ void issue_qk(float (&scores)[GROUPS][8], SharedStorage& shared, int stage) {
    warpgroup_fence();
    wgmma_m64n64k16<0>(scores, shared.Q[0], shared.K[stage][0]);
#pragma unroll
    for (int k = 1; k < HALF_D / WGMMA_K; ++k) {
        wgmma_m64n64k16<1>(scores, shared.Q[0] + k * WGMMA_K, shared.K[stage][0] + k * WGMMA_K);
    }
#pragma unroll
    for (int k = 0; k < HALF_D / WGMMA_K; ++k) {
        wgmma_m64n64k16<1>(scores, shared.Q[1] + k * WGMMA_K, shared.K[stage][1] + k * WGMMA_K);
    }
    warpgroup_commit();
}

__device__ __forceinline__ void issue_pv(float (&output)[2][GROUPS][8], SharedStorage& shared, int value_stage) {
    warpgroup_fence();
#pragma unroll
    for (int k = 0; k < BN / WGMMA_K; ++k) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            wgmma_m64n64k16<1>(output[half], shared.P + k * WGMMA_K,
                                shared.Vt[value_stage] + half * 64 * BN + k * WGMMA_K);
        }
    }
    warpgroup_commit();
}

__device__ __forceinline__ void transpose_value(SharedStorage& shared, int stage, int consumer_thread) {
#pragma unroll 4
    for (int index = consumer_thread; index < D * BN; index += CONSUMER_THREADS) {
        const int column = index / BN;
        const int token = index % BN;
        const int half = column / HALF_D;
        const int row = column % HALF_D;
        // P@V consumes V as [output column, key token], in the same swizzled
        // K-major layout that QK uses for its two shared-memory operands.
        shared.Vt[stage][half * HALF_D * BN + swizzle_128b_bf16(row, token)] =
            shared.V[stage][half][token * HALF_D + row];
    }
    consumer_sync();
}

template <bool First, bool Masked>
__device__ __forceinline__ void online_softmax(float (&scores)[GROUPS][8], float (&row_max)[2], float (&row_sum)[2], float (&previous_scale)[2],
                                               int query_start, int key_start, int consumer_thread, float scale_log2) {
    const int lane = consumer_thread & 31;
    const int warp = consumer_thread >> 5;
    const int top_row = warp * 16 + lane / 4;
    const int bottom_row = top_row + 8;
    float current_max[2] = {-FLT_MAX, -FLT_MAX};

#pragma unroll
    for (int group = 0; group < GROUPS; ++group) {
        const int column = group * 16 + 2 * (lane & 3);
        if constexpr (Masked) {
            scores[group][0] = query_start + top_row >= key_start + column ? scores[group][0] : -FLT_MAX;
            scores[group][1] = query_start + top_row >= key_start + column + 1 ? scores[group][1] : -FLT_MAX;
            scores[group][4] = query_start + top_row >= key_start + column + 8 ? scores[group][4] : -FLT_MAX;
            scores[group][5] = query_start + top_row >= key_start + column + 9 ? scores[group][5] : -FLT_MAX;
            scores[group][2] = query_start + bottom_row >= key_start + column ? scores[group][2] : -FLT_MAX;
            scores[group][3] = query_start + bottom_row >= key_start + column + 1 ? scores[group][3] : -FLT_MAX;
            scores[group][6] = query_start + bottom_row >= key_start + column + 8 ? scores[group][6] : -FLT_MAX;
            scores[group][7] = query_start + bottom_row >= key_start + column + 9 ? scores[group][7] : -FLT_MAX;
        }
        current_max[0] = fmaxf(current_max[0], fmaxf(fmaxf(scores[group][0], scores[group][1]), fmaxf(scores[group][4], scores[group][5])));
        current_max[1] = fmaxf(current_max[1], fmaxf(fmaxf(scores[group][2], scores[group][3]), fmaxf(scores[group][6], scores[group][7])));
    }
    current_max[0] = fmaxf(current_max[0], __shfl_xor_sync(0xffffffff, current_max[0], 1));
    current_max[0] = fmaxf(current_max[0], __shfl_xor_sync(0xffffffff, current_max[0], 2));
    current_max[1] = fmaxf(current_max[1], __shfl_xor_sync(0xffffffff, current_max[1], 1));
    current_max[1] = fmaxf(current_max[1], __shfl_xor_sync(0xffffffff, current_max[1], 2));

    const float next_max[2] = {First ? current_max[0] : fmaxf(row_max[0], current_max[0]),
                               First ? current_max[1] : fmaxf(row_max[1], current_max[1])};
    previous_scale[0] = First ? 1.0F : exp2f((row_max[0] - next_max[0]) * scale_log2);
    previous_scale[1] = First ? 1.0F : exp2f((row_max[1] - next_max[1]) * scale_log2);
    float local_sum[2] = {};
    const float maximum_scaled[2] = {next_max[0] * scale_log2, next_max[1] * scale_log2};

#pragma unroll
    for (int group = 0; group < GROUPS; ++group) {
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const int row_id = item == 0 || item == 1 || item == 4 || item == 5 ? 0 : 1;
            scores[group][item] = exp2f(__fmaf_rn(scores[group][item], scale_log2, -maximum_scaled[row_id]));
            local_sum[row_id] += scores[group][item];
        }
    }
    local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], 1);
    local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], 2);
    local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], 1);
    local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], 2);
    row_sum[0] = (First ? 0.0F : row_sum[0] * previous_scale[0]) + local_sum[0];
    row_sum[1] = (First ? 0.0F : row_sum[1] * previous_scale[1]) + local_sum[1];
    row_max[0] = next_max[0];
    row_max[1] = next_max[1];
}

__device__ __forceinline__ void write_probabilities(SharedStorage& shared, const float (&scores)[GROUPS][8], int consumer_thread) {
    const int lane = consumer_thread & 31;
    const int warp = consumer_thread >> 5;
    const int row = warp * 16 + lane / 4;
#define STORE_P(Row, Column, Value) shared.P[swizzle_128b_bf16((Row), (Column))] = __float2bfloat16(Value)
#pragma unroll
    for (int group = 0; group < GROUPS; ++group) {
        const int column = group * 16 + 2 * (lane & 3);
        STORE_P(row, column, scores[group][0]);
        STORE_P(row, column + 1, scores[group][1]);
        STORE_P(row + 8, column, scores[group][2]);
        STORE_P(row + 8, column + 1, scores[group][3]);
        STORE_P(row, column + 8, scores[group][4]);
        STORE_P(row, column + 9, scores[group][5]);
        STORE_P(row + 8, column + 8, scores[group][6]);
        STORE_P(row + 8, column + 9, scores[group][7]);
    }
#undef STORE_P
    consumer_sync();
}

__device__ __forceinline__ void scale_output(float (&output)[2][GROUPS][8], const float (&factor)[2]) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int group = 0; group < GROUPS; ++group) {
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                const int row_id = item == 0 || item == 1 || item == 4 || item == 5 ? 0 : 1;
                output[half][group][item] *= factor[row_id];
            }
        }
    }
}

__device__ __forceinline__ int tensor_index(int batch, int token, int head, int column, int sequence_length, int heads) {
    return ((batch * sequence_length + token) * heads + head) * D + column;
}

__device__ __forceinline__ void store_pair(bf16* destination, float first, float second) {
    *reinterpret_cast<__nv_bfloat162*>(destination) = __floats2bfloat162_rn(first, second);
}

__device__ __forceinline__ void store_output(bf16* output, float* logsumexp, float (&accumulator)[2][GROUPS][8], const float (&row_max)[2],
                                             const float (&row_sum)[2], int batch, int query_head, int query_start, int sequence_length,
                                             int query_heads, int consumer_thread, float scale) {
    const int lane = consumer_thread & 31;
    const int warp = consumer_thread >> 5;
    const int row = warp * 16 + lane / 4;
    const float inverse_sum[2] = {1.0F / row_sum[0], 1.0F / row_sum[1]};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int group = 0; group < GROUPS; ++group) {
            const int column = half * 64 + group * 16 + 2 * (lane & 3);
            store_pair(output + tensor_index(batch, query_start + row, query_head, column, sequence_length, query_heads),
                       accumulator[half][group][0] * inverse_sum[0], accumulator[half][group][1] * inverse_sum[0]);
            store_pair(output + tensor_index(batch, query_start + row + 8, query_head, column, sequence_length, query_heads),
                       accumulator[half][group][2] * inverse_sum[1], accumulator[half][group][3] * inverse_sum[1]);
            store_pair(output + tensor_index(batch, query_start + row, query_head, column + 8, sequence_length, query_heads),
                       accumulator[half][group][4] * inverse_sum[0], accumulator[half][group][5] * inverse_sum[0]);
            store_pair(output + tensor_index(batch, query_start + row + 8, query_head, column + 8, sequence_length, query_heads),
                       accumulator[half][group][6] * inverse_sum[1], accumulator[half][group][7] * inverse_sum[1]);
        }
    }
    if ((lane & 3) == 0) {
        const int batch_head = batch * query_heads + query_head;
        logsumexp[batch_head * sequence_length + query_start + row] = row_max[0] * scale + __logf(row_sum[0]);
        logsumexp[batch_head * sequence_length + query_start + row + 8] = row_max[1] * scale + __logf(row_sum[1]);
    }
}

#endif

__global__ __launch_bounds__(NUM_THREADS) void flash_attention_forward_sm90_kernel(
    bf16* output, float* logsumexp, const __grid_constant__ CUtensorMap Q_map, const __grid_constant__ CUtensorMap K_map,
    const __grid_constant__ CUtensorMap V_map, int sequence_length, int query_heads, int key_value_heads, float scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(16) unsigned char storage[];
    const unsigned int storage_address = static_cast<unsigned int>(__cvta_generic_to_shared(storage));
    const unsigned int aligned_address = (storage_address + SMEM_ALIGNMENT - 1) & ~(SMEM_ALIGNMENT - 1);
    auto& shared = *reinterpret_cast<SharedStorage*>(__cvta_shared_to_generic(aligned_address));

    __shared__ __align__(8) uint64_t query_full;
    __shared__ __align__(8) uint64_t full[STAGES];
    __shared__ __align__(8) uint64_t empty[STAGES];
    if (threadIdx.x == 0) {
        barrier_init(&query_full, 1);
        for (int stage = 0; stage < STAGES; ++stage) {
            barrier_init(&full[stage], 1);
            barrier_init(&empty[stage], 1);
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();

    const int query_block = blockIdx.x;
    const int query_start = query_block * BM;
    const int query_head = blockIdx.y;
    const int batch = blockIdx.z;
    const int key_value_head = query_head / (query_heads / key_value_heads);
    const int warpgroup = threadIdx.x / 128;
    const int warpgroup_thread = threadIdx.x % 128;

    if (warpgroup == 0) {
        if (warpgroup_thread == 0) {
            barrier_expect_bytes(&query_full, 2 * TMA_TILE_BYTES);
            tma_load(shared.Q[0], &Q_map, &query_full, query_start, 0, query_head, batch);
            tma_load(shared.Q[1], &Q_map, &query_full, query_start, 1, query_head, batch);
            int stage = 0;
            int phase = 0;
            for (int iteration = 0; iteration <= query_block; ++iteration) {
                const int key_start = (query_block - iteration) * BN;
                barrier_wait(&empty[stage], phase);
                barrier_expect_bytes(&full[stage], KV_TRANSACTION_BYTES);
                tma_load(shared.K[stage][0], &K_map, &full[stage], key_start, 0, key_value_head, batch);
                tma_load(shared.K[stage][1], &K_map, &full[stage], key_start, 1, key_value_head, batch);
                tma_load(shared.V[stage][0], &V_map, &full[stage], key_start, 0, key_value_head, batch);
                tma_load(shared.V[stage][1], &V_map, &full[stage], key_start, 1, key_value_head, batch);
                advance_stage(stage, phase);
            }
        }
        return;
    }

    if (warpgroup_thread == 0) {
        for (int stage = 0; stage < STAGES; ++stage) barrier_arrive(&empty[stage]);
    }
    barrier_wait(&query_full, 0);

    float output_accumulator[2][GROUPS][8] = {};
    float row_max[2] = {-FLT_MAX, -FLT_MAX};
    float row_sum[2] = {};
    float previous_scale[2];
    const float scale_log2 = scale * LOG2E;
    int stage = 0;
    int phase = 0;

    barrier_wait(&full[stage], phase);
    transpose_value(shared, stage, warpgroup_thread);
    float current_scores[GROUPS][8] = {};
    issue_qk(current_scores, shared, stage);
    warpgroup_wait<0>();
    if (warpgroup_thread == 0) barrier_arrive(&empty[stage]);
    online_softmax<true, true>(current_scores, row_max, row_sum, previous_scale, query_start, query_start, warpgroup_thread, scale_log2);
    write_probabilities(shared, current_scores, warpgroup_thread);
    int value_stage = stage;
    advance_stage(stage, phase);

    // QK(next) is the older WGMMA group and PV(current) is newer. Waiting for
    // one group exposes scores to CUDA cores while Tensor Cores finish PV.
    for (int iteration = 1; iteration <= query_block; ++iteration) {
        const int key_start = (query_block - iteration) * BN;
        barrier_wait(&full[stage], phase);
        transpose_value(shared, stage, warpgroup_thread);
        float next_scores[GROUPS][8] = {};
        issue_qk(next_scores, shared, stage);
        issue_pv(output_accumulator, shared, value_stage);
        warpgroup_wait<1>();
        if (warpgroup_thread == 0) barrier_arrive(&empty[stage]);
        online_softmax<false, false>(next_scores, row_max, row_sum, previous_scale, query_start, key_start, warpgroup_thread, scale_log2);
        warpgroup_wait<0>();
        scale_output(output_accumulator, previous_scale);
        write_probabilities(shared, next_scores, warpgroup_thread);
        value_stage = stage;
        advance_stage(stage, phase);
    }

    issue_pv(output_accumulator, shared, value_stage);
    warpgroup_wait<0>();
    store_output(output, logsumexp, output_accumulator, row_max, row_sum, batch, query_head, query_start, sequence_length, query_heads,
                 warpgroup_thread, scale);
#endif
}

}  // namespace

void flash_attention_forward_sm90_cuda(bf16* output, float* logsumexp, const bf16* query, const bf16* key, const bf16* value, int batch_size,
                                       int sequence_length, int query_heads, int key_value_heads, int head_size, float scale, cudaStream_t stream) {
    if (head_size != D) throw std::runtime_error("SM90 flash attention requires head_size = 128");
    if (batch_size <= 0 || sequence_length <= 0 || sequence_length % BM != 0) {
        throw std::runtime_error("SM90 flash attention requires positive B and T divisible by 64");
    }
    if (query_heads <= 0 || key_value_heads <= 0 || query_heads % key_value_heads != 0) {
        throw std::runtime_error("SM90 flash attention requires query_heads divisible by key_value_heads");
    }

    const CUtensorMap Q_map = make_qkv_map<true>(query, batch_size, sequence_length, query_heads);
    const CUtensorMap K_map = make_qkv_map<true>(key, batch_size, sequence_length, key_value_heads);
    const CUtensorMap V_map = make_qkv_map<false>(value, batch_size, sequence_length, key_value_heads);
    CUDA_CHECK(cudaFuncSetAttribute(flash_attention_forward_sm90_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    const dim3 grid(sequence_length / BM, query_heads, batch_size);
    flash_attention_forward_sm90_kernel<<<grid, NUM_THREADS, SMEM_BYTES, stream>>>(
        output, logsumexp, Q_map, K_map, V_map, sequence_length, query_heads, key_value_heads, scale);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpu_kernels
