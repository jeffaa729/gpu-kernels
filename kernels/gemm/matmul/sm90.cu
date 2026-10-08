#include "common.cuh"
#include "cuda_common.h"

#include <cuda.h>
#include <cudaTypedefs.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace gpu_kernels {
namespace {

using bf16 = __nv_bfloat16;

// Kernel 10 adds a shared-memory epilogue and asynchronous TMA output stores
// to the Kernel 9 compute pipeline while retaining the Kernel 7 scheduler.
constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 64;

constexpr int WGMMA_M = 64;
constexpr int WGMMA_N = BN;
constexpr int WGMMA_K = 16; // Number of K elements consumed by one WGMMA instruction

// Warpgroup 0 produces TMA tiles; warpgroups 1 and 2 consume them with WGMMA.
constexpr int PRODUCER_THREADS = 128;
constexpr int NUM_CONSUMERS = 2;
constexpr int CONSUMER_THREADS = NUM_CONSUMERS * 128;
constexpr int NUM_THREADS = PRODUCER_THREADS + CONSUMER_THREADS;
// The wider B tile limits the circular buffer to three 48-KiB stages.
constexpr int STAGES = 3;

constexpr unsigned int SMEM_ALIGNMENT = 1024;

constexpr int PERSISTENT_BLOCKS = 128;
constexpr int GROUP_M = 16;
constexpr int GROUP_N = 8;

static_assert(PERSISTENT_BLOCKS == GROUP_M * GROUP_N);
static_assert(PRODUCER_THREADS == 128 && CONSUMER_THREADS == 256 && NUM_THREADS == 384);
static_assert(BM / NUM_CONSUMERS == WGMMA_M);
static_assert(BK == 64 && BN == 256 && STAGES == 3);
static_assert(BK % WGMMA_K == 0);

// Matmul7 reuses this queue across logical output tiles assigned to one CTA.
// Both operands are K-contiguous, matching fast.cu's A[M,K] and B[N,K].
struct SharedStorage {
    alignas(SMEM_ALIGNMENT) bf16 A[STAGES][BM * BK];
    alignas(SMEM_ALIGNMENT) bf16 B[STAGES][BN * BK];
    alignas(SMEM_ALIGNMENT) bf16 C[BM * BN];
};
constexpr size_t SMEM_BYTES = sizeof(SharedStorage) + SMEM_ALIGNMENT - 1;

template <int TileRows, int TileColumns, bool Swizzle = true>
CUtensorMap make_tensor_map(const bf16* pointer, int rows, int columns) {
    CUtensorMap map;
    void* address = const_cast<bf16*>(pointer);
    static_assert(TileColumns >= 64 && TileColumns % 64 == 0);
    const uint64_t global_shape[5] = {64, static_cast<uint64_t>(rows), static_cast<uint64_t>(columns / 64), 1, 1};
    const uint64_t global_stride[4] = {static_cast<uint64_t>(columns) * sizeof(bf16), 64 * sizeof(bf16), 0, 0};
    const uint32_t box_shape[5] = {64, TileRows, TileColumns / 64, 1, 1};
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

// Raw shared-memory barriers: one producer arrival, two consumer-group arrivals.
// Wait parity identifies the generation of a stage as the queue wraps.
__device__ __forceinline__ uint32_t barrier_address(const uint64_t* bar) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(bar));
}

__device__ __forceinline__ void barrier_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(barrier_address(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void barrier_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(barrier_address(bar)) : "memory");
}

__device__ __forceinline__ void barrier_expect_bytes(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(barrier_address(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void barrier_wait(uint64_t* bar, int phase) {
    asm volatile(
        "{ .reg .pred done;\n"
        "wait_loop:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1;\n"
        "@!done bra wait_loop;\n}"
        :: "r"(barrier_address(bar)), "r"(phase) : "memory");
}

// The descriptor views K as 64-element chunks; no physical repacking is needed.
__device__ __forceinline__ void tma_load(bf16* dst, const CUtensorMap* map, uint64_t* bar, int k, int row) {
    const uint32_t destination = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {0, %3, %4, 0, 0}], [%2];"
        :: "r"(destination), "l"(map), "r"(barrier_address(bar)), "r"(row), "r"(k / 64) : "memory");
}

// TMA reads the complete C tile from shared memory and writes it asynchronously.
// C uses an unswizzled [64, BN, BM / 64] tensor-map layout: one 64-row band
// from each consumer warpgroup is contiguous in shared memory.
__device__ __forceinline__ void tma_store(const CUtensorMap* map, bf16* src, int row, int column) {
    const uint32_t source = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group "
        "[%0, {0, %2, %3, 0, 0}], [%1];"
        :: "l"(map), "r"(source), "r"(row), "r"(column / 64) : "memory");
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

// Optimization: use the 128-byte-swizzled shared-memory layout produced by TMA.
template <int LeadingBytes>
__device__ __forceinline__ uint64_t make_smem_descriptor(bf16* pointer) {
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
    uint64_t descriptor = encode_descriptor(address);
    descriptor |= encode_descriptor(LeadingBytes) << 16U;
    descriptor |= encode_descriptor(1024) << 32U;
    descriptor |= 1ULL << 62U;
    return descriptor;
}

__device__ __forceinline__ void warpgroup_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// increase or decrease register limit
template <uint32_t Count>
__device__ __forceinline__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(Count));
}

template <uint32_t Count>
__device__ __forceinline__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(Count));
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma_m64n256k16(float (&accumulator)[16][8], bf16* A, bf16* B) {
    const uint64_t A_descriptor = make_smem_descriptor<16>(A);
    const uint64_t B_descriptor = make_smem_descriptor<16>(B);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
        " %104, %105, %106, %107, %108, %109, %110, %111,  "
        " %112, %113, %114, %115, %116, %117, %118, %119,  "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130,    %131,  %132,  %133,  %134;\n"
        "}\n"
        :   "+f"(accumulator[0][0]), "+f"(accumulator[0][1]), "+f"(accumulator[0][2]), "+f"(accumulator[0][3]), "+f"(accumulator[0][4]), "+f"(accumulator[0][5]), "+f"(accumulator[0][6]), "+f"(accumulator[0][7]),
            "+f"(accumulator[1][0]), "+f"(accumulator[1][1]), "+f"(accumulator[1][2]), "+f"(accumulator[1][3]), "+f"(accumulator[1][4]), "+f"(accumulator[1][5]), "+f"(accumulator[1][6]), "+f"(accumulator[1][7]),
            "+f"(accumulator[2][0]), "+f"(accumulator[2][1]), "+f"(accumulator[2][2]), "+f"(accumulator[2][3]), "+f"(accumulator[2][4]), "+f"(accumulator[2][5]), "+f"(accumulator[2][6]), "+f"(accumulator[2][7]),
            "+f"(accumulator[3][0]), "+f"(accumulator[3][1]), "+f"(accumulator[3][2]), "+f"(accumulator[3][3]), "+f"(accumulator[3][4]), "+f"(accumulator[3][5]), "+f"(accumulator[3][6]), "+f"(accumulator[3][7]),
            "+f"(accumulator[4][0]), "+f"(accumulator[4][1]), "+f"(accumulator[4][2]), "+f"(accumulator[4][3]), "+f"(accumulator[4][4]), "+f"(accumulator[4][5]), "+f"(accumulator[4][6]), "+f"(accumulator[4][7]),
            "+f"(accumulator[5][0]), "+f"(accumulator[5][1]), "+f"(accumulator[5][2]), "+f"(accumulator[5][3]), "+f"(accumulator[5][4]), "+f"(accumulator[5][5]), "+f"(accumulator[5][6]), "+f"(accumulator[5][7]),
            "+f"(accumulator[6][0]), "+f"(accumulator[6][1]), "+f"(accumulator[6][2]), "+f"(accumulator[6][3]), "+f"(accumulator[6][4]), "+f"(accumulator[6][5]), "+f"(accumulator[6][6]), "+f"(accumulator[6][7]),
            "+f"(accumulator[7][0]), "+f"(accumulator[7][1]), "+f"(accumulator[7][2]), "+f"(accumulator[7][3]), "+f"(accumulator[7][4]), "+f"(accumulator[7][5]), "+f"(accumulator[7][6]), "+f"(accumulator[7][7]),
            "+f"(accumulator[8][0]), "+f"(accumulator[8][1]), "+f"(accumulator[8][2]), "+f"(accumulator[8][3]), "+f"(accumulator[8][4]), "+f"(accumulator[8][5]), "+f"(accumulator[8][6]), "+f"(accumulator[8][7]),
            "+f"(accumulator[9][0]), "+f"(accumulator[9][1]), "+f"(accumulator[9][2]), "+f"(accumulator[9][3]), "+f"(accumulator[9][4]), "+f"(accumulator[9][5]), "+f"(accumulator[9][6]), "+f"(accumulator[9][7]),
            "+f"(accumulator[10][0]), "+f"(accumulator[10][1]), "+f"(accumulator[10][2]), "+f"(accumulator[10][3]), "+f"(accumulator[10][4]), "+f"(accumulator[10][5]), "+f"(accumulator[10][6]), "+f"(accumulator[10][7]),
            "+f"(accumulator[11][0]), "+f"(accumulator[11][1]), "+f"(accumulator[11][2]), "+f"(accumulator[11][3]), "+f"(accumulator[11][4]), "+f"(accumulator[11][5]), "+f"(accumulator[11][6]), "+f"(accumulator[11][7]),
            "+f"(accumulator[12][0]), "+f"(accumulator[12][1]), "+f"(accumulator[12][2]), "+f"(accumulator[12][3]), "+f"(accumulator[12][4]), "+f"(accumulator[12][5]), "+f"(accumulator[12][6]), "+f"(accumulator[12][7]),
            "+f"(accumulator[13][0]), "+f"(accumulator[13][1]), "+f"(accumulator[13][2]), "+f"(accumulator[13][3]), "+f"(accumulator[13][4]), "+f"(accumulator[13][5]), "+f"(accumulator[13][6]), "+f"(accumulator[13][7]),
            "+f"(accumulator[14][0]), "+f"(accumulator[14][1]), "+f"(accumulator[14][2]), "+f"(accumulator[14][3]), "+f"(accumulator[14][4]), "+f"(accumulator[14][5]), "+f"(accumulator[14][6]), "+f"(accumulator[14][7]),
            "+f"(accumulator[15][0]), "+f"(accumulator[15][1]), "+f"(accumulator[15][2]), "+f"(accumulator[15][3]), "+f"(accumulator[15][4]), "+f"(accumulator[15][5]), "+f"(accumulator[15][6]), "+f"(accumulator[15][7])
        : "l"(A_descriptor), "l"(B_descriptor), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

// custom scheduler that hold a iteration counter, each block has to request multiple tiles
struct TileScheduler {
    int iteration;
    int tiles_m;
    int tiles_n;

    __device__ TileScheduler(int M, int N) : iteration(0), tiles_m(M / BM), tiles_n(N / BN) {}

    __device__ int next() {
        const int linear = iteration * PERSISTENT_BLOCKS + blockIdx.x;
        ++iteration;

        if (linear >= tiles_m * tiles_n) {
            return -1;
        }

        const int group_size = GROUP_M * GROUP_N;
        const int group = linear / group_size;
        const int position = linear % group_size;

        const int groups_n = tiles_n / GROUP_N;
        const int group_m = group / groups_n;
        const int group_n = group % groups_n;

        const int local_m = position / GROUP_N;
        const int local_n = position % GROUP_N;

        const int tile_m = group_m * GROUP_M + local_m;
        const int tile_n = group_n * GROUP_N + local_n;
        return tile_m * tiles_n + tile_n;
    }
};

#endif

__global__ __launch_bounds__(NUM_THREADS) void gemm_bf16_kernel(const __grid_constant__ CUtensorMap C_map, const __grid_constant__ CUtensorMap A_map, const __grid_constant__ CUtensorMap B_map, int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(16) unsigned char storage[];
    // Align the actual shared address: static barriers can shift the dynamic base.
    // The launch reserves padding so rounding up keeps every TMA stage in bounds.
    const unsigned int storage_address = static_cast<unsigned int>(__cvta_generic_to_shared(storage));
    const unsigned int aligned_address = (storage_address + SMEM_ALIGNMENT - 1) & ~(SMEM_ALIGNMENT - 1);
    auto& shared = *reinterpret_cast<SharedStorage*>(__cvta_shared_to_generic(aligned_address));

    // Full waits for one producer and all TMA bytes; empty waits for two groups.
    __shared__ __align__(8) uint64_t full[STAGES];
    __shared__ __align__(8) uint64_t empty[STAGES];
    if (threadIdx.x == 0) {
        for (int stage = 0; stage < STAGES; ++stage) {
            barrier_init(&full[stage], 1);
            barrier_init(&empty[stage], NUM_CONSUMERS);
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();

    // Every warpgroup follows the same logical-tile sequence independently.
    TileScheduler scheduler(M, N);
    const int tiles_n = N / BN;
    const int k_tiles = K / BK;
    const int warpgroup_id = threadIdx.x / 128;
    const int warpgroup_thread = threadIdx.x % 128;

    // Producer warpgroup: release registers; lane zero fills the three-stage ring.
    if (warpgroup_id == 0) {
        warpgroup_reg_dealloc<24>();
        if (warpgroup_thread == 0) {
            int stage = 0, phase = 0;
            for (int tile_id = scheduler.next(); tile_id >= 0; tile_id = scheduler.next()) {
                const int tile_m = tile_id / tiles_n;
                const int tile_n = tile_id % tiles_n;

                for (int tile_k = 0; tile_k < k_tiles; ++tile_k) {
                    barrier_wait(&empty[stage], phase);
                    barrier_expect_bytes(&full[stage], sizeof(shared.A[stage]) + sizeof(shared.B[stage]));
                    tma_load(shared.A[stage], &A_map, &full[stage], tile_k * BK, tile_m * BM);
                    tma_load(shared.B[stage], &B_map, &full[stage], tile_k * BK, tile_n * BN);
                    advance_stage(stage, phase);
                }
            }
        }
        return;
    }

    // Two consumers receive the register budget and own separate 64x256 row bands.
    warpgroup_reg_alloc<240>();
    const int consumer_id = warpgroup_id - 1;
    for (int stage = 0; stage < STAGES; ++stage) {
        if (warpgroup_thread == 0) barrier_arrive(&empty[stage]);
    }

    float accumulator[WGMMA_N / 16][8];
    int stage = 0, phase = 0;
    static_assert(sizeof(accumulator) * CONSUMER_THREADS == BM * BN * sizeof(float));
    const int lane = warpgroup_thread & 31;
    const int warp = warpgroup_thread >> 5;
    const int row = warp * 16 + lane / 4;

    for (int tile_id = scheduler.next(); tile_id >= 0; tile_id = scheduler.next()) {
        const int tile_m = tile_id / tiles_n;
        const int tile_n = tile_id % tiles_n;

        // First WGMMA overwrites C (ScaleD=0), avoiding explicit register clearing.
        // Peel the first K tile so all remaining iterations accumulate unconditionally.
        {
            barrier_wait(&full[stage], phase);
            bf16* A_tile = shared.A[stage] + consumer_id * WGMMA_M * BK;
            warpgroup_fence();
            wgmma_m64n256k16<0, 1, 1, 0, 0>(accumulator, A_tile, shared.B[stage]);
#pragma unroll
            for (int k_it = 1; k_it < BK / WGMMA_K; ++k_it) {
                wgmma_m64n256k16<1, 1, 1, 0, 0>(accumulator, A_tile + k_it * WGMMA_K, shared.B[stage] + k_it * WGMMA_K);
            }
            warpgroup_commit();
            warpgroup_wait();
            if (warpgroup_thread == 0) barrier_arrive(&empty[stage]);
            advance_stage(stage, phase);
        }

        for (int tile_k = 1; tile_k < k_tiles; ++tile_k) {
            barrier_wait(&full[stage], phase);

            bf16* A_tile = shared.A[stage] + consumer_id * WGMMA_M * BK;
            warpgroup_fence();
#pragma unroll
            for (int k_it = 0; k_it < BK / WGMMA_K; ++k_it) {
                wgmma_m64n256k16<1, 1, 1, 0, 0>(
                    accumulator, A_tile + k_it * WGMMA_K, shared.B[stage] + k_it * WGMMA_K);
            }
            warpgroup_commit();
            warpgroup_wait();

            if (warpgroup_thread == 0) barrier_arrive(&empty[stage]);
            advance_stage(stage, phase);
        }

        // Wait before reusing C shared memory from the previous asynchronous store.
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        bf16* consumer_C = shared.C + consumer_id * WGMMA_M * BN;
#pragma unroll
        for (int group = 0; group < WGMMA_N / 16; ++group) {
            const int column = group * 16 + 2 * (lane & 3);
// Kernel 10: stage the WGMMA register layout in shared memory for one TMA store.
#define STORE(Row, Column, Value) consumer_C[(Column) * WGMMA_M + (Row)] = __float2bfloat16(Value)
            STORE(row, column, accumulator[group][0]);
            STORE(row + 8, column, accumulator[group][2]);
            STORE(row, column + 1, accumulator[group][1]);
            STORE(row + 8, column + 1, accumulator[group][3]);
            STORE(row, column + 8, accumulator[group][4]);
            STORE(row + 8, column + 8, accumulator[group][6]);
            STORE(row, column + 9, accumulator[group][5]);
            STORE(row + 8, column + 9, accumulator[group][7]);
#undef STORE
        }
        asm volatile("bar.sync 10, %0;" :: "n"(CONSUMER_THREADS) : "memory");
        if (threadIdx.x == PRODUCER_THREADS) {
            tma_store(&C_map, shared.C, tile_n * BN, tile_m * BM);
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
    }
#endif
}

}  // namespace

void gemm_bf16_sm90_cuda(bf16* C, const bf16* A, const bf16* B, int M, int N, int K, cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0 || M % (BM * GROUP_M) != 0 || N % (BN * GROUP_N) != 0 || K % BK != 0) {
        throw std::invalid_argument(
            "SM90 persistent BF16 GEMM requires M and N divisible by 2048 and K divisible by 64.");
    }
    const CUtensorMap A_map = make_tensor_map<BM, BK>(A, M, K);
    const CUtensorMap B_map = make_tensor_map<BN, BK>(B, N, K);
    const CUtensorMap C_map = make_tensor_map<BN, BM, false>(C, N, M);
    CUDA_CHECK(cudaFuncSetAttribute(gemm_bf16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    gemm_bf16_kernel<<<PERSISTENT_BLOCKS, NUM_THREADS, SMEM_BYTES, stream>>>(C_map, A_map, B_map, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace gpu_kernels
