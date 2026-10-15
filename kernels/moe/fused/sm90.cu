#include "fused_moe.h"
#include "cuda_common.h"
#include "common.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/bfloat16.h>
#include <cuda/atomic>
#include <type_traits>

namespace gpu_kernels {
namespace {
using namespace cute;
using BF16 = cutlass::bfloat16_t;
using Barrier = cutlass::arch::ClusterBarrier;
constexpr int WG_M = 64, MATH_THREADS = 256, THREADS = 384;
constexpr int STORE_STAGES = 2;
constexpr bool L2_N_MAJOR = true;  // Set false for the original M-major ablation.
constexpr bool L1_N_MAJOR = true;  // Reuse weight columns across activation tiles.

// Keep Python's block_m selection identical; metadata and both GEMMs share
// this small-batch tile policy.
__host__ __device__ constexpr int moe_block_m(int tokens) {
    return tokens <= 512 ? 64 : 128;
}

// Small batches split columns between warpgroups; large batches split rows.
template <int BM>
struct Tile {
    static_assert(BM == 64 || BM == 128, "Supported two-warpgroup tiles");
    static constexpr bool SPLIT_N = BM == 64;
    // Wider math tiles use a three-stage BK64 queue.
    static constexpr bool WIDE = !SPLIT_N;
    static constexpr int WG_N = WIDE ? 256 : 128;
    static constexpr int BN = SPLIT_N ? 256 : WG_N;
    static constexpr int BK = WIDE ? 64 : 128;
    static constexpr int STAGES = WIDE ? 3 : 2;
    using MmaOp = std::conditional_t<WIDE,
        SM90_64x256x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>,
        SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>>;
    using Mma = decltype(make_tiled_mma(MmaOp{}));
    using LayoutA = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF16>{}, Shape<Int<BM>, Int<BK>, Int<STAGES>>{}));
    using LayoutB = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF16>{}, Shape<Int<BN>, Int<BK>, Int<STAGES>>{}));
    using LayoutC = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF16>{}, Shape<Int<BM>, Int<BN / 2>, Int<STORE_STAGES>>{}));
    static_assert(BN % 16 == 0, "Complete 8-gate/8-up interleave groups");
    static_assert(cosize_v<LayoutC> == BM * BN, "Two L1 output slots reuse the L2 scatter buffer");
    struct Shared {
        alignas(1024) BF16 A[cosize_v<LayoutA>];
        alignas(1024) BF16 B[cosize_v<LayoutB>];
        alignas(1024) BF16 gate_up[BM * BN];  // L1: two SW128 slots; L2: row-major.
        alignas(128) BF16 send[2][2048];
        uint64_t full[STAGES], empty[STAGES], pull[2];
    };
};

// Reset separately: scatter CTAs must never race with counter initialization.
__global__ void metadata_reset_kernel(MoePeers peers, MoeShape s, int* counts) {
    const int BM = moe_block_m(s.tokens);
    const int expert = blockIdx.x;
    if (threadIdx.x == 0) counts[expert] = 0;
    for (int i = threadIdx.x; i < s.capacity / BM; i += blockDim.x)
        peers.ready[expert * (s.capacity / BM) + i] = 0;
}

// Partition routes, build shared histograms, reserve global ranges and scatter.
// Each receiver reads peer IDs and stores capacity-strided routes and weights.
constexpr int METADATA_THREADS = 256, METADATA_ROUTES_PER_THREAD = 4;
__global__ void metadata_kernel(MoePeers peers, MoeShape s, int* routes, int* counts) {
    const int local_experts = s.experts / s.world;
    extern __shared__ int metadata_storage[];
    int* histogram = metadata_storage;
    int* bases = histogram + local_experts;
    for (int e = threadIdx.x; e < local_experts; e += blockDim.x)
        histogram[e] = 0;
    __syncthreads();

    int experts[METADATA_ROUTES_PER_THREAD], slots[METADATA_ROUTES_PER_THREAD];
    const int routes_per_rank = s.tokens * s.topk;
    const int64_t first_route = int64_t(blockIdx.x) * blockDim.x * METADATA_ROUTES_PER_THREAD + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < METADATA_ROUTES_PER_THREAD; ++i) {
        experts[i] = -1;
        const int64_t route = first_route + i * blockDim.x;
        if (route < int64_t(s.world) * routes_per_rank) {
            const int rank = route / routes_per_rank, local = route % routes_per_rank;
            const int expert = int(peers.ids[rank][local]) - s.rank * local_experts;
            if (expert >= 0 && expert < local_experts) {
                experts[i] = expert;
                slots[i] = atomicAdd(histogram + expert, 1);
            }
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < local_experts; e += blockDim.x) {
        if (histogram[e] != 0)
            bases[e] = atomicAdd(counts + e, histogram[e]);
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < METADATA_ROUTES_PER_THREAD; ++i) {
        const int expert = experts[i];
        if (expert >= 0) {
            const int route = int(first_route + i * blockDim.x);
            const int rank = route / routes_per_rank, local = route % routes_per_rank;
            const int dst = expert * s.capacity + bases[expert] + slots[i];
            routes[dst] = route;
            peers.route_weights[dst] = peers.weights[rank][local];
        }
    }
}

// Same-stream launch after all expert counts are complete. No host count read:
// offsets are rebuilt on every invocation, including CUDA graph replay.
// Each expert receives a BM-aligned slice of the full token pool.
__global__ void offsets_kernel(MoePeers peers, MoeShape s, const int* counts) {
    const int BM = moe_block_m(s.tokens);
    int64_t offset = 0;
    if (s.pool_capacity <= 0) { asm volatile("trap;"); return; }
    peers.offsets[0] = 0;
    for (int e = 0; e < s.experts / s.world; ++e) {
        const int64_t count = counts[e];
        const int64_t padded = (count + BM - 1) / BM * BM;
        if (count < 0 || count > s.capacity || padded > s.pool_capacity - offset) {
            asm volatile("trap;");
            return;
        }
        offset += padded;
        peers.offsets[e + 1] = offset;
    }
}

// Schedule only populated token tiles. A capacity-strided schedule can send
// every expert's first tile to the same dispatcher and serialize communication.
template <int BM>
__device__ __forceinline__ int tile_expert(int& tile, const int* counts) {
    int expert = 0;
    while (tile >= (counts[expert] + BM - 1) / BM) {
        tile -= (counts[expert] + BM - 1) / BM;
        ++expert;
    }
    return expert;
}

// Iterating n outside m reuses a weight column across an expert's activation tiles.
// Find the expert before transposing its local task index.
template <bool First, int BM>
__device__ __forceinline__ int task_expert(int task, int ntiles, int width,
        const int* counts, int& tile, int& n) {
    tile = task / ntiles;
    const int expert = tile_expert<BM>(tile, counts);
    n = task % ntiles;
    if constexpr (First ? L1_N_MAJOR : L2_N_MAJOR) {
        const int local_task = tile * ntiles + n;
        const int num_m_tiles = (counts[expert] + BM - 1) / BM;
        tile = local_task % num_m_tiles;
        n = local_task / num_m_tiles;
    }
    n *= width;
    return expert;
}

template <bool First, int BM>
__global__ __launch_bounds__(THREADS, 1) void phase_kernel(MoePeers peers, MoeShape s, const BF16* W,
        BF16* activation, const int* routes, const int* counts,
        const __grid_constant__ CUtensorMap map_A, const __grid_constant__ CUtensorMap map_B,
        const __grid_constant__ CUtensorMap map_C) {
    constexpr int BN = Tile<BM>::BN;
    constexpr int BK = Tile<BM>::BK, STAGES = Tile<BM>::STAGES, WG_N = Tile<BM>::WG_N;
    using Mma = typename Tile<BM>::Mma;
    using Shared = typename Tile<BM>::Shared;
    using LayoutA = typename Tile<BM>::LayoutA;
    using LayoutB = typename Tile<BM>::LayoutB;
    using LayoutC = typename Tile<BM>::LayoutC;
    extern __shared__ __align__(1024) unsigned char storage[];
    auto& shared = *reinterpret_cast<Shared*>(storage);
    const int tid = threadIdx.x, lane = tid % 128;
    const int K = First ? s.hidden : s.intermediate;
    const int N = First ? s.intermediate : s.hidden;
    const int width = First ? BN / 2 : BN;
    const int ntiles = N / width, mtiles = s.capacity / BM;
    __shared__ int token_tiles;
    if (tid == 0) {
        token_tiles = 0;
        for (int e = 0; e < s.experts / s.world; ++e) token_tiles += (counts[e] + BM - 1) / BM;
    }
    if (tid < STAGES) {
        Barrier::init(&shared.full[tid], 1);
        // Release each operand stage once per completed warpgroup.
        Barrier::init(&shared.empty[tid], MATH_THREADS / 128);
    }
    if (tid < 2) Barrier::init(&shared.pull[tid], 1);
    cutlass::arch::fence_barrier_init();
    __syncthreads();
    if (tid < 128) cutlass::arch::warpgroup_reg_dealloc<48>();
    const int tasks = token_tiles * ntiles;
    if constexpr (First) {
        if (tid < 64) {
            // Two dispatch warps share the CTA with
            // TMA loaders and math, rather than reserving whole SMs for copies.
            const int warp = tid / 32;
            int pull_phase = 0;
            for (int task = blockIdx.x; task < token_tiles; task += gridDim.x) {
                int tile = task;
                const int expert = tile_expert<BM>(tile, counts), m = tile * BM;
                const int count = counts[expert];
                if (m >= count) continue;
                // Warp leaders issue up to 4 KiB TMA pulls and local stores.
                // This avoids a separate remote round trip for every uint4 load.
                if (tid % 32 == 0) {
                    for (int row = warp; row < BM && m + row < count; row += 2) {
                        const int route = routes[expert * s.capacity + m + row];
                        const int rank = route / (s.tokens * s.topk);
                        const int token = route % (s.tokens * s.topk) / s.topk;
                        for (int col = 0; col < s.hidden; col += 2048) {
                            const int bytes = min(2048, s.hidden - col) * int(sizeof(BF16));
                            const BF16* src = static_cast<const BF16*>(peers.x[rank]) + token * s.hidden + col;
                            BF16* dst = static_cast<BF16*>(peers.pool) + (peers.offsets[expert] + m + row) * s.hidden + col;
                            BF16* temp = shared.send[warp];
                            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&shared.pull[warp], bytes);
                            SM90_BULK_COPY_G2S::copy(src, &shared.pull[warp], temp, bytes);
                            Barrier::wait(&shared.pull[warp], pull_phase);
                            pull_phase ^= 1;
                            SM90_BULK_COPY_S2G::copy(temp, dst, bytes);
                            tma_store_arrive();
                            // CuTe tma_store_wait waits only for source reads. Pool
                            // publication needs the stronger global-store completion.
                            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                        }
                    }
                }
                cutlass::arch::NamedBarrier::sync(64, 2);
                if (tid == 0) cuda::atomic_ref<int, cuda::thread_scope_device>(peers.ready[expert * mtiles + tile]).store(1, cuda::memory_order_release);
                cutlass::arch::NamedBarrier::sync(64, 2);
            }
            return;
        }
    }
    const int worker = blockIdx.x, workers = gridDim.x;
    auto A = make_tensor(make_smem_ptr(shared.A), LayoutA{});
    auto B = make_tensor(make_smem_ptr(shared.B), LayoutB{});
    // Keep barrier phases monotonic across tiles, allowing producer look-ahead.
    int sequence = 0;
    if (tid < 128) {
        // Single elected producer publishes all A/B loads into one transaction
        // barrier, after acquiring the dispatched activation tile for L1.
        if (tid != 64) return;
        for (int task = worker; task < tasks; task += workers) {
            int tile, n;
            const int expert = task_expert<First, BM>(task, ntiles, width, counts, tile, n);
            const int m = tile * BM;
            const int count = counts[expert];
            if (m >= count) continue;
            if constexpr (First) {
                auto ready = cuda::atomic_ref<int, cuda::thread_scope_device>(peers.ready[expert * mtiles + m / BM]);
                while (!ready.load(cuda::memory_order_acquire)) __nanosleep(64);
            }
            for (int k = 0; k < K; k += BK, ++sequence) {
                const int stage = sequence % STAGES, phase = sequence / STAGES % 2;
                if (sequence >= STAGES) Barrier::wait(&shared.empty[stage], phase ^ 1);
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &shared.full[stage], (BM + BN) * BK * sizeof(BF16));
                constexpr auto cache = uint64_t(TMA::CacheHintSm90::EVICT_NORMAL);
                CUTE_UNROLL
                for (int column = 0; column < BK; column += 64) {
                    SM90_TMA_LOAD_2D::copy(&map_A, &shared.full[stage], cache,
                        &A(0, column, stage), k + column, int(peers.offsets[expert]) + m);
                    if constexpr (First) {
                        // Python prepacking makes both halves one contiguous BN tile.
                        SM90_TMA_LOAD_2D::copy(&map_B, &shared.full[stage], cache,
                            &B(0, column, stage), k + column, expert * 2 * N + n * 2);
                    } else {
                        SM90_TMA_LOAD_2D::copy(&map_B, &shared.full[stage], cache,
                            &B(0, column, stage), k + column, expert * N + n);
                    }
                }
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<224>();
        const int math_tid = tid - 128, wg = math_tid / 128;
        const int row_offset = Tile<BM>::SPLIT_N ? 0 : wg * WG_M;
        const int col_offset = Tile<BM>::SPLIT_N ? wg * WG_N : 0;
        Mma mma;
        auto thread_mma = mma.get_slice(lane);
        auto A_wg = local_tile(A, Shape<Int<WG_M>, Int<BK>, Int<STAGES>>{}, make_coord(row_offset / WG_M, 0, 0));
        auto B_wg = local_tile(B, Shape<Int<WG_N>, Int<BK>, Int<STAGES>>{}, make_coord(col_offset / WG_N, 0, 0));
        auto fragmentA = thread_mma.make_fragment_A(thread_mma.partition_A(A_wg));
        auto fragmentB = thread_mma.make_fragment_B(thread_mma.partition_B(B_wg));
        auto coordinates = thread_mma.partition_C(make_identity_tensor(Shape<Int<WG_M>, Int<WG_N>>{}));
        auto accum = partition_fragment_C(mma, Shape<Int<WG_M>, Int<WG_N>>{});
        // m64n128/m64n256 C fragment: ((2,2,WG_N/8),1,1), four values per 8-column
        // band. For gate indices j%8<4, j+4 has the same row and column+8.
        // Gate/up interleaving keeps each pair within one thread's fragment.
        static_assert(decltype(size<0, 0>(accum))::value == 2 &&
                      decltype(size<0, 1>(accum))::value == 2 &&
                      decltype(size<0, 2>(accum))::value == WG_N / 8 &&
                      decltype(size<1>(accum))::value == 1 &&
                      decltype(size<2>(accum))::value == 1,
                      "Reverify gate/up pairing if the WGMMA fragment changes");
        auto C = make_tensor(make_smem_ptr(shared.gate_up), LayoutC{});
        int output_sequence = 0;  // Per output tile, independent of the K pipeline.
        for (int task = worker; task < tasks; task += workers) {
            int tile, n;
            const int expert = task_expert<First, BM>(task, ntiles, width, counts, tile, n);
            const int m = tile * BM;
            const int count = counts[expert];
            if (m >= count) continue;
            clear(accum);
            for (int k = 0; k < K; k += BK, ++sequence) {
                const int stage = sequence % STAGES;
                Barrier::wait(&shared.full[stage], sequence / STAGES % 2);
                warpgroup_fence_operand(accum);
                warpgroup_arrive();
                gemm(mma, fragmentA(_, _, _, stage), fragmentB(_, _, _, stage), accum);
                warpgroup_commit_batch();
                warpgroup_wait<0>();
                warpgroup_fence_operand(accum);
                if (lane == 0) Barrier::arrive(&shared.empty[stage]);
            }
            if constexpr (First) {
                // Apply rounded BF16 SwiGLU in registers, then use one issuer
                // to store the full swizzled tile after waiting for slot reuse.
                const int slot = output_sequence % STORE_STAGES;
                // Only tid128 owns these bulk groups. A full wait (no .read)
                // retires the older slot; the rendezvous protects all writers.
                if (math_tid == 0) asm volatile("cp.async.bulk.wait_group 1;" ::: "memory");
                cutlass::arch::NamedBarrier::sync(MATH_THREADS, 1);
                CUTE_UNROLL
                for (int base = 0; base < size(accum); base += 8) {
                    CUTE_UNROLL
                    for (int pair = 0; pair < 4; ++pair) {
                        const int j = base + pair;
                        auto coord = coordinates(j);
                        const int row = row_offset + get<0>(coord);
                        const int gate_col = col_offset + get<1>(coord);
                        const int col = gate_col / 16 * 8 + gate_col % 8;
                        BF16 value = BF16(0.f);
                        if (m + row < count) {
                            // Preserve GEMM1 BF16 rounding, clamp, expf/division,
                            // routing-weight multiplication, and final BF16 rounding.
                            float gate = fminf(float(BF16(accum(j))), s.clamp);
                            float up = fminf(fmaxf(float(BF16(accum(j + 4))), -s.clamp), s.clamp);
                            value = BF16(gate / (1.f + expf(-gate)) * up * peers.route_weights[expert * s.capacity + m + row]);
                        }
                        // Full BM store includes padding, without reading invalid weights.
                        C(row, col, slot) = value;
                    }
                }
                // Each writer publishes its generic shared stores to the async
                // proxy before the rendezvous lets the leader issue TMA.
                tma_store_fence();
                cutlass::arch::NamedBarrier::sync(MATH_THREADS, 1);
                if (math_tid == 0) {
                    // The output map's SW128 atom is 64 BF16 columns. N-split
                    // uses two atoms, committed together as ONE output-tile group.
                    CUTE_UNROLL
                    for (int col = 0; col < BN / 2; col += 64)
                        SM90_TMA_STORE_2D::copy(&map_C, &C(0, col, slot),
                            n + col, int(peers.offsets[expert]) + m);
                    tma_store_arrive();
                }
                cutlass::arch::NamedBarrier::sync(MATH_THREADS, 1);
                ++output_sequence;
            } else {
                // Each top-k route has a unique destination slot: no remote atomics.
                CUTE_UNROLL
                for (int j = 0; j < size(accum); ++j) {
                    auto coord = coordinates(j);
                    shared.gate_up[(row_offset + get<0>(coord)) * BN + col_offset + get<1>(coord)] = BF16(accum(j));
                }
                cutlass::arch::NamedBarrier::sync(MATH_THREADS, 1);
                for (int v = math_tid; v < BM * BN / 8; v += MATH_THREADS) {
                    const int row = v / (BN / 8), col = v % (BN / 8) * 8;
                    if (m + row < count) {
                        const int route = routes[expert * s.capacity + m + row];
                        const int rank = route / (s.tokens * s.topk), local = route % (s.tokens * s.topk);
                        BF16* dst = static_cast<BF16*>(peers.output[rank]) + local * N + n + col;
                        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(shared.gate_up + row * BN + col);
                    }
                }
                cutlass::arch::NamedBarrier::sync(MATH_THREADS, 1);
            }
        }
        if constexpr (First) {
            // Same-stream L2 may consume activation only after global completion,
            // not merely CuTe tma_store_wait<0>()'s shared-source read completion.
            if (math_tid == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
            cutlass::arch::NamedBarrier::sync(MATH_THREADS, 1);
        }
    }
}

__global__ void reduce_kernel(BF16* output, const BF16* slots, MoeShape s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.tokens * s.hidden) return;
    const int token = i / s.hidden, col = i % s.hidden;
    float sum = 0.f;
    for (int k = 0; k < s.topk; ++k) sum += float(slots[(token * s.topk + k) * s.hidden + col]);
    output[i] = BF16(sum);
}

template <bool First, int BM>
void launch(MoePeers peers, MoeShape shape, const void* weights, void* activation,
            const int* routes, const int* counts, int sms, cudaStream_t stream) {
    constexpr int BN = Tile<BM>::BN;
    using Shared = typename Tile<BM>::Shared;
    CUDA_CHECK(cudaFuncSetAttribute(phase_kernel<First, BM>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared)));
    auto W = static_cast<const BF16*>(weights);
    auto A = static_cast<BF16*>(activation);
    const int K = First ? shape.hidden : shape.intermediate;
    const int N = First ? 2 * shape.intermediate : shape.hidden;
    auto map_A = moe_tensor_map(First ? peers.pool : activation, int(shape.pool_capacity), K, BM);
    auto map_B = moe_tensor_map(weights, shape.experts / shape.world * N, K, BN);
    auto map_C = map_A;  // Unused by L2; avoid another descriptor encode there.
    if constexpr (First)
        map_C = moe_tensor_map(activation, int(shape.pool_capacity), shape.intermediate, BM);
    if constexpr (First) {
        // Cooperative launch guarantees dispatcher residency: waiting compute
        // CTAs cannot occupy the entire GPU and starve their pool producers.
        void* args[] = {&peers, &shape, &W, &A, &routes, &counts, &map_A, &map_B, &map_C};
        CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(phase_kernel<First, BM>),
            dim3(sms), dim3(THREADS), args, sizeof(Shared), stream));
    } else {
        phase_kernel<First, BM><<<sms, THREADS, sizeof(Shared), stream>>>(peers, shape, W, A, routes, counts, map_A, map_B, map_C);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <bool First>
void launch_for_tokens(MoePeers peers, MoeShape shape, const void* weights, void* activation,
                       const int* routes, const int* counts, int sms, cudaStream_t stream) {
    if (moe_block_m(shape.tokens) == 64)
        launch<First, 64>(peers, shape, weights, activation, routes, counts, sms, stream);
    else
        launch<First, 128>(peers, shape, weights, activation, routes, counts, sms, stream);
}
} // namespace

void moe_metadata_cuda(MoePeers p, MoeShape s, int* routes, int* counts, cudaStream_t stream) {
    const int local_experts = s.experts / s.world;
    metadata_reset_kernel<<<local_experts, METADATA_THREADS, 0, stream>>>(p, s, counts);
    CUDA_CHECK(cudaGetLastError());
    const int64_t num_routes = int64_t(s.world) * s.tokens * s.topk;
    constexpr int routes_per_cta = METADATA_THREADS * METADATA_ROUTES_PER_THREAD;
    const int blocks = int((num_routes + routes_per_cta - 1) / routes_per_cta);
    metadata_kernel<<<blocks, METADATA_THREADS, 2 * local_experts * sizeof(int), stream>>>(p, s, routes, counts);
    CUDA_CHECK(cudaGetLastError());
    offsets_kernel<<<1, 1, 0, stream>>>(p, s, counts);
    CUDA_CHECK(cudaGetLastError());
}
void moe_phase1_cuda(MoePeers p, MoeShape s, const void* w, void* a, const int* r, const int* c, int sms, cudaStream_t stream) {
    launch_for_tokens<true>(p, s, w, a, r, c, sms, stream);
}
void moe_phase2_cuda(MoePeers p, MoeShape s, const void* w, const void* a, const int* r, const int* c, int sms, cudaStream_t stream) {
    launch_for_tokens<false>(p, s, w, const_cast<void*>(a), r, c, sms, stream);
}
void moe_reduce_cuda(void* y, const void* slots, MoeShape s, cudaStream_t stream) {
    reduce_kernel<<<(s.tokens * s.hidden + 255) / 256, 256, 0, stream>>>(static_cast<BF16*>(y), static_cast<const BF16*>(slots), s);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace gpu_kernels
