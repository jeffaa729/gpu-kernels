#include "fused_moe.h"
#include "cuda_common.h"
#include "common.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/bfloat16.h>
#include <cuda/atomic>
#include <algorithm>
#include <type_traits>

namespace gpu_kernels {
namespace {
using namespace cute;
using BF16 = cutlass::bfloat16_t;
using Barrier = cutlass::arch::ClusterBarrier;
constexpr int DISPATCH_ELEMENTS = 4096;
constexpr int MAX_PIPELINE_EXPERTS = 256;

// Shape-aware tiling: use 64 token rows for small batches to reduce padded work,
// and 128 otherwise. Metadata, pool allocation, and GEMMs share this alignment.
__host__ __device__ constexpr int moe_block_m(MoeShape s) {
    return s.tokens <= 512 ? 64 : 128;
}

// Reset separately: scatter CTAs must never race with counter initialization.
__global__ void metadata_reset_kernel(MoePeers peers, MoeShape s, int* counts) {
    const int BM = moe_block_m(s);
    const int expert = blockIdx.x;
    if (threadIdx.x == 0) counts[expert] = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < s.world * s.tokens; i += gridDim.x * blockDim.x)
        peers.carrier_rows[i] = -1;
    for (int i = threadIdx.x; i < s.capacity / BM; i += blockDim.x) {
        peers.ready[expert * (s.capacity / BM) + i] = 0;
    }
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
            // Dispatch deduplication: select one local expert to fetch each input;
            // other local routes reuse its published token-pool row.
            if (local % s.topk == 0) {
                int carrier = local_experts;
                for (int k = 0; k < s.topk; ++k) {
                    const int e = int(peers.ids[rank][local + k]) - s.rank * local_experts;
                    if (e >= 0 && e < local_experts) carrier = min(carrier, e);
                }
                peers.carrier_experts[rank * s.tokens + local / s.topk] = carrier;
            }
            const int expert = int(peers.ids[rank][local]) - s.rank * local_experts;
            if (expert >= 0 && expert < local_experts) {
                experts[i] = expert;
                slots[i] = atomicAdd(histogram + expert, 1);
            }
        }
    }
    __syncthreads();
    // Hierarchical histogram: reserve one global range per populated expert/CTA,
    // rather than performing a global counter update for every route.
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

// GPU-side pool compaction: prefix offsets pack only populated expert tiles,
// avoiding full-capacity payload storage and CPU count readback on graph replay.
__global__ void offsets_kernel(MoePeers peers, MoeShape s, const int* counts) {
    const int BM = moe_block_m(s);
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

__global__ void reduce_kernel(BF16* output, const BF16* slots, MoeShape s) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= s.tokens * s.hidden) return;
    const int token = i / s.hidden, col = i % s.hidden;
    // Vectorized combine: load eight BF16 values per 16-byte access and accumulate
    // in FP32 registers, preserving each element's fixed top-k sum order.
    float sum[8] = {};
    for (int k = 0; k < s.topk; ++k) {
        const uint4 packed = *reinterpret_cast<const uint4*>(slots + (token * s.topk + k) * s.hidden + col);
        const BF16* values = reinterpret_cast<const BF16*>(&packed);
        CUTE_UNROLL
        for (int j = 0; j < 8; ++j) sum[j] += float(values[j]);
    }
    uint4 packed;
    BF16* values = reinterpret_cast<BF16*>(&packed);
    CUTE_UNROLL
    for (int j = 0; j < 8; ++j) values[j] = BF16(sum[j]);
    *reinterpret_cast<uint4*>(output + i) = packed;
}

namespace hopper {
constexpr int TM = 128, TK = 64, PIPE = 5, QUEUE = 2;
constexpr int NTHREADS = 384, MAX_EXPERTS = MAX_PIPELINE_EXPERTS;
// Shape-specialized WGMMA: keep the feature tile fixed while selecting 64 or
// 128 token columns, reducing padded Tensor Core work for small batches.
template <int TN>
using Op = std::conditional_t<TN == 64,
    SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>,
    SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>>;
template <int TN>
using Mma = decltype(make_tiled_mma(Op<TN>{}));
using Pipeline = cutlass::PipelineTmaAsync<PIPE>;
// Shared-memory swizzling: SW128 layouts match WGMMA/TMA operand accesses;
// padded alignment preserves the address requirements of asynchronous copies.
template <int Rows>
using Operand = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF16>{}, Shape<Int<Rows>, Int<TK>, Int<PIPE>>{}));
template <int TN>
using Output = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<BF16>{}, Shape<Int<TN>, Int<TM / 2>, _2>{}));

struct Work { int expert, m, n, count; };
template <int TN>
struct Shared {
    static_assert(TN == 64 || TN == 128);
    // Multi-stage buffering: five operand slots let TMA prefetch ahead of math;
    // two output slots let the math warpgroups stage separate FC1 epilogues.
    alignas(1024) BF16 A[cosize_v<Operand<TM>>], B[cosize_v<Operand<TN>>];
    alignas(1024) BF16 C[cosize_v<Output<TN>>];
    alignas(128) BF16 send[1][DISPATCH_ELEMENTS];
    // Reserved barrier slots retain the benchmarked shared-memory layout.
    uint64_t full[PIPE], empty[PIPE], pull[4];
    uint64_t work_full[QUEUE], work_empty[QUEUE], math_order[2], epi_order[2];
    Work work[QUEUE];
    int sizes[MAX_EXPERTS], starts[MAX_EXPERTS + 1];
    Pipeline::SharedStorage pipeline;
};
static_assert(sizeof(Shared<64>) <= 232448 && sizeof(Shared<128>) <= 232448);

template <int TN>
__device__ __forceinline__ Work consume(Shared<TN>& shared, int index, bool leader) {
    // Bounded descriptor queue: publish two tasks ahead and recycle a slot only
    // after both loaders and both math warpgroups have copied its descriptor.
    const int slot = index % QUEUE;
    Barrier::wait(&shared.work_full[slot], index / QUEUE % 2);
    Work result = shared.work[slot];
    // Every role copies the descriptor before the slot can be reused.
    if (threadIdx.x < 256) cutlass::arch::NamedBarrier::sync(128, 4 + threadIdx.x / 128);
    if (leader) Barrier::arrive(&shared.work_empty[slot]);
    return result;
}

template <int Group, int TN, class Fragment>
__device__ __forceinline__ void gather_group(const Fragment& accum, float (&gates)[4], float (&ups)[4]) {
    // Constant-index register gathers: template indices avoid runtime addressing
    // of the accumulator array, which can otherwise spill into local memory.
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        constexpr int m64_stride = TN / 2;
        const int j = (i / 2) * m64_stride + Group * 4 + i % 2;
        gates[i] = accum(j);
        ups[i] = accum(j + 2);
    }
}

// Compile-time phase specialization: FC1 fuses weighted SwiGLU; FC2 scatters
// expert outputs. Both phases reuse the same scheduler and operand pipeline.
template <bool First, int TN>
__global__ __launch_bounds__(NTHREADS, 1) void kernel(MoePeers peers, MoeShape s,
        const int* routes, const int* counts,
        const __grid_constant__ CUtensorMap map_A, const __grid_constant__ CUtensorMap map_B,
        const __grid_constant__ CUtensorMap map_C) {
    extern __shared__ __align__(1024) unsigned char storage[];
    auto& shared = *reinterpret_cast<Shared<TN>*>(storage);
    // Warp specialization: WG0/1 perform math and epilogues; WG2 contains
    // separate dispatch, weight-loader, activation-loader, and scheduler warps.
    const int tid = threadIdx.x, lane = tid % 128, warp = tid / 32;
    const int experts = s.experts / s.world;
    const int K = First ? s.hidden : s.intermediate;
    const int N = First ? 2 * s.intermediate : s.hidden;
    const int ntiles = N / TM, mtiles = s.capacity / TN;
    // Shared metadata caching: cooperative loads make expert sizes and compact
    // offsets available to every role without rereading global memory per task.
    for (int e = tid; e < experts; e += NTHREADS) {
        shared.sizes[e] = counts[e];
        shared.starts[e] = int(peers.offsets[e] / TN);
    }
    if (tid == 0) shared.starts[experts] = int(peers.offsets[experts] / TN);
    if (tid < PIPE) {
        Barrier::init(&shared.full[tid], 2);
        Barrier::init(&shared.empty[tid], 1);
    }
    if (tid < 4) Barrier::init(&shared.pull[tid], 1);
    if (tid < QUEUE) {
        Barrier::init(&shared.work_full[tid], 1);
        Barrier::init(&shared.work_empty[tid], 4);
        Barrier::init(&shared.math_order[tid], 128);
        Barrier::init(&shared.epi_order[tid], 128);
    }
    Pipeline::Params params;
    // Split-producer TMA accounting: weight and token tiles can differ in size;
    // each leader contributes exactly its own transfer bytes to the full barrier.
    params.transaction_bytes = (warp == 10 ? TN : TM) * TK * sizeof(BF16);
    params.role = tid < 256 ? Pipeline::ThreadCategory::Consumer : Pipeline::ThreadCategory::Producer;
    params.is_leader = tid == 288 || tid == 320;
    params.num_producers = 2;
    params.num_consumers = 128;
    Pipeline pipeline(shared.pipeline, params, Shape<_2, _1, _1>{});
    cutlass::arch::fence_barrier_init();
    __syncthreads();
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
    // Register redistribution: non-math warps release registers so the two
    // consumer warpgroups can keep their accumulators and epilogues in registers.
    if (tid >= 256) cutlass::arch::warpgroup_reg_dealloc<40>();
    const int tokens = shared.starts[experts];
    // Operand swap: A holds weight features and B holds token rows, so WGMMA
    // computes a feature-by-token tile before the epilogue restores token order.
    auto A = make_tensor(make_smem_ptr(shared.A), Operand<TM>{});
    auto B = make_tensor(make_smem_ptr(shared.B), Operand<TN>{});
    auto C = make_tensor(make_smem_ptr(shared.C), Output<TN>{});

    if (warp == 8) {
        if constexpr (First) {
            int expert = 0, first_row = 0, end_row = shared.sizes[0], phase = 0;
            // Token-strided dispatch: distribute actual routed rows across all
            // persistent CTAs instead of assigning whole expert tiles to one CTA.
            for (int row = blockIdx.x;; row += gridDim.x) {
                while (expert < experts && row >= end_row) {
                    first_row = end_row;
                    ++expert;
                    if (expert < experts) end_row += shared.sizes[expert];
                }
                if (expert == experts) break;
                const int local_row = row - first_row;
                if (tid % 32 == 0) {
                    const int route = routes[expert * s.capacity + local_row];
                    const int rank = route / (s.tokens * s.topk);
                    const int token = route % (s.tokens * s.topk) / s.topk;
                    const int source = rank * s.tokens + token;
                    // Deduplicated input pulls: fetch a token once per destination
                    // rank, then copy its local carrier row for other local experts.
                    const bool duplicate = expert != peers.carrier_experts[source];
                    int64_t carrier_row = -1;
                    if (duplicate) {
                        // Acquire/release publication makes the carrier payload
                        // visible before another CTA reads it through the async proxy.
                        auto carrier = cuda::atomic_ref<int64_t, cuda::thread_scope_device>(peers.carrier_rows[source]);
                        while ((carrier_row = carrier.load(cuda::memory_order_acquire)) < 0) __nanosleep(20);
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                    }
                    for (int col = 0; col < s.hidden; col += DISPATCH_ELEMENTS) {
                        // Bulk asynchronous dispatch: move up to 8 KiB through
                        // shared memory instead of issuing per-element peer loads.
                        const int bytes = min(DISPATCH_ELEMENTS, s.hidden - col) * int(sizeof(BF16));
                        auto* temp = shared.send[0];
                        auto* src = duplicate ? static_cast<const BF16*>(peers.pool) + carrier_row * s.hidden + col :
                            static_cast<const BF16*>(peers.x[rank]) + int64_t(token) * s.hidden + col;
                        auto* dst = static_cast<BF16*>(peers.pool) + (peers.offsets[expert] + local_row) * s.hidden + col;
                        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&shared.pull[0], bytes);
                        SM90_BULK_COPY_G2S::copy(src, &shared.pull[0], temp, bytes);
                        Barrier::wait(&shared.pull[0], phase);
                        phase ^= 1;
                        SM90_BULK_COPY_S2G::copy(temp, dst, bytes);
                        tma_store_arrive();
                        // Wait for global completion, not just shared-source reads,
                        // before publishing the carrier or incrementing tile readiness.
                        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                    }
                    if (!duplicate)
                        cuda::atomic_ref<int64_t, cuda::thread_scope_device>(peers.carrier_rows[source]).store(peers.offsets[expert] + local_row, cuda::memory_order_release);
                    cuda::atomic_ref<int, cuda::thread_scope_device>(peers.ready[expert * mtiles + local_row / TN]).fetch_add(1, cuda::memory_order_release);
                }
                __syncwarp();
            }
        }
        return;
    }

    if (warp == 11) {
        if (tid != 352) return;
        int expert = 0, index = 0;
        // Static persistent scheduling: each resident CTA advances by grid size,
        // reusing its pipeline without a global atomic task queue.
        for (int task = blockIdx.x; task < tokens * ntiles; task += gridDim.x, ++index) {
            while (task >= shared.starts[expert + 1] * ntiles) ++expert;
            const int local = task - shared.starts[expert] * ntiles;
            const int tiles = shared.starts[expert + 1] - shared.starts[expert];
            // Weight-major ordering: process token tiles within a feature tile
            // before moving to the next weight tile, encouraging weight-cache reuse.
            Work work{expert, local % tiles * TN, local / tiles * TM, shared.sizes[expert]};
            const int slot = index % QUEUE;
            if (index >= QUEUE) Barrier::wait(&shared.work_empty[slot], index / QUEUE % 2 ^ 1);
            shared.work[slot] = work;
            Barrier::arrive(&shared.work_full[slot]);
        }
        const int slot = index % QUEUE;
        if (index >= QUEUE) Barrier::wait(&shared.work_empty[slot], index / QUEUE % 2 ^ 1);
        shared.work[slot] = Work{-1, 0, 0, 0};
        Barrier::arrive(&shared.work_full[slot]);
        return;
    }

    if (warp == 9 || warp == 10) {
        if (tid % 32 != 0) return;
        const bool weight_loader = warp == 9;
        int sequence = 0;
        auto write_state = cutlass::make_producer_start_state<Pipeline>();
        for (int index = 0;; ++index) {
            Work work = consume(shared, index, true);
            if (work.expert < 0) break;
            if (!weight_loader && First) {
                // Tile-granular readiness: start a GEMM as soon as its own token
                // rows arrive, without waiting for the entire dispatch to finish.
                auto ready = cuda::atomic_ref<int, cuda::thread_scope_device>(peers.ready[work.expert * mtiles + work.m / TN]);
                while (ready.load(cuda::memory_order_acquire) < min(TN, work.count - work.m)) __nanosleep(20);
            }
            const int num_m_tiles = shared.starts[work.expert + 1] - shared.starts[work.expert];
            const int peer_task = (int(blockIdx.x) ^ 1) + index * int(gridDim.x) - shared.starts[work.expert] * ntiles;
            // Two-CTA TMA multicast: share one weight transfer only when both
            // cluster members need the same expert and feature tile.
            const bool multicast = peer_task >= 0 && peer_task < num_m_tiles * ntiles && peer_task / num_m_tiles == work.n / TM;
            for (int k = 0; k < K; k += TK, ++sequence) {
                const int stage = sequence % PIPE;
                pipeline.producer_acquire(write_state);
                auto* full = pipeline.producer_get_barrier(write_state);
                constexpr auto cache = uint64_t(TMA::CacheHintSm90::EVICT_NORMAL);
                if (weight_loader) {
                    if (multicast) {
                        if ((blockIdx.x & 1) == 0)
                            SM90_TMA_LOAD_MULTICAST_2D::copy(&map_A, full, 3, cache, &A(0, 0, stage), k, work.expert * N + work.n);
                    } else {
                        SM90_TMA_LOAD_2D::copy(&map_A, full, cache, &A(0, 0, stage), k, work.expert * N + work.n);
                    }
                } else {
                    SM90_TMA_LOAD_2D::copy(&map_B, full, cache, &B(0, 0, stage), k, int(peers.offsets[work.expert]) + work.m);
                }
                ++write_state;
            }
        }
        pipeline.producer_tail(write_state);
        return;
    }
    if (tid >= 256) return;

    cutlass::arch::warpgroup_reg_alloc<232>();
    const int wg = tid / 128;
    Mma<TN> mma;
    auto thread_mma = mma.get_slice(0);
    auto fragmentA = thread_mma.make_fragment_A(thread_mma.partition_A(A));
    auto fragmentB = thread_mma.make_fragment_B(thread_mma.partition_B(B));
    int sequence = 0, owned = 0;
    Pipeline::PipelineState read_state;
    for (int index = 0;; ++index) {
        Work work = consume(shared, index, lane == 0);
        if (work.expert < 0) break;
        // Ping-pong math: warpgroups alternate complete tiles so one group's
        // epilogue can execute while the other group issues the next GEMM.
        if (index % 2 != wg) { sequence += K / TK; read_state.advance(K / TK); continue; }
        if (wg != 0 || owned != 0) Barrier::wait(&shared.math_order[wg], (owned - (wg == 0)) & 1);
        auto release_state = read_state;
        auto accum = partition_fragment_C(mma, Shape<Int<TM>, Int<TN>>{});
        // First-MMA initialization avoids a separate accumulator-clear loop;
        // subsequent instructions accumulate BF16 products into FP32 registers.
        mma.accumulate_ = GMMA::ScaleOut::Zero;
        warpgroup_fence_operand(accum);
        {
            const int stage = sequence % PIPE;
            pipeline.consumer_wait(read_state);
            warpgroup_fence_operand(accum);
            warpgroup_arrive();
            CUTE_UNROLL
            for (int part = 0; part < size<2>(fragmentA); ++part) {
                gemm(mma, fragmentA(_, _, part, stage), fragmentB(_, _, part, stage), accum);
                mma.accumulate_ = GMMA::ScaleOut::One;
            }
            warpgroup_commit_batch();
            ++read_state;
            ++sequence;
        }
        warpgroup_fence_operand(accum);
        #pragma unroll 1
        for (int k = TK; k < K; k += TK, ++sequence) {
            const int stage = sequence % PIPE;
            pipeline.consumer_wait(read_state);
            warpgroup_fence_operand(accum);
            warpgroup_arrive();
            CUTE_UNROLL
            for (int part = 0; part < size<2>(fragmentA); ++part) {
                gemm(mma, fragmentA(_, _, part, stage), fragmentB(_, _, part, stage), accum);
            }
            warpgroup_commit_batch();
            ++read_state;
            // Bounded asynchronous MMA: retain one outstanding batch while
            // releasing only operand stages whose earlier MMA has completed.
            warpgroup_wait<1>();
            warpgroup_fence_operand(accum);
            pipeline.consumer_release(release_state);
            ++release_state;
        }
        warpgroup_fence_operand(accum);
        warpgroup_wait<0>();
        warpgroup_fence_operand(accum);
        pipeline.consumer_release(release_state);
        // The peer may start WGMMA while this group converts and stores its output.
        Barrier::arrive(&shared.math_order[1 - wg]);
        if (wg != 0 || owned != 0) Barrier::wait(&shared.epi_order[wg], (owned - (wg == 0)) & 1);
        warpgroup_fence_operand(accum);
        if constexpr (First) {
            // Chunked epilogue: constant-index gathers feed a small runtime loop,
            // limiting live temporaries without dynamically indexing registers.
            #pragma unroll 1
            for (int group = 0; group < TN / 8; ++group) {
                float gates[4], ups[4];
                switch (group) {
                    case 0: gather_group<0, TN>(accum, gates, ups); break;
                    case 1: gather_group<1, TN>(accum, gates, ups); break;
                    case 2: gather_group<2, TN>(accum, gates, ups); break;
                    case 3: gather_group<3, TN>(accum, gates, ups); break;
                    case 4: gather_group<4, TN>(accum, gates, ups); break;
                    case 5: gather_group<5, TN>(accum, gates, ups); break;
                    case 6: gather_group<6, TN>(accum, gates, ups); break;
                    case 7: gather_group<7, TN>(accum, gates, ups); break;
                    case 8: if constexpr (TN == 128) gather_group<8, TN>(accum, gates, ups); break;
                    case 9: if constexpr (TN == 128) gather_group<9, TN>(accum, gates, ups); break;
                    case 10: if constexpr (TN == 128) gather_group<10, TN>(accum, gates, ups); break;
                    case 11: if constexpr (TN == 128) gather_group<11, TN>(accum, gates, ups); break;
                    case 12: if constexpr (TN == 128) gather_group<12, TN>(accum, gates, ups); break;
                    case 13: if constexpr (TN == 128) gather_group<13, TN>(accum, gates, ups); break;
                    case 14: if constexpr (TN == 128) gather_group<14, TN>(accum, gates, ups); break;
                    case 15: if constexpr (TN == 128) gather_group<15, TN>(accum, gates, ups); break;
                }
                CUTE_UNROLL
                for (int i = 0; i < 4; ++i) {
                    const int token = group * 8 + (lane % 4) * 2 + i % 2;
                    const int col = (i / 2) * 32 + lane / 32 * 8 + lane % 32 / 4;
                    const float weight = work.m + token < work.count ?
                        peers.route_weights[work.expert * s.capacity + work.m + token] : 0.f;
                    // Precision-preserving fusion: round GEMM outputs to BF16,
                    // then clamp, apply SwiGLU/routing weight, and round once more.
                    float gate = fminf(float(BF16(gates[i])), s.clamp);
                    float up = fminf(fmaxf(float(BF16(ups[i])), -s.clamp), s.clamp);
                    C(token, col, wg) = BF16(gate / (1.f + expf(-gate)) * up * weight);
                }
            }
            tma_store_fence();
            cutlass::arch::NamedBarrier::sync(128, 2 + wg);
            if (lane == 0) {
                // Fused TMA epilogue: store the weighted activation directly;
                // gate/up tensors never need a separate global-memory pass.
                SM90_TMA_STORE_2D::copy(&map_C, &C(0, 0, wg), work.n / 2, int(peers.offsets[work.expert]) + work.m);
                tma_store_arrive();
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
            }
            cutlass::arch::NamedBarrier::sync(128, 2 + wg);
        } else {
            // Shared-memory retiling: convert the WGMMA fragment layout into
            // contiguous token rows suitable for aligned vector peer stores.
            CUTE_UNROLL
            for (int group = 0; group < TN / 8; ++group) {
                CUTE_UNROLL
                for (int token_sel = 0; token_sel < 2; ++token_sel) {
                    const int token = group * 8 + lane % 4 * 2 + token_sel;
                    CUTE_UNROLL
                    for (int half = 0; half < 2; ++half) {
                        const int j = half * (TN / 2) + group * 4 + token_sel;
                        const int col = half * 64 + lane / 32 * 16 + lane % 32 / 4;
                        shared.C[token * TM + col] = BF16(accum(j));
                        shared.C[token * TM + col + 8] = BF16(accum(j + 2));
                    }
                }
            }
            cutlass::arch::NamedBarrier::sync(128, 6 + wg);
            for (int v = lane; v < TN * TM / 8; v += 128) {
                const int token = work.m + v / (TM / 8), col = v % (TM / 8) * 8;
                if (token < work.count) {
                    const int route = routes[work.expert * s.capacity + token];
                    const int rank = route / (s.tokens * s.topk), local = route % (s.tokens * s.topk);
                    // Atomics-free scatter: each route owns a distinct output slot;
                    // uint4 writes eight BF16 values in one aligned 16-byte access.
                    BF16* dst = static_cast<BF16*>(peers.output[rank]) + int64_t(local) * N + work.n + col;
                    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(shared.C + v * 8);
                }
            }
            cutlass::arch::NamedBarrier::sync(128, 6 + wg);
        }
        Barrier::arrive(&shared.epi_order[1 - wg]);
        ++owned;
    }
}

template <bool First, int TN>
void launch(MoePeers peers, MoeShape shape, const void* weights, void* activation,
        const int* routes, const int* counts, int sms, cudaStream_t stream) {
    const int K = First ? shape.hidden : shape.intermediate;
    const int N = First ? 2 * shape.intermediate : shape.hidden;
    auto map_A = moe_tensor_map(weights, shape.experts / shape.world * N, K, TM);
    auto map_B = moe_tensor_map(First ? peers.pool : activation, int(shape.pool_capacity), K, TN);
    auto map_C = map_B;
    if constexpr (First) map_C = moe_tensor_map(activation, int(shape.pool_capacity), shape.intermediate, TN);
    CUDA_CHECK(cudaFuncSetAttribute(kernel<First, TN>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared<TN>)));
    cudaLaunchAttribute attributes[2]{};
    attributes[0].id = cudaLaunchAttributeClusterDimension;
    attributes[0].val.clusterDim = {2, 1, 1};
    attributes[1].id = cudaLaunchAttributeCooperative;
    attributes[1].val.cooperative = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(sms / 2 * 2);
    config.blockDim = dim3(NTHREADS);
    config.dynamicSmemBytes = sizeof(Shared<TN>);
    config.stream = stream;
    config.attrs = attributes;
    config.numAttrs = 2;
    int active_clusters = 0;
    // Residency-limited cooperative launch: keep all producer CTAs resident so
    // consumers cannot occupy the device while waiting for unscheduled dispatch.
    CUDA_CHECK(cudaOccupancyMaxActiveClusters(&active_clusters, kernel<First, TN>, &config));
    config.gridDim = dim3(std::min(active_clusters, sms / 2) * 2);
    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel<First, TN>, peers, shape, routes, counts, map_A, map_B, map_C));
    CUDA_CHECK(cudaGetLastError());
}
} // namespace hopper

template <bool First>
void launch_for_tokens(MoePeers peers, MoeShape shape, const void* weights, void* activation,
                       const int* routes, const int* counts, int sms, cudaStream_t stream) {
    // One optimized pipeline, specialized only for the selected token-tile size.
    if (moe_block_m(shape) == 64)
        hopper::launch<First, 64>(peers, shape, weights, activation, routes, counts, sms, stream);
    else
        hopper::launch<First, 128>(peers, shape, weights, activation, routes, counts, sms, stream);
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
    reduce_kernel<<<(s.tokens * s.hidden / 8 + 255) / 256, 256, 0, stream>>>(static_cast<BF16*>(y), static_cast<const BF16*>(slots), s);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace gpu_kernels
