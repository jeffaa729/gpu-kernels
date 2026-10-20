#include "nvfp4.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace gpu_kernels {
namespace nvfp4_sm100 {

void cuda_check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

void driver_check(CUresult result) {
    if (result != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(result, &message);
        throw std::runtime_error(message);
    }
}

namespace ptx {

template <typename T>
static __device__ __forceinline__ uint32_t to_shared(T* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

static __device__ __forceinline__ uint32_t mapa_shared_cluster(
        uint32_t local_addr, uint32_t cta_rank) {
    uint32_t mapped;
    asm("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(mapped) : "r"(local_addr), "r"(cta_rank));
    return mapped;
}

static __device__ __forceinline__ void bar_sync(uint32_t id, uint32_t count) {
    asm volatile("barrier.cta.sync %0, %1;" :: "r"(id), "r"(count));
}

// Publish mbarrier initialization before either CTA issues peer-targeted transfers.
static __device__ __forceinline__ void cluster_sync_rel_acq() {
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
}

static __device__ __forceinline__ uint32_t cluster_rank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

// Read the lane special register directly to preserve uniform issue-role control.
static __device__ __forceinline__ uint32_t lane_id() {
    uint32_t r;
    asm("mov.u32 %0, %%laneid;" : "=r"(r));
    return r;
}

static __device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                 :: "r"(to_shared(bar)), "r"(count));
}

static __device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
    asm volatile("fence.mbarrier_init.release.cluster;");
}

static __device__ __forceinline__ void mbar_arrive_cluster_release(
        uint64_t* bar, uint32_t cta_rank) {
    const uint32_t mapped = mapa_shared_cluster(to_shared(bar), cta_rank);
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0], 1;"
                 :: "r"(mapped));
}

static __device__ __forceinline__ void tcgen05_wait_ld_then_arrive_release(
        uint64_t* bar, uint32_t cta_rank, bool do_arrive) {
    const uint32_t mapped = mapa_shared_cluster(to_shared(bar), cta_rank);
    asm volatile(
        "{\n\t"
        ".reg .pred %%pdo;\n\t"
        "setp.ne.s32 %%pdo, %1, 0;\n\t"
        "tcgen05.wait::ld.sync.aligned;\n\t"
        "@%%pdo mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0], 1;\n\t"
        "}\n\t"
        :: "r"(mapped), "r"((int)do_arrive) : "memory");
}

// Parity distinguishes queue generations; this wait is not a timeout.
static __device__ __forceinline__ void mbar_wait_parity(
        uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1, 10000000;\n\t"
        "@!p bra WAIT_%=;\n\t}\n"
        :: "r"(to_shared(bar)), "r"(parity));
}

static __device__ __forceinline__ void mbar_arrive_expect_tx_cluster(
        uint64_t* bar, uint32_t bytes) {
    const uint32_t mbar_addr = to_shared(bar) & 0xFEFFFFFFu;
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes));
}

// TMA loads packed operands into the issuing CTA and completes on the even peer.
static __device__ __forceinline__ void cp_async_bulk_tensor_3d_load_multicast(
        uint32_t dst_smem, const CUtensorMap* tmap,
        int32_t x, int32_t y, int32_t z, uint64_t* bar,
        uint16_t multicast_mask) {
    const uint32_t mbar_addr = to_shared(bar) & 0xFEFFFFFFu;
    asm volatile(
        "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%4, %5, %6}], [%2], %3;"
        :: "r"(dst_smem), "l"(tmap), "r"(mbar_addr), "h"(multicast_mask),
           "r"(x), "r"(y), "r"(z)
        : "memory");
}

static __device__ __forceinline__ void cp_async_bulk_tensor_3d_load_2sm_bit24(
        uint32_t dst_smem, const CUtensorMap* tmap,
        int32_t x, int32_t y, int32_t z, uint64_t* bar) {
    const uint32_t mbar_addr = to_shared(bar) & 0xFEFFFFFFu;
    asm volatile(
        "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst_smem), "l"(tmap), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr)
        : "memory");
}

// TMA reads the standard VEC16 layout; SFB is multicast to both cluster members.
static __device__ __forceinline__ void cp_async_bulk_tensor_4d_load_multicast(
        uint32_t dst_smem, const CUtensorMap* tmap,
        int32_t x, int32_t y, int32_t z2, int32_t w, uint64_t* bar,
        uint16_t multicast_mask) {
    const uint32_t mbar_addr = to_shared(bar) & 0xFEFFFFFFu;
    asm volatile(
        "cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%4, %5, %6, %7}], [%2], %3;"
        :: "r"(dst_smem), "l"(tmap), "r"(mbar_addr), "h"(multicast_mask),
           "r"(x), "r"(y), "r"(z2), "r"(w)
        : "memory");
}

// Allocate Tensor Memory collectively for the two-CTA MMA and its scale operands.
static __device__ __forceinline__ void tcgen05_alloc_2sm(
        uint32_t smem_addr_for_taddr, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_addr_for_taddr), "r"(n_cols));
}

static __device__ __forceinline__ void tcgen05_dealloc_2sm(
        uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(n_cols));
}

static __device__ __forceinline__ void tcgen05_relinquish_2sm() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
}

// Each lane drains 32 FP32 values from its Tensor Memory row into registers.
static __device__ __forceinline__ void tcgen05_ld_32x32b_x32(
        uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                 " {%0, %1, %2, %3, %4, %5, %6, %7,"
                 "  %8, %9, %10, %11, %12, %13, %14, %15,"
                 "  %16, %17, %18, %19, %20, %21, %22, %23,"
                 "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
                   "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]),
                   "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                   "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
                   "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                   "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
                   "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
                 : "r"(taddr));
}

// Convert two FP32 accumulators into one packed FP16 register.
static __device__ __forceinline__ uint32_t cvt_pack_f16x2(float a, float b) {
    uint32_t d;
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(a), "f"(b));
    return d;
}


// Write-once output bypasses L1 allocation and receives early L2 eviction priority.
static __device__ __forceinline__ void st_global_na_ef_b256(
        void* ptr, uint4 lo, uint4 hi) {
    asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], "
                 "{%1, %2, %3, %4, %5, %6, %7, %8};"
                 :: "l"(ptr), "r"(lo.x), "r"(lo.y), "r"(lo.z), "r"(lo.w),
                    "r"(hi.x), "r"(hi.y), "r"(hi.z), "r"(hi.w));
}


}  // namespace ptx

static __device__ __forceinline__ void mbar_wait_relaxed(
        uint32_t bar_smem, uint32_t parity) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1, 10000000;\n\t"
        "@!p bra WAIT_%=;\n\t}\n"
        :: "r"(bar_smem), "r"(parity) : "memory");
}


static __device__ __forceinline__ void cp_sf(
        uint32_t el, uint32_t dst, uint32_t lo, uint32_t hi) {
    const uint64_t desc = (uint64_t(hi) << 32) | uint64_t(lo);
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.u32 p, %2, 0;\n\t"
        "@p tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
        :: "r"(dst), "l"(desc), "r"(el) : "memory");
}

static __device__ __forceinline__ void commit_consumed(
        uint32_t el, uint32_t smbase, uint32_t mb_off, uint16_t mask) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.u32 p, %2, 0;\n\t"
        "@p tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t}\n"
        :: "r"(smbase + mb_off), "h"(mask), "r"(el) : "memory");
}

static __device__ __forceinline__ void mma_cell(
        uint32_t el, uint32_t d,
        uint32_t desc_a_lo, uint32_t desc_a_hi,
        uint32_t desc_b_lo, uint32_t desc_b_hi,
        uint32_t idesc, uint32_t sf_a, uint32_t sf_b, uint32_t acc_pred) {
    const uint64_t desc_a = (uint64_t(desc_a_hi) << 32) | uint64_t(desc_a_lo);
    const uint64_t desc_b = (uint64_t(desc_b_hi) << 32) | uint64_t(desc_b_lo);
    asm volatile(
        "{\n\t.reg .pred p, acc;\n\t"
        "setp.ne.u32 p, %7, 0;\n\t"
        "setp.ne.b32 acc, %4, 0;\n\t"
        "@p tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X"
        "  [%0], %1, %2, %3, [%5], [%6], acc;\n\t}\n"
        :: "r"(d), "l"(desc_a), "l"(desc_b), "r"(idesc),
           "r"(acc_pred), "r"(sf_a), "r"(sf_b), "r"(el) : "memory");
}


// One two-CTA cluster computes a 256x256 tile; each CTA owns 128 output rows.
constexpr int BM = 128, BN = 256, THREADS = 224;
// B200 scale windows cover K128, so eleven slots match the reference's
// seven K192 windows in lookahead distance without changing MMA granularity.
constexpr int AB_STAGES = 6, SF_STAGES = 11;
constexpr int AB_BYTES = 128;
constexpr int AB_K = 2 * AB_BYTES, SF_K = 128;
constexpr int AB_STRIDE = BM * AB_BYTES;
constexpr int SFA_BYTES = 1024, SFB_BYTES = 2048;
constexpr int SF_COLUMNS = 24, TMEM_COLUMNS = 512;
constexpr int ACC_BUFFERS = 2;
constexpr int ACC_STRIDE = 232;
constexpr int SF_BASE = 488;
constexpr uint32_t AB_TX = 2 * (BM + BM) * AB_BYTES;
constexpr uint32_t SF_TX = 2 * (SFA_BYTES + SFB_BYTES);
static_assert((ACC_BUFFERS - 1) * ACC_STRIDE + BN <= SF_BASE);
static_assert(SF_BASE + SF_COLUMNS <= TMEM_COLUMNS);

// Separate circular queues let A/B and scale TMA run ahead independently.
struct SharedStorage {
    struct alignas(1024) Header {
        uint64_t ab_ready[AB_STAGES], ab_free[AB_STAGES];
        uint64_t sf_ready[SF_STAGES], sf_free[SF_STAGES];
        uint64_t acc_ready, acc_free, dealloc;
        uint32_t tmem;
    } hdr;
    alignas(1024) uint8_t A[AB_STAGES][AB_STRIDE];
    alignas(1024) uint8_t B[AB_STAGES][AB_STRIDE];
    alignas(1024) uint8_t SFA[SF_STAGES][SFA_BYTES];
    alignas(1024) uint8_t SFB[SF_STAGES][SFB_BYTES];
};
static_assert(sizeof(SharedStorage) <= 232448);

// K64 uses relative leading-stride mode, not B300's K96 absolute-address mode.
// The descriptor and TMA agree on the same K-major swizzle and eight-row stride.
__device__ __forceinline__ uint64_t operand_descriptor(uint32_t base, int byte_offset = 0) {
    return uint64_t(((base + byte_offset) & 0x3FFFFU) >> 4) |
           (uint64_t(8 * AB_BYTES / 16) << 32) | (1ULL << 46) |
           (2ULL << 61);
}

// Unswizzled 512-byte scale atoms use the native 32x128b.warpx4 copy layout.
__device__ __forceinline__ uint64_t scale_descriptor(uint32_t base) {
    return uint64_t((base & 0x3FFFFU) >> 4) | 0x400800010000ULL;
}

__device__ __forceinline__ void advance(int& slot, int& phase, int count) {
    if (++slot == count) { slot = 0; phase ^= 1; }
}

// SFA: two K64 words; SFB: interleave the two 128-column N partitions per word.
// Each MMA reads four scale bytes per row with scale-factor ID zero.
__device__ __forceinline__ void copy_scales(uint32_t base, uint32_t tmem, int slot) {
    const uint64_t A = scale_descriptor(base + offsetof(SharedStorage, SFA) + slot * SFA_BYTES);
    const uint64_t B = scale_descriptor(base + offsetof(SharedStorage, SFB) + slot * SFB_BYTES);
    cp_sf(1, tmem + SF_BASE,     uint32_t(A),      uint32_t(A >> 32));
    cp_sf(1, tmem + SF_BASE + 4, uint32_t(A) + 32, uint32_t(A >> 32));
    cp_sf(1, tmem + SF_BASE + 8,  uint32_t(B),      uint32_t(B >> 32));
    cp_sf(1, tmem + SF_BASE + 12, uint32_t(B) + 64, uint32_t(B >> 32));
    cp_sf(1, tmem + SF_BASE + 16, uint32_t(B) + 32, uint32_t(B >> 32));
    cp_sf(1, tmem + SF_BASE + 20, uint32_t(B) + 96, uint32_t(B >> 32));
}

// The first K64 MMA overwrites TMEM, avoiding an accumulator-clear instruction.
__device__ __forceinline__ void issue_mma(uint32_t base, uint32_t tmem, uint32_t output,
                                          int ab_slot, int atom, int scale_word, bool accumulate) {
    const uint64_t A = operand_descriptor(base + offsetof(SharedStorage, A) + ab_slot * AB_STRIDE, atom * 32);
    const uint64_t B = operand_descriptor(base + offsetof(SharedStorage, B) + ab_slot * AB_STRIDE, atom * 32);
    constexpr uint32_t IDESC = (1U << 7) | (1U << 10) | (32U << 17) | (16U << 24);
    static_assert(IDESC == 0x10400480U);
    mma_cell(1, output, uint32_t(A), uint32_t(A >> 32), uint32_t(B), uint32_t(B >> 32),
             IDESC, tmem + SF_BASE + scale_word * 4, tmem + SF_BASE + 8 + scale_word * 8, accumulate);
}

struct Cursor { int slot = 0, phase = 0; };

// One issuer orders scale copies and MMAs in the same asynchronous instruction stream.
// K256 windows contain four exact K64 MMAs: no straddling descriptors or K96 tail.
__device__ __forceinline__ void compute_tile(SharedStorage& smem, uint32_t tmem,
                                             Cursor& ab, Cursor& sf, int& parity, int K) {
    const uint32_t base = ptx::to_shared(&smem);
    const int old_parity = parity;
    parity ^= 1;
    const uint32_t output = tmem + parity * ACC_STRIDE;
    #pragma unroll 1
    for (int k = 0; k < K; k += AB_K) {
        // Unroll the feed geometry: two K64 MMAs per scale slot, four per K256 window.
        #pragma unroll
        for (int pair = 0; pair < AB_K / SF_K; ++pair) {
            mbar_wait_relaxed(ptx::to_shared(&smem.hdr.sf_ready[sf.slot]), sf.phase);
            copy_scales(base, tmem, sf.slot);
            commit_consumed(1, base, offsetof(SharedStorage, hdr.sf_free) + sf.slot * 8, 3);
            if (pair == 0) mbar_wait_relaxed(ptx::to_shared(&smem.hdr.ab_ready[ab.slot]), ab.phase);
            if (k == 0 && pair == 0) mbar_wait_relaxed(ptx::to_shared(&smem.hdr.acc_free), old_parity);
            issue_mma(base, tmem, output, ab.slot, pair * 2,     0, k != 0 || pair != 0);
            issue_mma(base, tmem, output, ab.slot, pair * 2 + 1, 1, true);
            advance(sf.slot, sf.phase, SF_STAGES);
        }
        commit_consumed(1, base, offsetof(SharedStorage, hdr.ab_free) + ab.slot * 8, 3);
        advance(ab.slot, ab.phase, AB_STAGES);
    }
    commit_consumed(1, base, offsetof(SharedStorage, hdr.acc_ready), 3);
}


// L2 ownership is calibrated from measured requester latency, never hard-coded SM IDs.
__constant__ int route_table[4096], expected_side[128], sm_side[256];
__device__ unsigned placement_errors;

__device__ __forceinline__ int uniform_tile(int work) {
    unsigned tile;
    asm("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(tile) : "r"(route_table[work]));
    return int(tile);
}


// Seven warps, four roles: 0-3 drain TMEM, 4 issues MMA, 5 loads A/B, 6 loads scales.
// Exact block shape enables uniform compiler reasoning; launch grid is in cluster units.
__global__ __block_size__((THREADS, 1, 1)) __cluster_dims__(2, 1, 1) __launch_bounds__(THREADS, 1)
void gemm_kernel(__half* C, const __grid_constant__ CUtensorMap A_map,
                 const __grid_constant__ CUtensorMap B_map,
                 const __grid_constant__ CUtensorMap SFA_map,
                 const __grid_constant__ CUtensorMap SFB_map, int M, int N, int K) {
    extern __shared__ __align__(1024) char storage[];
    auto& smem = *reinterpret_cast<SharedStorage*>(storage);
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const uint32_t rank = ptx::cluster_rank();
    const int clusters = gridDim.z, cluster = blockIdx.z;
    const int m_tiles = M / 256, tiles = m_tiles * (N / 256);
    const uint32_t base = ptx::to_shared(&smem);
    if (threadIdx.x == 0) {
        unsigned sm;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        if (sm_side[sm] != expected_side[cluster]) atomicAdd(&placement_errors, 1U);
    }

    // Transaction counts include both CTAs and both SFB multicast destinations.
    if (warp == 0 && ptx::lane_id() == 0) {
        #pragma unroll
        for (int i = 0; i < AB_STAGES; ++i) {
            ptx::mbar_init(&smem.hdr.ab_ready[i], 1);
            ptx::mbar_init(&smem.hdr.ab_free[i], 1);
        }
        #pragma unroll
        for (int i = 0; i < SF_STAGES; ++i) {
            ptx::mbar_init(&smem.hdr.sf_ready[i], 1);
            ptx::mbar_init(&smem.hdr.sf_free[i], 1);
        }
        ptx::mbar_init(&smem.hdr.acc_ready, 1);
        ptx::mbar_init(&smem.hdr.acc_free, 8);
        ptx::mbar_init(&smem.hdr.dealloc, 32);
        ptx::fence_mbarrier_init_release_cluster();
    }
    ptx::cluster_sync_rel_acq();

    uint32_t tmem = 0;
    if (warp <= 4) {
        if (warp == 0) ptx::tcgen05_alloc_2sm(ptx::to_shared(&smem.hdr.tmem), TMEM_COLUMNS);
        ptx::bar_sync(2, 160);
        tmem = smem.hdr.tmem;
    }

    // Producers carry their cursors across tiles and can prefetch the next output tile.
    if (warp == 5 && ptx::lane_id() == 0) {
        int slot = 0, phase = 1;
        for (int work = cluster; work < tiles; work += clusters) {
            const int tile = route_table[work];
            const int m = 2 * (tile % m_tiles) + rank;
            const int n = 2 * (tile / m_tiles) + rank;
            for (int k = 0; k < K; k += AB_K) {
                ptx::mbar_wait_parity(&smem.hdr.ab_free[slot], phase);
                if (rank == 0) ptx::mbar_arrive_expect_tx_cluster(&smem.hdr.ab_ready[slot], AB_TX);
                ptx::cp_async_bulk_tensor_3d_load_multicast(base + offsetof(SharedStorage, A) + slot * AB_STRIDE,
                    &A_map, k / 2, m * BM, 0, &smem.hdr.ab_ready[slot], uint16_t(1U << rank));
                ptx::cp_async_bulk_tensor_3d_load_2sm_bit24(base + offsetof(SharedStorage, B) + slot * AB_STRIDE,
                    &B_map, k / 2, n * BM, 0, &smem.hdr.ab_ready[slot]);
                advance(slot, phase, AB_STAGES);
            }
        }
        const int last = (slot + AB_STAGES - 1) % AB_STAGES;
        ptx::mbar_wait_parity(&smem.hdr.ab_free[last], slot == 0 ? phase : phase ^ 1);
    } else if (warp == 6 && ptx::lane_id() == 0) {
        int slot = 0, phase = 1;
        for (int work = cluster; work < tiles; work += clusters) {
            const int tile = route_table[work];
            const int m = 2 * (tile % m_tiles) + rank;
            const int n = 2 * (tile / m_tiles) + rank;
            for (int k = 0; k < K; k += SF_K) {
                ptx::mbar_wait_parity(&smem.hdr.sf_free[slot], phase);
                if (rank == 0) ptx::mbar_arrive_expect_tx_cluster(&smem.hdr.sf_ready[slot], SF_TX);
                ptx::cp_async_bulk_tensor_4d_load_multicast(base + offsetof(SharedStorage, SFA) + slot * SFA_BYTES,
                    &SFA_map, 0, 0, k / 64, m, &smem.hdr.sf_ready[slot], uint16_t(1U << rank));
                ptx::cp_async_bulk_tensor_4d_load_multicast(base + offsetof(SharedStorage, SFB) + slot * SFB_BYTES + rank * 1024,
                    &SFB_map, 0, 0, k / 64, n, &smem.hdr.sf_ready[slot], uint16_t(3));
                advance(slot, phase, SF_STAGES);
            }
        }
        const int last = (slot + SF_STAGES - 1) % SF_STAGES;
        ptx::mbar_wait_parity(&smem.hdr.sf_free[last], slot == 0 ? phase : phase ^ 1);
    } else if (warp == 4 && rank == 0 && ptx::lane_id() == 0) {
        Cursor ab, sf;
        int parity = 1;
        for (int work = cluster; work < tiles; work += clusters)
            compute_tile(smem, tmem, ab, sf, parity, K);
        mbar_wait_relaxed(ptx::to_shared(&smem.hdr.acc_free), parity);
    } else if (warp < 4) {
        int phase = 0, buffer = 0;
        const uint32_t row_address = uint32_t(warp * 32) << 16;
        for (int work = cluster; work < tiles; work += clusters) {
            const int tile = uniform_tile(work);
            const int row = (2 * (tile % m_tiles) + rank) * BM + warp * 32 + lane;
            const int column = (tile / m_tiles) * BN;
            mbar_wait_relaxed(ptx::to_shared(&smem.hdr.acc_ready), phase);
            phase ^= 1;
            #pragma unroll 1
            for (int i = 0; i < 8; ++i) {
                // Drain the shared 24-column edge first; one x32 load covers the overlap.
                const int band = buffer == 0 ? 7 - i : i;
                uint32_t values[32];
                ptx::tcgen05_ld_32x32b_x32(tmem + buffer * ACC_STRIDE + band * 32 + row_address, values);
                if (i == 0) {
                    ptx::tcgen05_wait_ld_then_arrive_release(&smem.hdr.acc_free, 0, ptx::lane_id() == 0);
                }
                uint32_t packed[16];
                #pragma unroll
                for (int j = 0; j < 16; ++j)
                    packed[j] = ptx::cvt_pack_f16x2(__int_as_float(values[2 * j + 1]), __int_as_float(values[2 * j]));
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    void* destination = C + size_t(row) * N + column + band * 32 + j * 16;
                    uint4 lo{packed[j * 8], packed[j * 8 + 1], packed[j * 8 + 2], packed[j * 8 + 3]};
                    uint4 hi{packed[j * 8 + 4], packed[j * 8 + 5], packed[j * 8 + 6], packed[j * 8 + 7]};
                    ptx::st_global_na_ef_b256(destination, lo, hi);
                }
            }
            buffer = (buffer + 1) % ACC_BUFFERS;
        }
        // Peer teardown prevents an SM from freeing TMEM while its partner still reads it.
        ptx::bar_sync(3, 128);
        if (warp == 0) {
            ptx::tcgen05_relinquish_2sm();
            ptx::mbar_arrive_cluster_release(&smem.hdr.dealloc, rank ^ 1U);
            ptx::mbar_wait_parity(&smem.hdr.dealloc, 0);
            ptx::tcgen05_dealloc_2sm(tmem, TMEM_COLUMNS);
        }
    }
}


// Same block/cluster/resource shape as GEMM, used once before captured launches.
__global__ __block_size__((THREADS, 1, 1)) __cluster_dims__(2, 1, 1) __launch_bounds__(THREADS, 1)
void cluster_census(unsigned* smids) {
    extern __shared__ char pad[];
    if (threadIdx.x == 0) {
        pad[0] = 0;
        unsigned sm;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        smids[blockIdx.z * 2 + ptx::cluster_rank()] = sm;
    }
}

// A dependent atomic chain exposes near/far requester latency to the same cache line.
__global__ void latency_census(unsigned* probe, volatile unsigned* turn,
                               unsigned long long* timings, unsigned* smids) {
    extern __shared__ char pad[];
    if (threadIdx.x != 0) return;
    pad[0] = 0;
    while (*turn != blockIdx.x) {}
    unsigned value = 0, sm;
    unsigned long long start, stop;
    #pragma unroll
    for (int i = 0; i < 8; ++i) value = atomicAdd(probe + (value >> 31), 1U);
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    for (int i = 0; i < 64; ++i) value = atomicAdd(probe + (value >> 31), 1U);
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stop));
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    timings[blockIdx.x] = (stop - start) / 64 + (value >> 31);
    smids[blockIdx.x] = sm;
    __threadfence();
    *turn = blockIdx.x + 1;
}

// The reference's serpentine pockets group nearby output tiles for A/B reuse.
std::vector<int> pocket_order(int rows, int columns, int height, int width) {
    std::vector<int> order;
    for (int i = 0; i < (rows + height - 1) / height; ++i)
        for (int j = 0; j < (columns + width - 1) / width; ++j) {
            const int band = i % 2 ? (columns + width - 1) / width - 1 - j : j;
            for (int m = i * height; m < std::min((i + 1) * height, rows); ++m)
                for (int n = band * width; n < std::min((band + 1) * width, columns); ++n)
                    order.push_back(m + n * rows);
        }
    return order;
}

void prepare_schedule(const Nvfp4Params& p, int sms, cudaStream_t stream) {
    static int last_device = -1, last_M = 0, last_N = 0;
    int device;
    cuda_check(cudaGetDevice(&device));
    if (device == last_device && p.M == last_M && p.N == last_N) return;
    const int rows = p.M / 256, columns = p.N / 256, tiles = rows * columns;
    std::vector<int> route(tiles), sides(p.clusters, 0);
    std::array<int, 256> labels{};
    std::iota(route.begin(), route.end(), 0);

    // Small correctness cases use raster order; ownership needs a full resident grid.
    if (tiles >= sms / 2) {
        unsigned* scratch = nullptr;
        unsigned long long* d_times = nullptr;
        // Keep the latency probe and spin counter on separate cache lines,
        // as in the reference's separately allocated census buffers.
        constexpr int turn_offset = 32, ids_offset = 64;
        cuda_check(cudaMalloc(&scratch, size_t(ids_offset + 2 * sms) * sizeof(unsigned)));
        cuda_check(cudaMalloc(&d_times, size_t(sms) * sizeof(unsigned long long)));
        cuda_check(cudaMemsetAsync(scratch, 0, size_t(ids_offset + 2 * sms) * sizeof(unsigned), stream));
        constexpr int fat_smem = 130 * 1024;
        cuda_check(cudaFuncSetAttribute(latency_census, cudaFuncAttributeMaxDynamicSharedMemorySize, fat_smem));
        latency_census<<<sms, 32, fat_smem, stream>>>(scratch, scratch + turn_offset, d_times, scratch + ids_offset);
        cuda_check(cudaGetLastError());
        std::vector<unsigned long long> times(sms);
        std::vector<unsigned> smids(sms), cluster_smids(2 * p.clusters);
        cuda_check(cudaMemcpyAsync(times.data(), d_times, size_t(sms) * sizeof(times[0]), cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaMemcpyAsync(smids.data(), scratch + ids_offset, size_t(sms) * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaStreamSynchronize(stream));
        auto sorted = times;
        std::sort(sorted.begin(), sorted.end());
        int cut = sms / 2;
        for (int i = sms / 4; i < 3 * sms / 4; ++i)
            if (sorted[i + 1] - sorted[i] > sorted[cut + 1] - sorted[cut]) cut = i;
        const auto threshold = (sorted[cut] + sorted[cut + 1]) / 2;
        for (int i = 0; i < sms; ++i) labels[smids[i]] = times[i] > threshold;
        cuda_check(cudaFuncSetAttribute(cluster_census, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));
        cluster_census<<<dim3(1, 1, p.clusters), THREADS, sizeof(SharedStorage), stream>>>(scratch + ids_offset);
        cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpyAsync(cluster_smids.data(), scratch + ids_offset, cluster_smids.size() * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaStreamSynchronize(stream));
        cuda_check(cudaFree(d_times));
        cuda_check(cudaFree(scratch));

        std::array<std::vector<int>, 2> ids, pools;
        int side0_work = 0;
        bool same_side = true;
        for (int c = 0; c < p.clusters; ++c) {
            sides[c] = labels[cluster_smids[2 * c]];
            same_side &= sides[c] == labels[cluster_smids[2 * c + 1]];
            ids[sides[c]].push_back(c);
            if (sides[c] == 0) side0_work += (tiles + p.clusters - 1 - c) / p.clusters;
        }
        // SM-side timing is empirical, not an architectural promise. Raster is valid if grouping fails.
        if (same_side && !ids[0].empty() && !ids[1].empty()) {
            const int boundary_m = side0_work / columns, boundary_n = side0_work % columns;
            // fast.cu gemm9.cuh::pick_schedule uses owned 8x8 below the
            // reorder knee for 8192, and owned 8x4 for large near-square grids.
            const uint64_t operand_bytes = uint64_t(p.M + p.N) * p.K * 9 / 16;
            const int lo = std::min(rows, columns), hi = std::max(rows, columns);
            const bool owned_8x4 = operand_bytes >= (128ULL << 20) && lo >= 48 && hi * 4 <= lo * 5;
            const auto order = owned_8x4 ? pocket_order(rows, columns, 8, 4) :
                rows == 32 && columns == 32 ? pocket_order(rows, columns, 8, 8) : route;
            for (int tile : order) {
                const int m = tile % rows, n = tile / rows;
                const int side = m < boundary_m || (m == boundary_m && n < boundary_n) ? 0 : 1;
                pools[side].push_back(tile);
            }
            for (int side = 0; side < 2; ++side) {
                auto& clusters = ids[side];
                std::sort(clusters.begin(), clusters.end(), [&](int a, int b) {
                    const int a_trips = (tiles + p.clusters - 1 - a) / p.clusters;
                    const int b_trips = (tiles + p.clusters - 1 - b) / p.clusters;
                    return a_trips != b_trips ? a_trips > b_trips : a < b;
                });
                for (int rank = 0; rank < int(clusters.size()); ++rank) {
                    int trip = 0;
                    for (int work = clusters[rank]; work < tiles; work += p.clusters, ++trip)
                        route[work] = pools[side][rank + trip * clusters.size()];
                }
            }
        } else {
            labels.fill(0);
            std::fill(sides.begin(), sides.end(), 0);
        }
    }
    cuda_check(cudaMemcpyToSymbolAsync(route_table, route.data(), route.size() * sizeof(int), 0, cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyToSymbolAsync(sm_side, labels.data(), sizeof(labels), 0, cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyToSymbolAsync(expected_side, sides.data(), sides.size() * sizeof(int), 0, cudaMemcpyHostToDevice, stream));
    const unsigned zero = 0;
    cuda_check(cudaMemcpyToSymbolAsync(placement_errors, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice, stream));
    cuda_check(cudaStreamSynchronize(stream));
    last_device = device; last_M = p.M; last_N = p.N;
}


// Maps describe existing cuBLASLt-compatible allocations; no operand repack is needed.
CUtensorMap operand_map(const uint8_t* pointer, int rows, int K) {
    const uint64_t shape[3] = {uint64_t(K / 2), uint64_t(rows), 1};
    const uint64_t strides[2] = {uint64_t(K / 2), uint64_t(rows) * K / 2};
    const uint32_t box[3] = {AB_BYTES, BM, 1}, steps[3] = {1, 1, 1};
    CUtensorMap map{};
    driver_check(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
        const_cast<uint8_t*>(pointer), shape, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

CUtensorMap scale_map(const uint8_t* pointer, int rows, int K) {
    const uint64_t shape[4] = {128, 4, uint64_t(K / 64), uint64_t(rows / 128)};
    const uint64_t strides[3] = {128, 512, uint64_t(K / 64) * 512};
    const uint32_t box[4] = {128, 4, 2, 1}, steps[4] = {1, 1, 1, 1};
    CUtensorMap map{};
    driver_check(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
        const_cast<uint8_t*>(pointer), shape, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

void prepare(Nvfp4Params& p, const uint8_t* A, const uint8_t* B, const uint8_t* SFA, const uint8_t* SFB,
             int M, int N, int K, cudaStream_t stream) {
    int device;
    cuda_check(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device));
    p.M = M; p.N = N; p.K = K;
    p.clusters = std::min(properties.multiProcessorCount / 2, (M / 256) * (N / 256));
    p.A = operand_map(A, M, K); p.B = operand_map(B, N, K);
    p.SFA = scale_map(SFA, M, K); p.SFB = scale_map(SFB, N, K);
    cuda_check(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));
    // Size the persistent grid from actual cluster residency, not a hard-coded SM count.
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim = {2, 1, 1};
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(2, 1, p.clusters);
    config.blockDim = dim3(THREADS);
    config.dynamicSmemBytes = sizeof(SharedStorage);
    config.attrs = &attribute;
    config.numAttrs = 1;
    int active_clusters = 0;
    cuda_check(cudaOccupancyMaxActiveClusters(&active_clusters, gemm_kernel, &config));
    p.clusters = std::min(active_clusters, (M / 256) * (N / 256));
    prepare_schedule(p, properties.multiProcessorCount, stream);
}

void launch(const Nvfp4Params& p, __half* C, cudaStream_t stream) {
    gemm_kernel<<<dim3(1, 1, p.clusters), THREADS, sizeof(SharedStorage), stream>>>(C, p.A, p.B, p.SFA, p.SFB, p.M, p.N, p.K);
    cuda_check(cudaGetLastError());
}

void validate() {
    unsigned errors;
    cuda_check(cudaMemcpyFromSymbol(&errors, placement_errors, sizeof(errors)));
    if (errors) throw std::runtime_error("L2 ownership placement changed; rerun this stage with an idle GPU.");
}
}  // namespace nvfp4_sm100

Nvfp4Implementation nvfp4_sm100_implementation() {
    return {nvfp4_sm100::prepare, nvfp4_sm100::launch, nvfp4_sm100::validate};
}
}  // namespace gpu_kernels
