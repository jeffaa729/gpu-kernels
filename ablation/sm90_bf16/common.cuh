#pragma once
#include "../api.cuh"
#include <cuda.h>
#include <cstdint>
#include <stdexcept>
#include <string>

// WGMMA descriptors and the TMA ring follow the existing production SM90 GEMM.
// Technique sequence: https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
// Original fast.cu author: Pranjal Shankhdhar; see LICENSE.fast-cu for MIT terms.
namespace ablation::hopper {
constexpr int BM = 128, BK = 64, ALIGNMENT = 1024;
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

template <int ScaleD>
__device__ __forceinline__ void wgmma128(float (&accumulator)[8][8], bf16* A, bf16* B) {
    const uint64_t ad = make_smem_descriptor<16>(A), bd = make_smem_descriptor<16>(B);
    asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, %66, 1, 1, 0, 0;"
        :
        "+f"(accumulator[0][0]), "+f"(accumulator[0][1]), "+f"(accumulator[0][2]), "+f"(accumulator[0][3]), "+f"(accumulator[0][4]), "+f"(accumulator[0][5]), "+f"(accumulator[0][6]), "+f"(accumulator[0][7]),
        "+f"(accumulator[1][0]), "+f"(accumulator[1][1]), "+f"(accumulator[1][2]), "+f"(accumulator[1][3]), "+f"(accumulator[1][4]), "+f"(accumulator[1][5]), "+f"(accumulator[1][6]), "+f"(accumulator[1][7]),
        "+f"(accumulator[2][0]), "+f"(accumulator[2][1]), "+f"(accumulator[2][2]), "+f"(accumulator[2][3]), "+f"(accumulator[2][4]), "+f"(accumulator[2][5]), "+f"(accumulator[2][6]), "+f"(accumulator[2][7]),
        "+f"(accumulator[3][0]), "+f"(accumulator[3][1]), "+f"(accumulator[3][2]), "+f"(accumulator[3][3]), "+f"(accumulator[3][4]), "+f"(accumulator[3][5]), "+f"(accumulator[3][6]), "+f"(accumulator[3][7]),
        "+f"(accumulator[4][0]), "+f"(accumulator[4][1]), "+f"(accumulator[4][2]), "+f"(accumulator[4][3]), "+f"(accumulator[4][4]), "+f"(accumulator[4][5]), "+f"(accumulator[4][6]), "+f"(accumulator[4][7]),
        "+f"(accumulator[5][0]), "+f"(accumulator[5][1]), "+f"(accumulator[5][2]), "+f"(accumulator[5][3]), "+f"(accumulator[5][4]), "+f"(accumulator[5][5]), "+f"(accumulator[5][6]), "+f"(accumulator[5][7]),
        "+f"(accumulator[6][0]), "+f"(accumulator[6][1]), "+f"(accumulator[6][2]), "+f"(accumulator[6][3]), "+f"(accumulator[6][4]), "+f"(accumulator[6][5]), "+f"(accumulator[6][6]), "+f"(accumulator[6][7]),
        "+f"(accumulator[7][0]), "+f"(accumulator[7][1]), "+f"(accumulator[7][2]), "+f"(accumulator[7][3]), "+f"(accumulator[7][4]), "+f"(accumulator[7][5]), "+f"(accumulator[7][6]), "+f"(accumulator[7][7])
        : "l"(ad), "l"(bd), "n"(ScaleD));
}

template <int ScaleD>
__device__ __forceinline__ void wgmma256(float (&accumulator)[16][8], bf16* A, bf16* B) {
    const uint64_t ad = make_smem_descriptor<16>(A), bd = make_smem_descriptor<16>(B);
    asm volatile("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, %130, 1, 1, 0, 0;"
        :
        "+f"(accumulator[0][0]), "+f"(accumulator[0][1]), "+f"(accumulator[0][2]), "+f"(accumulator[0][3]), "+f"(accumulator[0][4]), "+f"(accumulator[0][5]), "+f"(accumulator[0][6]), "+f"(accumulator[0][7]),
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
        : "l"(ad), "l"(bd), "n"(ScaleD));
}

template <int BN, int ScaleD>
__device__ __forceinline__ void multiply(float (&accumulator)[BN / 16][8], bf16* A, bf16* B) {
    if constexpr (BN == 128) wgmma128<ScaleD>(accumulator, A, B);
    else wgmma256<ScaleD>(accumulator, A, B);
}

template <int Stages>
__device__ __forceinline__ void advance(int& stage, int& phase) {
    if (++stage == Stages) { stage = 0; phase ^= 1; }
}
#endif
} // namespace ablation::hopper

