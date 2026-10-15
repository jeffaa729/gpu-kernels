#pragma once

#include <cuda.h>
#include <stdexcept>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/bfloat16.h>

namespace gpu_kernels {

// Same 128-byte inner atoms as DeepGEMM common/tma_copy.cuh::copy_nd.
// Repeated atoms form the swizzled CuTe WGMMA operand tile in shared memory.
inline CUtensorMap moe_tensor_map(const void* pointer, int rows, int columns, int tile_rows) {
    CUtensorMap map;
    const uint64_t dimensions[] = {uint64_t(columns), uint64_t(rows)};
    const uint64_t strides[] = {uint64_t(columns) * sizeof(cutlass::bfloat16_t)};
    const uint32_t box[] = {64, uint32_t(tile_rows)}, element_strides[] = {1, 1};
    const auto result = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<void*>(pointer), dimensions, strides, box, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) throw std::runtime_error("MoE TMA tensor-map encoding failed");
    return map;
}

} // namespace gpu_kernels
