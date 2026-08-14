#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/python.h>
#include "exceptions.hpp"

namespace blockwise {
template <typename T>
static T ceil_div(const T& a, const T& b) {
    return (a + b - 1) / b;
}
template <typename T>
static constexpr T align(const T& a, const T& b) {
    return ceil_div(a, b) * b;
}

static int get_tma_aligned_size(const int& x, const int& element_size) {
    constexpr int kNumTMAAlignmentBytes = 16;
    DG_HOST_ASSERT(kNumTMAAlignmentBytes % element_size == 0);
    return align(x, kNumTMAAlignmentBytes / element_size);
}
static std::pair<int, int> get_inner_outer_dims(const cute::UMMA::Major& major, const int& k, const int& mn) {
    return major == cute::UMMA::Major::K ? std::make_pair(k, mn) : std::make_pair(mn, k);
}

static int get_non_contiguous_dim(const cute::UMMA::Major& major) {
    return major == cute::UMMA::Major::K ? -2 : -1;
}

static int get_compiled_dim(const int& dim, const char& name, const std::string& compiled_dims) {
    for (const char& c: compiled_dims) {
        if (name == c)
            return dim;
    }
    return 0;
}
static CUtensorMapDataType aten_dtype_to_tensor_map_dtype(const at::ScalarType& dtype,
                                                          const bool& allow_tf32) {
    if (allow_tf32 and dtype == torch::kFloat)
        return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32;

    switch (dtype) {
        case torch::kInt:           return CU_TENSOR_MAP_DATA_TYPE_INT32;
        case torch::kFloat:         return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        case torch::kBFloat16:      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        case torch::kFloat8_e4m3fn: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
        default: DG_HOST_UNREACHABLE("Unsupported dtype");
    }
}

static CUtensorMapSwizzle mode_into_tensor_map_swizzle(const int& mode, const int& base) {
#if CUDA_VERSION >= 12080
    if (base != 0) {
        DG_HOST_ASSERT(base == 32 and mode == 128);
        return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
    }
#endif

    DG_HOST_ASSERT(base == 0);
    switch (mode) {
        case   0:
        case  16: return CU_TENSOR_MAP_SWIZZLE_NONE;
        case  32: return CU_TENSOR_MAP_SWIZZLE_32B;
        case  64: return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        default: DG_HOST_UNREACHABLE("Unsupported swizzling mode");
    }
}

static CUtensorMap make_tma_2d_desc(const torch::Tensor& t,
                                    int gmem_inner_dim, int gmem_outer_dim,
                                    int smem_inner_dim, int smem_outer_dim,
                                    const int& gmem_outer_stride,
                                    const int& swizzle_mode, const int& swizzle_base = 0,
                                    const bool& allow_tf32 = false) {
    const auto& elem_size = static_cast<int>(t.element_size());
    if (swizzle_mode != 0)
        smem_inner_dim = swizzle_mode / elem_size;

    CUtensorMap tensor_map;
    const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim), static_cast<cuuint64_t>(gmem_outer_dim)};
    const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim), static_cast<cuuint32_t>(smem_outer_dim)};
    const cuuint64_t gmem_strides[1] = {static_cast<cuuint64_t>(gmem_outer_stride * elem_size), };
    const cuuint32_t elem_strides[2] = {1, 1};
    // if (get_env<int>("DG_JIT_DEBUG")) {
    //     printf("Making TMA desc: global memory: %d %d, shared memory: %d %d, outer stride: %d, swizzle: %d (base: %d), elem size: %d\n",
    //            gmem_inner_dim, gmem_outer_dim, smem_inner_dim, smem_outer_dim,
    //            gmem_outer_stride, swizzle_mode, swizzle_base, elem_size);
    // }
    cuTensorMapEncodeTiled(
        &tensor_map, aten_dtype_to_tensor_map_dtype(t.scalar_type(), allow_tf32),
        2, t.data_ptr(), gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tensor_map;
}

static CUtensorMap make_tma_3d_desc(const torch::Tensor& t,
                                    const int& gmem_dim_0, const int& gmem_dim_1, const int& gmem_dim_2,
                                    const int& smem_dim_0, const int& smem_dim_1, const int& smem_dim_2,
                                    const int& gmem_stride_0, const int& gmem_stride_1,
                                    const int& swizzle_mode, const int& swizzle_base = 0,
                                    const bool& allow_tf32 = false) {
    const auto& elem_size = static_cast<int>(t.element_size());
    if (swizzle_mode != 0)
        DG_HOST_ASSERT(smem_dim_0 == swizzle_mode / elem_size);

    CUtensorMap tensor_map;
    const cuuint64_t gmem_dims[3] = {static_cast<cuuint64_t>(gmem_dim_0), static_cast<cuuint64_t>(gmem_dim_1), static_cast<cuuint64_t>(gmem_dim_2),};
    const cuuint32_t smem_dims[3] = {static_cast<cuuint32_t>(smem_dim_0), static_cast<cuuint32_t>(smem_dim_1), static_cast<cuuint32_t>(smem_dim_2)};
    const cuuint64_t gmem_strides[2] = {static_cast<cuuint64_t>(gmem_stride_0 * elem_size), static_cast<cuuint64_t>(gmem_stride_1 * elem_size)};
    const cuuint32_t elem_strides[3] = {1, 1, 1};
    // if (get_env<int>("DG_JIT_DEBUG")) {
    //     printf("Making 3D TMA desc: global memory: %d %d %d, shared memory: %d %d %d, outer stride: %d %d, swizzle: %d, elem size: %d\n",
    //            gmem_dim_0, gmem_dim_1, gmem_dim_2, smem_dim_0, smem_dim_1, smem_dim_2,
    //            gmem_stride_0, gmem_stride_1, swizzle_mode, elem_size);
    // }
    cuTensorMapEncodeTiled(
        &tensor_map, aten_dtype_to_tensor_map_dtype(t.scalar_type(), allow_tf32),
        3, t.data_ptr(), gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tensor_map;
}

static CUtensorMap make_tma_a_desc(const cute::UMMA::Major& major,
                                   const torch::Tensor& t,
                                   const int& shape_m, const int& shape_k,
                                   const int& block_m, const int& block_k,
                                   const int& outer_stride,
                                   const int& num_groups,
                                   const int& swizzle_mode, const int& swizzle_base = 0,
                                   const bool& allow_tf32 = false) {
    if (num_groups > 1)
        DG_HOST_ASSERT(major == cute::UMMA::Major::K);
    const auto& [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, shape_k, shape_m * num_groups);
    const auto& [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_k, block_m);
    return make_tma_2d_desc(t,
                            gmem_inner_dim, gmem_outer_dim,
                            smem_inner_dim, smem_outer_dim,
                            outer_stride,
                            swizzle_mode, swizzle_base,
                            allow_tf32);
}

static CUtensorMap make_tma_b_desc(const cute::UMMA::Major& major,
                                   const torch::Tensor& t,
                                   const int& shape_n, const int& shape_k,
                                   const int& block_n, const int& block_k,
                                   const int& outer_stride,
                                   const int& num_groups,
                                   const int& swizzle_mode, const int& swizzle_base = 0,
                                   const bool& allow_tf32 = false) {
    const auto& [gmem_inner_dim, gmem_outer_dim] = get_inner_outer_dims(major, shape_k, shape_n);
    const auto& [smem_inner_dim, smem_outer_dim] = get_inner_outer_dims(major, block_k, block_n);

    // `num_groups` is always applied into the outer dimensions
    return make_tma_2d_desc(t,
                            gmem_inner_dim, gmem_outer_dim * num_groups,
                            smem_inner_dim, smem_outer_dim,
                            outer_stride,
                            swizzle_mode, swizzle_base,
                            allow_tf32);
}

static CUtensorMap make_tma_cd_desc(const torch::Tensor& t,
                                    const int& shape_m, const int& shape_n,
                                    const int& block_m, const int& block_n,
                                    const int& outer_stride,
                                    const int& num_groups,
                                    const int& swizzle_mode, const int& swizzle_base = 0,
                                    const bool& allow_tf32 = false) {
    // Swizzling requires the inner box dim to be less or equal than `kSwizzleCDMode`
    // bytes, so `BLOCK_N * sizeof(T) / kSwizzleCDMode` TMA stores are required
    return make_tma_2d_desc(t,
                            shape_n, shape_m * num_groups,
                            block_n, block_m,
                            outer_stride,
                            swizzle_mode, swizzle_base,
                            allow_tf32);
}

static CUtensorMap make_tma_sf_desc(const cute::UMMA::Major& major,
                                    const torch::Tensor& t,
                                    int shape_mn, int shape_k,
                                    const int& block_mn, const int& block_k,
                                    const int& num_groups,
                                    const int& swizzle_mode, const int& swizzle_base = 0,
                                    const bool& allow_tf32 = false) {
    DG_HOST_ASSERT(major == cute::UMMA::Major::MN);

    // TODO: maybe swizzle SF as well
    DG_HOST_ASSERT(swizzle_mode == 0);

    shape_mn = get_tma_aligned_size(shape_mn, static_cast<int>(t.element_size()));
    return make_tma_2d_desc(t,
                            shape_mn, ceil_div(shape_k, block_k * (t.scalar_type() == torch::kFloat ? 1 : 4)) * num_groups,
                            block_mn, 1,
                            shape_mn,
                            swizzle_mode, swizzle_base,
                            allow_tf32);
}

} // namespace deep_gemm
