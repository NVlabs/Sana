

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>

#include <cute/arch/mma_sm100_desc.hpp>
#include "runtime_utils.hpp"

#include "config.hpp"
#include "static_switch.hpp"

namespace deep_gemm{
template<int N, int K>
void sm90_fp8_gemm_1d2d_bias_launch(int num_sms, int num_threads, int cluster_dim, int smem_size, cudaStream_t stream, float* sfb, float* bias, int* grouped_layout,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        const CUtensorMap tensor_map_a,
                        const CUtensorMap tensor_map_b,
                        const CUtensorMap tensor_map_d,
                        const CUtensorMap tensor_map_sfa);
};

namespace blockwise{

static void sm90_fp8_gemm_1d2d_bias(const torch::Tensor& a, const torch::Tensor& sfa,
                               const torch::Tensor& b, const torch::Tensor& sfb,
                               const torch::Tensor& bias,
                               const std::optional<torch::Tensor>& c,
                               const torch::Tensor& d,
                               const int& m, const int& n, const int& k, const int num_sms) {
    // DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kBFloat16);
    const auto& config = GemmConfig<90>();

    // Requires no TMA splits
    // DG_HOST_ASSERT(config.smem_config.swizzle_a_mode == config.block_k);
    // DG_HOST_ASSERT(config.smem_config.swizzle_b_mode == config.block_k);
    int smem_size = k == 16384 || k == 8192 ? 216624 : config.smem_config.smem_size;     
    const auto& tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a, m, k,
                                               config.block_m,
                                               config.block_k,
                                               static_cast<int>(a.stride(-2)), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(cute::UMMA::Major::K, b, n, k,
                                               config.block_n,
                                               config.block_k,
                                               static_cast<int>(b.stride(-2)), 1,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_d = make_tma_cd_desc(d, m, static_cast<int>(d.size(-1)),
                                                config.block_m,
                                                config.block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, config.block_k, 1, 0);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // Launch    
    DIM_SWITCH(k, K, 
        DIM_SWITCH(n, N, 
            deep_gemm::sm90_fp8_gemm_1d2d_bias_launch<N, K>(num_sms, config.thread_config.num_threads, config.multicast_config.num_multicast, smem_size, stream, (float*)sfb.data_ptr(), (float*)bias.data_ptr(), nullptr, m, n, k, tensor_map_a, tensor_map_b, tensor_map_d, tensor_map_sfa);)
    )
}
};