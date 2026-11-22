// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
//
// SPDX-License-Identifier: Apache-2.0

// NVFP4 GEMM tile configuration 5: 256x256x256 2sm, cluster 2x1x1, KernelTmaWarpSpecialized2SmNvf4Sm100.

#include "nvfp4_gemm_sm110.cuh"

namespace sana_robot::nvfp4_sm110 {

void run_config_5(at::Tensor& out, const at::Tensor& a, const at::Tensor& b, const at::Tensor& sfa,
                  const at::Tensor& sfb, const at::Tensor& alpha, const c10::optional<at::Tensor>& bias) {
  using Config = Nvfp4GemmConfig<Shape<_256, _256, _256>, Shape<_2, _1, _1>,
                                 cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100,
                                 cutlass::epilogue::TmaWarpSpecialized2Sm, false>;
  run_nvfp4_gemm<Config>(out, a, b, sfa, sfb, alpha, bias);
}

}  // namespace sana_robot::nvfp4_sm110
