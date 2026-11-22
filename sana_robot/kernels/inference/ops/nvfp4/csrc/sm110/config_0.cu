// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
//
// SPDX-License-Identifier: Apache-2.0

// NVFP4 GEMM tile configuration 0: 128x128x256 1sm, cluster 1x1x1, KernelTmaWarpSpecialized1SmNvf4Sm100.

#include "nvfp4_gemm_sm110.cuh"

namespace sana_robot::nvfp4_sm110 {

void run_config_0(at::Tensor& out, const at::Tensor& a, const at::Tensor& b, const at::Tensor& sfa,
                  const at::Tensor& sfb, const at::Tensor& alpha, const c10::optional<at::Tensor>& bias) {
  using Config = Nvfp4GemmConfig<Shape<_128, _128, _256>, Shape<_1, _1, _1>,
                                 cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100,
                                 cutlass::epilogue::TmaWarpSpecialized1Sm, false>;
  run_nvfp4_gemm<Config>(out, a, b, sfa, sfb, alpha, bias);
}

}  // namespace sana_robot::nvfp4_sm110
