// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
//
// SPDX-License-Identifier: Apache-2.0

// NVFP4 GEMM tile configuration 2: 128x64x128, KernelTmaWarpSpecializedPingpongNvf4Sm120.

#include "nvfp4_gemm_sm120.cuh"

namespace sana_robot::nvfp4_sm120 {

void run_config_2(at::Tensor& out, const at::Tensor& a, const at::Tensor& b, const at::Tensor& sfa,
                  const at::Tensor& sfb, const at::Tensor& alpha, const c10::optional<at::Tensor>& bias) {
  using Config = Nvfp4GemmConfig<Shape<_128, _64, _128>, cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120, false>;
  run_nvfp4_gemm<Config>(out, a, b, sfa, sfb, alpha, bias);
}

}  // namespace sana_robot::nvfp4_sm120
