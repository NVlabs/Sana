// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
//
// SPDX-License-Identifier: Apache-2.0

// Python seam of the sm_110 NVFP4 GEMM: gemm(out, a, b, sfa, sfb, alpha, bias, config) writes out in place.

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "nvfp4_gemm_sm110.cuh"

namespace sana_robot::nvfp4_sm110 {

namespace {

using RunConfig = void (*)(at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                           const at::Tensor&, const c10::optional<at::Tensor>&);

// config index -> (name, entry)
const char* kConfigNames[] = {
    "128x128x256 1sm",
    "128x256x256 1sm",
    "128x64x256 1sm",
    "128x192x256 1sm",
    "256x128x256 2sm",
    "256x256x256 2sm",
    "swapAB 128x64x256 1sm",
    "swapAB 128x128x256 1sm",
};
const RunConfig kConfigRuns[] = {
    run_config_0,
    run_config_1,
    run_config_2,
    run_config_3,
    run_config_4,
    run_config_5,
    run_config_6,
    run_config_7,
};
constexpr int64_t kConfigs = sizeof(kConfigRuns) / sizeof(kConfigRuns[0]);
static_assert(sizeof(kConfigNames) / sizeof(kConfigNames[0]) == kConfigs, "one name per configuration");

void check(const at::Tensor& t, const char* name, at::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda(), "nvfp4 gemm: ", name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == dtype, "nvfp4 gemm: ", name, " must be ", dtype, ", got ", t.scalar_type());
  TORCH_CHECK(t.is_contiguous(), "nvfp4 gemm: ", name, " must be contiguous");
}

}  // namespace

void gemm(at::Tensor out, const at::Tensor& a, const at::Tensor& b, const at::Tensor& sfa, const at::Tensor& sfb,
          const at::Tensor& alpha, const c10::optional<at::Tensor>& bias, int64_t config) {
  check(out, "out", at::kBFloat16);
  check(a, "a", at::kByte);
  check(b, "b", at::kByte);
  check(sfa, "sfa", at::kByte);
  check(sfb, "sfb", at::kByte);
  check(alpha, "alpha", at::kFloat);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "nvfp4 gemm: a and b are [rows, K / 2]");
  TORCH_CHECK(a.size(1) == b.size(1), "nvfp4 gemm: a and b disagree on K: ", a.size(1) * 2, " vs ", b.size(1) * 2);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == a.size(0) && out.size(1) == b.size(0), "nvfp4 gemm: out must be [",
              a.size(0), ", ", b.size(0), "]");
  TORCH_CHECK(alpha.numel() == 1, "nvfp4 gemm: alpha is one fp32 element");
  if (bias.has_value()) {
    check(*bias, "bias", at::kBFloat16);
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == b.size(0), "nvfp4 gemm: bias must be [", b.size(0), "]");
  }
  TORCH_CHECK(config >= 0 && config < kConfigs, "nvfp4 gemm: config ", config, " outside [0, ", kConfigs, ")");
  const c10::cuda::CUDAGuard guard(a.device());
  kConfigRuns[config](out, a, b, sfa, sfb, alpha, bias);
}

std::vector<std::string> configs() { return std::vector<std::string>(kConfigNames, kConfigNames + kConfigs); }

}  // namespace sana_robot::nvfp4_sm110

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &sana_robot::nvfp4_sm110::gemm, "NVFP4 GEMM (sm_110a): out = alpha * (a . b^T) + bias",
        pybind11::arg("out"), pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("sfa"), pybind11::arg("sfb"),
        pybind11::arg("alpha"), pybind11::arg("bias"), pybind11::arg("config"));
  m.def("configs", &sana_robot::nvfp4_sm110::configs, "the tile configurations by index");
}
