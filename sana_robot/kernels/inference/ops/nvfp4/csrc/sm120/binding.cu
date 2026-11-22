// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
//
// SPDX-License-Identifier: Apache-2.0

// Python seam of the sm_120 NVFP4 GEMM: gemm(out, a, b, sfa, sfb, alpha, bias, config) writes out in place.

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "nvfp4_gemm_sm120.cuh"

namespace sana_robot::nvfp4_sm120 {

namespace {

constexpr int64_t kConfigs = 8;
const char* kConfigNames[kConfigs] = {
    "128x128x128 cooperative",
    "128x128x128 pingpong",
    "128x64x128 pingpong",
    "128x256x128 cooperative",
    "128x128x256 cooperative",
    "swapAB 128x64x128 pingpong",
    "swapAB 128x32x128 pingpong",
    "swapAB 128x128x128 cooperative",
};

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
  switch (config) {
    case 0: run_config_0(out, a, b, sfa, sfb, alpha, bias); break;
    case 1: run_config_1(out, a, b, sfa, sfb, alpha, bias); break;
    case 2: run_config_2(out, a, b, sfa, sfb, alpha, bias); break;
    case 3: run_config_3(out, a, b, sfa, sfb, alpha, bias); break;
    case 4: run_config_4(out, a, b, sfa, sfb, alpha, bias); break;
    case 5: run_config_5(out, a, b, sfa, sfb, alpha, bias); break;
    case 6: run_config_6(out, a, b, sfa, sfb, alpha, bias); break;
    case 7: run_config_7(out, a, b, sfa, sfb, alpha, bias); break;
  }
}

std::vector<std::string> configs() { return std::vector<std::string>(kConfigNames, kConfigNames + kConfigs); }

}  // namespace sana_robot::nvfp4_sm120

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &sana_robot::nvfp4_sm120::gemm, "NVFP4 GEMM (sm_120a): out = alpha * (a . b^T) + bias",
        pybind11::arg("out"), pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("sfa"), pybind11::arg("sfb"),
        pybind11::arg("alpha"), pybind11::arg("bias"), pybind11::arg("config"));
  m.def("configs", &sana_robot::nvfp4_sm120::configs, "the tile configurations by index");
}
