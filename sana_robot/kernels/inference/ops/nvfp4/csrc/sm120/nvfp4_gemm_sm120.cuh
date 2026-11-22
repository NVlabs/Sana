// Copyright 2026 NVIDIA CORPORATION & AFFILIATES
//
// SPDX-License-Identifier: Apache-2.0

// NVFP4 GEMM on RTX 5090 and RTX PRO 6000 (sm_120a) with CUTLASS's block-scaled warp-specialized kernels:
//
//   out[M, N] (bf16) = alpha * (A[M, K] . B[N, K]^T) + bias[N]
//
// A and B are packed E2M1 (two per byte, [rows, K / 2]) with one E4M3 scale per 16 elements along K, the scales in the
// Sm1xx block-scaled layout (128 x 4 tiles; the layout cuBLASLt's VEC16_UE4M3 reads). alpha is a device fp32 scalar
// (the product of the two tensor scales, written by the activation quantizer), read by the epilogue: no host sync.
// The bias, if any, is added after alpha. Each tile configuration is instantiated in its own translation unit
// (config_*.cu). A SwapAB configuration computes out^T = B . A^T (the weight as the kernel's A, the tokens on the
// kernel's N, where tiles may be narrower than the 128-row minimum of the block-scaled M tile) into a column-major D,
// which is out in row-major order; its bias is per row of out^T (LinCombPerRowBias).

#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace sana_robot::nvfp4_sm120 {

using namespace cute;

template <class TileShape, class KernelSchedule, bool SwapAB_ = false>
struct Nvfp4GemmConfig {
  static constexpr bool SwapAB = SwapAB_;
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentA = 32;
  static constexpr int AlignmentB = 32;
  using ElementD = cutlass::bfloat16_t;
  using ElementC = cutlass::bfloat16_t;
  using ElementBias = cutlass::bfloat16_t;
  using LayoutD = cute::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using ClusterShape = Shape<_1, _1, _1>;
  using Fusion = cute::conditional_t<
      SwapAB,
      cutlass::epilogue::fusion::LinCombPerRowBias<ElementD, ElementCompute, ElementBias, ElementC, ElementCompute>,
      cutlass::epilogue::fusion::LinCombPerColBias<ElementD, ElementCompute, ElementBias, ElementC, ElementCompute>>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute, ElementC, LayoutD, AlignmentD, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <class Config>
void run_nvfp4_gemm(at::Tensor& out, const at::Tensor& a, const at::Tensor& b, const at::Tensor& sfa,
                    const at::Tensor& sfb, const at::Tensor& alpha, const c10::optional<at::Tensor>& bias) {
  using Gemm = typename Config::Gemm;
  using Kernel = typename Gemm::GemmKernel;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using StrideD = typename Kernel::StrideD;
  using BlkScaledConfig = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using DataA = typename Config::ElementA::DataType;
  using DataB = typename Config::ElementB::DataType;
  using ScaleA = typename Config::ElementA::ScaleFactorType;
  using ScaleB = typename Config::ElementB::ScaleFactorType;

  // the kernel's problem: (M, N) = (tokens, features), or (features, tokens) with SwapAB
  const at::Tensor& op_a = Config::SwapAB ? b : a;
  const at::Tensor& op_b = Config::SwapAB ? a : b;
  const at::Tensor& op_sfa = Config::SwapAB ? sfb : sfa;
  const at::Tensor& op_sfb = Config::SwapAB ? sfa : sfb;
  const int m = static_cast<int>(op_a.size(0));
  const int k = static_cast<int>(op_a.size(1)) * 2;
  const int n = static_cast<int>(op_b.size(0));

  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  auto layout_sfa = BlkScaledConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  auto layout_sfb = BlkScaledConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
  TORCH_CHECK(static_cast<int64_t>(size(filter_zeros(layout_sfa))) <= op_sfa.numel(),
              "nvfp4 gemm: the kernel's A scales hold ", op_sfa.numel(), " bytes, the layout needs ",
              size(filter_zeros(layout_sfa)));
  TORCH_CHECK(static_cast<int64_t>(size(filter_zeros(layout_sfb))) <= op_sfb.numel(),
              "nvfp4 gemm: the kernel's B scales hold ", op_sfb.numel(), " bytes, the layout needs ",
              size(filter_zeros(layout_sfb)));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {reinterpret_cast<DataA const*>(op_a.data_ptr()), stride_a, reinterpret_cast<DataB const*>(op_b.data_ptr()),
       stride_b, reinterpret_cast<ScaleA const*>(op_sfa.data_ptr()), layout_sfa,
       reinterpret_cast<ScaleB const*>(op_sfb.data_ptr()), layout_sfb},
      {{}, nullptr, stride_c, reinterpret_cast<typename Config::ElementD*>(out.data_ptr()), stride_d}};
  auto& fusion = arguments.epilogue.thread;
  fusion.alpha_ptr = alpha.data_ptr<float>();
  fusion.beta = 0.0f;
  fusion.bias_ptr = bias.has_value() ? reinterpret_cast<typename Config::ElementBias const*>(bias->data_ptr()) : nullptr;

  Gemm gemm;
  TORCH_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess,
              "nvfp4 gemm: CUTLASS cannot implement M=", m, " N=", n, " K=", k, " with this tile configuration");
  const size_t workspace_bytes = Gemm::get_workspace_size(arguments);
  at::Tensor workspace = at::empty({static_cast<int64_t>(workspace_bytes)}, a.options().dtype(at::kByte));
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  cutlass::Status status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "nvfp4 gemm: initialize failed: ",
              cutlassGetStatusString(status));
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "nvfp4 gemm: run failed: ", cutlassGetStatusString(status));
}

// One entry per instantiated configuration (config_<index>.cu).
void run_config_0(at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                  const at::Tensor&, const c10::optional<at::Tensor>&);
void run_config_1(at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                  const at::Tensor&, const c10::optional<at::Tensor>&);
void run_config_2(at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                  const at::Tensor&, const c10::optional<at::Tensor>&);
void run_config_3(at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                  const at::Tensor&, const c10::optional<at::Tensor>&);
void run_config_4(at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                  const at::Tensor&, const c10::optional<at::Tensor>&);
void run_config_5(at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                  const at::Tensor&, const c10::optional<at::Tensor>&);
void run_config_6(at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                  const at::Tensor&, const c10::optional<at::Tensor>&);
void run_config_7(at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                  const at::Tensor&, const c10::optional<at::Tensor>&);

}  // namespace sana_robot::nvfp4_sm120
