/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "nvfp4_scaled_mm_common.cuh"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include "cutlass/epilogue/thread/activation.h"

// Config(half_t/bfloat16_t) for M <= 128
template <typename T>
struct KernelConfigM128 {
  using OutputType = T;
  using MmaTileShape = Shape<_128, _256, _256>;
  using ClusterShape = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;  // Avoid register spilling
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
template <typename T>
const dim3 KernelConfigM128<T>::preferred_cluster(1, 4, 1);
template <typename T>
const dim3 KernelConfigM128<T>::fallback_cluster(1, 2, 1);

// Config(half_t/bfloat16_t) for M <= 256
template <typename T>
struct KernelConfigM256 {
  using OutputType = T;
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;  // Avoid register spilling
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
template <typename T>
const dim3 KernelConfigM256<T>::preferred_cluster(2, 4, 1);
template <typename T>
const dim3 KernelConfigM256<T>::fallback_cluster(2, 1, 1);

// Config(half_t/bfloat16_t) for 256 < M <= 1024
template <typename T>
struct KernelConfigDefault {
  using OutputType = T;
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;  // Avoid register spilling
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
template <typename T>
const dim3 KernelConfigDefault<T>::preferred_cluster(2, 4, 1);
template <typename T>
const dim3 KernelConfigDefault<T>::fallback_cluster(2, 1, 1);

// Config(half_t/bfloat16_t) for M > 1024: 1x4 cluster reduces M-tail waste.
template <typename T>
struct KernelConfigLargeM {
  using OutputType = T;
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
template <typename T>
const dim3 KernelConfigLargeM<T>::preferred_cluster(2, 4, 1);
template <typename T>
const dim3 KernelConfigLargeM<T>::fallback_cluster(2, 1, 1);

struct KernelConfigFp32 {
  using OutputType = float;
  using MmaTileShape = Shape<_128, _128, _256>;
  using ClusterShape = Shape<int, int, _1>;
  using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
const dim3 KernelConfigFp32::preferred_cluster = dim3(1, 4, 1);
const dim3 KernelConfigFp32::fallback_cluster = dim3(1, 2, 1);

template <typename KernelConfig, typename FusionOp>
struct Fp4GemmSm100 {
  using Config = KernelConfig;
  using OutputType = typename KernelConfig::OutputType;

  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = OutputType;
  using ElementC = OutputType;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename KernelConfig::MmaTileShape;
  using ClusterShape = typename KernelConfig::ClusterShape;
  using EpilogueTile = typename KernelConfig::EpilogueTile;
  using EpilogueSchedule = typename KernelConfig::EpilogueSchedule;
  using MainloopSchedule = typename KernelConfig::MainloopSchedule;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      MmaTileShape,
      ClusterShape,
      EpilogueTile,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      EpilogueSchedule,
      FusionOp>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutATag,
      AlignmentA,
      ElementB,
      LayoutBTag,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));
};

template <typename KernelConfig>
using Fp4GemmSm100Linear = Fp4GemmSm100<
    KernelConfig,
    cutlass::epilogue::fusion::LinearCombination<typename KernelConfig::OutputType, float, void, float>>;

template <typename KernelConfig>
using Fp4GemmSm100BiasGelu = Fp4GemmSm100<
    KernelConfig,
    cutlass::epilogue::fusion::LinCombPerColBiasEltAct<
        cutlass::epilogue::thread::GELU_taylor,
        typename KernelConfig::OutputType,
        float,
        typename KernelConfig::OutputType,
        void,
        float>>;

template <typename KernelConfig>
using Fp4GemmSm100PerColResidualGate = Fp4GemmSm100<
    KernelConfig,
    cutlass::epilogue::fusion::PerColResAddPerColBiasEltAct<
        cutlass::epilogue::thread::Identity,
        typename KernelConfig::OutputType,
        float,
        typename KernelConfig::OutputType,
        typename KernelConfig::OutputType,
        typename KernelConfig::OutputType>>;

template <typename T>
typename T::Gemm::Arguments args_from_options(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int64_t M,
    int64_t N,
    int64_t K) {
  using ElementA = typename T::Gemm::ElementA;
  using ElementB = typename T::Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementD = typename T::Gemm::ElementD;
  using ElementCompute = float;
  using StrideA = typename T::StrideA;
  using StrideB = typename T::StrideB;
  using StrideD = typename T::StrideD;
  using Sm1xxBlkScaledConfig = typename T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename T::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       nullptr,
       stride_D,
       static_cast<ElementD*>(D.data_ptr()),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());
  using KernelConfig = typename T::Config;
  arguments.hw_info.cluster_shape = KernelConfig::preferred_cluster;
  arguments.hw_info.cluster_shape_fallback = KernelConfig::fallback_cluster;
  return arguments;
}

template <typename T>
typename T::Gemm::Arguments args_from_options_bias_gelu(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView bias,
    int64_t M,
    int64_t N,
    int64_t K) {
  using ElementA = typename T::Gemm::ElementA;
  using ElementB = typename T::Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementD = typename T::Gemm::ElementD;
  using ElementCompute = float;
  using StrideA = typename T::StrideA;
  using StrideB = typename T::StrideB;
  using StrideD = typename T::StrideD;
  using Sm1xxBlkScaledConfig = typename T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename T::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       nullptr,
       stride_D,
       static_cast<ElementD*>(D.data_ptr()),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());
  fusion_args.beta = ElementCompute(0);
  fusion_args.bias_ptr = static_cast<ElementD const*>(bias.data_ptr());
  using KernelConfig = typename T::Config;
  arguments.hw_info.cluster_shape = KernelConfig::preferred_cluster;
  arguments.hw_info.cluster_shape_fallback = KernelConfig::fallback_cluster;
  return arguments;
}

template <typename T>
typename T::Gemm::Arguments args_from_options_per_col_residual_gate(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView bias_gate,
    int64_t M,
    int64_t N,
    int64_t K) {
  using ElementA = typename T::Gemm::ElementA;
  using ElementB = typename T::Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementD = typename T::Gemm::ElementD;
  using ElementCompute = float;
  using StrideA = typename T::StrideA;
  using StrideB = typename T::StrideB;
  using StrideC = typename T::StrideC;
  using StrideD = typename T::StrideD;
  using Sm1xxBlkScaledConfig = typename T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename T::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       static_cast<ElementD const*>(residual.data_ptr()),
       stride_C,
       static_cast<ElementD*>(D.data_ptr()),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementD const*>(gate.data_ptr());
  fusion_args.beta = ElementD(1);
  fusion_args.beta_ptr = nullptr;
  fusion_args.bias_ptr = static_cast<ElementD const*>(bias_gate.data_ptr());
  using KernelConfig = typename T::Config;
  arguments.hw_info.cluster_shape = KernelConfig::preferred_cluster;
  arguments.hw_info.cluster_shape_fallback = KernelConfig::fallback_cluster;
  return arguments;
}

template <typename T>
typename T::Gemm::Arguments args_from_options_batched_per_col_residual_gate(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView bias_gate,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t L) {
  using ElementA = typename T::Gemm::ElementA;
  using ElementB = typename T::Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementD = typename T::Gemm::ElementD;
  using StrideA = typename T::StrideA;
  using StrideB = typename T::StrideB;
  using StrideC = typename T::StrideC;
  using StrideD = typename T::StrideD;
  using Sm1xxBlkScaledConfig = typename T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  int l = static_cast<int>(L);
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, l});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, l});
  cute::get<2>(stride_B) = int64_t(0);
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, l});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, l});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, l));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, l));

  typename T::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, l},
      {// Mainloop arguments
       static_cast<ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       static_cast<ElementD const*>(residual.data_ptr()),
       stride_C,
       static_cast<ElementD*>(D.data_ptr()),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementD const*>(gate.data_ptr());
  fusion_args.dAlpha = {cute::_0{}, bool(1), static_cast<int64_t>(n)};
  fusion_args.beta = ElementD(1);
  fusion_args.beta_ptr = nullptr;
  fusion_args.bias_ptr = static_cast<ElementD const*>(bias_gate.data_ptr());
  fusion_args.dBias = {cute::_0{}, cute::_1{}, static_cast<int64_t>(n)};
  using KernelConfig = typename T::Config;
  arguments.hw_info.cluster_shape = KernelConfig::preferred_cluster;
  arguments.hw_info.cluster_shape_fallback = KernelConfig::fallback_cluster;
  return arguments;
}

template <typename T>
void runGemm(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  typename T::Gemm gemm;
  auto arguments = args_from_options<T>(D, A, B, A_sf, B_sf, alpha, m, n, k);

  size_t workspace_size = T::Gemm::get_workspace_size(arguments);
  auto workspace_tensor = alloc_workspace_tensor(workspace_size, A.device());
  void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

  CUTLASS_CHECK(gemm.can_implement(arguments));

  CUTLASS_CHECK(gemm.initialize(arguments, workspace, stream));

  CUTLASS_CHECK(gemm.run(arguments, workspace, stream));
}

template <typename T>
void runGemmBiasGelu(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView bias,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  typename T::Gemm gemm;
  auto arguments = args_from_options_bias_gelu<T>(D, A, B, A_sf, B_sf, alpha, bias, m, n, k);

  size_t workspace_size = T::Gemm::get_workspace_size(arguments);
  auto workspace_tensor = alloc_workspace_tensor(workspace_size, A.device());
  void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace, stream));
  CUTLASS_CHECK(gemm.run(arguments, workspace, stream));
}

template <typename T>
void runGemmPerColResidualGate(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView bias_gate,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  typename T::Gemm gemm;
  auto arguments =
      args_from_options_per_col_residual_gate<T>(D, A, B, A_sf, B_sf, alpha, residual, gate, bias_gate, m, n, k);

  size_t workspace_size = T::Gemm::get_workspace_size(arguments);
  auto workspace_tensor = alloc_workspace_tensor(workspace_size, A.device());
  void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace, stream));
  CUTLASS_CHECK(gemm.run(arguments, workspace, stream));
}

template <typename T>
void runGemmBatchedPerColResidualGate(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView bias_gate,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t l,
    cudaStream_t stream) {
  typename T::Gemm gemm;
  auto arguments = args_from_options_batched_per_col_residual_gate<T>(
      D, A, B, A_sf, B_sf, alpha, residual, gate, bias_gate, m, n, k, l);

  size_t workspace_size = T::Gemm::get_workspace_size(arguments);
  auto workspace_tensor = alloc_workspace_tensor(workspace_size, A.device());
  void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace, stream));
  CUTLASS_CHECK(gemm.run(arguments, workspace, stream));
}

template <typename OutType>
void cutlassFp4GemmDispatchSm100(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  if (m <= 128) {
    runGemm<Fp4GemmSm100Linear<KernelConfigM128<OutType>>>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else if (m <= 256) {
    runGemm<Fp4GemmSm100Linear<KernelConfigM256<OutType>>>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else if (m <= 1024) {
    // m in (256, 1024]: 2x4 cluster balances SM occupancy and data reuse
    runGemm<Fp4GemmSm100Linear<KernelConfigDefault<OutType>>>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else {
    // m in (1024, inf): 1x4 cluster eliminates M-tail waste for FLUX-class shapes
    runGemm<Fp4GemmSm100Linear<KernelConfigLargeM<OutType>>>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  }
}

template <>
void cutlassFp4GemmDispatchSm100<float>(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  runGemm<Fp4GemmSm100Linear<KernelConfigFp32>>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
}

template <typename OutType>
void cutlassFp4GemmBiasGeluDispatchSm100(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView bias,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  if (m <= 128) {
    runGemmBiasGelu<Fp4GemmSm100BiasGelu<KernelConfigM128<OutType>>>(D, A, B, A_sf, B_sf, alpha, bias, m, n, k, stream);
  } else if (m <= 256) {
    runGemmBiasGelu<Fp4GemmSm100BiasGelu<KernelConfigM256<OutType>>>(D, A, B, A_sf, B_sf, alpha, bias, m, n, k, stream);
  } else if (m <= 1024) {
    runGemmBiasGelu<Fp4GemmSm100BiasGelu<KernelConfigDefault<OutType>>>(
        D, A, B, A_sf, B_sf, alpha, bias, m, n, k, stream);
  } else {
    runGemmBiasGelu<Fp4GemmSm100BiasGelu<KernelConfigLargeM<OutType>>>(
        D, A, B, A_sf, B_sf, alpha, bias, m, n, k, stream);
  }
}

template <typename OutType>
void cutlassFp4GemmPerColResidualGateDispatchSm100(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView bias_gate,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  if (m <= 128) {
    runGemmPerColResidualGate<Fp4GemmSm100PerColResidualGate<KernelConfigM128<OutType>>>(
        D, A, B, A_sf, B_sf, alpha, residual, gate, bias_gate, m, n, k, stream);
  } else if (m <= 256) {
    runGemmPerColResidualGate<Fp4GemmSm100PerColResidualGate<KernelConfigM256<OutType>>>(
        D, A, B, A_sf, B_sf, alpha, residual, gate, bias_gate, m, n, k, stream);
  } else if (m <= 1024) {
    runGemmPerColResidualGate<Fp4GemmSm100PerColResidualGate<KernelConfigDefault<OutType>>>(
        D, A, B, A_sf, B_sf, alpha, residual, gate, bias_gate, m, n, k, stream);
  } else {
    runGemmPerColResidualGate<Fp4GemmSm100PerColResidualGate<KernelConfigLargeM<OutType>>>(
        D, A, B, A_sf, B_sf, alpha, residual, gate, bias_gate, m, n, k, stream);
  }
}

template <typename OutType>
void cutlassFp4GemmBatchedPerColResidualGateDispatchSm100(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView bias_gate,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t l,
    cudaStream_t stream) {
  if (m <= 128) {
    runGemmBatchedPerColResidualGate<Fp4GemmSm100PerColResidualGate<KernelConfigM128<OutType>>>(
        D, A, B, A_sf, B_sf, alpha, residual, gate, bias_gate, m, n, k, l, stream);
  } else if (m <= 256) {
    runGemmBatchedPerColResidualGate<Fp4GemmSm100PerColResidualGate<KernelConfigM256<OutType>>>(
        D, A, B, A_sf, B_sf, alpha, residual, gate, bias_gate, m, n, k, l, stream);
  } else if (m <= 1024) {
    runGemmBatchedPerColResidualGate<Fp4GemmSm100PerColResidualGate<KernelConfigDefault<OutType>>>(
        D, A, B, A_sf, B_sf, alpha, residual, gate, bias_gate, m, n, k, l, stream);
  } else {
    runGemmBatchedPerColResidualGate<Fp4GemmSm100PerColResidualGate<KernelConfigLargeM<OutType>>>(
        D, A, B, A_sf, B_sf, alpha, residual, gate, bias_gate, m, n, k, l, stream);
  }
}

#endif  // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
