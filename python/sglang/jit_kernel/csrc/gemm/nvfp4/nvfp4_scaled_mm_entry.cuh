/* Copyright 2025 SGLang Team. All Rights Reserved.

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

#include <sgl_kernel/tensor.h>

void cutlass_scaled_fp4_mm_sm100a_sm120a(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha);

void cutlass_scaled_fp4_mm_bias_gelu_sm100a_sm120a(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView bias);

void cutlass_scaled_fp4_mm_per_col_residual_gate_sm100a_sm120a(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView bias_gate);

void cutlass_scaled_fp4_mm_batched_per_col_residual_gate_sm100a_sm120a(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView bias_gate,
    int64_t batch_size,
    int64_t m_per_batch);

void cutlass_scaled_fp4_mm(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha) {
  cutlass_scaled_fp4_mm_sm100a_sm120a(D, A, B, A_sf, B_sf, alpha);
}

void cutlass_scaled_fp4_mm_bias_gelu(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView bias) {
  cutlass_scaled_fp4_mm_bias_gelu_sm100a_sm120a(D, A, B, A_sf, B_sf, alpha, bias);
}

void cutlass_scaled_fp4_mm_per_col_residual_gate(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView bias_gate) {
  cutlass_scaled_fp4_mm_per_col_residual_gate_sm100a_sm120a(D, A, B, A_sf, B_sf, alpha, residual, gate, bias_gate);
}

void cutlass_scaled_fp4_mm_batched_per_col_residual_gate(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView bias_gate,
    int64_t batch_size,
    int64_t m_per_batch) {
  cutlass_scaled_fp4_mm_batched_per_col_residual_gate_sm100a_sm120a(
      D, A, B, A_sf, B_sf, alpha, residual, gate, bias_gate, batch_size, m_per_batch);
}
