# Copyright 2024 MIT Han Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from ..utils.custom_autotune import custom_autotune


def get_cuda_autotune_config():
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_H_": 8, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_H_": 8, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_H_": 4, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_H_": 2, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_H_": 4, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_H_": 1, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_H_": 1, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_H_": 2, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        # Good config for fp8 inputs.
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_H_": 8, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_H_": 4, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_H_": 2, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_H_": 8, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_H_": 4, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_H_": 2, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_H_": 4, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_H_": 1, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
    ]


def get_autotune_config():
    return get_cuda_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@custom_autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K", "H_", "C_"],
)
@triton.jit
def proj_divide_bwd_kernel(
    # Pointers to matrices
    grad_y_ptr,
    project_weight_ptr,
    vk_q_ptr,
    grad_vk_q_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    H_,
    C_,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_grad_y_m,
    stride_grad_y_k,  #
    stride_project_weight_k,
    stride_project_weight_n,  #
    stride_vk_q_m,
    stride_vk_q_h_,
    stride_vk_q_c_,
    eps,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C_: tl.constexpr,
    BLOCK_SIZE_H_: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_h_ = (pid_n * BLOCK_SIZE_H_ + tl.arange(0, BLOCK_SIZE_H_)) % H_
    offs_c_ = tl.arange(0, BLOCK_SIZE_C_)
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    # offs_hc_ = tl.reshape(offs_n, BLOCK_SIZE_H_, BLOCK_SIZE_C_)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    grad_y_ptrs = grad_y_ptr + (
        offs_m[:, None] * stride_grad_y_m + offs_k[None, :] * stride_grad_y_k
    )  # BLOCK_SIZE_M, BLOCK_SIZE_K
    project_weight_ptrs = project_weight_ptr + (
        offs_n[None, :] * stride_project_weight_n + offs_k[:, None] * stride_project_weight_k
    )  # BLOCK_SIZE_K, BLOCK_SIZE_N

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        grad_y = tl.load(grad_y_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        project_weight = tl.load(project_weight_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0).to(
            grad_y_ptr.dtype.element_ty
        )
        # We accumulate along the K dimension.
        accumulator = tl.dot(grad_y, project_weight, accumulator)
        # Advance the ptrs to the next K block.
        grad_y_ptrs += BLOCK_SIZE_K * stride_grad_y_k
        project_weight_ptrs += BLOCK_SIZE_K * stride_project_weight_k
    grad_proj_input = accumulator.to(grad_vk_q_ptr.dtype.element_ty)  # BLOCK_SIZE_M, BLOCK_SIZE_N
    grad_proj_input = tl.reshape(
        grad_proj_input, BLOCK_SIZE_M, BLOCK_SIZE_H_, BLOCK_SIZE_C_
    )  # BLOCK_SIZE_M, BLOCK_SIZE_H_, C_

    vk_q_numerator_ptrs = (
        vk_q_ptr
        + offs_m[:, None, None] * stride_vk_q_m
        + offs_h_[None, :, None] * stride_vk_q_h_
        + offs_c_[None, None, :] * stride_vk_q_c_
    )  # BLOCK_SIZE_M, BLOCK_SIZE_H_, C_
    vk_q_denominator_ptrs = (
        vk_q_ptr
        + offs_m[:, None, None] * stride_vk_q_m
        + offs_h_[None, :, None] * stride_vk_q_h_
        + BLOCK_SIZE_C_ * stride_vk_q_c_
    )  # BLOCK_SIZE_M, BLOCK_SIZE_H_, 1
    vk_q_numerator = tl.load(vk_q_numerator_ptrs)
    vk_q_denominator = tl.load(vk_q_denominator_ptrs) + eps

    grad_vk_q_numerator = grad_proj_input / vk_q_denominator
    grad_vk_q_denominator = -tl.sum(grad_vk_q_numerator * vk_q_numerator, axis=2, keep_dims=True) / vk_q_denominator

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_h_ = pid_n * BLOCK_SIZE_H_ + tl.arange(0, BLOCK_SIZE_H_)
    grad_vk_q_numerator_ptrs = (
        grad_vk_q_ptr
        + offs_m[:, None, None] * stride_vk_q_m
        + offs_h_[None, :, None] * stride_vk_q_h_
        + offs_c_[None, None, :] * stride_vk_q_c_
    )
    grad_vk_q_denominator_ptrs = (
        grad_vk_q_ptr
        + offs_m[:, None, None] * stride_vk_q_m
        + offs_h_[None, :, None] * stride_vk_q_h_
        + BLOCK_SIZE_C_ * stride_vk_q_c_
    )
    grad_vk_q_mask = (offs_m[:, None, None] < M) & (offs_h_[None, :, None] < H_)
    tl.store(grad_vk_q_numerator_ptrs, grad_vk_q_numerator, mask=grad_vk_q_mask)
    tl.store(grad_vk_q_denominator_ptrs, grad_vk_q_denominator, mask=grad_vk_q_mask)


def proj_divide_bwd(grad_y: torch.Tensor, proj_weight: torch.Tensor, vk_q: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Input:
        grad_y: (B, N, H*C)
        proj_weight: (H*C, H*C)
        vk_q: (B, N, H, C+1)
    Output:
        grad_vk_q: (B, N, H, C+1)
    """
    assert vk_q.is_contiguous()  # to ensure the stride of vk_q and grad_vk_q are the same

    assert grad_y.dim() == 3 and proj_weight.dim() == 2 and vk_q.dim() == 4
    assert grad_y.shape[0] == vk_q.shape[0]
    assert grad_y.shape[1] == vk_q.shape[1]
    assert grad_y.shape[2] == proj_weight.shape[0] == proj_weight.shape[1] == vk_q.shape[2] * (vk_q.shape[3] - 1)

    B_, N_, H_, C1_ = vk_q.shape
    C_ = C1_ - 1
    assert C_ == 32, "currently only support C=32, to ensure reduction for C in each thread"

    M, K, N = B_ * N_, H_ * C_, H_ * C_

    # Allocates output.
    grad_vk_q = torch.empty_like(vk_q)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    proj_divide_bwd_kernel[grid](
        grad_y,
        proj_weight,
        vk_q,
        grad_vk_q,  #
        M,
        N,
        K,
        H_,
        C_,  #
        grad_y.stride(1),
        grad_y.stride(2),  #
        proj_weight.stride(0),
        proj_weight.stride(1),  #
        grad_vk_q.stride(1),
        grad_vk_q.stride(2),
        grad_vk_q.stride(3),  #
        eps,
        BLOCK_SIZE_C_=C_,
    )

    return grad_vk_q
