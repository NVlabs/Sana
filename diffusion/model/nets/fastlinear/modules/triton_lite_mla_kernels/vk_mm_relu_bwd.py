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
        triton.Config({"BLOCK_SIZE_N": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_N": 256}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_N": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 64}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_N": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 32}, num_stages=5, num_warps=2),
    ]


def get_autotune_config():
    return get_cuda_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_C1`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@custom_autotune(
    configs=get_autotune_config(),
    key=["B", "N", "H", "C"],
)
@triton.jit
def vk_mm_relu_bwd_kernel(
    # Pointers to matrices
    grad_vk_ptr,
    k_ptr,
    v_ptr,
    k_relu_mask_ptr,
    grad_k_ptr,
    grad_v_ptr,  #
    # Matrix dimensions
    B,
    N,
    H,
    C,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_vk_b,
    stride_vk_h,
    stride_vk_c1,
    stride_vk_c,
    stride_k_b,
    stride_k_n,
    stride_k_h,
    stride_k_c,
    stride_grad_k_b,
    stride_grad_k_n,
    stride_grad_k_h,
    stride_grad_k_c,
    # Meta-parameters
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  #
):
    """
    Input:
        grad_vk: (B, H, C+1, C), fp32
        k: (B, N, H, C), fp16
        v: (B, N, H, C), fp16
        k_relu_mask: (B, N, H, C), bool
    Output:
        grad_k: (B, N, H, C), fp16
        grad_v: (B, N, H, C), fp16
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    pid_b, pid_h = pid // H, pid % H

    offs_c = tl.arange(0, BLOCK_SIZE_C)
    c_mask = offs_c < C
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    grad_vk_ptrs = (
        grad_vk_ptr
        + pid_b * stride_vk_b
        + pid_h * stride_vk_h
        + offs_c[:, None] * stride_vk_c1
        + offs_c[None, :] * stride_vk_c
    )  # Cv, Ck
    grad_vk = tl.load(grad_vk_ptrs, mask=c_mask[:, None] & c_mask[None, :], other=0.0)  # Cv, Ck
    grad_vk_last_row_ptrs = (
        grad_vk_ptr + pid_b * stride_vk_b + pid_h * stride_vk_h + C * stride_vk_c1 + offs_c * stride_vk_c
    )  # Ck
    grad_vk_last_row = tl.load(grad_vk_last_row_ptrs, mask=c_mask, other=0.0)  # Ck
    k_offs = (
        pid_b * stride_k_b + pid_h * stride_k_h + offs_n[:, None] * stride_k_n + offs_c[None, :] * stride_k_c
    )  # n, C
    grad_k_offs = (
        pid_b * stride_grad_k_b
        + pid_h * stride_grad_k_h
        + offs_n[:, None] * stride_grad_k_n
        + offs_c[None, :] * stride_grad_k_c
    )  # n, C

    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        n_mask = offs_n < N - n * BLOCK_SIZE_N
        nc_mask = n_mask[:, None] & c_mask[None, :]

        k = tl.load(k_ptr + k_offs, mask=nc_mask, other=0.0).to(tl.float32)  # n, Ck
        grad_v = tl.dot(k, tl.trans(grad_vk)).to(grad_v_ptr.dtype.element_ty)  # n, Cv
        tl.store(grad_v_ptr + grad_k_offs, grad_v, mask=nc_mask)

        v = tl.load(v_ptr + k_offs, mask=nc_mask, other=0.0).to(tl.float32)  # n, Ck
        grad_k = tl.dot(v, grad_vk) + grad_vk_last_row  # n, Ck
        k_relu_mask = tl.load(k_relu_mask_ptr + k_offs, mask=nc_mask, other=0)  # n, Ck
        grad_k = tl.where(k_relu_mask, grad_k, 0).to(grad_k_ptr.dtype.element_ty)  # n, Ck
        tl.store(grad_k_ptr + grad_k_offs, grad_k, mask=nc_mask)

        k_offs += BLOCK_SIZE_N * stride_k_n
        grad_k_offs += BLOCK_SIZE_N * stride_grad_k_n


def vk_mm_relu_bwd(
    grad_vk: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_relu_mask: torch.Tensor,
    grad_k: torch.Tensor,
    grad_v: torch.Tensor,
) -> None:
    """
    Input:
        grad_vk: (B, H, C+1, C), fp32
        k: (B, N, H, C), fp16
        v: (B, N, H, C), fp16
        k_relu_mask: (B, N, H, C), bool
        grad_k: (B, N, H, C), fp16
        grad_v: (B, N, H, C), fp16
    """

    assert grad_vk.dim() == 4 and k.dim() == 4 and v.dim() == 4 and k_relu_mask.dim() == 4
    assert k.shape == v.shape == k_relu_mask.shape
    assert grad_vk.shape[0] == k.shape[0]  # B
    assert grad_vk.shape[1] == k.shape[2]  # N
    assert grad_vk.shape[2] - 1 == grad_vk.shape[3] == k.shape[3]  # C

    assert k.stride() == v.stride() == k_relu_mask.stride()

    B, N, H, C = k.shape
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (B * H,)
    vk_mm_relu_bwd_kernel[grid](
        grad_vk,
        k,
        v,
        k_relu_mask,
        grad_k,
        grad_v,  #
        B,
        N,
        H,
        C,  #
        grad_vk.stride(0),
        grad_vk.stride(1),
        grad_vk.stride(2),
        grad_vk.stride(3),  #
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),  #
        grad_k.stride(0),
        grad_k.stride(1),
        grad_k.stride(2),
        grad_k.stride(3),  #
        BLOCK_SIZE_C=triton.next_power_of_2(C),
    )
