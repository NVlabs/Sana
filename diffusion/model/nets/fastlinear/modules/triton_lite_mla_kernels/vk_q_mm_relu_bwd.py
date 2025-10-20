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
def vk_q_mm_relu_bwd_kernel(
    # Pointers to matrices
    grad_vk_q_ptr,
    vk_ptr,
    q_ptr,
    q_relu_mask_ptr,
    grad_vk_ptr,
    grad_q_ptr,  #
    # Matrix dimensions
    B,
    N,
    H,
    C,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_vk_q_b,
    stride_vk_q_n,
    stride_vk_q_h,
    stride_vk_q_c1,
    stride_vk_b,
    stride_vk_h,
    stride_vk_c1,
    stride_vk_c,
    stride_q_b,
    stride_q_n,
    stride_q_h,
    stride_q_c,
    stride_grad_q_b,
    stride_grad_q_n,
    stride_grad_q_h,
    stride_grad_q_c,
    # Meta-parameters
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_C1: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  #
):
    """
    Input:
        grad_vk_q: (B, N, H, C+1), fp32
        vk: (B, H, C+1, C), fp32
        q: (B, N, H, C), fp16
        q_relu_mask: (B, N, H, C), bool
    Output:
        grad_vk: (B, H, C+1, C), fp32
        grad_q: (B, N, H, C), fp16
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    pid_b, pid_h = pid // H, pid % H

    offs_c = tl.arange(0, BLOCK_SIZE_C)
    c_mask = offs_c < C
    offs_c1 = tl.arange(0, BLOCK_SIZE_C1)
    c1_mask = offs_c1 < C + 1
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    # n_mask = offs_n < N
    grad_vk_q_ptrs = (
        grad_vk_q_ptr
        + pid_b * stride_vk_q_b
        + pid_h * stride_vk_q_h
        + offs_n[:, None] * stride_vk_q_n
        + offs_c1[None, :] * stride_vk_q_c1
    )  # n, C1
    vk_offs = (
        pid_b * stride_vk_b + pid_h * stride_vk_h + offs_c1[:, None] * stride_vk_c1 + offs_c[None, :] * stride_vk_c
    )  # C1, C
    q_offs = (
        pid_b * stride_q_b + pid_h * stride_q_h + offs_c[:, None] * stride_q_c + offs_n[None, :] * stride_q_n
    )  # C, n
    grad_q_offs = (
        pid_b * stride_grad_q_b
        + pid_h * stride_grad_q_h
        + offs_c[:, None] * stride_grad_q_c
        + offs_n[None, :] * stride_grad_q_n
    )  # C, n

    vk = tl.load(vk_ptr + vk_offs, mask=c1_mask[:, None] & c_mask[None, :], other=0.0)  # C1, C
    grad_vk = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_C1), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        n_mask = offs_n < N - n * BLOCK_SIZE_N

        grad_vk_q = tl.load(grad_vk_q_ptrs, mask=n_mask[:, None] & c1_mask[None, :], other=0.0)  # n, C1
        q = tl.load(q_ptr + q_offs, mask=c_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)  # C, n
        q_relu_mask = tl.load(q_relu_mask_ptr + q_offs, mask=c_mask[:, None] & n_mask[None, :], other=0)  # C, n

        grad_q = tl.trans(tl.dot(grad_vk_q, vk))  # n, C -> C, n
        grad_q = tl.where(q_relu_mask, grad_q, 0).to(grad_q_ptr.dtype.element_ty)  # C, n
        grad_vk = tl.dot(q, grad_vk_q, grad_vk)

        tl.store(grad_q_ptr + grad_q_offs, grad_q, mask=c_mask[:, None] & n_mask[None, :])

        grad_vk_q_ptrs += BLOCK_SIZE_N * stride_vk_q_n
        q_offs += BLOCK_SIZE_N * stride_q_n
        grad_q_offs += BLOCK_SIZE_N * stride_grad_q_n

    tl.store(grad_vk_ptr + vk_offs, tl.trans(grad_vk), mask=c1_mask[:, None] & c_mask[None, :])


def vk_q_mm_relu_bwd(
    grad_vk_q: torch.Tensor, vk: torch.Tensor, q: torch.Tensor, q_relu_mask: torch.Tensor, grad_q: torch.Tensor
) -> torch.Tensor:
    """
    Input:
        grad_vk_q: (B, N, H, C+1), fp32
        vk: (B, H, C+1, C), fp32
        q: (B, N, H, C), fp16
        q_relu_mask: (B, N, H, C), bool
        grad_q: (B, N, H, C), fp16
    Output:
        grad_vk: (B, H, C+1, C), fp32
    """

    assert grad_vk_q.dim() == 4 and vk.dim() == 4 and q.dim() == 4 and q_relu_mask.dim() == 4
    assert q.shape == q_relu_mask.shape
    assert grad_vk_q.shape[0] == vk.shape[0] == q.shape[0]  # B
    assert grad_vk_q.shape[1] == q.shape[1]  # N
    assert grad_vk_q.shape[2] == vk.shape[1] == q.shape[2]  # N
    assert grad_vk_q.shape[3] - 1 == vk.shape[2] - 1 == vk.shape[3] == q.shape[3]  # C

    B, N, H, C = q.shape
    # Allocates output.
    grad_vk = torch.empty_like(vk)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (B * H,)
    vk_q_mm_relu_bwd_kernel[grid](
        grad_vk_q,
        vk,
        q,
        q_relu_mask,
        grad_vk,
        grad_q,  #
        B,
        N,
        H,
        C,  #
        grad_vk_q.stride(0),
        grad_vk_q.stride(1),
        grad_vk_q.stride(2),
        grad_vk_q.stride(3),  #
        vk.stride(0),
        vk.stride(1),
        vk.stride(2),
        vk.stride(3),  #
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),  #
        grad_q.stride(0),
        grad_q.stride(1),
        grad_q.stride(2),
        grad_q.stride(3),  #
        BLOCK_SIZE_C=triton.next_power_of_2(C),
        BLOCK_SIZE_C1=triton.next_power_of_2(C + 1),
    )

    return grad_vk
