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
        triton.Config({"BLOCK_SIZE_N": 256, "BLOCK_SIZE_D": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_N": 256, "BLOCK_SIZE_D": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_D": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_D": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_D": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_D": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_D": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_D": 32}, num_stages=5, num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({"BLOCK_SIZE_N": 256, "BLOCK_SIZE_D": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_D": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_D": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 256, "BLOCK_SIZE_D": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_D": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_D": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_D": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_D": 64}, num_stages=4, num_warps=4),
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
    key=["B", "N", "H", "D"],
)
@triton.jit
def vk_q_mm_divide_fwd_kernel_fp32(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    c_mid_ptr,
    # Matrix dimensions
    B,
    N,
    H,
    D,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_ab,
    stride_ah,
    stride_ac1,
    stride_ad,
    stride_bb,
    stride_bn,
    stride_bh,
    stride_bd,
    stride_cb,
    stride_cn,
    stride_ch,
    stride_cc,
    stride_cmidb,
    stride_cmidn,
    stride_cmidh,
    stride_cmidc,
    eps,
    # Meta-parameters
    BLOCK_SIZE_C1: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,  #
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_b, pid_h, pid_n = pid // num_pid_n // H, (pid // num_pid_n) % H, pid % num_pid_n

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_C1, BLOCK_SIZE_D] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_D, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_ac = tl.arange(0, BLOCK_SIZE_C1) % D
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    a_ptrs = a_ptr + (
        pid_b * stride_ab + pid_h * stride_ah + offs_ac[:, None] * stride_ac1 + offs_d[None, :] * stride_ad
    )
    a1_ptrs = a_ptr + (pid_b * stride_ab + pid_h * stride_ah + D * stride_ac1 + offs_d[:, None] * stride_ad)
    b_ptrs = b_ptr + (
        pid_b * stride_bb + pid_h * stride_bh + offs_d[:, None] * stride_bd + offs_bn[None, :] * stride_bn
    )
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_C1, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_C1, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator1 = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_d[None, :] < D - d * BLOCK_SIZE_D, other=0.0).to(tl.float32)
        a1 = tl.load(a1_ptrs, mask=offs_d[:, None] < D - d * BLOCK_SIZE_D, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=offs_d[:, None] < D - d * BLOCK_SIZE_D, other=0.0).to(tl.float32)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        accumulator1 += tl.sum(a1 * b, axis=0)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_D * stride_ad
        a1_ptrs += BLOCK_SIZE_D * stride_ad
        b_ptrs += BLOCK_SIZE_D * stride_bd
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = (accumulator / (accumulator1 + eps)).to(c_ptr.dtype.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cc = tl.arange(0, BLOCK_SIZE_C1)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cb * pid_b + stride_ch * pid_h + stride_cc * offs_cc[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cc[:, None] < D) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

    c_mid_ptrs = (
        c_mid_ptr
        + stride_cmidb * pid_b
        + stride_cmidh * pid_h
        + stride_cmidc * offs_cc[:, None]
        + stride_cmidn * offs_cn[None, :]
    )
    tl.store(c_mid_ptrs, accumulator, mask=c_mask)
    c_mid_ptrs_lastrow = (
        c_mid_ptr + stride_cmidb * pid_b + stride_cmidh * pid_h + stride_cmidc * D + stride_cmidn * offs_cn
    )
    tl.store(c_mid_ptrs_lastrow, accumulator1, mask=offs_cn < N)


def vk_q_mm_divide_fwd(
    a: torch.Tensor, b: torch.Tensor, eps: float, compute_dtype: torch.dtype, output_dtype: torch.dtype
) -> torch.Tensor:
    """
    a: (B, H, C+1, D) # C=D
    b: (B, N, H, D)
    """
    # Check constraints.
    assert a.dim() == 4 and b.dim() == 4
    assert (
        a.shape[0] == b.shape[0]
        and a.shape[1] == b.shape[2]
        and a.shape[3] == b.shape[3]
        and a.shape[2] == a.shape[3] + 1
    )

    B, N, H, D = b.shape
    # Allocates output.
    c_mid = torch.empty((B, N, H, D + 1), device=a.device, dtype=compute_dtype)
    c = torch.empty((B, N, H, D), device=a.device, dtype=output_dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (B * H * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    if compute_dtype == torch.float:
        vk_q_mm_divide_fwd_kernel_fp32[grid](
            a,
            b,
            c,
            c_mid,  #
            B,
            N,
            H,
            D,  #
            a.stride(0),
            a.stride(1),
            a.stride(2),
            a.stride(3),  #
            b.stride(0),
            b.stride(1),
            b.stride(2),
            b.stride(3),  #
            c.stride(0),
            c.stride(1),
            c.stride(2),
            c.stride(3),  #
            c_mid.stride(0),
            c_mid.stride(1),
            c_mid.stride(2),
            c_mid.stride(3),  #
            eps,
            BLOCK_SIZE_C1=triton.next_power_of_2(D),
        )
    else:
        raise NotImplementedError()
    return c, c_mid
