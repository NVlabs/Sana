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
        triton.Config({"BLOCK_SIZE_C": 256, "BLOCK_SIZE_N": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_C": 256, "BLOCK_SIZE_N": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_C": 128, "BLOCK_SIZE_N": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_C": 64, "BLOCK_SIZE_N": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_C": 128, "BLOCK_SIZE_N": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_C": 32, "BLOCK_SIZE_N": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_C": 32, "BLOCK_SIZE_N": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_C": 64, "BLOCK_SIZE_N": 32}, num_stages=5, num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({"BLOCK_SIZE_C": 256, "BLOCK_SIZE_N": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_C": 128, "BLOCK_SIZE_N": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_C": 64, "BLOCK_SIZE_N": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_C": 256, "BLOCK_SIZE_N": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_C": 128, "BLOCK_SIZE_N": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_C": 64, "BLOCK_SIZE_N": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_C": 128, "BLOCK_SIZE_N": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_C": 32, "BLOCK_SIZE_N": 64}, num_stages=4, num_warps=4),
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
def pad_vk_mm_fwd_kernel_fp32_fp32(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    B,
    N,
    H,
    C,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_ab,
    stride_an,
    stride_ah,
    stride_ac,  #
    stride_bb,
    stride_bn,
    stride_bh,
    stride_bc,  #
    stride_cb,
    stride_ch,
    stride_cc1,
    stride_cc,
    # Meta-parameters
    BLOCK_SIZE_C1: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_bc = tl.cdiv(C, BLOCK_SIZE_C)
    pid_b, pid_h, pid_bc = pid // num_pid_bc // H, (pid // num_pid_bc) % H, pid % num_pid_bc

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_C1, BLOCK_SIZE_N] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_N, BLOCK_SIZE_C] pointers
    # See above `Pointer Arithmetic` section for details
    offs_ac = tl.arange(0, BLOCK_SIZE_C1) % C
    offs_bc = (pid_bc * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)) % C
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + (
        pid_b * stride_ab + pid_h * stride_ah + offs_ac[:, None] * stride_ac + offs_n[None, :] * stride_an
    )
    b_ptrs = b_ptr + (
        pid_b * stride_bb + pid_h * stride_bh + offs_n[:, None] * stride_bn + offs_bc[None, :] * stride_bc
    )
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_C1, BLOCK_SIZE_C]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_C1, BLOCK_SIZE_C), dtype=tl.float32)
    accumulator1 = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_n[None, :] < N - n * BLOCK_SIZE_N, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=offs_n[:, None] < N - n * BLOCK_SIZE_N, other=0.0).to(tl.float32)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        accumulator1 += tl.sum(b, axis=0)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_N * stride_an
        b_ptrs += BLOCK_SIZE_N * stride_bn
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator
    c1 = accumulator1

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cc1 = tl.arange(0, BLOCK_SIZE_C1)
    offs_cc = pid_bc * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    c_ptrs = (
        c_ptr + stride_cb * pid_b + stride_ch * pid_h + stride_cc1 * offs_cc1[:, None] + stride_cc * offs_cc[None, :]
    )
    c_mask = (offs_cc1[:, None] < C) & (offs_cc[None, :] < C)
    tl.store(c_ptrs, c, mask=c_mask)
    c1_ptrs = c_ptr + stride_cb * pid_b + stride_ch * pid_h + stride_cc1 * C + stride_cc * offs_cc
    c1_mask = offs_cc < C
    tl.store(c1_ptrs, c1, mask=c1_mask)


def pad_vk_mm_fwd(a, b, compute_dtype: torch.dtype, output_dtype: torch.dtype):
    """
    Input:
        v: (B, N, H, C)
        k: (B, N, H, C)
    Output:
        vk: (B, H, C+1, C)
    """
    # Check constraints.
    assert a.dim() == 4 and b.dim() == 4
    assert a.shape == b.shape, "Incompatible dimensions"
    B, N, H, C = a.shape
    # Allocates output.
    c = torch.empty((B, H, C + 1, C), device=a.device, dtype=output_dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (B * H * triton.cdiv(C, META["BLOCK_SIZE_C"]),)
    if compute_dtype == torch.float and output_dtype == torch.float:
        pad_vk_mm_fwd_kernel_fp32_fp32[grid](
            a,
            b,
            c,  #
            B,
            N,
            H,
            C,  #
            a.stride(-4),
            a.stride(-3),
            a.stride(-2),
            a.stride(-1),  #
            b.stride(-4),
            b.stride(-3),
            b.stride(-2),
            b.stride(-1),  #
            c.stride(-4),
            c.stride(-3),
            c.stride(-2),
            c.stride(-1),  #
            BLOCK_SIZE_C1=triton.next_power_of_2(C),
        )
    else:
        raise NotImplementedError()
    return c
