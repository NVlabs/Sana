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
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2
        ),
        # Good config for fp8 inputs.
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
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
    key=["M", "N", "K"],
)
@triton.jit
def linear_glu_fwd_kernel(
    # Pointers to matrices
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_x_m` is how much to increase `x_ptr`
    # by to get the element one row down (A has M rows).
    stride_x_m,
    stride_x_k,  #
    stride_weight_n,
    stride_weight_k,  #
    stride_bias_n,
    stride_y_m,
    stride_y_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    """
    Input:
        x: (..., C)
        weight: (2*D, C)
        bias: (2*D,)
    Output:
        y: (..., D)
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
    # `x_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `weight_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_x_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_weight_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_x_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k)  # BLOCK_SIZE_M, BLOCK_SIZE_K
    weight_ptrs = weight_ptr + (
        offs_weight_n[None, :] * stride_weight_n + offs_k[:, None] * stride_weight_k
    )  # BLOCK_SIZE_K, BLOCK_SIZE_N
    weight_1_ptrs = weight_ptr + (
        (N + offs_weight_n[None, :]) * stride_weight_n + offs_k[:, None] * stride_weight_k
    )  # BLOCK_SIZE_K, BLOCK_SIZE_N

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator_1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        weight = tl.load(weight_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        weight_1 = tl.load(weight_1_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(x, weight, accumulator)
        accumulator_1 = tl.dot(x, weight_1, accumulator_1)
        # Advance the ptrs to the next K block.
        x_ptrs += BLOCK_SIZE_K * stride_x_k
        weight_ptrs += BLOCK_SIZE_K * stride_weight_k
        weight_1_ptrs += BLOCK_SIZE_K * stride_weight_k

    bias_ptrs = bias_ptr + (offs_weight_n * stride_bias_n)  # BLOCK_SIZE_N
    bias_1_ptrs = bias_ptr + ((N + offs_weight_n) * stride_bias_n)  # BLOCK_SIZE_N
    bias = tl.load(bias_ptrs)
    bias_1 = tl.load(bias_1_ptrs)
    accumulator += bias
    accumulator_1 += bias_1

    y = accumulator * accumulator_1 * tl.sigmoid(accumulator_1).to(y_ptr.dtype.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_y_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_y_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_offs = stride_y_m * offs_y_m[:, None] + stride_y_n * offs_y_n[None, :]
    y_mask = (offs_y_m[:, None] < M) & (offs_y_n[None, :] < N)
    tl.store(y_ptr + y_offs, y, mask=y_mask)


def linear_glu_fwd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Input:
        x: (..., C)
        weight: (2*D, C)
        bias: (2*D,)
    Output:
        y: (..., D)
    """
    assert x.dim() >= 1 and weight.dim() == 2 and bias.dim() == 1
    assert x.shape[-1] == weight.shape[-1]  # C
    assert weight.shape[0] == bias.shape[0]  # D
    assert weight.shape[0] % 2 == 0  # D
    M, K, N = torch.prod(torch.tensor(x.shape[:-1])).item(), x.shape[-1], weight.shape[0] // 2

    # Allocates output.
    y = torch.empty(x.shape[:-1] + (N,), device=x.device, dtype=x.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    if x.dtype == weight.dtype == bias.dtype:
        linear_glu_fwd_kernel[grid](
            x,
            weight,
            bias,
            y,  #
            M,
            N,
            K,  #
            x.stride(-2),
            x.stride(-1),  #
            weight.stride(0),
            weight.stride(1),  #
            bias.stride(0),
            y.stride(-2),
            y.stride(-1),
        )
    else:
        raise NotImplementedError(f"data type {x.dtype} {weight.dtype} {bias.dtype} is not support")
    return y
