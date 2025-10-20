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

# from ..utils.custom_autotune import custom_autotune


def get_cuda_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE_H": 128, "BLOCK_SIZE_W": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_H": 64, "BLOCK_SIZE_W": 256}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_H": 128, "BLOCK_SIZE_W": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_H": 128, "BLOCK_SIZE_W": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_H": 64, "BLOCK_SIZE_W": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_H": 128, "BLOCK_SIZE_W": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_H": 64, "BLOCK_SIZE_W": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_H": 32, "BLOCK_SIZE_W": 64}, num_stages=5, num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({"BLOCK_SIZE_H": 128, "BLOCK_SIZE_W": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_H": 256, "BLOCK_SIZE_W": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_H": 256, "BLOCK_SIZE_W": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_H": 64, "BLOCK_SIZE_W": 256}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_H": 128, "BLOCK_SIZE_W": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_H": 128, "BLOCK_SIZE_W": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_H": 64, "BLOCK_SIZE_W": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_H": 128, "BLOCK_SIZE_W": 32}, num_stages=4, num_warps=4),
    ]


def get_autotune_config():
    return get_cuda_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_H`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=["B", "H", "W", "C", "K"],
)
@triton.jit
def depthwise_conv_fwd_kernel(
    # Pointers to matrices
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    # Matrix dimensions
    B,
    H,
    W,
    C,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_x_m` is how much to increase `x_ptr`
    # by to get the element one row down (A has M rows).
    stride_x_b,
    stride_x_h,
    stride_x_w,
    stride_x_c,  #
    stride_weight_c,
    stride_weight_k1,
    stride_weight_k2,  #
    stride_bias_c,
    # Meta-parameters
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,  #
):
    """
    Input:
        x: (B, H, W, C)
        weight: (C, K, K)
        bias: (C,)
    Output:
        y: (B, H, W, C)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_h = tl.cdiv(H, BLOCK_SIZE_H)
    num_pid_w = tl.cdiv(W, BLOCK_SIZE_W)
    pid_bc, pid_hw = pid // (num_pid_h * num_pid_w), pid % (num_pid_h * num_pid_w)
    pid_b, pid_c, pid_h, pid_w = pid_bc // C, pid_bc % C, pid_hw // num_pid_w, pid_hw % num_pid_w

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    offs_xy = (
        pid_b * stride_x_b + offs_h[:, None] * stride_x_h + offs_w[None, :] * stride_x_w + pid_c * stride_x_c
    )  # BLOCK_SIZE_H, BLOCK_SIZE_W

    K_2 = K // 2
    accumulator = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    for kh in range(-K_2, K_2 + 1):
        mask_h = (offs_h >= -kh) & (offs_h < H - kh)
        for kw in range(-K_2, K_2 + 1):
            mask_w = (offs_w >= -kw) & (offs_w < W - kw)
            weight = tl.load(
                weight_ptr + pid_c * stride_weight_c + (kh + K_2) * stride_weight_k1 + (kw + K_2) * stride_weight_k2
            )
            x = tl.load(
                x_ptr + offs_xy + kh * stride_x_h + kw * stride_x_w, mask=mask_h[:, None] & mask_w[None, :], other=0.0
            )
            accumulator += weight * x
    bias = tl.load(bias_ptr + pid_c * stride_bias_c)
    y = (accumulator + bias).to(y_ptr.dtype.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    y_mask = (offs_h[:, None] < H) & (offs_w[None, :] < W)
    tl.store(y_ptr + offs_xy, y, mask=y_mask)


def depthwise_conv_fwd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Input:
        x: (B, H, W, C)
        weight: (C, K, K)
        bias: (C,)
    Output:
        y: (B, H, W, C)
    """
    assert x.dim() == 4 and weight.dim() == 3 and bias.dim() == 1
    assert x.shape[-1] == weight.shape[0] == bias.shape[0]  # C
    assert weight.shape[1] == weight.shape[2]  # K
    B, H, W, C = x.shape
    K = weight.shape[1]

    # Allocates output.
    y = torch.empty_like(x)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (B * C * triton.cdiv(H, META["BLOCK_SIZE_H"]) * triton.cdiv(W, META["BLOCK_SIZE_W"]),)
    if x.dtype == weight.dtype == bias.dtype:
        depthwise_conv_fwd_kernel[grid](
            x,
            weight,
            bias,
            y,  #
            B,
            H,
            W,
            C,
            K,  #
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),  #
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),  #
            bias.stride(0),
        )
    else:
        raise NotImplementedError(f"data type {x.dtype} {weight.dtype} {bias.dtype} is not support")
    return y


def debug():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.manual_seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda")
    dtype = torch.float16

    conv = torch.nn.Conv2d(
        in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512, device=device, dtype=dtype
    )
    x = torch.randn(16, 512, 32, 32, device=device, dtype=dtype).to(memory_format=torch.channels_last)
    ref_y = conv(x)
    tri_y = depthwise_conv_fwd(x.permute(0, 2, 3, 1), conv.weight[:, 0], conv.bias).permute(0, 3, 1, 2)


if __name__ == "__main__":
    debug()

"""
python -m modules.depthwise_conv_fwd
"""
