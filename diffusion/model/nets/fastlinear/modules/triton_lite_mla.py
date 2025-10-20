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

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .triton_lite_mla_kernels.linear_relu_fwd import linear_relu_fwd
from .triton_lite_mla_kernels.mm import matmul  # for autocast
from .triton_lite_mla_kernels.pad_vk_mm_fwd import pad_vk_mm_fwd
from .triton_lite_mla_kernels.proj_divide_bwd import proj_divide_bwd
from .triton_lite_mla_kernels.vk_mm_relu_bwd import vk_mm_relu_bwd
from .triton_lite_mla_kernels.vk_q_mm_divide_fwd import vk_q_mm_divide_fwd
from .triton_lite_mla_kernels.vk_q_mm_relu_bwd import vk_q_mm_relu_bwd


class TritonLiteMLAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        qkv_weight: torch.Tensor,
        proj_weight: torch.Tensor,
        proj_bias: Optional[torch.Tensor],
        num_heads: int,
        head_dim: int,
        eps: float,
    ) -> torch.Tensor:
        ctx.x_dtype, ctx.qkv_weight_dtype, ctx.proj_dtype = x.dtype, qkv_weight.dtype, proj_weight.dtype
        if torch.is_autocast_enabled():
            autocast_dtype = torch.get_autocast_gpu_dtype()
            x = x.to(autocast_dtype)
            qkv_weight = qkv_weight.to(autocast_dtype)
            proj_weight = proj_weight.to(autocast_dtype)
            if proj_bias is not None:
                proj_bias = proj_bias.to(autocast_dtype)
        B, N, C = x.shape
        qkv, relu_mask = linear_relu_fwd(x, qkv_weight)  # B, N, 3*C. autocast is processed here
        qkv, relu_mask = qkv.view(B, N, 3, C), relu_mask.view(B, N, 3, C)
        q, k, v = qkv.unbind(2)  # B, N, C
        k = k.reshape(B, N, num_heads, head_dim)
        v = v.reshape(B, N, num_heads, head_dim)
        q = q.reshape(B, N, num_heads, head_dim)
        vk = pad_vk_mm_fwd(v, k, torch.float, torch.float)
        proj_input, vk_q = vk_q_mm_divide_fwd(vk, q, eps, torch.float, qkv.dtype)
        proj_input = proj_input.view(B, N, C)
        y = F.linear(proj_input, proj_weight, proj_bias)
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            ctx.save_for_backward(x, qkv_weight, relu_mask, v, k, vk, q, vk_q, proj_input, proj_weight)
            ctx.eps = eps
        if torch.get_autocast_gpu_dtype() == torch.float16:
            y = y.clip(-65504, 65504)
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, qkv_weight, relu_mask, v, k, vk, q, vk_q, proj_input, proj_weight = ctx.saved_tensors
        B, N, H, C1 = vk_q.shape
        C = C1 - 1

        grad_proj_weight = (
            (grad_y.reshape(-1, H * C).T @ proj_input.view(-1, H * C)).to(ctx.proj_dtype)
            if ctx.needs_input_grad[2]
            else None
        )
        grad_proj_bias = grad_y.sum((0, 1)).to(ctx.proj_dtype) if ctx.needs_input_grad[3] else None
        #
        grad_vk_q = proj_divide_bwd(grad_y, proj_weight, vk_q, ctx.eps)
        del grad_y, vk_q

        grad_qkv = torch.empty(B, N, 3, H, C, dtype=q.dtype, device=q.device)
        grad_vk = vk_q_mm_relu_bwd(grad_vk_q, vk, q, relu_mask[:, :, 0].view(B, N, H, C), grad_qkv[:, :, 0])
        del grad_vk_q, vk

        vk_mm_relu_bwd(grad_vk, k, v, relu_mask[:, :, 1].view(B, N, H, C), grad_qkv[:, :, 1], grad_qkv[:, :, 2])
        del grad_vk, q, k, v, relu_mask

        grad_qkv_weight = (
            (grad_qkv.view(B * N, 3 * H * C).T @ x.view(B * N, H * C)).to(ctx.qkv_weight_dtype)
            if ctx.needs_input_grad[1]
            else None
        )
        grad_x = (grad_qkv.view(B, N, 3 * H * C) @ qkv_weight).to(ctx.x_dtype) if ctx.needs_input_grad[0] else None
        del grad_qkv

        return grad_x, grad_qkv_weight, grad_proj_weight, grad_proj_bias, None, None, None


class TritonLiteMLA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps=1e-15,
        use_bias=False,
    ):
        super().__init__()
        self.dim, self.num_heads, self.head_dim, self.eps = dim, num_heads, dim // num_heads, eps
        if use_bias:
            raise NotImplementedError(f"use_bias is not supported for TritonLiteMLA")
        self.qkv = nn.Linear(dim, dim * 3, bias=use_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask=None, HW=None, block_id=None) -> torch.Tensor:
        return TritonLiteMLAFunction.apply(
            x, self.qkv.weight, self.proj.weight, self.proj.bias, self.num_heads, self.head_dim, self.eps
        )

    @property
    def module_str(self) -> str:
        _str = type(self).__name__ + "("
        eps = f"{self.eps:.1E}"
        _str += f"i={self.in_dim},o={self.out_dim},h={self.heads},d={self.dim},eps={eps}"
        return _str

    def __repr__(self):
        return f"EPS{self.eps}-" + super().__repr__()
