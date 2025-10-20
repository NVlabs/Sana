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
from torch import nn
from torch.nn import functional as F

from .triton_lite_mla_kernels.linear_relu_fwd import linear_relu_fwd
from .triton_lite_mla_kernels.pad_vk_mm_fwd import pad_vk_mm_fwd
from .triton_lite_mla_kernels.vk_q_mm_divide_fwd import vk_q_mm_divide_fwd


class TritonLiteMLAFwdFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        qkv_weight: torch.Tensor,
        proj_weight: torch.Tensor,
        proj_bias: torch.Tensor,
        num_heads: int,
        head_dim: int,
        eps: float,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv, relu_mask = linear_relu_fwd(x, qkv_weight)  # .view(B, N, 3, C) # B, N, 3, C
        qkv, relu_mask = qkv.view(B, N, 3, C), relu_mask.view(B, N, 3, C)
        q, k, v = qkv.unbind(2)  # B, N, C
        k = k.reshape(B, N, num_heads, head_dim)
        v = v.reshape(B, N, num_heads, head_dim)
        q = q.reshape(B, N, num_heads, head_dim)
        vk = pad_vk_mm_fwd(v, k, torch.float, torch.float)
        proj_input, vk_q = vk_q_mm_divide_fwd(vk, q, eps, torch.float, x.dtype)
        proj_input = proj_input.view(B, N, C)
        y = F.linear(proj_input, proj_weight, proj_bias)
        ctx.save_for_backward(x, qkv_weight, relu_mask, v, k, vk, q, vk_q, proj_input, proj_weight)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, qkv_weight, relu_mask, v, k, vk, q, vk_q, proj_input, proj_weight = ctx.saved_tensors
        B, N, H, C1 = vk_q.shape
        C = C1 - 1

        grad_proj_weight = grad_y.reshape(-1, H * C).T @ proj_input.view(-1, H * C)
        grad_proj_bias = grad_y.sum((0, 1))
        #
        grad_proj_input = grad_y @ proj_weight
        grad_vk_q_numerator = grad_proj_input.view(B, N, H, C) / (vk_q[:, :, :, -1:] + ctx.eps)
        grad_vk_q_denominator = (
            -(grad_proj_input.view(B, N, H, C) * vk_q[:, :, :, :-1]).sum(-1, keepdim=True)
            / (vk_q[:, :, :, -1:] + ctx.eps) ** 2
        )
        grad_vk_q = torch.cat([grad_vk_q_numerator, grad_vk_q_denominator], dim=-1)

        grad_q = (grad_vk_q.permute(0, 2, 1, 3) @ vk).permute(0, 2, 1, 3)
        grad_vk = grad_vk_q.permute(0, 2, 3, 1) @ q.float().permute(0, 2, 1, 3)
        grad_q.mul_(relu_mask[:, :, 0].view(B, N, H, C))

        grad_v = (grad_vk @ k.float().permute(0, 2, 3, 1)).permute(0, 3, 1, 2)[:, :, :, :-1]
        grad_k = ((v.float().permute(0, 2, 1, 3) @ grad_vk[:, :, :-1]) + grad_vk[:, :, -1:]).permute(0, 2, 1, 3)
        grad_k.mul_(relu_mask[:, :, 1].view(B, N, H, C))

        grad_qkv = torch.stack([grad_q, grad_k, grad_v], dim=2).view(B, N, 3 * H * C).to(x.dtype)
        grad_qkv_weight = grad_qkv.view(B * N, 3 * H * C).T @ x.view(B * N, H * C)
        grad_x = grad_qkv @ qkv_weight

        return grad_x, grad_qkv_weight, grad_proj_weight, grad_proj_bias, None, None, None


class TritonLiteMLAFwd(nn.Module):
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
        return TritonLiteMLAFwdFunction.apply(
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
