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

import os
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class LiteMLA(nn.Module):
    r"""Lightweight multiscale linear attention"""

    PAD_VAL = 1

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=32,
        kernel_func="relu",
        scales: Optional[Tuple[int]] = (5,),
        eps=1e-15,
        use_bias=False,
        norm=(None, "bn2d"),
        act=(None, None),
    ):
        heads = heads or int(out_dim // dim * heads_ratio)
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = dim
        self.scales = scales
        self.eps = eps

        self.aggreg = None
        scales = ()
        self.kernel_func = nn.ReLU(inplace=False)

        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=use_bias)
        self.proj = nn.Linear(out_dim, out_dim)

    @torch.cuda.amp.autocast(enabled=os.environ.get("AUTOCAST_LINEAR_ATTN", False) == "true")
    def attn_matmul(self, q, k, v: torch.Tensor) -> torch.Tensor:
        # lightweight linear attention
        q = self.kernel_func(q)  # B, h, h_d, N
        k = self.kernel_func(k)

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=LiteMLA.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)
        if out.dtype in [torch.float16, torch.bfloat16]:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(0, 2, 3, 1)
        # B, 3, C, N --> B, C, N
        q, k, v = qkv.unbind(1)
        dtype = q.dtype

        q = q.reshape(B, C // self.dim, self.dim, N)  # b, h, h_d, N
        k = k.reshape(B, C // self.dim, self.dim, N).transpose(-1, -2)  # b, h, N, h_d
        v = v.reshape(B, C // self.dim, self.dim, N)  # b, h, h_d, N

        out = self.attn_matmul(q, k, v).to(dtype)

        out = out.view(B, C, N).permute(0, 2, 1)  # B, N, C
        out = self.proj(out)

        return out

    @property
    def module_str(self) -> str:
        _str = type(self).__name__ + "("
        eps = f"{self.eps:.1E}"
        _str += f"i={self.in_dim},o={self.out_dim},h={self.heads},d={self.dim},eps={eps}"
        return _str

    def __repr__(self):
        return f"EPS{self.eps}-" + super().__repr__()
