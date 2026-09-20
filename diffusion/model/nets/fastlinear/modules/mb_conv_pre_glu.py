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

from .nn.act import build_act, get_act_name
from .nn.conv import ConvLayer
from .nn.norm import build_norm, get_norm_name
from .utils.model import get_same_padding, val2tuple


class MBConvPreGLU(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size=3,
        stride=1,
        mid_dim=None,
        expand=6,
        padding: int or None = None,
        use_bias=False,
        norm=(None, None, "ln2d"),
        act=("silu", "silu", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)

        mid_dim = mid_dim or round(in_dim * expand)

        self.inverted_conv = ConvLayer(
            in_dim,
            mid_dim * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act=None,
        )
        self.glu_act = build_act(act[0], inplace=False)
        self.depth_conv = ConvLayer(
            mid_dim,
            mid_dim,
            kernel_size,
            stride=stride,
            groups=mid_dim,
            padding=padding,
            use_bias=use_bias[1],
            norm=norm[1],
            act=act[1],
        )
        self.point_conv = ConvLayer(
            mid_dim,
            out_dim,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act=act[2],
        )

    def forward(self, x: torch.Tensor, HW=None) -> torch.Tensor:
        B, N, C = x.shape
        if HW is None:
            H = W = int(N**0.5)
        else:
            H, W = HW

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        x = self.inverted_conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.depth_conv(x)
        x = self.point_conv(x)

        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x

    @property
    def module_str(self) -> str:
        _str = f"{self.depth_conv.kernel_size}{type(self).__name__}("
        _str += f"in={self.inverted_conv.in_dim},mid={self.depth_conv.in_dim},out={self.point_conv.out_dim},s={self.depth_conv.stride}"
        _str += (
            f",norm={get_norm_name(self.inverted_conv.norm)}"
            f"+{get_norm_name(self.depth_conv.norm)}"
            f"+{get_norm_name(self.point_conv.norm)}"
        )
        _str += (
            f",act={get_act_name(self.inverted_conv.act)}"
            f"+{get_act_name(self.depth_conv.act)}"
            f"+{get_act_name(self.point_conv.act)}"
        )
        _str += f",glu_act={get_act_name(self.glu_act)})"
        return _str
