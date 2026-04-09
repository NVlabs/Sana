# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
GLUMBConv with 1x1 Conv replaced by Linear layers for better efficiency.
The 1x1 Conv2d is mathematically equivalent to Linear layer.
"""


import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp

from diffusion.model.act import build_act, get_act_name
from diffusion.model.norms import build_norm, get_norm_name
from diffusion.model.utils import get_same_padding, val2tuple


class LinearLayer(nn.Module):
    """Linear layer with optional normalization and activation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bias=True,
        norm=None,
        act=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias

        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.norm = build_norm(norm, num_features=out_dim)
        self.act = build_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ConvLayer(nn.Module):
    """Conv layer for depthwise convolution (kernel_size > 1)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        padding: int or None = None,
        use_bias=False,
        dropout=0.0,
        conv_type="2d",
        norm="bn2d",
        act="relu",
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.use_bias = use_bias

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        if conv_type == "2d":
            self.conv = nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=(kernel_size, kernel_size),
                stride=(stride, stride),
                padding=padding,
                dilation=(dilation, dilation),
                groups=groups,
                bias=use_bias,
            )
        elif conv_type == "3d":
            self.conv = nn.Conv3d(
                in_dim,
                out_dim,
                kernel_size=(kernel_size, kernel_size, kernel_size),
                stride=(stride, stride, stride),
                padding=padding,
                dilation=(dilation, dilation, dilation),
                groups=groups,
                bias=use_bias,
            )
        else:
            self.conv = None

        self.norm = build_norm(norm, num_features=out_dim)
        self.act = build_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class GLUMBConvLinear(nn.Module):
    """
    GLUMBConv with 1x1 Conv replaced by Linear layers.

    Original GLUMBConv structure:
    - inverted_conv: Conv2d(in_features, hidden_features*2, kernel=1x1) -> replaced with Linear
    - depth_conv: Conv2d(hidden_features*2, hidden_features*2, kernel=3x3, groups=hidden_features*2) -> keep Conv
    - point_conv: Conv2d(hidden_features, out_features, kernel=1x1) -> replaced with Linear

    Weight mapping (for checkpoint conversion):
    - inverted_conv.conv.weight: [out, in, 1, 1] -> inverted_conv.linear.weight: [out, in]
    - inverted_conv.conv.bias: [out] -> inverted_conv.linear.bias: [out]
    - point_conv.conv.weight: [out, in, 1, 1] -> point_conv.linear.weight: [out, in]
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_feature=None,
        kernel_size=3,
        stride=1,
        padding: int or None = None,
        use_bias=False,
        norm=(None, None, None),
        act=("silu", "silu", None),
        dilation=1,
    ):
        out_feature = out_feature or in_features
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)

        self.hidden_features = hidden_features
        self.out_feature = out_feature

        self.glu_act = build_act(act[1], inplace=False)

        # Replace 1x1 conv with Linear
        self.inverted_conv = LinearLayer(
            in_features,
            hidden_features * 2,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
        )

        # Keep depthwise conv (kernel_size=3)
        self.depth_conv = ConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size,
            stride=stride,
            groups=hidden_features * 2,
            padding=padding,
            use_bias=use_bias[1],
            norm=norm[1],
            act=None,
            dilation=dilation,
        )

        # Replace 1x1 conv with Linear
        self.point_conv = LinearLayer(
            hidden_features,
            out_feature,
            use_bias=use_bias[2],
            norm=norm[2],
            act=act[2],
        )

    def forward(self, x: torch.Tensor, HW=None) -> torch.Tensor:
        B, N, C = x.shape
        if HW is None:
            H = W = int(N**0.5)
        elif len(HW) == 2:
            H, W = HW
        elif len(HW) == 3:
            T, H, W = HW

        # inverted_conv: Linear on sequence dimension (B, N, C) -> (B, N, hidden*2)
        x = self.inverted_conv(x)

        # Reshape for depthwise conv: (B, N, hidden*2) -> (B*T or B, hidden*2, H, W)
        if HW is not None and len(HW) == 3:
            x = x.reshape(B * T, H, W, -1).permute(0, 3, 1, 2)
        else:
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # depth_conv: keep as Conv2d (depthwise 3x3)
        x = self.depth_conv(x)

        # GLU activation
        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        # Reshape back to sequence for point_conv: (B*T or B, hidden, H, W) -> (B, N, hidden)
        if HW is not None and len(HW) == 3:
            x = x.reshape(B * T, self.hidden_features, H * W).permute(0, 2, 1)
            x = x.reshape(B, N, self.hidden_features)
        else:
            x = x.reshape(B, self.hidden_features, N).permute(0, 2, 1)

        # point_conv: Linear on sequence dimension (B, N, hidden) -> (B, N, out)
        x = self.point_conv(x)

        return x


class GLUMBConvLinearTemp(GLUMBConvLinear):
    """GLUMBConvLinear with temporal convolution support."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_feature=None,
        kernel_size=3,
        stride=1,
        padding: int or None = None,
        use_bias=False,
        norm=(None, None, None),
        act=("silu", "silu", None),
        t_kernel_size=3,
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_feature=out_feature,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bias=use_bias,
            norm=norm,
            act=act,
        )

        out_feature = out_feature or in_features
        t_padding = t_kernel_size // 2
        self.t_conv = nn.Conv2d(
            out_feature,
            out_feature,
            kernel_size=(t_kernel_size, 1),
            stride=1,
            padding=(t_padding, 0),
            bias=False,
        )

        nn.init.zeros_(self.t_conv.weight)

    def forward(self, x: torch.Tensor, HW=None, **kwargs) -> torch.Tensor:
        B, N, C = x.shape

        assert len(HW) == 3, "HW must be a tuple of (T, H, W)"
        T, H, W = HW

        # inverted_conv: Linear (B, N, C) -> (B, N, hidden*2)
        x = self.inverted_conv(x)

        # Reshape for depth_conv: (B, T*H*W, hidden*2) -> (B*T, hidden*2, H, W)
        x = x.reshape(B * T, H, W, -1).permute(0, 3, 1, 2)
        x = self.depth_conv(x)

        # Space aggregation (GLU)
        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        # Reshape for point_conv: (B*T, hidden, H, W) -> (B*T, H*W, hidden)
        x = x.reshape(B * T, self.hidden_features, H * W).permute(0, 2, 1)
        x = self.point_conv(x)  # (B*T, H*W, out_feature)

        # Temporal aggregation: reshape for t_conv
        x = x.reshape(B, T, H * W, self.out_feature).permute(0, 3, 1, 2)  # B, out, T, H*W
        x_out = x + self.t_conv(x)

        x_out = x_out.permute(0, 2, 3, 1).reshape(B, N, self.out_feature)

        return x_out


if __name__ == "__main__":
    # Test GLUMBConvLinear
    model = GLUMBConvLinear(
        2240,
        2240 * 5 // 2,  # hidden_features = 5600
        2240,
        use_bias=(True, True, False),
        norm=(None, None, None),
        act=("silu", "silu", None),
    ).cuda()

    input_tensor = torch.randn(4, 1024, 2240).cuda()  # (B, H*W, C)
    output = model(input_tensor, HW=(32, 32))
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
