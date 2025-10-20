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

from typing import List

# This file is modified from https://github.com/PixArt-alpha/PixArt-sigma
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp

from diffusion.model.act import build_act, get_act_name
from diffusion.model.norms import build_norm, get_norm_name
from diffusion.model.utils import get_same_padding, val2tuple


class ConvLayer(nn.Module):
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


class GLUMBConv(nn.Module):
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

        self.glu_act = build_act(act[1], inplace=False)
        self.inverted_conv = ConvLayer(
            in_features,
            hidden_features * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
        )
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
        self.point_conv = ConvLayer(
            hidden_features,
            out_feature,
            1,
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
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        elif len(HW) == 3:
            T, H, W = HW
            x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)

        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        if len(HW) == 3:
            x = x.reshape(B * T, C, H * W).permute(0, 2, 1)
            x = x.reshape(B, N, C)
        else:
            x = x.reshape(B, C, N).permute(0, 2, 1)

        return x


class GLUMBConvTemp(GLUMBConv):
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
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)

        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        # Space aggregation
        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate
        x = self.point_conv(x)

        # Temporal aggregation
        x_reshaped = x.view(B, T, C, H * W).permute(0, 2, 1, 3)
        x_out = x_reshaped + self.t_conv(x_reshaped)

        x_out = x_out.permute(0, 2, 3, 1).reshape(B, N, C)

        return x_out


class ChunkGLUMBConvTemp(GLUMBConvTemp):
    def forward(self, x: torch.Tensor, HW=None, chunk_index: List[int] = [0]) -> torch.Tensor:
        B, N, C = x.shape

        assert len(HW) == 3, "HW must be a tuple of (T, H, W)"
        T, H, W = HW
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)

        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        # Space aggregation
        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate
        x = self.point_conv(x)

        # Temporal aggregation
        x_reshaped = x.view(B, T, C, H * W).permute(0, 2, 1, 3)  # B, C, T, H*W
        padding_size = self.t_conv.kernel_size[0] // 2
        # add the last chunk index
        chunk_index = chunk_index[:]
        chunk_index.append(T)
        chunk_sizes = torch.diff(torch.tensor(chunk_index)).tolist()  # [f1, f2-f1, f3-f2, ...]
        x_reshaped_list = x_reshaped.split(chunk_sizes, dim=-2)
        # for the first chunk, padding padding_size zero to the right
        # for the other chunks, padding padding_size zero to the right, padding the padding_size items in the last chunk to the left
        padded_x_reshaped_list = []
        padded_x_reshaped_list.append(
            torch.cat(
                [x_reshaped_list[0], torch.zeros(B, C, padding_size, H * W).to(x_reshaped.device, x_reshaped.dtype)],
                dim=-2,
            )
        )
        for i in range(1, len(x_reshaped_list)):
            prev_chunk = x_reshaped_list[i - 1][
                :, :, -padding_size:, :
            ]  # .detach() seems not necessary, since we will drop it
            cur_chunk = x_reshaped_list[i]
            padded_x_reshaped_list.append(
                torch.cat(
                    [
                        prev_chunk,
                        cur_chunk,
                        torch.zeros(B, C, padding_size, H * W).to(x_reshaped.device, x_reshaped.dtype),
                    ],
                    dim=-2,
                )
            )
        x_reshaped_t_conv = torch.cat(padded_x_reshaped_list, dim=-2)
        t_conv_out = self.t_conv(x_reshaped_t_conv)

        # Remove padding from the output
        # Calculate the expected output size after convolution
        padded_chunk_sizes = []
        padded_chunk_sizes.append(chunk_sizes[0] + padding_size)  # First chunk: original + right padding
        for i in range(1, len(chunk_sizes)):
            padded_chunk_sizes.append(
                padding_size + chunk_sizes[i] + padding_size
            )  # Other chunks: left + original + right padding

        # After convolution, the output size depends on the convolution parameters
        # For typical temporal convolution with same padding, output size should match input size
        # Split the convolved output back into chunks
        t_conv_out_list = t_conv_out.split(padded_chunk_sizes, dim=-2)

        # Remove padding from each chunk
        unpadded_chunks = []
        for i, chunk in enumerate(t_conv_out_list):
            if i == 0:
                # First chunk: remove right padding
                unpadded_chunk = chunk[:, :, : chunk_sizes[i], :]
            else:
                # Other chunks: remove left and right padding
                start_idx = padding_size
                end_idx = start_idx + chunk_sizes[i]
                unpadded_chunk = chunk[:, :, start_idx:end_idx, :]
            unpadded_chunks.append(unpadded_chunk)

        # Concatenate the unpadded chunks
        t_conv_out_final = torch.cat(unpadded_chunks, dim=-2)

        # Verify the output has the correct temporal dimension
        assert t_conv_out_final.shape[-2] == T, f"Expected temporal dimension {T}, got {t_conv_out_final.shape[-2]}"

        x_out = x_reshaped + t_conv_out_final

        x_out = x_out.permute(0, 2, 3, 1).reshape(B, N, C)

        return x_out


class CachedGLUMBConvTemp(GLUMBConvTemp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_cache = None

    def forward(self, x: torch.Tensor, HW=None, save_kv_cache=False, **kwargs) -> torch.Tensor:
        B, N, C = x.shape

        assert len(HW) == 3, "HW must be a tuple of (T, H, W)"
        T, H, W = HW
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)

        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        # Space aggregation
        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate
        x = self.point_conv(x)

        # Temporal aggregation
        x_reshaped = x.view(B, T, C, H * W).permute(0, 2, 1, 3)  # B,C,T,HW
        padding_size = self.t_conv.kernel_size[0] // 2
        x_t_conv_in = x_reshaped
        padded_size = 0
        # Use internal cache with the same logic as before
        if self.kv_cache is not None:

            if self.kv_cache[2] is not None:
                # Use previous chunk's temporal convolution cache
                x_t_conv_in = torch.cat([self.kv_cache[2], x_reshaped], dim=2)  # B,C,P+T,HW
                padded_size = self.kv_cache[2].shape[2]

            if save_kv_cache:  # Save current chunk's cache for next chunk
                self.kv_cache[2] = x_reshaped[:, :, -padding_size:, :].detach().clone()
                # print(f"CachedGLUMBConvTemp: Saved internal tconv cache, shape: {self.kv_cache[2].shape}")

        t_conv_out = self.t_conv(x_t_conv_in)[:, :, padded_size:]
        x_out = x_reshaped + t_conv_out

        x_out = x_out.permute(0, 2, 3, 1).reshape(B, N, C)

        return x_out


class SlimGLUMBConv(GLUMBConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 移除 self.inverted_conv 层
        del self.inverted_conv
        self.out_dim = self.point_conv.out_dim

    def forward(self, x: torch.Tensor, HW=None) -> torch.Tensor:
        B, N, C = x.shape
        if HW is None:
            H = W = int(N**0.5)
        else:
            H, W = HW

        # 直接使用 x，跳过 self.inverted_conv 层的调用
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        x = x.reshape(B, self.out_dim, N).permute(0, 2, 1)

        return x


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


class DWMlp(Mlp):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding=None,
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            bias=bias,
            drop=drop,
        )
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.conv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=hidden_features,
            bias=bias,
        )

    def forward(self, x, HW=None):
        B, N, C = x.shape
        if HW is None:
            H = W = int(N**0.5)
        else:
            H, W = HW
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = x.reshape(B, H, W, self.hidden_features).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(B, self.hidden_features, N).permute(0, 2, 1)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Mlp(Mlp):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.0):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            bias=bias,
            drop=drop,
        )

    def forward(self, x, HW=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


if __name__ == "__main__":
    model = GLUMBConv(
        1152,
        1152 * 4,
        1152,
        use_bias=(True, True, False),
        norm=(None, None, None),
        act=("silu", "silu", None),
    ).cuda()
    input = torch.randn(4, 256, 1152).cuda()
    output = model(input)
