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

# This file is modified from https://github.com/PixArt-alpha/PixArt-sigma
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp

from diffusion.model.act import build_act, get_act_name
from diffusion.model.norms import build_norm, get_norm_name
from diffusion.model.registry import FFN_BLOCKS
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


# Safe element-count threshold for a single conv call: PyTorch's 2D conv kernels
# (both cuDNN and the ATEN fallback) use 32-bit indexing internally, so very
# large ``(BT, C, H, W)`` inputs (e.g. minute-scale video at default CFG) can
# overflow. Empirically a single call up to ~1 B elements is safe; above that
# we chunk along the leading dim. Set so short videos stay on the original
# fused path (no chunking, no overhead) and long videos transparently split.
_INT32_SAFE_CONV_ELEMENTS = 1 << 30  # 1,073,741,824


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

    def _apply_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Fused spatial pipeline: inverted_conv -> depth_conv -> GLU -> point_conv."""
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        a, g = torch.chunk(x, 2, dim=1)
        g = self.glu_act(g)
        return self.point_conv(a * g)

    def _apply_spatial_autochunked(self, x: torch.Tensor) -> torch.Tensor:
        """Run :meth:`_apply_spatial`, chunking dim 0 to keep each call under
        PyTorch's 32-bit conv indexing limit. No-op for short inputs."""
        BT, _, H, W = x.shape
        # Conservative estimate of the largest intermediate (after inverted_conv).
        elements_per_bt = self.inverted_conv.conv.out_channels * H * W
        max_bt = max(1, _INT32_SAFE_CONV_ELEMENTS // elements_per_bt)
        if BT <= max_bt:
            return self._apply_spatial(x)
        return torch.cat([self._apply_spatial(x[s : s + max_bt]) for s in range(0, BT, max_bt)], dim=0)

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

        x = self._apply_spatial_autochunked(x)

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
        del kwargs
        B, N, C = x.shape

        assert len(HW) == 3, "HW must be a tuple of (T, H, W)"
        T, H, W = HW
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)

        x = self._apply_spatial_autochunked(x)

        # Temporal aggregation
        x_reshaped = x.view(B, T, C, H * W).permute(0, 2, 1, 3)
        try:
            from diffusion.distributed.context_parallel.config import cp_enabled, get_cp_group
            from diffusion.distributed.context_parallel.halo_exchange import cp_halo_exchange
        except Exception:
            cp_active = False
        else:
            cp_active = cp_enabled() and get_cp_group() is not None

        if cp_active and self.t_conv.padding[0] > 0:
            halo = int(self.t_conv.padding[0])
            x_halo = cp_halo_exchange(x_reshaped, left_size=halo, right_size=halo, dim=2, group=get_cp_group())
            t_out = self.t_conv(x_halo)[:, :, halo : halo + T, :]
        else:
            t_out = self.t_conv(x_reshaped)
        x_out = x_reshaped + t_out

        x_out = x_out.permute(0, 2, 3, 1).reshape(B, N, C)

        return x_out


class ChunkGLUMBConvTemp(GLUMBConvTemp):
    def forward(self, x: torch.Tensor, HW=None, chunk_index=None, **kwargs) -> torch.Tensor:
        del kwargs
        if chunk_index is None:
            chunk_index = [0]
        B, N, C = x.shape

        assert len(HW) == 3, "HW must be a tuple of (T, H, W)"
        T, H, W = HW
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)

        x = self._apply_spatial_autochunked(x)

        x_reshaped = x.view(B, T, C, H * W).permute(0, 2, 1, 3)  # B, C, T, H*W
        padding_size = self.t_conv.kernel_size[0] // 2
        chunk_boundaries = [int(idx) for idx in chunk_index] + [T]
        chunk_sizes = [end - start for start, end in zip(chunk_boundaries, chunk_boundaries[1:])]
        x_reshaped_list = x_reshaped.split(chunk_sizes, dim=-2)

        padded_x_reshaped_list = []
        padded_x_reshaped_list.append(
            torch.cat([x_reshaped_list[0], x_reshaped.new_zeros(B, C, padding_size, H * W)], dim=-2)
        )
        for i in range(1, len(x_reshaped_list)):
            prev_chunk = x_reshaped_list[i - 1][:, :, -padding_size:, :]
            cur_chunk = x_reshaped_list[i]
            padded_x_reshaped_list.append(
                torch.cat(
                    [
                        prev_chunk,
                        cur_chunk,
                        x_reshaped.new_zeros(B, C, padding_size, H * W),
                    ],
                    dim=-2,
                )
            )
        x_reshaped_t_conv = torch.cat(padded_x_reshaped_list, dim=-2)
        t_conv_out = self.t_conv(x_reshaped_t_conv)

        padded_chunk_sizes = [chunk_sizes[0] + padding_size] + [
            padding_size + chunk_size + padding_size for chunk_size in chunk_sizes[1:]
        ]
        t_conv_out_list = t_conv_out.split(padded_chunk_sizes, dim=-2)

        unpadded_chunks = []
        for i, chunk in enumerate(t_conv_out_list):
            if i == 0:
                unpadded_chunk = chunk[:, :, : chunk_sizes[i], :]
            else:
                start_idx = padding_size
                end_idx = start_idx + chunk_sizes[i]
                unpadded_chunk = chunk[:, :, start_idx:end_idx, :]
            unpadded_chunks.append(unpadded_chunk)

        t_conv_out_final = torch.cat(unpadded_chunks, dim=-2)
        assert t_conv_out_final.shape[-2] == T, f"Expected temporal dimension {T}, got {t_conv_out_final.shape[-2]}"

        x_out = x_reshaped + t_conv_out_final

        x_out = x_out.permute(0, 2, 3, 1).reshape(B, N, C)

        return x_out


@FFN_BLOCKS.register_module()
class CachedGLUMBConvTemp(GLUMBConvTemp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, HW=None, save_kv_cache=False, kv_cache=None, **kwargs) -> torch.Tensor:
        B, N, C = x.shape

        assert len(HW) == 3, "HW must be a tuple of (T, H, W)"
        T, H, W = HW
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)

        x = self._apply_spatial_autochunked(x)

        # Temporal aggregation
        x_reshaped = x.view(B, T, C, H * W).permute(0, 2, 1, 3)  # B,C,T,HW
        padding_size = self.t_conv.kernel_size[0] // 2
        x_t_conv_in = x_reshaped
        padded_size = 0
        # Tconv state lives in the last slot of the per-block KV cache list.
        # The streaming sampler's 10-slot layout uses indices 0-3 / 4-6 for
        # main + camera attention state and the final slot (-1) for the
        # temporal short conv left context written here.
        if kv_cache is not None:
            if kv_cache[-1] is not None:
                # Slice to the actual conv padding window in case the cache
                # spans multiple past chunks.
                x_t_conv_in = torch.cat([kv_cache[-1][:, :, -padding_size:], x_reshaped], dim=2)  # B,C,P+T,HW
                padded_size = x_t_conv_in.shape[2] - x_reshaped.shape[2]

            if save_kv_cache:  # Save current chunk's cache for next chunk
                kv_cache[-1] = x_reshaped[:, :, -padding_size:, :].detach().clone()

        t_conv_out = self.t_conv(x_t_conv_in)[:, :, padded_size:]
        x_out = x_reshaped + t_conv_out

        x_out = x_out.permute(0, 2, 3, 1).reshape(B, N, C)

        if kv_cache is not None:
            return x_out, kv_cache

        return x_out


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
