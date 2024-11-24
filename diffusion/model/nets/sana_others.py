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
from timm.models.layers import DropPath

from diffusion.model.nets.basic_modules import DWMlp, MBConvPreGLU, Mlp
from diffusion.model.nets.sana_blocks import Attention, FlashAttention, MultiHeadCrossAttention, t2i_modulate
from diffusion.utils.import_utils import is_triton_module_available

_triton_modules_available = False
if is_triton_module_available():
    from diffusion.model.nets.fastlinear.modules import TritonLiteMLA

    _triton_modules_available = True


class SanaMSPABlock(nn.Module):
    """
    A Sana block with adaptive layer norm zero (adaLN-Zero) conditioning.
    reference VIT-22B
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L224
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        input_size=None,
        sampling=None,
        sr_ratio=1,
        qk_norm=False,
        attn_type="flash",
        ffn_type="mlp",
        mlp_acts=("silu", "silu", None),
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size * 3, elementwise_affine=False, eps=1e-6)
        if attn_type == "flash":
            # flash self attention
            self.attn = FlashAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                sampling=sampling,
                sr_ratio=sr_ratio,
                qk_norm=qk_norm,
                **block_kwargs,
            )
            print("currently not support parallel attn")
            exit()
        elif attn_type == "linear":
            # linear self attention
            # TODO: Here the num_heads set to 36 for tmp used
            self_num_heads = hidden_size // 32
            # self.attn = LiteLA(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8)
            self.attn = SlimLiteLA(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8)
        elif attn_type == "triton_linear":
            # linear self attention with triton kernel fusion
            self_num_heads = hidden_size // 32
            self.attn = TritonLiteMLA(hidden_size, num_heads=self_num_heads, eps=1e-8)
            print("currently not support parallel attn")
            exit()
        elif attn_type == "vanilla":
            # vanilla self attention
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
            print("currently not support parallel attn")
            exit()
        else:
            raise ValueError(f"{attn_type} type is not defined.")

        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(int(hidden_size * mlp_ratio * 2), elementwise_affine=False, eps=1e-6)
        if ffn_type == "dwmlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = DWMlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
            print("currently not support parallel attn")
            exit()
        elif ffn_type == "glumbconv":
            self.mlp = SlimGLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
            )
        elif ffn_type == "mlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
            print("currently not support parallel attn")
            exit()
        elif ffn_type == "mbconvpreglu":
            self.mlp = MBConvPreGLU(
                in_dim=hidden_size,
                out_dim=hidden_size,
                mid_dim=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=None,
                act=("silu", "silu", None),
            )
            print("currently not support parallel attn")
            exit()
        else:
            raise ValueError(f"{ffn_type} type is not defined.")
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        # parallel layers
        self.mlp_ratio = mlp_ratio
        self.in_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.in_proj = nn.Linear(hidden_size, (hidden_size * 3 + int(hidden_size * mlp_ratio * 2)))
        self.in_split = [hidden_size * 3] + [int(hidden_size * mlp_ratio * 2)]

    def forward(self, x, y, t, mask=None, HW=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        # original Attention code
        # x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), HW=HW))
        # x = x + self.cross_attn(x, y, mask)
        # x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp), HW=HW))

        # combine GLUMBConv fc1 & qkv projections
        # x_1 = self.in_norm(x)
        # x_1 = self.in_proj(x_1)
        x_1 = self.in_proj(self.in_norm(x))
        qkv, x_mlp = torch.split(x_1, self.in_split, dim=-1)

        qkv = t2i_modulate(self.norm1(qkv), shift_msa.repeat(1, 1, 3), scale_msa.repeat(1, 1, 3))
        x_mlp = t2i_modulate(
            self.norm2(x_mlp),
            shift_mlp.repeat(1, 1, int(self.mlp_ratio * 2)),
            scale_mlp.repeat(1, 1, int(self.mlp_ratio * 2)),
        )
        # qkv = self.norm1(qkv)
        # x_mlp = self.norm2(x_mlp)

        # branch 1
        x_attn = gate_msa * self.attn(qkv, HW=HW)
        x_attn = x_attn + self.cross_attn(x_attn, y, mask)

        # branch 2
        x_mlp = gate_mlp * self.mlp(x_mlp, HW=HW)

        # Add residual w/ drop path & layer scale applied
        x = x + self.drop_path(x_attn + x_mlp)

        return x


class SanaMSPABlock(nn.Module):
    """
    A Sana block with adaptive layer norm zero (adaLN-Zero) conditioning.
    reference VIT-22B
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L224
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        input_size=None,
        sampling=None,
        sr_ratio=1,
        qk_norm=False,
        attn_type="flash",
        ffn_type="mlp",
        mlp_acts=("silu", "silu", None),
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size * 3, elementwise_affine=False, eps=1e-6)
        if attn_type == "flash":
            # flash self attention
            self.attn = FlashAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                sampling=sampling,
                sr_ratio=sr_ratio,
                qk_norm=qk_norm,
                **block_kwargs,
            )
            print("currently not support parallel attn")
            exit()
        elif attn_type == "linear":
            # linear self attention
            # TODO: Here the num_heads set to 36 for tmp used
            self_num_heads = hidden_size // 32
            # self.attn = LiteLA(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8)
            self.attn = SlimLiteLA(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8)
        elif attn_type == "triton_linear":
            # linear self attention with triton kernel fusion
            self_num_heads = hidden_size // 32
            self.attn = TritonLiteMLA(hidden_size, num_heads=self_num_heads, eps=1e-8)
            print("currently not support parallel attn")
            exit()
        elif attn_type == "vanilla":
            # vanilla self attention
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
            print("currently not support parallel attn")
            exit()
        else:
            raise ValueError(f"{attn_type} type is not defined.")

        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(int(hidden_size * mlp_ratio * 2), elementwise_affine=False, eps=1e-6)
        if ffn_type == "dwmlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = DWMlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
            print("currently not support parallel attn")
            exit()
        elif ffn_type == "glumbconv":
            self.mlp = SlimGLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
            )
        elif ffn_type == "mlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
            print("currently not support parallel attn")
            exit()
        elif ffn_type == "mbconvpreglu":
            self.mlp = MBConvPreGLU(
                in_dim=hidden_size,
                out_dim=hidden_size,
                mid_dim=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=None,
                act=("silu", "silu", None),
            )
            print("currently not support parallel attn")
            exit()
        else:
            raise ValueError(f"{ffn_type} type is not defined.")
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        # parallel layers
        self.mlp_ratio = mlp_ratio
        self.in_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.in_proj = nn.Linear(hidden_size, (hidden_size * 3 + int(hidden_size * mlp_ratio * 2)))
        self.in_split = [hidden_size * 3] + [int(hidden_size * mlp_ratio * 2)]

    def forward(self, x, y, t, mask=None, HW=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x_1 = self.in_proj(self.in_norm(x))
        qkv, x_mlp = torch.split(x_1, self.in_split, dim=-1)

        qkv = t2i_modulate(self.norm1(qkv), shift_msa.repeat(1, 1, 3), scale_msa.repeat(1, 1, 3))
        x_mlp = t2i_modulate(
            self.norm2(x_mlp),
            shift_mlp.repeat(1, 1, int(self.mlp_ratio * 2)),
            scale_mlp.repeat(1, 1, int(self.mlp_ratio * 2)),
        )

        # branch 1
        x_attn = gate_msa * self.attn(qkv, HW=HW)
        x_attn = x_attn + self.cross_attn(x_attn, y, mask)

        # branch 2
        x_mlp = gate_mlp * self.mlp(x_mlp, HW=HW)

        # Add residual w/ drop path & layer scale applied
        x = x + self.drop_path(x_attn + x_mlp)

        return x
