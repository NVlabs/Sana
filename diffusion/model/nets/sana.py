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
import os

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath

from diffusion.model.builder import MODELS
from diffusion.model.nets.basic_modules import DWMlp, GLUMBConv, MBConvPreGLU, Mlp
from diffusion.model.nets.sana_blocks import (
    Attention,
    CaptionEmbedder,
    FlashAttention,
    LiteLA,
    MultiHeadCrossAttention,
    PatchEmbed,
    T2IFinalLayer,
    TimestepEmbedder,
    t2i_modulate,
)
from diffusion.model.norms import RMSNorm
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from diffusion.utils.dist_utils import get_rank
from diffusion.utils.import_utils import is_triton_module_available
from diffusion.utils.logger import get_root_logger

_triton_modules_available = False
if is_triton_module_available():
    from diffusion.model.nets.fastlinear.modules import TritonLiteMLA, TritonMBConvPreGLU

    _triton_modules_available = True


class SanaBlock(nn.Module):
    """
    A Sana block with global shared adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0,
        input_size=None,
        qk_norm=False,
        attn_type="flash",
        ffn_type="mlp",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if attn_type == "flash":
            # flash self attention
            self.attn = FlashAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=qk_norm,
                **block_kwargs,
            )
        elif attn_type == "linear":
            # linear self attention
            # TODO: Here the num_heads set to 36 for tmp used
            self_num_heads = hidden_size // linear_head_dim
            self.attn = LiteLA(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm)
        elif attn_type == "triton_linear":
            if not _triton_modules_available:
                raise ValueError(
                    f"{attn_type} type is not available due to _triton_modules_available={_triton_modules_available}."
                )
            # linear self attention with triton kernel fusion
            # TODO: Here the num_heads set to 36 for tmp used
            self_num_heads = hidden_size // linear_head_dim
            self.attn = TritonLiteMLA(hidden_size, num_heads=self_num_heads, eps=1e-8)
        elif attn_type == "vanilla":
            # vanilla self attention
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        else:
            raise ValueError(f"{attn_type} type is not defined.")

        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        if ffn_type == "dwmlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = DWMlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
        elif ffn_type == "glumbconv":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
            )
        elif ffn_type == "glumbconv_dilate":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
                dilation=2,
            )
        elif ffn_type == "mbconvpreglu":
            self.mlp = MBConvPreGLU(
                in_dim=hidden_size,
                out_dim=hidden_size,
                mid_dim=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=None,
                act=("silu", "silu", None),
            )
        elif ffn_type == "triton_mbconvpreglu":
            if not _triton_modules_available:
                raise ValueError(
                    f"{ffn_type} type is not available due to _triton_modules_available={_triton_modules_available}."
                )
            self.mlp = TritonMBConvPreGLU(
                in_dim=hidden_size,
                out_dim=hidden_size,
                mid_dim=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=None,
                act=("silu", "silu", None),
            )
        elif ffn_type == "mlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
        else:
            raise ValueError(f"{ffn_type} type is not defined.")
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


#############################################################################
#                                 Core Sana Model                                #
#################################################################################
@MODELS.register_module()
class Sana(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path: float = 0.0,
        caption_channels=2304,
        pe_interpolation=1.0,
        config=None,
        model_max_length=120,
        qk_norm=False,
        y_norm=False,
        norm_eps=1e-5,
        attn_type="flash",
        ffn_type="mlp",
        use_pe=True,
        y_norm_scale_factor=1.0,
        patch_embed_kernel=None,
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        **kwargs,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pe_interpolation = pe_interpolation
        self.depth = depth
        self.use_pe = use_pe
        self.y_norm = y_norm
        self.fp32_attention = kwargs.get("use_fp32_attention", False)

        kernel_size = patch_embed_kernel or patch_size
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, kernel_size=kernel_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size // self.patch_size
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )
        if self.y_norm:
            self.attention_y_norm = RMSNorm(hidden_size, scale_factor=y_norm_scale_factor, eps=norm_eps)
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                SanaBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    input_size=(input_size // patch_size, input_size // patch_size),
                    qk_norm=qk_norm,
                    attn_type=attn_type,
                    ffn_type=ffn_type,
                    mlp_acts=mlp_acts,
                    linear_head_dim=linear_head_dim,
                )
                for i in range(depth)
            ]
        )
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        if config:
            logger = get_root_logger(os.path.join(config.work_dir, "train_log.log"))
            logger = logger.warning
        else:
            logger = print
        if get_rank() == 0:
            logger(
                f"use pe: {use_pe}, position embed interpolation: {self.pe_interpolation}, base size: {self.base_size}"
            )
            logger(
                f"attention type: {attn_type}; ffn type: {ffn_type}; "
                f"autocast linear attn: {os.environ.get('AUTOCAST_LINEAR_ATTN', False)}"
            )

    def forward(self, x, timestep, y, mask=None, data_info=None, **kwargs):
        """
        Forward pass of Sana.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        if self.use_pe:
            x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        else:
            x = self.x_embedder(x)
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, 1, L, D)
        if self.y_norm:
            y = self.attention_y_norm(y)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, y, t0, y_lens)  # (N, T, D) #support grad checkpoint
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def __call__(self, *args, **kwargs):
        """
        This method allows the object to be called like a function.
        It simply calls the forward method.
        """
        return self.forward(*args, **kwargs)

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, mask)
        return model_out.chunk(2, dim=1)[0] if self.pred_sigma else model_out

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.use_pe:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.x_embedder.num_patches**0.5),
                pe_interpolation=self.pe_interpolation,
                base_size=self.base_size,
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, pe_interpolation=1.0, base_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / pe_interpolation
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / pe_interpolation
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   Sana Configs                              #
#################################################################################
@MODELS.register_module()
def Sana_600M_P1_D28(**kwargs):
    return Sana(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)


@MODELS.register_module()
def Sana_1600M_P1_D20(**kwargs):
    # 20 layers, 1648.48M
    return Sana(depth=20, hidden_size=2240, patch_size=1, num_heads=20, **kwargs)
