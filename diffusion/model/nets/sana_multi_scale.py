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

import torch
import torch.nn as nn
from timm.models.layers import DropPath

from diffusion.model.builder import MODELS
from diffusion.model.nets.basic_modules import DWMlp, GLUMBConv, Mlp
from diffusion.model.nets.sana import Sana, get_2d_sincos_pos_embed
from diffusion.model.nets.sana_blocks import (
    Attention,
    CaptionEmbedder,
    FlashAttention,
    LiteLA,
    MultiHeadCrossAttention,
    MultiHeadCrossVallinaAttention,
    PatchEmbedMS,
    RopePosEmbed,
    T2IFinalLayer,
    t2i_modulate,
)
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.utils.import_utils import is_triton_module_available, is_xformers_available

_triton_modules_available = False
if is_triton_module_available():
    from diffusion.model.nets.fastlinear.modules import TritonLiteMLA, TritonMBConvPreGLU

    _triton_modules_available = True

_xformers_available = False if os.environ.get("DISABLE_XFORMERS", "0") == "1" else is_xformers_available()
if _xformers_available:
    import xformers.ops


class SanaMSBlock(nn.Module):
    """
    A Sana block with global shared adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        qk_norm=False,
        attn_type="flash",
        ffn_type="mlp",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        cross_norm=False,
        cross_attn_type="flash",
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
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
            self_num_heads = hidden_size // linear_head_dim
            self.attn = TritonLiteMLA(hidden_size, num_heads=self_num_heads, eps=1e-8)
        elif attn_type == "vanilla":
            # vanilla self attention
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        else:
            raise ValueError(f"{attn_type} type is not defined.")

        if cross_attn_type in ["flash", "linear"]:
            self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, qk_norm=cross_norm, **block_kwargs)
        elif cross_attn_type == "vanilla":
            self.cross_attn = MultiHeadCrossVallinaAttention(hidden_size, num_heads, qk_norm=cross_norm, **block_kwargs)
        else:
            raise ValueError(f"{cross_attn_type} type is not defined.")
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
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
        elif ffn_type == "mlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
        else:
            raise ValueError(f"{ffn_type} type is not defined.")
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, x, y, t, mask=None, HW=None, image_rotary_emb=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x = x + self.drop_path(
            gate_msa
            * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), HW=HW, image_rotary_emb=image_rotary_emb)
        )
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp), HW=HW))

        return x


#############################################################################
#                                 Core Sana Model                                #
#################################################################################
@MODELS.register_module()
class SanaMS(Sana):
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
        learn_sigma=True,
        pred_sigma=True,
        drop_path: float = 0.0,
        caption_channels=2304,
        pe_interpolation=1.0,
        config=None,
        model_max_length=300,
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
        cross_norm=False,
        cross_attn_type="flash",
        logvar=False,
        logvar_scale_factor=1.0,
        cfg_embed=False,
        cfg_embed_scale=1.0,
        lr_scale=None,
        timestep_norm_scale_factor=1.0,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            learn_sigma=learn_sigma,
            pred_sigma=pred_sigma,
            drop_path=drop_path,
            caption_channels=caption_channels,
            pe_interpolation=pe_interpolation,
            config=config,
            model_max_length=model_max_length,
            qk_norm=qk_norm,
            y_norm=y_norm,
            norm_eps=norm_eps,
            attn_type=attn_type,
            ffn_type=ffn_type,
            use_pe=use_pe,
            y_norm_scale_factor=y_norm_scale_factor,
            patch_embed_kernel=patch_embed_kernel,
            mlp_acts=mlp_acts,
            linear_head_dim=linear_head_dim,
            cross_norm=cross_norm,
            cross_attn_type=cross_attn_type,
            cfg_embed=cfg_embed,
            timestep_norm_scale_factor=timestep_norm_scale_factor,
            **kwargs,
        )
        self.h = self.w = 0
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.pos_embed_ms = None
        self.cfg_embed_scale = cfg_embed_scale

        kernel_size = patch_embed_kernel or patch_size
        self.x_embedder = PatchEmbedMS(patch_size, in_channels, hidden_size, kernel_size=kernel_size, bias=True)
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                SanaMSBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=qk_norm,
                    attn_type=attn_type,
                    ffn_type=ffn_type,
                    mlp_acts=mlp_acts,
                    linear_head_dim=linear_head_dim,
                    cross_norm=cross_norm,
                    cross_attn_type=cross_attn_type,
                )
                for i in range(depth)
            ]
        )
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)
        self.logvar_linear = None
        if logvar:
            self.logvar_scale_factor = logvar_scale_factor
            self.logvar_linear = nn.Linear(hidden_size, 1)

        self.lr_scale = lr_scale

        self.initialize()

    def forward(self, x, timestep, y, mask=None, data_info=None, return_logvar=False, jvp=False, **kwargs):
        """
        Forward pass of Sana.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        bs = x.shape[0]
        x = x.to(self.dtype)
        if self.timestep_norm_scale_factor != 1.0:
            timestep = (timestep.float() / self.timestep_norm_scale_factor).to(torch.float32)
        else:
            timestep = timestep.long().to(torch.float32)
        y = y.to(self.dtype)
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        x = self.x_embedder(x)
        image_pos_embed = None
        if self.use_pe:
            if self.pos_embed_type == "sincos":
                if self.pos_embed_ms is None or self.pos_embed_ms.shape[1:] != x.shape[1:]:
                    self.pos_embed_ms = (
                        torch.from_numpy(
                            get_2d_sincos_pos_embed(
                                self.pos_embed.shape[-1],
                                (self.h, self.w),
                                pe_interpolation=self.pe_interpolation,
                                base_size=self.base_size,
                            )
                        )
                        .unsqueeze(0)
                        .to(x.device)
                        .to(self.dtype)
                    )
                x += self.pos_embed_ms  # (N, T, D), where T = H * W / patch_size ** 2
            elif self.pos_embed_type == "3d_rope":
                self.pos_embed_ms = RopePosEmbed(theta=10000, axes_dim=[0, 16, 16])
                latent_image_ids = self.pos_embed_ms._prepare_latent_image_ids(bs, self.h, self.w, x.device, x.dtype)
                image_pos_embed = self.pos_embed_ms(latent_image_ids)
                x += image_pos_embed
            else:
                raise ValueError(f"Unknown pos_embed_type: {self.pos_embed_type}")

        t = self.t_embedder(timestep)  # (N, D)
        if self.cfg_embedder:
            cfg_embed = self.cfg_embedder(data_info["cfg_scale"] * self.cfg_embed_scale)
            t += cfg_embed

        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training, mask=mask)  # (N, D)
        if self.y_norm:
            y = self.attention_y_norm(y)

        if mask is not None:
            mask = mask.to(torch.int16)
            mask = mask.repeat(y.shape[0] // mask.shape[0], 1) if mask.shape[0] != y.shape[0] else mask
            mask = mask.squeeze(1).squeeze(1)
            if _xformers_available:
                y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
                y_lens = mask.sum(dim=1).tolist()
            else:
                y_lens = mask
        elif _xformers_available:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])
        else:
            raise ValueError(f"Attention type is not available due to _xformers_available={_xformers_available}.")

        for block in self.blocks:
            if jvp:
                x = block(x, y, t0, y_lens, (self.h, self.w), image_pos_embed, **kwargs)
            # gradient checkpointing is not supported for JVP
            else:
                x = auto_grad_checkpoint(
                    block,
                    x,
                    y,
                    t0,
                    y_lens,
                    (self.h, self.w),
                    image_pos_embed,
                    **kwargs,
                    use_reentrant=False,
                )  # (N, T, D) #support grad checkpoint

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        if return_logvar and self.logvar_linear is not None:
            logvar = self.logvar_linear(t) * self.logvar_scale_factor
            return x, logvar

        return x

    def __call__(self, *args, **kwargs):
        """
        This method allows the object to be called like a function.
        It simply calls the forward method.
        """
        return self.forward(*args, **kwargs)

    def forward_with_dpmsolver(self, x, timestep, y, data_info, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, data_info=data_info, **kwargs)
        return model_out.chunk(2, dim=1)[0] if self.pred_sigma else model_out

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        assert self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.h, self.w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, self.h * p, self.w * p))
        return imgs

    def initialize(self):
        super().initialize_weights()
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

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


class SanaMSCM(SanaMS):
    def forward(self, x, timestep, y, data_info=None, return_logvar=False, **kwargs):

        # TrigFlow --> Flow Transformation
        # the input now is [0, np.pi/2], arctan(N(P_mean, P_std))
        t = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))

        pretrain_timestep = t * 1000  # stabilize large resolution training
        t = t.view(-1, 1, 1, 1)

        x = x * torch.sqrt(t**2 + (1 - t) ** 2)

        # forward in original flow
        if return_logvar:
            model_out, logvar = super().forward(
                x, pretrain_timestep, y, data_info=data_info, return_logvar=return_logvar, **kwargs
            )
        else:
            model_out = super().forward(x, pretrain_timestep, y, data_info=data_info, **kwargs)

        # Flow --> TrigFlow Transformation
        trigflow_model_out = ((1 - 2 * t) * x + (1 - 2 * t + 2 * t**2) * model_out) / torch.sqrt(
            t**2 + (1 - t) ** 2
        )

        if return_logvar:
            return trigflow_model_out, logvar
        else:
            return trigflow_model_out


#################################################################################
#                                   Sana Multi-scale Configs                              #
#################################################################################


@MODELS.register_module()
def SanaMS_600M_P1_D28(**kwargs):
    return SanaMS(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)


@MODELS.register_module()
def SanaMS_600M_P2_D28(**kwargs):
    return SanaMS(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


@MODELS.register_module()
def SanaMS_600M_P4_D28(**kwargs):
    return SanaMS(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


@MODELS.register_module()
def SanaMS_1600M_P1_D20(**kwargs):
    # 20 layers, 1648.48M
    return SanaMS(depth=20, hidden_size=2240, patch_size=1, num_heads=20, **kwargs)


@MODELS.register_module()
def SanaMS_1600M_P2_D20(**kwargs):
    # 28 layers, 1648.48M
    return SanaMS(depth=20, hidden_size=2240, patch_size=2, num_heads=20, **kwargs)


@MODELS.register_module()
def SanaMS_2400M_P1_D30(**kwargs):
    return SanaMS(depth=30, hidden_size=2240, patch_size=1, num_heads=20, **kwargs)


@MODELS.register_module()
def SanaMS_3200M_P1_D40(**kwargs):
    return SanaMS(depth=40, hidden_size=2240, patch_size=1, num_heads=20, **kwargs)


@MODELS.register_module()
def SanaMS_4800M_P1_D60(**kwargs):
    # 60 layers, 4800M
    return SanaMS(depth=60, hidden_size=2240, patch_size=1, num_heads=20, **kwargs)


# TrigFlow/sCM model
@MODELS.register_module()
def SanaMSCM_600M_P1_D28(**kwargs):
    return SanaMSCM(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)


@MODELS.register_module()
def SanaMSCM_1600M_P1_D20(**kwargs):
    return SanaMSCM(depth=20, hidden_size=2240, patch_size=1, num_heads=20, **kwargs)


@MODELS.register_module()
def SanaMSCM_2400M_P1_D30(**kwargs):
    # 30 layers, 2400M
    return SanaMSCM(depth=30, hidden_size=2240, patch_size=1, num_heads=20, **kwargs)
