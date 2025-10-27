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

import math

# This file is modified from https://github.com/PixArt-alpha/PixArt-sigma
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from diffusion.model.builder import MODELS
from diffusion.model.nets.basic_modules import CachedGLUMBConvTemp, ChunkGLUMBConvTemp, GLUMBConv, GLUMBConvTemp, Mlp
from diffusion.model.nets.sana_blocks import (
    CachedCausalAttention,
    CaptionEmbedder,
    CausalWanRotaryPosEmbed,
    ChunkAttention,
    ChunkCausalAttention,
    ChunkedLiteLAReLURope,
    ClipVisionProjection,
    FlashAttention,
    LiteLA,
    LiteLAReLURope,
    MultiHeadCrossAttention,
    MultiHeadCrossAttentionImageEmbed,
    PatchEmbedMS3D,
    RopePosEmbed,
    T2IFinalLayer,
    WanRotaryPosEmbed,
    WanRotaryTemporalPosEmbed,
    WindowAttention,
    t2i_modulate,
)
from diffusion.model.nets.sana_multi_scale import Sana, get_2d_sincos_pos_embed
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.model.wan.model import BlockHook
from diffusion.utils.dist_utils import get_rank
from diffusion.utils.import_utils import is_triton_module_available, is_xformers_available

_triton_modules_available = False
if is_triton_module_available():
    from diffusion.model.nets.fastlinear.modules import TritonLiteMLA, TritonMBConvPreGLU

    _triton_modules_available = True

_xformers_available = False if os.environ.get("DISABLE_XFORMERS", "0") == "1" else is_xformers_available()
if _xformers_available:
    import xformers.ops


class SanaVideoMSBlock(nn.Module):
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
        cross_attn_image_embeds=False,
        t_kernel_size=3,
        additional_flash_attn=False,
        flash_attn_window_count=None,
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
        elif attn_type == "chunkcausal":
            self_num_heads = hidden_size // linear_head_dim
            self.attn = ChunkCausalAttention(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm)
        elif attn_type == "cachedcausal":
            self_num_heads = hidden_size // linear_head_dim
            self.attn = CachedCausalAttention(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm)
        elif attn_type == "chunkattn":
            self.attn = ChunkAttention(
                hidden_size, hidden_size, heads=hidden_size // linear_head_dim, eps=1e-8, qk_norm=qk_norm
            )
        elif attn_type == "linear":
            # linear self attention
            # TODO: Here the num_heads set to 36 for tmp used
            self_num_heads = hidden_size // linear_head_dim
            self.attn = LiteLA(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm)
        elif attn_type == "LiteLAReLURope":
            # linear self attention with first relu kernel and then rope
            self_num_heads = hidden_size // linear_head_dim
            self.attn = LiteLAReLURope(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm)
        elif attn_type == "ChunkedLiteLAReLURope":
            # linear self attention with first relu kernel and then rope
            self_num_heads = hidden_size // linear_head_dim
            self.attn = ChunkedLiteLAReLURope(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm)
        else:
            self.attn = None

        if additional_flash_attn == "flash":
            self.learnable_fa_scale = nn.Parameter(torch.ones(1) * 100)
            self.flash_attn_additional = FlashAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=qk_norm,
                **block_kwargs,
            )
        elif additional_flash_attn == "window_flash":
            self.learnable_fa_scale = nn.Parameter(torch.ones(1) * 100)
            self.flash_attn_additional = WindowAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=qk_norm,
                window_count=flash_attn_window_count,
                pad_if_needed=True,
                **block_kwargs,
            )
        else:
            self.flash_attn_additional = None

        # Cross Attention
        self.cross_attn_image_embeds = cross_attn_image_embeds
        if cross_attn_image_embeds:
            self.cross_attn = MultiHeadCrossAttentionImageEmbed(
                hidden_size, num_heads, qk_norm=cross_norm, **block_kwargs
            )
        else:
            self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, qk_norm=cross_norm, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # MLP
        if ffn_type == "glumbconv":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
            )
        elif ffn_type == "GLUMBConvTemp":
            self.mlp = GLUMBConvTemp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
                t_kernel_size=t_kernel_size,
            )
        elif ffn_type == "ChunkGLUMBConvTemp":
            self.mlp = ChunkGLUMBConvTemp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
                t_kernel_size=t_kernel_size,
            )
        elif ffn_type == "CachedGLUMBConvTemp":
            self.mlp = CachedGLUMBConvTemp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
                t_kernel_size=t_kernel_size,
            )
        elif ffn_type == "mlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
        else:
            self.mlp = None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
        self.block_hook: Optional[BlockHook] = None

    def forward_frame_aware(self, x, y, t, mask=None, THW=None, rotary_emb=None, chunk_index=None, **kwargs):
        B, N, C = x.shape
        num_frames = t.shape[2]

        t = t.reshape(B, num_frames, 6, -1)  # B,F,6,D
        # scale_shift_table: 6, hidden_size -> 1,1,6,hidden_size
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None, :, :] + t
        ).chunk(
            6, dim=-2
        )  # each chunk: B,F,1,D
        self_attn_kwargs = {
            "HW": THW,
            "rotary_emb": rotary_emb,
        }
        if chunk_index is not None:
            self_attn_kwargs["chunk_index"] = chunk_index[:]  # NOTE: important, copy the list
        x = x + self.drop_path(
            (
                gate_msa
                * self.attn(
                    t2i_modulate(self.norm1(x).reshape(B, num_frames, -1, C), shift_msa, scale_msa).reshape(B, N, C),
                    **self_attn_kwargs,
                ).reshape(B, num_frames, -1, C)
            ).reshape(B, N, C)
        )

        if self.flash_attn_additional:
            x = x + self.flash_attn_additional(x, HW=THW)

        if self.cross_attn_image_embeds:
            x = x + self.cross_attn(x, y, mask=mask, image_embeds=kwargs.get("image_embeds", None))
        else:
            x = x + self.cross_attn(x, y, mask=mask)

        mlp_kwargs = {
            "HW": THW,
        }
        if chunk_index is not None:
            mlp_kwargs["chunk_index"] = chunk_index[:]  # NOTE: important, copy the list
        x = x + self.drop_path(
            (
                gate_mlp
                * self.mlp(
                    t2i_modulate(self.norm2(x).reshape(B, num_frames, -1, C), shift_mlp, scale_mlp).reshape(B, N, C),
                    **mlp_kwargs,
                ).reshape(B, num_frames, -1, C)
            ).reshape(B, N, C)
        )

        return x

    def forward(self, x, y, t, mask=None, THW=None, rotary_emb=None, chunk_index=None, **kwargs):
        if len(t.shape) > 2:
            return self.forward_frame_aware(
                x,
                y,
                t,
                mask=mask,
                THW=THW,
                rotary_emb=rotary_emb,
                chunk_index=chunk_index,
                **kwargs,
            )
        intermediate_feats = {
            "x_in": x,
            "x_self_attn": None,
            "x_cross_attn": None,
            "x_ffn": None,
        }
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x_sa_in = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        self_attn_kwargs = {
            "HW": THW,
            "rotary_emb": rotary_emb,
        }
        if chunk_index is not None:
            self_attn_kwargs["chunk_index"] = chunk_index[:]  # NOTE: important, copy the list
        save_kv_cache = kwargs.get("save_kv_cache", None)
        if save_kv_cache is not None:
            self_attn_kwargs["save_kv_cache"] = save_kv_cache
        x_sa = self.attn(x_sa_in, **self_attn_kwargs)

        intermediate_feats["x_self_attn"] = x_sa

        if self.flash_attn_additional:
            x_sa = x_sa + self.learnable_fa_scale * self.flash_attn_additional(x_sa_in, rotary_emb=rotary_emb, HW=THW)

        x = x + self.drop_path(gate_msa * x_sa)

        if self.cross_attn_image_embeds:
            x = x + self.cross_attn(x, y, mask=mask, image_embeds=kwargs.get("image_embeds", None))
        else:
            x = x + self.cross_attn(x, y, mask=mask)

        intermediate_feats["x_cross_attn"] = x

        mlp_kwargs = {
            "HW": THW,
        }
        if chunk_index is not None:
            mlp_kwargs["chunk_index"] = chunk_index[:]  # NOTE: important, copy the list
        if save_kv_cache is not None:
            mlp_kwargs["save_kv_cache"] = save_kv_cache
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp), **mlp_kwargs))

        intermediate_feats["x_ffn"] = x

        if self.block_hook is not None:
            self.block_hook(**intermediate_feats)

        return x


#############################################################################
#                                 Core Sana Model                                #
#################################################################################
@MODELS.register_module()
class SanaMSVideo(Sana):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=(1, 2, 2),
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
        cross_attn_image_embeds=False,
        image_embed_channels=1152,
        pos_embed_type="wan_rope",
        rope_fhw_dim=None,
        t_kernel_size=3,
        flash_attn_window_count=None,
        pack_latents=False,
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
            pos_embed_type=pos_embed_type,
            **kwargs,
        )
        self.patch_size = patch_size
        self.h = self.w = 0
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.pos_embed_ms = None
        self.pack_latents = pack_latents
        self.pos_embed_type = pos_embed_type

        kernel_size = patch_embed_kernel or patch_size
        x_embedder_in_channels = in_channels
        if self.pack_latents:
            x_embedder_in_channels = x_embedder_in_channels * 2 * 2
            self.out_channels = in_channels * 2 * 2

        self.x_embedder = PatchEmbedMS3D(
            patch_size, x_embedder_in_channels, hidden_size, kernel_size=kernel_size, bias=True
        )

        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )
        if cross_attn_image_embeds:
            self.image_embedder = ClipVisionProjection(image_embed_channels, hidden_size)
        else:
            self.image_embedder = None

        if attn_type in ["flash"]:
            attention_head_dim = hidden_size // num_heads
        else:
            attention_head_dim = linear_head_dim
        if self.use_pe:
            self.rope = self.get_rope(pos_embed_type, attention_head_dim, patch_size, rope_fhw_dim)
        else:
            self.rope = None

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule

        attn_type_list = [attn_type] * depth

        self.blocks = nn.ModuleList(
            [
                SanaVideoMSBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=qk_norm,
                    attn_type=attn_type_list[i],
                    ffn_type=ffn_type,
                    mlp_acts=mlp_acts,
                    linear_head_dim=linear_head_dim,
                    cross_norm=cross_norm,
                    cross_attn_image_embeds=cross_attn_image_embeds,
                    t_kernel_size=t_kernel_size,
                    flash_attn_window_count=flash_attn_window_count,
                )
                for i in range(depth)
            ]
        )

        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        if get_rank() == 0:
            if ffn_type == "GLUMBConvTemp":
                self.logger(f"{ffn_type} Temporal kernal: {t_kernel_size}")

        self.initialize()

    def get_rope(self, pos_embed_type, attention_head_dim, patch_size, rope_fhw_dim):
        if pos_embed_type == "wan_rope":
            rope = WanRotaryPosEmbed(
                attention_head_dim=attention_head_dim, patch_size=patch_size, max_seq_len=1024, fhw_dim=rope_fhw_dim
            )
        elif pos_embed_type == "casual_wan_rope":
            rope = CausalWanRotaryPosEmbed(
                attention_head_dim=attention_head_dim, patch_size=patch_size, max_seq_len=1024
            )
        elif pos_embed_type == "wan_temporal_rope":
            rope = WanRotaryTemporalPosEmbed(
                attention_head_dim=attention_head_dim, patch_size=patch_size, max_seq_len=1024
            )
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

        return rope

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width, frame):
        latents = latents.view(batch_size, num_channels_latents, frame, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 4, 6, 2, 3, 5)
        latents = latents.reshape(batch_size, num_channels_latents * 4, frame, height // 2, width // 2)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, frame):
        batch_size, channels, frame, H, W = latents.shape

        assert height % 2 == 0 and width % 2 == 0
        # latent height and width to be divisible by 2.
        latents = latents.view(batch_size, channels // 4, 2, 2, frame, height // 2, width // 2)
        latents = latents.permute(0, 1, 4, 5, 2, 6, 3)
        latents = latents.reshape(batch_size, channels // (2 * 2), frame, height, width)

        return latents

    @staticmethod
    def _unpack_latents_additional_layers(latents, height, width, frame):
        batch_size, num_patches, channels = latents.shape

        # latent height and width to be divisible by 2.
        latents = latents.view(batch_size, frame, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 1, 2, 5, 3, 6, 4)
        latents = latents.reshape(batch_size, frame, height, width, channels // (2 * 2))
        latents = latents.view(batch_size, num_patches * 2 * 2, channels // (2 * 2))

        return latents

    def _apply_positional_embedding(self, x, bs, start_f=None, end_f=None):
        """Apply positional embedding to input tensor.

        Args:
            x: Input tensor (N, T, D)
            bs: Batch size
            start_f: Start frame index for casual_wan_rope (optional)
            end_f: End frame index for casual_wan_rope (optional)

        Returns:
            x with positional embedding added (for sincos type)
            image_pos_embed for other types (or None)
        """
        image_pos_embed = None

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
            x = x + self.pos_embed_ms  # (N, T, D), where T = H * W / patch_size ** 2

        elif self.pos_embed_type == "flux_rope":
            self.pos_embed_ms = RopePosEmbed(theta=10000, axes_dim=[12, 10, 10])
            latent_image_ids = self.pos_embed_ms._prepare_latent_image_ids(
                bs, self.h, self.w, x.device, x.dtype, frame=self.f
            )
            image_pos_embed = self.pos_embed_ms(latent_image_ids)

        elif self.pos_embed_type == "wan_rope":
            image_pos_embed = self.rope((self.f, self.h, self.w), x.device)

        elif self.pos_embed_type == "casual_wan_rope":
            image_pos_embed = self.rope(((start_f, end_f), self.h, self.w), x.device)

        elif self.pos_embed_type == "wan_temporal_rope":
            image_pos_embed = self.rope((self.f, self.h, self.w), x.device)

        else:
            raise ValueError(f"Unknown pos_embed_type: {self.pos_embed_type}")

        return x, image_pos_embed

    def forward(self, x, timestep, y, mask=None, **kwargs):
        """
        Forward pass of Sana.
        x: (N, C, T, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps or (N, 1, F) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        bs = x.shape[0]
        x = x.to(self.dtype)
        if self.timestep_norm_scale_factor != 1.0:
            timestep = (timestep.float() / self.timestep_norm_scale_factor).to(torch.float32)
        else:
            timestep = timestep.long().to(torch.float32)
        y = y.to(self.dtype)
        self.f, self.h, self.w = (
            x.shape[-3] // self.patch_size[0],
            x.shape[-2] // self.patch_size[1],
            x.shape[-1] // self.patch_size[2],
        )

        data_info = kwargs.get("data_info", {})
        if data_info.get("image_vae_embeds", None) is not None:
            x = torch.cat([x, data_info["image_vae_embeds"].to(self.dtype)], dim=1)
        if data_info.get("image_embeds", None) is not None:
            image_embeds = data_info["image_embeds"].to(self.dtype)
            image_embeds = self.image_embedder(image_embeds)
            kwargs["image_embeds"] = image_embeds

        if self.pack_latents:
            x = self._pack_latents(x, bs, self.in_channels, self.h, self.w, self.f)
            self.h = self.h // 2
            self.w = self.w // 2
        if self.x_embedder.patch_size != self.x_embedder.kernel_size and self.x_embedder.kernel_size == (1, 2, 2):
            x = F.pad(x, (0, 1, 0, 1, 0, 0))

        x = self.x_embedder(x)
        image_pos_embed = None
        if self.use_pe:
            x, image_pos_embed = self._apply_positional_embedding(x, bs)

        t = self.t_embedder(timestep.flatten())  # (N, D)
        t0 = self.t_block(t)
        t = t.unflatten(dim=0, sizes=timestep.shape)
        t0 = t0.unflatten(dim=0, sizes=timestep.shape)
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

        for i, block in enumerate(self.blocks):
            x = auto_grad_checkpoint(
                block,
                x,
                y,
                t0,
                y_lens,
                (self.f, self.h, self.w),
                image_pos_embed,
                **kwargs,
                use_reentrant=False,
            )  # (N, T, D) #support grad checkpoint

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        if self.pack_latents:
            x = self._unpack_latents(x, self.h * 2, self.w * 2, self.f)

        return x

    def forward_long(self, x, timestep, y, mask=None, **kwargs):
        """
        Forward pass of Sana.
        x: (N, C, T, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps or (N, 1, F) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        # print("Use Self forcing long generation")
        bs = x.shape[0]
        x = x.to(self.dtype)
        if self.timestep_norm_scale_factor != 1.0:
            timestep = (timestep.float() / self.timestep_norm_scale_factor).to(torch.float32)
        else:
            timestep = timestep.long().to(torch.float32)
        y = y.to(self.dtype)
        start_f = kwargs.get("start_f", None)
        end_f = kwargs.get("end_f", None)
        if start_f is not None and end_f is not None:
            assert self.pos_embed_type == "casual_wan_rope"
            start_f = start_f // self.patch_size[0]
            end_f = end_f // self.patch_size[0]

        self.f, self.h, self.w = (
            x.shape[-3] // self.patch_size[0],
            x.shape[-2] // self.patch_size[1],
            x.shape[-1] // self.patch_size[2],
        )

        data_info = kwargs.get("data_info", {})
        if data_info.get("image_vae_embeds", None) is not None:
            x = torch.cat([x, data_info["image_vae_embeds"].to(self.dtype)], dim=1)
        if data_info.get("image_embeds", None) is not None:
            image_embeds = data_info["image_embeds"].to(self.dtype)
            image_embeds = self.image_embedder(image_embeds)
            kwargs["image_embeds"] = image_embeds

        if self.pack_latents:
            x = self._pack_latents(x, bs, self.in_channels, self.h, self.w, self.f)
            self.h = self.h // 2
            self.w = self.w // 2
        if self.x_embedder.patch_size != self.x_embedder.kernel_size and self.x_embedder.kernel_size == (1, 2, 2):
            x = F.pad(x, (0, 1, 0, 1, 0, 0))

        x = self.x_embedder(x)
        image_pos_embed = None
        if self.use_pe:
            x, image_pos_embed = self._apply_positional_embedding(x, bs, start_f, end_f)

        t = self.t_embedder(timestep.flatten())  # (N, D)
        t0 = self.t_block(t)
        t = t.unflatten(dim=0, sizes=timestep.shape)
        t0 = t0.unflatten(dim=0, sizes=timestep.shape)
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

        for i, block in enumerate(self.blocks):
            x = auto_grad_checkpoint(
                block,
                x,
                y,
                t0,
                y_lens,
                (self.f, self.h, self.w),
                image_pos_embed,
                **kwargs,
                use_reentrant=False,
            )  # (N, T, D) #support grad checkpoint

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        if self.pack_latents:
            x = self._unpack_latents(x, self.h * 2, self.w * 2, self.f)

        return x

    def __call__(self, *args, **kwargs):
        """
        This method allows the object to be called like a function.
        It simply calls the forward method.
        """
        if "start_f" in kwargs:
            return self.forward_long(*args, **kwargs)
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
        p_f, p_h, p_w = self.x_embedder.patch_size
        h, w = self.h, self.w
        assert self.f * self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.f, h, w, p_f, p_h, p_w, c))
        x = torch.einsum("nfhwopqc->ncfohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, self.f * p_f, h * p_h, w * p_w))

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

        # Initialize cfg embedder
        if self.cfg_embedder:
            nn.init.normal_(self.cfg_embedder.mlp[0].weight, std=0.02)
            nn.init.zeros_(self.cfg_embedder.mlp[2].weight)
            if hasattr(self.cfg_embedder.mlp[2], "bias") and self.cfg_embedder.mlp[2].bias is not None:
                nn.init.zeros_(self.cfg_embedder.mlp[2].bias)

        for block in self.blocks:
            if hasattr(block, "flash_attn_additional") and block.flash_attn_additional is not None:
                nn.init.zeros_(block.flash_attn_additional.proj.weight)
                nn.init.zeros_(block.flash_attn_additional.proj.bias)

            if hasattr(block, "cross_attn") and hasattr(block.cross_attn, "image_kv_linear"):
                nn.init.zeros_(block.cross_attn.image_kv_linear.weight)
                nn.init.zeros_(block.cross_attn.image_kv_linear.bias)

    def load_state_dict(self, state_dict, strict=True):
        """when the channel in FFN is not the same as the checkpoint, load the checkpoint"""
        current_state_dict = self.state_dict()
        new_state_dict = {}

        for key, current_param in current_state_dict.items():
            checkpoint_param = state_dict.get(key)
            if checkpoint_param is None:
                if strict:
                    raise KeyError(f"Missing key in state dict: {key}")
                continue
            try:
                new_param = torch.zeros_like(current_param)

                if current_param.shape == checkpoint_param.shape:
                    new_param.copy_(checkpoint_param)
                    new_state_dict[key] = checkpoint_param
                    continue
                else:
                    self.logger(
                        f"Loading {key} from checkpoint, shape: {checkpoint_param.shape}, current_param.shape: {current_param.shape}"
                    )
                if "x_embedder.proj.weight" in key:
                    new_param[: checkpoint_param.shape[0], : checkpoint_param.shape[1]] = checkpoint_param
                elif "x_embedder.proj.bias" in key:
                    new_param[: checkpoint_param.shape[0]] = checkpoint_param
                elif "attn.qkv.weight" in key:
                    old_hidden_size = checkpoint_param.shape[1]
                    new_hidden_size = current_param.shape[1]
                    # split qkv into 3 parts
                    for i in range(3):
                        start_idx = i * old_hidden_size
                        new_start_idx = i * new_hidden_size
                        new_param[new_start_idx : new_start_idx + old_hidden_size, :old_hidden_size] = checkpoint_param[
                            start_idx : start_idx + old_hidden_size
                        ]
                elif "attn.qkv.bias" in key:
                    old_hidden_size = checkpoint_param.shape[0] // 3
                    new_hidden_size = current_param.shape[0] // 3
                    new_param[:old_hidden_size] = checkpoint_param[:old_hidden_size]
                    new_param[new_hidden_size : new_hidden_size + old_hidden_size] = checkpoint_param[
                        old_hidden_size : 2 * old_hidden_size
                    ]
                    new_param[2 * new_hidden_size : 2 * new_hidden_size + old_hidden_size] = checkpoint_param[
                        2 * old_hidden_size :
                    ]
                elif "q_norm.weight" in key:
                    new_param[: checkpoint_param.shape[0]] = checkpoint_param
                elif "q_norm.bias" in key:
                    new_param[: checkpoint_param.shape[0]] = checkpoint_param
                elif "k_norm.weight" in key:
                    new_param[: checkpoint_param.shape[0]] = checkpoint_param
                elif "k_norm.bias" in key:
                    new_param[: checkpoint_param.shape[0]] = checkpoint_param
                elif "cross_attn.q_linear.weight" in key:
                    new_param[: checkpoint_param.shape[0], : checkpoint_param.shape[1]] = checkpoint_param
                elif "cross_attn.q_linear.bias" in key:
                    new_param[: checkpoint_param.shape[0]] = checkpoint_param
                elif "cross_attn.kv_linear.weight" in key:
                    new_param[: checkpoint_param.shape[0], : checkpoint_param.shape[1]] = checkpoint_param
                elif "cross_attn.kv_linear.bias" in key:
                    new_param[: checkpoint_param.shape[0]] = checkpoint_param
                elif "attn.proj.weight" in key:
                    old_hidden_size = checkpoint_param.shape[0]
                    new_param[:old_hidden_size, :old_hidden_size] = checkpoint_param
                elif "attn.proj.bias" in key:
                    old_hidden_size = checkpoint_param.shape[0]
                    new_param[:old_hidden_size] = checkpoint_param
                elif "scale_shift_table" in key:
                    # scale_shift_table shape: [6, hidden_size]
                    old_hidden_size = checkpoint_param.shape[1]
                    new_param[:, :old_hidden_size] = checkpoint_param
                elif "final_layer.linear.weight" in key:
                    new_param[:, : checkpoint_param.shape[1]] = checkpoint_param
                elif "final_layer.linear.bias" in key:
                    new_param[: checkpoint_param.shape[0]] = checkpoint_param
                elif "t_embedder.mlp.0.weight" in key:
                    new_param[: checkpoint_param.shape[0]] = checkpoint_param
                elif "t_embedder.mlp.0.bias" in key:
                    new_param[: checkpoint_param.shape[0]] = checkpoint_param
                elif "t_embedder.mlp.2.weight" in key:
                    new_param[: checkpoint_param.shape[0], : checkpoint_param.shape[1]] = checkpoint_param
                elif "t_embedder.mlp.2.bias" in key:
                    new_param[: checkpoint_param.shape[0]] = checkpoint_param
                elif "t_block.1.weight" in key:
                    # t_block.1.weight shape: [6 * hidden_size, hidden_size]
                    old_hidden_size = checkpoint_param.shape[1]
                    new_hidden_size = current_param.shape[1]
                    # split t_block.1.weight into 6 parts
                    for i in range(6):
                        start_idx = i * old_hidden_size
                        new_start_idx = i * new_hidden_size
                        new_param[new_start_idx : new_start_idx + old_hidden_size, :old_hidden_size] = checkpoint_param[
                            start_idx : start_idx + old_hidden_size
                        ]
                elif "t_block.1.bias" in key:
                    # t_block.1.bias shape: [6 * hidden_size]
                    old_hidden_size = checkpoint_param.shape[0] // 6
                    new_hidden_size = current_param.shape[0] // 6
                    # split t_block.1.bias into 6 parts
                    for i in range(6):
                        start_idx = i * old_hidden_size
                        new_start_idx = i * new_hidden_size
                        new_param[new_start_idx : new_start_idx + old_hidden_size] = checkpoint_param[
                            start_idx : start_idx + old_hidden_size
                        ]
                elif "t_block.2.weight" in key:
                    new_param[: checkpoint_param.shape[0], : checkpoint_param.shape[1]] = checkpoint_param
                elif "y_embedder.y_proj.fc1.weight" in key:
                    new_param[: checkpoint_param.shape[0]] = checkpoint_param
                elif "y_embedder.y_proj.fc1.bias" in key:
                    new_param[: checkpoint_param.shape[0]] = checkpoint_param
                elif "y_embedder.y_proj.fc2.weight" in key:
                    new_param[: checkpoint_param.shape[0], : checkpoint_param.shape[1]] = checkpoint_param
                elif "y_embedder.y_proj.fc2.bias" in key:
                    new_param[: checkpoint_param.shape[0]] = checkpoint_param
                elif "y_embedder.y_embedding" in key:
                    pass
                elif "attention_y_norm.weight" in key:
                    new_param[: checkpoint_param.shape[0]] = checkpoint_param
                elif (
                    "inverted_conv.conv.weight" in key
                    or "inverted_conv.conv.bias" in key
                    or "depth_conv.conv.bias" in key
                ):
                    num_old_channels = checkpoint_param.shape[0] // 2
                    num_new_channels = new_param.shape[0] // 2
                    if new_param.dim() == 1:
                        new_param[:num_old_channels] = checkpoint_param[:num_old_channels]
                        new_param[num_new_channels : num_new_channels + num_old_channels] = checkpoint_param[
                            num_old_channels:
                        ]
                    else:
                        new_param[:num_old_channels, : checkpoint_param.shape[1]] = checkpoint_param[:num_old_channels]
                        new_param[
                            num_new_channels : num_new_channels + num_old_channels, : checkpoint_param.shape[1]
                        ] = checkpoint_param[num_old_channels:]
                elif "depth_conv.conv.weight" in key:
                    assert checkpoint_param.shape[1] == 1
                    num_old_channels = checkpoint_param.shape[0] // 2
                    new_param[:num_old_channels] = checkpoint_param[:num_old_channels]
                    new_param[num_new_channels : num_new_channels + num_old_channels] = checkpoint_param[
                        num_old_channels:
                    ]
                elif "point_conv.conv.weight" in key:
                    new_param[: checkpoint_param.shape[0], : checkpoint_param.shape[1]] = checkpoint_param
                elif "t_conv.weight" in key:
                    if new_param.shape[2] != checkpoint_param.shape[2]:
                        new_t_kernel_size = new_param.shape[2]
                        original_t_kernel_size = checkpoint_param.shape[2]
                        discrepancy = new_t_kernel_size - original_t_kernel_size
                        if discrepancy < 0 or discrepancy % 2 != 0:
                            raise ValueError(
                                f"Discrepancy {discrepancy} is not even or is negative, please check the t_kernel_size"
                            )
                        new_param[
                            : checkpoint_param.shape[0],
                            : checkpoint_param.shape[1],
                            discrepancy // 2 : -discrepancy // 2,
                        ] = checkpoint_param
                        # self.logger(f"Loading {key} with t_kernel_size {new_t_kernel_size} from checkpoint with t_kernel_size {original_t_kernel_size}")
                    else:
                        new_param[: checkpoint_param.shape[0], : checkpoint_param.shape[1]] = checkpoint_param
                else:
                    raise KeyError(f"Unhandled key: {key}")

            except Exception as e:
                print(f"Error loading {key}: {e}")
                new_param = checkpoint_param

            new_state_dict[key] = new_param

        return super().load_state_dict(new_state_dict, strict=strict)

    def register_block_hook(self, layers=None, device="cpu", detach=True, score_only=True):
        for i, block in enumerate(self.blocks):
            if layers is None or i in layers:
                block.block_hook = BlockHook(device, detach, score_only)

    def get_block_output(self):
        block_outputs = {}
        for i, block in enumerate(self.blocks):
            if block.block_hook is not None:
                block_outputs[i] = block.block_hook.get_output()
                block.block_hook.clear()
        return block_outputs


#################################################################################
#                             Sana Multi-scale Configs                          #
#################################################################################


@MODELS.register_module()
def SanaMSVideo_600M_P1_D28(**kwargs):
    return SanaMSVideo(depth=28, hidden_size=1152, patch_size=(1, 1, 1), num_heads=16, **kwargs)


@MODELS.register_module()
def SanaMSVideo_600M_P2_D28(**kwargs):
    return SanaMSVideo(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)


@MODELS.register_module()
def SanaMSVideo_2000M_P1_D20(**kwargs):
    # 20 layers, 1648.48M
    return SanaMSVideo(depth=20, hidden_size=2240, patch_size=(1, 1, 1), num_heads=20, **kwargs)


@MODELS.register_module()
def SanaMSVideo_2000M_P2_D20(**kwargs):
    # 20 layers, 1648.48M
    return SanaMSVideo(depth=20, hidden_size=2240, patch_size=(1, 2, 2), num_heads=20, **kwargs)


@MODELS.register_module()
def SanaMSVideo_2000M_P2S1_D20(**kwargs):
    # 20 layers, 1648.48M
    return SanaMSVideo(
        depth=20, hidden_size=2240, patch_size=(1, 1, 1), num_heads=20, patch_embed_kernel=(1, 2, 2), **kwargs
    )


@MODELS.register_module()
def SanaMSVideo_4000M_P2_D28(**kwargs):
    # 28 layers, 3459.49M
    return SanaMSVideo(depth=28, hidden_size=2560, patch_size=(1, 2, 2), num_heads=20, **kwargs)


@MODELS.register_module()
def SanaMSVideo_4800M_P1_D60(**kwargs):
    # 60 layers, 4800M
    return SanaMSVideo(depth=60, hidden_size=2240, patch_size=(1, 1, 1), num_heads=20, **kwargs)


@MODELS.register_module()
def SanaMSVideo_4800M_P2_D60(**kwargs):
    # 60 layers, 4800M
    return SanaMSVideo(depth=60, hidden_size=2240, patch_size=(1, 2, 2), num_heads=20, **kwargs)
