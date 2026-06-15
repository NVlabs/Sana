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


import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from diffusion.model.builder import MODELS
from diffusion.model.nets.sana_blocks import (
    CaptionEmbedder,
    ClipVisionProjection,
    PatchEmbedMS3D,
    T2IFinalLayer,
)
from diffusion.model.nets.sana_multi_scale_video import (
    SanaMSVideo,
    SanaVideoMSBlock,
)
from diffusion.model.registry import ATTENTION_BLOCKS, FFN_BLOCKS
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.utils.dist_utils import get_rank
from diffusion.utils.import_utils import is_xformers_available

_xformers_available = False if os.environ.get("DISABLE_XFORMERS", "0") == "1" else is_xformers_available()
if _xformers_available:
    import xformers.ops


def get_softmax_layer_indices(depth: int, softmax_ratio: float = 0.25) -> List[int]:
    """
    Calculate which layer indices should use softmax attention.

    By default, 25% of layers use softmax attention, evenly distributed.
    For a 20-layer model: layers [4, 9, 14, 19] would use softmax.

    Args:
        depth: Total number of layers
        softmax_ratio: Ratio of layers to use softmax attention (default 0.25)

    Returns:
        List of layer indices that should use softmax attention
    """
    if softmax_ratio == 0:
        return []
    num_softmax_layers = max(1, int(depth * softmax_ratio))
    step = depth / num_softmax_layers
    indices = [int((i + 1) * step) - 1 for i in range(num_softmax_layers)]
    return indices


class SanaV2VVideoMSBlock(SanaVideoMSBlock):
    """Sana video block with V2V-only registry fallbacks.

    This keeps custom V2V attention/FFN names local to ``SanaMSVideoV2V``
    instead of changing the base Sana-Video block used by other releases.
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
        super().__init__(
            hidden_size,
            num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            qk_norm=qk_norm,
            attn_type=attn_type,
            ffn_type=ffn_type,
            mlp_acts=mlp_acts,
            linear_head_dim=linear_head_dim,
            cross_norm=cross_norm,
            cross_attn_image_embeds=cross_attn_image_embeds,
            t_kernel_size=t_kernel_size,
            additional_flash_attn=additional_flash_attn,
            flash_attn_window_count=flash_attn_window_count,
            **block_kwargs,
        )

        if self.attn is None:
            attn_cls = ATTENTION_BLOCKS.get(attn_type) if attn_type else None
            if attn_cls is not None:
                self.attn = attn_cls(
                    in_dim=hidden_size,
                    out_dim=hidden_size,
                    heads=hidden_size // linear_head_dim,
                    eps=1e-8,
                    qk_norm=qk_norm,
                )

        if self.mlp is None:
            ffn_cls = FFN_BLOCKS.get(ffn_type) if ffn_type else None
            if ffn_cls is not None:
                self.mlp = ffn_cls(
                    in_features=hidden_size,
                    hidden_features=int(hidden_size * mlp_ratio),
                    use_bias=(True, True, False),
                    norm=(None, None, None),
                    act=mlp_acts,
                    t_kernel_size=t_kernel_size,
                )


#############################################################################
#                                 Core Sana Model                                #
#################################################################################
@MODELS.register_module()
class SanaMSVideoV2V(SanaMSVideo):
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
        flash_attn_layer_idx=None,
        flash_attn_layer_type=None,
        flash_attn_window_count=None,
        addition_layers_num=0,
        pack_latents=False,
        additional_inchannels=0,
        softmax_ratio: float = 0.0,
        softmax_layer_indices: Optional[List[int]] = None,
        softmax_attn_type="GDNSoftmaxAttention",
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
            rope_fhw_dim=rope_fhw_dim,
            t_kernel_size=t_kernel_size,
            flash_attn_layer_idx=flash_attn_layer_idx,
            flash_attn_layer_type=flash_attn_layer_type,
            flash_attn_window_count=flash_attn_window_count,
            addition_layers_num=addition_layers_num,
            pack_latents=pack_latents,
            additional_inchannels=additional_inchannels,
            **kwargs,
        )
        self.patch_size = patch_size
        self.h = self.w = 0
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.pos_embed_ms = None
        self.pack_latents = pack_latents
        self.addition_layers_num = addition_layers_num

        kernel_size = patch_embed_kernel or patch_size
        x_embedder_in_channels = (
            in_channels + additional_inchannels
            if additional_inchannels is not None and additional_inchannels > 0
            else in_channels
        )
        if self.pack_latents:
            x_embedder_in_channels = x_embedder_in_channels * 2 * 2
            self.out_channels = in_channels * 2 * 2
        elif self.addition_layers_num > 0:
            self.out_channels = in_channels

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

        # Calculate which layers use softmax attention
        if softmax_layer_indices is not None:
            self.softmax_layer_indices = softmax_layer_indices
        else:
            self.softmax_layer_indices = get_softmax_layer_indices(depth, softmax_ratio)

        if attn_type in ["flash", "FlexLinearAttention", "flex"]:
            attention_head_dim = hidden_size // num_heads
        else:
            attention_head_dim = linear_head_dim
        if self.use_pe:
            self.rope = self.get_rope(pos_embed_type, attention_head_dim, patch_size, rope_fhw_dim)
        else:
            self.rope = None

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule

        # insert flash attention layers
        if flash_attn_layer_idx is not None and flash_attn_layer_type is not None:
            assert int(flash_attn_layer_idx[-1]) < depth
            additional_flash_attn = [
                flash_attn_layer_type if i in flash_attn_layer_idx else False for i in range(depth)
            ]
        else:
            additional_flash_attn = [False] * depth

        # visualize qkv
        self.save_qkv = False
        self.qkv_store_buffer = {}

        # diagonal mask
        self.diagonal_mask = None
        attn_type_list = [attn_type] * depth
        if attn_type in ["flex", "FlexLinearAttention"]:
            attn_type_list[0] = "flash"
            attn_type_list[1] = "flash"

        for i in self.softmax_layer_indices:
            attn_type_list[i] = softmax_attn_type

        self.use_flex_attention = len([_attn_type for _attn_type in attn_type_list if "flex" in _attn_type.lower()]) > 0

        self.blocks = nn.ModuleList(
            [
                SanaV2VVideoMSBlock(
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
                    additional_flash_attn=additional_flash_attn[i],
                    flash_attn_window_count=flash_attn_window_count,
                )
                for i in range(depth)
            ]
        )

        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        if get_rank() == 0:
            if ffn_type == "GLUMBConvTemp":
                self.logger(f"{ffn_type} Temporal kernal: {t_kernel_size}")
            if flash_attn_layer_idx is not None:
                self.logger(f"additional flash attn layer idx: {flash_attn_layer_idx}, type: {flash_attn_layer_type}")
                if flash_attn_layer_type == "window_flash":
                    self.logger(f"flash attn window count: {flash_attn_window_count}")

        self.initialize()
        self.save_block_output = False
        self.block_output_buffer = {}

    def create_flexattention_chunkcausal_mask(self, x, THW, chunk_index=None):
        """
        Args:
            x: input tensor, shape (B, N, C)
            THW: tuple (f, h, w)
            chunk_indices: list or tensor, containing the start frame index of each chunk. If None, view each frame as a separate chunk.
        Returns:
            block_mask: BlockMask object
        """
        B, N, C = x.shape
        f, h, w = THW
        BLOCK_SIZE = 128
        chunk_id_map = torch.zeros(f, h, w, dtype=torch.long, device=x.device)
        if chunk_index is None:
            chunk_indices = range(f)
        else:
            chunk_indices = chunk_index

        for i, start_idx in enumerate(chunk_indices):
            chunk_id_map[start_idx:] = i
        chunk_id_map = chunk_id_map.view(f * h * w)

        pad_len = (BLOCK_SIZE - (N % BLOCK_SIZE)) % BLOCK_SIZE
        if pad_len > 0:
            # chunk_indices always larger than chunk_id_map, since the start index cannot be smaller than the frame index
            padding = torch.full((pad_len,), chunk_indices[-1] + 1, device=chunk_id_map.device)
            chunk_id_map = torch.cat([chunk_id_map, padding])

        def chunk_causal_mask_mod(b, h, q, kv):
            return chunk_id_map[q] >= chunk_id_map[kv]

        block_mask = create_block_mask(
            chunk_causal_mask_mod, B=None, H=None, Q_LEN=N, KV_LEN=N, device=x.device, BLOCK_SIZE=BLOCK_SIZE
        )

        return block_mask

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
        if self.save_qkv:
            self.qkv_store_buffer[int(timestep[0].item())] = {}
        if self.save_block_output:
            self.inference_timestep = int(timestep[0].item())

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

        if self.use_flex_attention:
            block_mask = self.create_flexattention_chunkcausal_mask(
                x, (self.f, self.h, self.w), kwargs.get("chunk_index", None)
            )
        else:
            block_mask = None

        for i, block in enumerate(self.blocks):
            if self.save_qkv:
                block.attn.qkv_store_buffer = {}

            x = auto_grad_checkpoint(
                block,
                x,
                y,
                t0,
                y_lens,
                (self.f, self.h, self.w),
                image_pos_embed,
                block_mask=block_mask,
                **kwargs,
                use_reentrant=False,
            )  # (N, T, D) #support grad checkpoint

            if self.save_qkv:
                self.qkv_store_buffer[int(timestep[0].item())][f"block_{i}"] = block.attn.qkv_store_buffer
                block.attn.qkv_store_buffer = None

        if self.addition_layers_num > 0:
            x = self.upsample_layer(x)
            x = self._unpack_latents_additional_layers(x, self.h * 2, self.w * 2, self.f)
            if self.pos_embed_type == "wan_rope":
                image_pos_embed = self.rope((self.f, self.h * 2, self.w * 2), x.device)
            else:
                raise ValueError(f"Unknown pos_embed_type: {self.pos_embed_type}")
            for i, block in enumerate(self.addition_layers):
                x = auto_grad_checkpoint(
                    block,
                    x,
                    y,
                    t0,
                    y_lens,
                    (self.f, self.h * 2, self.w * 2),
                    image_pos_embed,
                    block_mask=block_mask if i > 1 else None,
                    **kwargs,
                    use_reentrant=False,
                )

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        if self.pack_latents:
            x = self._unpack_latents(x, self.h * 2, self.w * 2, self.f)

        if self.save_block_output:
            block_output = self.get_block_output()
            self.block_output_buffer[self.inference_timestep] = block_output
        return x


#################################################################################
#                             Sana Multi-scale Configs                          #
#################################################################################


@MODELS.register_module()
def SanaMSVideoV2V_2000M_P1_D20(**kwargs):
    # 20 layers, 2B
    return SanaMSVideoV2V(depth=20, hidden_size=2240, patch_size=(1, 1, 1), num_heads=20, **kwargs)
