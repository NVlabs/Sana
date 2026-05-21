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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from diffusion.model.builder import MODELS
from diffusion.model.nets.basic_modules import GLUMBConv, GLUMBConvTemp, Mlp
from diffusion.model.nets.sana_blocks import (
    CaptionEmbedder,
    CausalWanRotaryPosEmbed,
    ClipVisionProjection,
    FlashAttention,
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
from diffusion.model.registry import ATTENTION_BLOCKS, FFN_BLOCKS
from diffusion.model.utils import auto_grad_checkpoint, create_block_mask_cached, generate_temporal_head_mask_mod
from diffusion.model.wan.model import BlockHook
from diffusion.utils.dist_utils import get_rank
from diffusion.utils.import_utils import is_xformers_available

from .sana_camctrl_blocks import (
    _maybe_drop_cam_branch,
    _process_camera_conditions_ucpe,
    prepare_prope_fns,
)
from .sana_gdn_blocks_triton import (
    BidirectionalGDNUCPESinglePathLiteLABothTriton,
)
from .sana_gdn_camctrl_blocks import (
    BidirectionalSoftmaxUCPESinglePathLiteLA,
)

# xformers is OFF by default on this stack (see diffusion/model/nets/sana_blocks.py for rationale).
# Opt in with ENABLE_XFORMERS=1.
_xformers_available = (
    os.environ.get("ENABLE_XFORMERS", "0") == "1"
    and os.environ.get("DISABLE_XFORMERS", "0") != "1"
    and is_xformers_available()
)
if _xformers_available:
    import xformers.ops


class DeltaActionEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_size, act_layer=nn.GELU):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            act_layer(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.mlp(x)


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class FP32NormProxy(nn.Module):
    def __init__(self, norm_module):
        super().__init__()
        self.norm = norm_module

    def forward(self, x):
        return self.norm(x.float()).type_as(x)


class SanaVideoMSCamCtrlBlock(nn.Module):
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
        camctrl_type=None,
        patch_size=(1, 2, 2),
        cam_attn_compress=2,
        fp32_norm=False,
        chunk_size=10,
        chunk_split_strategy="uniform",
        use_delta_pose_additive=False,
        use_chunk_plucker_post_attn=False,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.chunk_split_strategy = chunk_split_strategy

        if use_delta_pose_additive:
            self.delta_pose_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            nn.init.zeros_(self.delta_pose_proj.weight)
            nn.init.zeros_(self.delta_pose_proj.bias)

        if use_chunk_plucker_post_attn:
            self.plucker_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            nn.init.zeros_(self.plucker_proj.weight)
            nn.init.zeros_(self.plucker_proj.bias)

        if fp32_norm:
            self.norm1 = FP32LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if camctrl_type == "BidirectionalGDNUCPESinglePathLiteLABothTriton":
            # Both main and camera branches route through fused Triton kernels
            # (``fused_bigdn_func`` for main, ``cam_prep_func`` +
            # ``cam_scan_func`` for camera).  Module shapes and state-dict
            # keys are identical -- inference-only, no CP, no frame_valid_mask,
            # requires ``k_conv_only=True``.
            self_num_heads = hidden_size // linear_head_dim
            self.attn = BidirectionalGDNUCPESinglePathLiteLABothTriton(
                hidden_size,
                hidden_size,
                heads=self_num_heads,
                cam_dim=hidden_size // cam_attn_compress,
                cam_heads=max(1, self_num_heads // cam_attn_compress),
                eps=1e-8,
                qk_norm=qk_norm,
                patch_size=patch_size,
                **block_kwargs,
            )
        elif camctrl_type == "BidirectionalSoftmaxUCPESinglePathLiteLA":
            self_num_heads = hidden_size // linear_head_dim
            self.attn = BidirectionalSoftmaxUCPESinglePathLiteLA(
                hidden_size,
                hidden_size,
                heads=self_num_heads,
                cam_dim=hidden_size // cam_attn_compress,
                cam_heads=max(1, self_num_heads // cam_attn_compress),
                eps=1e-8,
                qk_norm=qk_norm,
                patch_size=patch_size,
                **block_kwargs,
            )
        else:
            # attn_type registered via ATTENTION_BLOCKS (e.g. "BidirectionalGDNTriton").
            attn_cls = ATTENTION_BLOCKS.get(attn_type)
            if attn_cls is None:
                raise ValueError(f"Unknown attn_type: {attn_type}")
            self.attn = attn_cls(
                hidden_size,
                hidden_size,
                heads=hidden_size // linear_head_dim,
                eps=1e-8,
                qk_norm=qk_norm,
            )

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
        if fp32_norm:
            self.norm2 = FP32LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if fp32_norm and self.attn is not None:
            if hasattr(self.attn, "q_norm"):
                self.attn.q_norm = FP32NormProxy(self.attn.q_norm)
            if hasattr(self.attn, "k_norm"):
                self.attn.k_norm = FP32NormProxy(self.attn.k_norm)
            if hasattr(self.attn, "norm_q"):
                self.attn.norm_q = FP32NormProxy(self.attn.norm_q)
            if hasattr(self.attn, "norm_k"):
                self.attn.norm_k = FP32NormProxy(self.attn.norm_k)

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
        elif ffn_type == "mlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
        else:
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
            else:
                self.mlp = None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
        self.block_hook: Optional[BlockHook] = None

    @staticmethod
    def _build_frame_token_mask(
        frame_valid_mask: Optional[torch.Tensor],
        *,
        B: int,
        T: int,
        N: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Convert frame-valid mask to token mask shaped ``(B, N, 1)``."""
        if frame_valid_mask is None:
            return None

        m = frame_valid_mask
        if m.ndim == 5:
            m = m[:, 0, :, 0, 0]
        elif m.ndim == 3 and m.shape[1] == 1:
            m = m[:, 0, :]
        elif m.ndim != 2:
            raise ValueError(
                "frame_valid_mask must be shaped (B, 1, T, 1, 1), (B, 1, T), or (B, T); "
                f"got shape={list(frame_valid_mask.shape)}"
            )

        if m.shape[0] != B or m.shape[1] != T:
            raise ValueError(f"frame_valid_mask shape mismatch: expected (B={B}, T={T}), got {list(m.shape)}")
        if T <= 0 or N % T != 0:
            raise ValueError(f"Invalid token/frame layout: N={N}, T={T}")

        S = N // T
        return m.to(device=device, dtype=dtype).view(B, T, 1).expand(B, T, S).reshape(B, N, 1)

    def forward_frame_aware(
        self, x, y, t, mask=None, THW=None, rotary_emb=None, block_mask=None, chunk_index=None, **kwargs
    ):
        B, N, C = x.shape
        num_frames = t.shape[2]
        frame_valid_mask = kwargs.get("frame_valid_mask", None)
        frame_token_mask = self._build_frame_token_mask(
            frame_valid_mask,
            B=B,
            T=num_frames,
            N=N,
            device=x.device,
            dtype=x.dtype,
        )
        if frame_token_mask is not None:
            x = x * frame_token_mask

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
            "block_mask": block_mask,
            "camera_conditions": kwargs.get("camera_conditions", None),
            "prope_fns": kwargs.get("prope_fns", None),
            "camera_embedding": kwargs.get("camera_embedding", None),
            "frame_valid_mask": frame_valid_mask,
        }
        cam_branch_drop_prob = kwargs.get("cam_branch_drop_prob", None)
        if cam_branch_drop_prob is not None:
            self_attn_kwargs["cam_branch_drop_prob"] = cam_branch_drop_prob
        if chunk_index is not None:
            self_attn_kwargs["chunk_index"] = chunk_index[:]  # NOTE: important, copy the list
        if kwargs.get("chunk_index_global", None) is not None:
            self_attn_kwargs["chunk_index_global"] = kwargs.get("chunk_index_global")
        chunk_split_strategy = kwargs.get("chunk_split_strategy", getattr(self, "chunk_split_strategy", "uniform"))
        if chunk_split_strategy is not None:
            self_attn_kwargs["chunk_split_strategy"] = chunk_split_strategy

        chunk_size = kwargs.get("chunk_size", getattr(self, "chunk_size", 10))
        if chunk_size is not None:
            self_attn_kwargs["chunk_size"] = chunk_size

        x_norm1 = self.norm1(x).reshape(B, num_frames, -1, C)
        x_msa_in = t2i_modulate(x_norm1, shift_msa, scale_msa).reshape(B, N, C)
        if frame_token_mask is not None:
            x_msa_in = x_msa_in * frame_token_mask
        attn_out = self.attn(x_msa_in, **self_attn_kwargs).reshape(B, num_frames, -1, C)
        attn_out = (gate_msa * attn_out).reshape(B, N, C)
        if frame_token_mask is not None:
            attn_out = attn_out * frame_token_mask
        x = x + self.drop_path(attn_out)
        if frame_token_mask is not None:
            x = x * frame_token_mask

        delta_pose_emb = kwargs.get("delta_pose_emb", None)
        if delta_pose_emb is not None and hasattr(self, "delta_pose_proj"):
            S = N // num_frames
            dpe = delta_pose_emb.unsqueeze(2).expand(-1, -1, S, -1).reshape(B, N, C)
            x = x + self.delta_pose_proj(dpe)

        plucker_emb = kwargs.get("plucker_emb", None)
        if plucker_emb is not None and hasattr(self, "plucker_proj"):
            x = x + self.plucker_proj(plucker_emb)

        if self.flash_attn_additional:
            x = x + self.flash_attn_additional(x, HW=THW)
            if frame_token_mask is not None:
                x = x * frame_token_mask

        if self.cross_attn_image_embeds:
            x = x + self.cross_attn(x, y, mask=mask, image_embeds=kwargs.get("image_embeds", None))
        else:
            x = x + self.cross_attn(x, y, mask=mask)
        if frame_token_mask is not None:
            x = x * frame_token_mask

        mlp_kwargs = {
            "HW": THW,
            "frame_valid_mask": frame_valid_mask,
        }
        if chunk_index is not None:
            mlp_kwargs["chunk_index"] = chunk_index[:]  # NOTE: important, copy the list
        if kwargs.get("chunk_index_global", None) is not None:
            mlp_kwargs["chunk_index_global"] = kwargs.get("chunk_index_global")
        if chunk_split_strategy is not None:
            mlp_kwargs["chunk_split_strategy"] = chunk_split_strategy

        chunk_size = kwargs.get("chunk_size", getattr(self, "chunk_size", 10))
        if chunk_size is not None:
            mlp_kwargs["chunk_size"] = chunk_size

        x_norm2 = self.norm2(x).reshape(B, num_frames, -1, C)
        x_mlp_in = t2i_modulate(x_norm2, shift_mlp, scale_mlp).reshape(B, N, C)
        if frame_token_mask is not None:
            x_mlp_in = x_mlp_in * frame_token_mask
        mlp_out = self.mlp(x_mlp_in, **mlp_kwargs).reshape(B, num_frames, -1, C)
        mlp_out = (gate_mlp * mlp_out).reshape(B, N, C)
        if frame_token_mask is not None:
            mlp_out = mlp_out * frame_token_mask
        x = x + self.drop_path(mlp_out)
        if frame_token_mask is not None:
            x = x * frame_token_mask

        return x

    def forward(self, x, y, t, mask=None, THW=None, rotary_emb=None, block_mask=None, chunk_index=None, **kwargs):
        if len(t.shape) > 2:
            return self.forward_frame_aware(
                x,
                y,
                t,
                mask=mask,
                THW=THW,
                rotary_emb=rotary_emb,
                block_mask=block_mask,
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
        frame_valid_mask = kwargs.get("frame_valid_mask", None)
        frame_token_mask = (
            self._build_frame_token_mask(
                frame_valid_mask,
                B=B,
                T=THW[0],
                N=N,
                device=x.device,
                dtype=x.dtype,
            )
            if THW is not None
            else None
        )
        if frame_token_mask is not None:
            x = x * frame_token_mask
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x_sa_in = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        if frame_token_mask is not None:
            x_sa_in = x_sa_in * frame_token_mask
        self_attn_kwargs = {
            "HW": THW,
            "rotary_emb": rotary_emb,
            "block_mask": block_mask,
            "camera_conditions": kwargs.get("camera_conditions", None),
            "prope_fns": kwargs.get("prope_fns", None),
            "frame_valid_mask": frame_valid_mask,
        }
        cam_branch_drop_prob = kwargs.get("cam_branch_drop_prob", None)
        if cam_branch_drop_prob is not None:
            self_attn_kwargs["cam_branch_drop_prob"] = cam_branch_drop_prob
        if chunk_index is not None:
            self_attn_kwargs["chunk_index"] = chunk_index[:]  # NOTE: important, copy the list
        if kwargs.get("chunk_index_global", None) is not None:
            self_attn_kwargs["chunk_index_global"] = kwargs.get("chunk_index_global")
        chunk_split_strategy = kwargs.get("chunk_split_strategy", getattr(self, "chunk_split_strategy", "uniform"))
        if chunk_split_strategy is not None:
            self_attn_kwargs["chunk_split_strategy"] = chunk_split_strategy

        chunk_size = kwargs.get("chunk_size", getattr(self, "chunk_size", 10))
        if chunk_size is not None:
            self_attn_kwargs["chunk_size"] = chunk_size

        if frame_token_mask is not None:
            x_sa = x_sa * frame_token_mask

        intermediate_feats["x_self_attn"] = x_sa

        if self.flash_attn_additional:
            x_sa = x_sa + self.learnable_fa_scale * self.flash_attn_additional(x_sa_in, rotary_emb=rotary_emb, HW=THW)
            if frame_token_mask is not None:
                x_sa = x_sa * frame_token_mask

        x = x + self.drop_path(gate_msa * x_sa)
        if frame_token_mask is not None:
            x = x * frame_token_mask

        delta_pose_emb = kwargs.get("delta_pose_emb", None)
        if delta_pose_emb is not None and hasattr(self, "delta_pose_proj"):
            T_dp = delta_pose_emb.shape[1]
            S_dp = N // T_dp
            dpe = delta_pose_emb.unsqueeze(2).expand(-1, -1, S_dp, -1).reshape(B, N, C)
            x = x + self.delta_pose_proj(dpe)

        plucker_emb = kwargs.get("plucker_emb", None)
        if plucker_emb is not None and hasattr(self, "plucker_proj"):
            x = x + self.plucker_proj(plucker_emb)

        if self.cross_attn_image_embeds:
            x = x + self.cross_attn(x, y, mask=mask, image_embeds=kwargs.get("image_embeds", None))
        else:
            x = x + self.cross_attn(x, y, mask=mask)
        if frame_token_mask is not None:
            x = x * frame_token_mask

        intermediate_feats["x_cross_attn"] = x

        mlp_kwargs = {
            "HW": THW,
            "frame_valid_mask": frame_valid_mask,
        }
        if chunk_index is not None:
            mlp_kwargs["chunk_index"] = chunk_index[:]  # NOTE: important, copy the list
        if kwargs.get("chunk_index_global", None) is not None:
            mlp_kwargs["chunk_index_global"] = kwargs.get("chunk_index_global")
        if chunk_split_strategy is not None:
            mlp_kwargs["chunk_split_strategy"] = chunk_split_strategy

        chunk_size = kwargs.get("chunk_size", getattr(self, "chunk_size", 10))
        if chunk_size is not None:
            mlp_kwargs["chunk_size"] = chunk_size

        if frame_token_mask is not None:
            mlp_out = mlp_out * frame_token_mask
        x = x + self.drop_path(gate_mlp * mlp_out)
        if frame_token_mask is not None:
            x = x * frame_token_mask

        intermediate_feats["x_ffn"] = x

        if self.block_hook is not None:
            self.block_hook(**intermediate_feats)

        return x


_GDN_TO_SOFTMAX_CAMCTRL: dict[str, str] = {
    "BidirectionalGDNUCPESinglePathLiteLABothTriton": "BidirectionalSoftmaxUCPESinglePathLiteLA",
}


def _inject_softmax_layers(
    attn_type_list: list,
    camctrl_type_list: list,
    softmax_every_n: int,
) -> tuple:
    """Replace every ``softmax_every_n``-th block's camctrl variant with its softmax counterpart.

    Pattern: for ``softmax_every_n=4``, blocks 3, 7, 11, ... (0-indexed at n-1) use
    softmax attention; the remaining blocks keep GDN. Blocks whose camctrl_type has
    no softmax mapping are left as-is.
    """
    attn_out = list(attn_type_list)
    camctrl_out = list(camctrl_type_list)
    for i in range(len(attn_out)):
        if (i + 1) % softmax_every_n != 0:
            continue
        if camctrl_out[i] in _GDN_TO_SOFTMAX_CAMCTRL:
            camctrl_out[i] = _GDN_TO_SOFTMAX_CAMCTRL[camctrl_out[i]]
    return attn_out, camctrl_out


class SanaMSVideoCamCtrl(Sana):
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
        pack_latents=False,
        camctrl_type: str = "PluckerPatchifyAdd",
        camctrl_layers_num: int = None,
        cam_attn_compress: int = 2,
        init_cam_from_base: bool = False,
        use_delta_actions: bool = False,
        delta_action_dim: int = 16 * 4,
        use_delta_translation: bool = False,
        fp32_norm: bool = False,
        chunk_size: int = 10,
        chunk_split_strategy: str = "uniform",
        conv_kernel_size: int = 4,
        k_conv_only: bool = True,
        softmax_every_n: int = 4,
        use_delta_pose_additive: bool = False,
        delta_pose_additive_dim: int = 64,
        use_chunk_plucker_input: bool = False,
        use_chunk_plucker_post_attn: bool = False,
        chunk_plucker_channels: int = 48,
        chunk_plucker_post_attn_blocks: int = -1,
        use_autograd_kernel: bool = False,
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
        self.chunk_size = chunk_size
        self.chunk_split_strategy = chunk_split_strategy
        self.patch_size = patch_size
        self.h = self.w = 0
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.pos_embed_ms = None
        self.pack_latents = pack_latents
        self.attn_type = attn_type

        self.camctrl_type = camctrl_type
        assert self.camctrl_type in [
            "BidirectionalGDNUCPESinglePathLiteLABothTriton",
            "BidirectionalSoftmaxUCPESinglePathLiteLA",
        ], f"Not supported camera control type: {self.camctrl_type}"

        self.camctrl_layers_num = camctrl_layers_num if camctrl_layers_num is not None else depth
        self.cam_attn_compress = cam_attn_compress
        self.init_cam_from_base = init_cam_from_base
        self.use_delta_actions = use_delta_actions
        self.use_delta_translation = use_delta_translation
        self.use_delta_pose_additive = use_delta_pose_additive

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

        if self.use_delta_actions:
            self.delta_action_embedder = DeltaActionEmbedder(
                input_dim=delta_action_dim,
                hidden_size=hidden_size,
                act_layer=approx_gelu,
            )
            nn.init.zeros_(self.delta_action_embedder.mlp[-1].weight)
            nn.init.zeros_(self.delta_action_embedder.mlp[-1].bias)

        if self.use_delta_translation:
            self.delta_translation_embedder = DeltaActionEmbedder(
                input_dim=3,
                hidden_size=hidden_size,
                act_layer=approx_gelu,
            )
            nn.init.zeros_(self.delta_translation_embedder.mlp[-1].weight)
            nn.init.zeros_(self.delta_translation_embedder.mlp[-1].bias)

        if self.use_delta_pose_additive:
            self.delta_pose_embedder = DeltaActionEmbedder(
                input_dim=delta_pose_additive_dim,
                hidden_size=hidden_size,
                act_layer=approx_gelu,
            )

        self.use_chunk_plucker_input = use_chunk_plucker_input
        self.use_chunk_plucker_post_attn = use_chunk_plucker_post_attn
        if self.use_chunk_plucker_input or self.use_chunk_plucker_post_attn:
            self.plucker_embedder = PatchEmbedMS3D(
                patch_size, chunk_plucker_channels, hidden_size, kernel_size=kernel_size, bias=True
            )
            nn.init.zeros_(self.plucker_embedder.proj.weight)
            nn.init.zeros_(self.plucker_embedder.proj.bias)

        # UCPE-style camera branch uses a 3-channel absmap (up_map + lat_map).
        self.raymap_embedder = PatchEmbedMS3D(patch_size, 3, hidden_size, kernel_size=kernel_size, bias=True)

        if cross_attn_image_embeds:
            self.image_embedder = ClipVisionProjection(image_embed_channels, hidden_size)
        else:
            self.image_embedder = None

        if attn_type in ["flash", "FlexLinearAttention", "flex"]:
            attention_head_dim = hidden_size // num_heads
        else:
            attention_head_dim = linear_head_dim

        if use_pe and pos_embed_type == "wan_rope":
            self.rope = WanRotaryPosEmbed(
                attention_head_dim=attention_head_dim, patch_size=patch_size, max_seq_len=1024, fhw_dim=rope_fhw_dim
            )
        elif use_pe and pos_embed_type == "casual_wan_rope":
            self.rope = CausalWanRotaryPosEmbed(
                attention_head_dim=attention_head_dim, patch_size=patch_size, max_seq_len=1024
            )
        elif use_pe and pos_embed_type == "wan_temporal_rope":
            self.rope = WanRotaryTemporalPosEmbed(
                attention_head_dim=attention_head_dim, patch_size=patch_size, max_seq_len=1024
            )
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
        self.softmax_every_n = softmax_every_n
        attn_type_list = [attn_type] * depth
        camctrl_type_list = [camctrl_type if i < self.camctrl_layers_num else None for i in range(depth)]
        if attn_type in ["flex", "FlexLinearAttention"]:
            attn_type_list[0] = "flash"
            attn_type_list[1] = "flash"

        if softmax_every_n > 0:
            attn_type_list, camctrl_type_list = _inject_softmax_layers(
                attn_type_list,
                camctrl_type_list,
                softmax_every_n,
            )
            if get_rank() == 0:
                self.logger(
                    f"Hybrid attention (softmax_every_n={softmax_every_n}):\n"
                    f"  attn_type_list = {attn_type_list}\n"
                    f"  camctrl_type_list = {camctrl_type_list}"
                )

        self.blocks = nn.ModuleList(
            [
                SanaVideoMSCamCtrlBlock(
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
                    camctrl_type=camctrl_type_list[i],
                    patch_size=patch_size,
                    cam_attn_compress=self.cam_attn_compress,
                    fp32_norm=fp32_norm,
                    chunk_size=chunk_size,
                    chunk_split_strategy=chunk_split_strategy,
                    conv_kernel_size=conv_kernel_size,
                    k_conv_only=k_conv_only,
                    use_delta_pose_additive=use_delta_pose_additive,
                    use_chunk_plucker_post_attn=(
                        use_chunk_plucker_post_attn
                        and (chunk_plucker_post_attn_blocks < 0 or i < chunk_plucker_post_attn_blocks)
                    ),
                    use_autograd_kernel=use_autograd_kernel,
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

    def _compute_rope_with_cp(self, device: torch.device, h: int, w: int) -> torch.Tensor:
        """Compute RoPE frequencies for the local frame window."""
        return self.rope((self.f, h, w), device)

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

        cam_embeds = kwargs.get("camera_conditions", None)
        cam_branch_drop_prob = kwargs.get("cam_branch_drop_prob", 0.0)
        if cam_embeds is not None and cam_branch_drop_prob:
            # Keep drop-path semantics consistent: when camera branch is dropped,
            # skip both camera-attention branch and camera embedding injection.
            cam_embeds = _maybe_drop_cam_branch(
                cam_embeds,
                cam_branch_drop_prob,
                self.training,
                x.device,
            )
            if cam_embeds is None:
                kwargs["camera_conditions"] = None
        if self.pack_latents:
            x = self._pack_latents(x, bs, self.in_channels, self.h, self.w, self.f)
            if cam_embeds is not None:
                cam_embeds = cam_embeds.to(self.dtype)

            self.h = self.h // 2
            self.w = self.w // 2

        if self.x_embedder.patch_size != self.x_embedder.kernel_size and self.x_embedder.kernel_size == (1, 2, 2):
            x = F.pad(x, (0, 1, 0, 1, 0, 0))
            if cam_embeds is not None:
                cam_embeds = F.pad(cam_embeds, (0, 1, 0, 1, 0, 0))

        x = self.x_embedder(x)
        if cam_embeds is not None:
            # Both surviving camctrl variants are UCPE-style: build raymats + 3-channel
            # absmap (up_map + lat_map) from the raw (B,F,20) camera conditions.
            raw_cam_conditions = cam_embeds
            cam_pos_embeds = kwargs.get("cam_pos_embeds", None)
            if cam_pos_embeds is not None and "absmap" in cam_pos_embeds:
                cam_embeds = cam_pos_embeds["absmap"]
                if "P" in cam_pos_embeds:
                    kwargs["raymats"] = cam_pos_embeds["P"]
            else:
                raymats, cam_embeds = _process_camera_conditions_ucpe(
                    raw_cam_conditions, bs, (self.f, self.h, self.w), self.patch_size
                )
                cam_embeds = cam_embeds.permute(0, 4, 1, 2, 3).to(self.dtype)
                kwargs["raymats"] = raymats
            _skip_absmap = getattr(self, "use_chunk_plucker_input", False) or getattr(
                self, "use_chunk_plucker_post_attn", False
            )
            if not _skip_absmap:
                cam_embeds = self.raymap_embedder(cam_embeds)
                x = x + cam_embeds
                kwargs["camera_embedding"] = cam_embeds
                kwargs["camera_conditions"] = raw_cam_conditions

        if getattr(self, "use_chunk_plucker_input", False) and "chunk_plucker" in kwargs:
            plucker_input = kwargs["chunk_plucker"].to(self.dtype)
            plucker_emb = self.plucker_embedder(plucker_input)
            x = x + plucker_emb

        if getattr(self, "use_chunk_plucker_post_attn", False) and "chunk_plucker" in kwargs:
            plucker_input = kwargs["chunk_plucker"].to(self.dtype)
            kwargs["plucker_emb"] = self.plucker_embedder(plucker_input)

        image_pos_embed = kwargs.get("pos_embeds", None)
        if self.use_pe and image_pos_embed is None:
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
            elif self.pos_embed_type == "flux_rope":
                self.pos_embed_ms = RopePosEmbed(theta=10000, axes_dim=[12, 10, 10])
                latent_image_ids = self.pos_embed_ms._prepare_latent_image_ids(
                    bs, self.h, self.w, x.device, x.dtype, frame=self.f
                )
                image_pos_embed = self.pos_embed_ms(latent_image_ids)
            elif self.pos_embed_type == "wan_rope":
                image_pos_embed = self._compute_rope_with_cp(x.device, self.h, self.w)
            elif self.pos_embed_type == "casual_wan_rope":
                image_pos_embed = self.rope((self.f, self.h, self.w), x.device)
            elif self.pos_embed_type == "wan_temporal_rope":
                image_pos_embed = self._compute_rope_with_cp(x.device, self.h, self.w)
            else:
                raise ValueError(f"Unknown pos_embed_type: {self.pos_embed_type}")
        elif image_pos_embed is not None:
            image_pos_embed = image_pos_embed.to(x.device)
            while image_pos_embed.ndim > 4:
                image_pos_embed = image_pos_embed.squeeze(1)

        # --- FSDP2 block timing (SANA_FSDP2_BLOCK_TIMING=1) ---
        import os as _os_fwd

        _fsdp2_block_timing = _os_fwd.environ.get("SANA_FSDP2_BLOCK_TIMING", "0") in ("1", "true")
        if _fsdp2_block_timing:
            import time as _time_fwd

            torch.cuda.synchronize()
            _t_embed_start = _time_fwd.perf_counter()

        t = self.t_embedder(timestep.flatten())  # (N, D)
        t0 = self.t_block(t)
        t = t.unflatten(dim=0, sizes=timestep.shape)
        t0 = t0.unflatten(dim=0, sizes=timestep.shape)

        # Compute delta embeddings for final_layer (stored separately, not touching t/t0)
        _delta_t_emb = None
        if getattr(self, "use_delta_actions", False) and "delta_actions" in kwargs:
            da = kwargs["delta_actions"].to(self.dtype)
            _delta_t_emb = self.delta_action_embedder(da)  # (B, T, D)

        if getattr(self, "use_delta_translation", False) and kwargs.get("camera_conditions") is not None:
            cam_cond = kwargs["camera_conditions"].to(self.dtype)
            c2w = cam_cond[:, :, :16].view(cam_cond.shape[0], cam_cond.shape[1], 4, 4)
            t_cam = c2w[:, :, :3, 3]  # (B, T, 3)
            delta_t = t_cam[:, 1:, :] - t_cam[:, :-1, :]
            delta_t = torch.cat([torch.zeros_like(delta_t[:, :1, :]), delta_t], dim=1)
            dt_emb = self.delta_translation_embedder(delta_t)  # (B, T, D)
            _delta_t_emb = dt_emb if _delta_t_emb is None else _delta_t_emb + dt_emb

        if getattr(self, "use_delta_pose_additive", False) and "delta_actions" in kwargs:
            da = kwargs["delta_actions"].to(self.dtype)
            kwargs["delta_pose_emb"] = self.delta_pose_embedder(da)  # (B, T, D)

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

        if self.diagonal_mask is not None:
            seq_len = x.shape[1]
            self.diagonal_mask = self.diagonal_mask.to(x.device)
            # self.diagonal_mask = torch.ones_like(self.diagonal_mask).bool().to(x.device)

            def mask_mod(b, h, q_idx, kv_idx):
                return self.diagonal_mask[q_idx, kv_idx].bool()

            block_mask = create_block_mask_cached(
                mask_mod, None, None, seq_len, seq_len, device=x.device, _compile=False
            )
        else:
            block_mask = None

        if kwargs.get("camera_conditions") is not None:
            # Pre-compute UCPE projection functions to share across blocks
            # (both surviving camctrl variants are UCPE-style).
            if self.attn_type in ["flash", "FlexLinearAttention", "flex"]:
                head_dim = self.hidden_size // self.num_heads
            else:
                head_dim = self.linear_head_dim

            cam_pos_embeds = kwargs.get("cam_pos_embeds", None)
            if cam_pos_embeds is not None:
                for k, v in cam_pos_embeds.items():
                    if isinstance(v, torch.Tensor):
                        v = v.to(x.device)
                        if k == "absmap":
                            while v.ndim > 5:
                                v = v.squeeze(1)
                        else:
                            while v.ndim > 4:
                                v = v.squeeze(1)
                        cam_pos_embeds[k] = v

            kwargs["prope_fns"] = prepare_prope_fns(
                camctrl_type="UCPE",
                head_dim=head_dim,
                camera_conditions=kwargs["camera_conditions"],
                HW=(self.f, self.h, self.w),
                patch_size=self.patch_size,
                rotary_emb=image_pos_embed,
                raymats=kwargs.get("raymats"),
                cam_pos_embeds=cam_pos_embeds,
            )

        if _fsdp2_block_timing:
            torch.cuda.synchronize()
            _t_pre_blocks = _time_fwd.perf_counter()
            print(f"[FSDP2-BT] embeddings+prep: {(_t_pre_blocks - _t_embed_start)*1000:.1f}ms", flush=True)

        for i, block in enumerate(self.blocks):
            if self.save_qkv:
                block.attn.qkv_store_buffer = {}

            if _fsdp2_block_timing:
                torch.cuda.synchronize()
                _t_blk_start = _time_fwd.perf_counter()

            x = auto_grad_checkpoint(
                block,
                x,
                y,
                t0,
                y_lens,
                (self.f, self.h, self.w),
                image_pos_embed,
                block_mask=block_mask if i > 1 else None,
                **kwargs,
                use_reentrant=False,
            )  # (N, T, D) #support grad checkpoint

            if _fsdp2_block_timing:
                torch.cuda.synchronize()
                _t_blk_end = _time_fwd.perf_counter()
                _blk_ms = (_t_blk_end - _t_blk_start) * 1000
                _attn_name = (
                    type(block.attn).__name__
                    if not hasattr(block, "_checkpoint_wrapped_module")
                    else type(getattr(block, "_checkpoint_wrapped_module", block).attn).__name__
                )
                print(f"[FSDP2-BT] block[{i}] ({_attn_name}): {_blk_ms:.1f}ms", flush=True)

            if self.save_qkv:
                self.qkv_store_buffer[int(timestep[0].item())][f"block_{i}"] = block.attn.qkv_store_buffer
                block.attn.qkv_store_buffer = None

        if _fsdp2_block_timing:
            torch.cuda.synchronize()
            _t_post_blocks = _time_fwd.perf_counter()
            print(f"[FSDP2-BT] all blocks: {(_t_post_blocks - _t_pre_blocks)*1000:.1f}ms", flush=True)

        if _delta_t_emb is not None:
            if t.ndim == 2:
                t = t.unsqueeze(1).expand(-1, _delta_t_emb.shape[1], -1)
            elif t.ndim == 4:
                t = t.squeeze(1)
            t = t + _delta_t_emb
            t = t.unsqueeze(1)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        if self.pack_latents:
            x = self._unpack_latents(x, self.h * 2, self.w * 2, self.f)

        if self.save_block_output:
            block_output = self.get_block_output()
            self.block_output_buffer[self.inference_timestep] = block_output
        return x

    # ------------------------------------------------------------------ #
    # AR KV-cache streaming forward
    # ------------------------------------------------------------------ #

    _SOFTMAX_OPTION_Y_TYPES: tuple = ()  # filled lazily; see _is_softmax_option_y_block

    @staticmethod
    def _is_softmax_option_y_block(block: nn.Module) -> bool:
        """Return ``True`` iff ``block.attn`` is a softmax-attention camctrl
        variant (``_SoftmaxUCPESinglePathLiteLA`` or registered aliases).

        Detected by class name (rather than ``isinstance``) to avoid the
        circular import that would result from importing the class here.
        """
        attn_cls_name = type(block.attn).__name__
        return attn_cls_name in (
            "_SoftmaxUCPESinglePathLiteLA",
            "BidirectionalSoftmaxUCPESinglePathLiteLA",
            "ChunkCausalSoftmaxUCPESinglePathLiteLA",
            "SoftmaxUCPELiteLA",
        )

    def __call__(self, *args, **kwargs):
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

    def create_diagonal_mask(self, N_pad, N, num_frames, block_size=1, mask_type="nlogn"):
        from tools.attn_mask.gen_nlogn_mask import (
            gen_linear_mask_shrinked,
            gen_log_mask_shrinked,
            gen_truncated_mask_shrinked,
        )

        if mask_type == "nlogn":
            diagonal_mask = gen_log_mask_shrinked(N, N, num_frames, block_size)
        elif mask_type == "linear":
            diagonal_mask = gen_linear_mask_shrinked(N, N, num_frames, block_size)
        elif mask_type == "truncated":
            diagonal_mask = gen_truncated_mask_shrinked(N, N, num_frames, block_size, max_frame_distance=8)
        else:
            raise ValueError(f'Unknown mask type: {mask_type}, only support "nlogn", "linear", "truncated"')
        padded_mask = torch.zeros((N_pad, N_pad), dtype=torch.bool)
        padded_mask[:N, :N] = diagonal_mask
        self.diagonal_mask = padded_mask
        return self.diagonal_mask

    def prepare_flexattention(
        self,
        cfg_size,
        num_head,
        head_dim,
        dtype,
        device,
        context_length,
        prompt_length,
        num_frame,
        frame_size,
        diag_width=1,
        multiplier=2,
    ):
        from torch.nn.attention.flex_attention import flex_attention

        assert diag_width == multiplier, f"{diag_width} is not equivalent to {multiplier}"

        seq_len = context_length + num_frame * frame_size
        query, key, value = (
            torch.zeros((cfg_size, num_head, seq_len, head_dim), dtype=dtype, device=device) for _ in range(3)
        )

        mask_mod = generate_temporal_head_mask_mod(context_length, prompt_length, num_frame, frame_size, mul=multiplier)
        block_mask = create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)

        hidden_states = flex_attention(query, key, value, block_mask=block_mask)

        return block_mask

    def load_diagonal_mask(self, *args, **kwargs):
        path = kwargs.get("path", None)
        if path is None:
            self.diagonal_mask = self.prepare_flexattention(*args, **kwargs)
        else:
            self.diagonal_mask = torch.load(path, map_location="cpu")

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

            if hasattr(block, "attn") and hasattr(block.attn, "prope_proj"):
                nn.init.zeros_(block.attn.prope_proj.weight)
                nn.init.zeros_(block.attn.prope_proj.bias)

            if hasattr(block, "attn") and hasattr(block.attn, "out_proj_cam"):
                nn.init.zeros_(block.attn.out_proj_cam.weight)
                nn.init.zeros_(block.attn.out_proj_cam.bias)

            if hasattr(block, "attn") and hasattr(block.attn, "_init_gdn_gates_for_linear_equiv"):
                block.attn._init_gdn_gates_for_linear_equiv()

        if hasattr(self, "raymap_embedder") and self.raymap_embedder is not None:
            nn.init.constant_(self.raymap_embedder.proj.weight, 0)
            if self.raymap_embedder.proj.bias is not None:
                nn.init.constant_(self.raymap_embedder.proj.bias, 0)

        if self.init_cam_from_base:
            self.init_cam_branch_from_base()

    def load_state_dict(self, state_dict, strict=True, **kwargs):
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
                        if discrepancy == 0:
                            new_param[: checkpoint_param.shape[0], : checkpoint_param.shape[1]] = checkpoint_param
                        elif discrepancy > 0:
                            if discrepancy % 2 != 0:
                                raise ValueError(
                                    f"Discrepancy {discrepancy} is not even, please check the t_kernel_size"
                                )
                            new_param[
                                : checkpoint_param.shape[0],
                                : checkpoint_param.shape[1],
                                discrepancy // 2 : -discrepancy // 2,
                            ] = checkpoint_param
                        else:
                            if (-discrepancy) % 2 != 0:
                                raise ValueError(
                                    f"Discrepancy {discrepancy} is not even, please check the t_kernel_size"
                                )
                            start = (-discrepancy) // 2
                            end = start + new_t_kernel_size
                            new_param[: checkpoint_param.shape[0], : checkpoint_param.shape[1]] = checkpoint_param[
                                :, :, start:end
                            ]
                        # self.logger(
                        #     f"Loading {key} with t_kernel_size {new_t_kernel_size} from checkpoint with t_kernel_size {original_t_kernel_size}"
                        # )
                    else:
                        new_param[: checkpoint_param.shape[0], : checkpoint_param.shape[1]] = checkpoint_param
                else:
                    raise KeyError(f"Unhandled key: {key}")

            except Exception as e:
                print(f"Error loading {key}: {e}")
                new_param = checkpoint_param

            new_state_dict[key] = new_param

        result = super().load_state_dict(new_state_dict, strict=strict, **kwargs)

        return result

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

    def init_cam_branch_from_base(self):
        for i, block in enumerate(self.blocks):
            if hasattr(block.attn, "init_cam_branch_weights"):
                block.attn.init_cam_branch_weights()


#################################################################################
#                             Sana Multi-scale Configs                          #
#################################################################################


@MODELS.register_module()
def SanaMSVideoCamCtrl_600M_P1_D28(**kwargs):
    return SanaMSVideoCamCtrl(depth=28, hidden_size=1152, patch_size=(1, 1, 1), num_heads=16, **kwargs)


@MODELS.register_module()
def SanaMSVideoCamCtrl_600M_P2_D28(**kwargs):
    return SanaMSVideoCamCtrl(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)


@MODELS.register_module()
def SanaMSVideoCamCtrl_1600M_P1_D20(**kwargs):
    # 20 layers, 1648.48M
    return SanaMSVideoCamCtrl(depth=20, hidden_size=2240, patch_size=(1, 1, 1), num_heads=20, **kwargs)


@MODELS.register_module()
def SanaMSVideoCamCtrl_1600M_P2_D20(**kwargs):
    # 20 layers, 1648.48M
    return SanaMSVideoCamCtrl(depth=20, hidden_size=2240, patch_size=(1, 2, 2), num_heads=20, **kwargs)


@MODELS.register_module()
def SanaMSVideoCamCtrl_1600M_P2S1_D20(**kwargs):
    # 20 layers, 1648.48M
    return SanaMSVideoCamCtrl(
        depth=20, hidden_size=2240, patch_size=(1, 1, 1), num_heads=20, patch_embed_kernel=(1, 2, 2), **kwargs
    )


@MODELS.register_module()
def SanaMSVideoCamCtrl_2000M_P2_D20(**kwargs):
    # 20 layers, 2016.26M
    return SanaMSVideoCamCtrl(depth=20, hidden_size=2304, patch_size=(1, 2, 2), num_heads=18, **kwargs)


@MODELS.register_module()
def SanaMSVideoCamCtrl_2800M_P2_D28(**kwargs):
    # 28 layers, 2861.13M
    return SanaMSVideoCamCtrl(depth=28, hidden_size=2240, patch_size=(1, 2, 2), num_heads=20, **kwargs)


@MODELS.register_module()
def SanaMSVideoCamCtrl_4000M_P2_D28(**kwargs):
    # 28 layers, 3459.49M
    return SanaMSVideoCamCtrl(depth=28, hidden_size=2560, patch_size=(1, 2, 2), num_heads=20, **kwargs)


@MODELS.register_module()
def SanaMSVideoCamCtrl_4800M_P1_D60(**kwargs):
    # 60 layers, 4800M
    return SanaMSVideoCamCtrl(depth=60, hidden_size=2240, patch_size=(1, 1, 1), num_heads=20, **kwargs)


@MODELS.register_module()
def SanaMSVideoCamCtrl_4800M_P2_D60(**kwargs):
    # 60 layers, 4800M
    return SanaMSVideoCamCtrl(depth=60, hidden_size=2240, patch_size=(1, 2, 2), num_heads=20, **kwargs)
