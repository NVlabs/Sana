# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

# Adapted from diffusers/models/transformers/transformer_wan.py for Bernini
# inference: variable-length attention with cu_seqlens, optional Ulysses
# sequence parallel, and latents that are patch-embedded by the caller.

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

from ..attention import varlen_attention
from ..parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    gen_cu_seqlens_for_cross_attn,
    get_parallel_state,
    pad_tensor,
    padding_tensor_for_seqeunce_parallel,
    slice_input_tensor,
    slice_input_tensor_scale_grad,
    unpad_tensor,
)


def _apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_rotated = torch.view_as_complex(x.to(torch.float64).unflatten(3, (-1, 2)))
    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
    return x_out.type_as(x)


class WanAttnProcessor2_0:
    """Attention processor using packed variable-length attention.

    Query/key/value are packed as ``[total_tokens, heads, head_dim]`` and the
    per-sample boundaries are given by ``cu_seqlens``. Under Ulysses sequence
    parallel, all-to-all collectives redistribute the sequence and head dims;
    on a single GPU those collectives are no-ops.
    """

    def _project_qkv(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        rotary_emb: Optional[torch.Tensor],
        origin_hidden_states_seq_len: Optional[int],
        is_cross_attn: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not is_cross_attn:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if not is_cross_attn:
            # Ulysses all-to-all: gather sequence, scatter heads (no-op single-GPU).
            query = gather_seq_scatter_heads(
                query, seq_dim=1, head_dim=2, unpadded_dim_size=origin_hidden_states_seq_len
            )
            key = gather_seq_scatter_heads(
                key, seq_dim=1, head_dim=2, unpadded_dim_size=origin_hidden_states_seq_len
            )
            value = gather_seq_scatter_heads(
                value, seq_dim=1, head_dim=2, unpadded_dim_size=origin_hidden_states_seq_len
            )

        # rotary_emb is only supplied for self-attention.
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, rotary_emb)
            key = _apply_rotary_emb(key, rotary_emb)

        return query, key, value

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        batch_image_vae_seqlen=None,
        text_features_length=None,
        origin_hidden_states_seq_len: Optional[int] = None,
        split_hidden_states_seq_len: Optional[int] = None,
        cu_seqlens_q_cache=None,
        max_seqlen_q_cache=None,
        cu_seqlens_k_cross_cache=None,
        cu_seqlens_q_cross_cache=None,
        max_seqlen_k_cross_cache=None,
        max_seqlen_q_cross_cache=None,
    ) -> torch.Tensor:
        is_cross_attn = encoder_hidden_states is not None

        query, key, value = self._project_qkv(
            attn, hidden_states, encoder_hidden_states, rotary_emb,
            origin_hidden_states_seq_len, is_cross_attn,
        )

        if not is_cross_attn:
            cu_seqlens_q = cu_seqlens_k = cu_seqlens_q_cache
            max_seqlen_q = max_seqlen_k = max_seqlen_q_cache
        else:
            cu_seqlens_q, max_seqlen_q = cu_seqlens_q_cross_cache, max_seqlen_q_cross_cache
            cu_seqlens_k, max_seqlen_k = cu_seqlens_k_cross_cache, max_seqlen_k_cross_cache

        query = query.squeeze(0).contiguous()
        key = key.squeeze(0).contiguous()
        value = value.squeeze(0).contiguous()

        if is_cross_attn and get_parallel_state().ulysses_enabled:
            # Drop query padding beyond this rank's real query length.
            padding_size = int(split_hidden_states_seq_len) - int(cu_seqlens_q[-1])
            if padding_size > 0:
                query = unpad_tensor(query, 0, padding_size)

        hidden_states = varlen_attention(
            query, key, value,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, causal=False,
        )
        hidden_states = hidden_states.unsqueeze(0)

        if get_parallel_state().ulysses_enabled:
            if not is_cross_attn:
                hidden_states = gather_heads_scatter_seq(hidden_states, head_dim=2, seq_dim=1)
            else:
                padding_size = int(split_hidden_states_seq_len) - hidden_states.shape[1]
                if padding_size > 0:
                    hidden_states = pad_tensor(hidden_states, 1, padding_size)

        hidden_states = hidden_states.flatten(2, 3).contiguous().type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanImageEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = nn.LayerNorm(out_features)

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")
        self.image_embedder = WanImageEmbedding(image_embed_dim, dim) if image_embed_dim is not None else None

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)

        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None and self.image_embedder is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
        use_src_id_rotary_emb: bool = False,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

        self.use_src_id_rotary_emb = use_src_id_rotary_emb
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor, source_id=None) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        self.freqs = self.freqs.to(hidden_states.device)
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)

        if self.use_src_id_rotary_emb:
            assert source_id is not None, "source_id is required when use_src_id_rotary_emb=True"
            # Compute the per-source rotary phase on the fly so that `source_id`
            # may be fractional. Rotary phase is continuous in the position, so a
            # fractional id (e.g. from interpolating many references into the
            # trained id range) lands inside the trained manifold rather than
            # extrapolating to unseen integer ids. Integer ids reproduce the old
            # precomputed-table behaviour exactly.
            pos = torch.tensor([float(source_id)], dtype=torch.float64, device=hidden_states.device)
            freqs_visual_id = get_1d_rotary_pos_embed(
                self.attention_head_dim, pos, self.theta,
                use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64,
            )
            freqs_visual_id = freqs_visual_id.view(1, 1, 1, -1).expand(ppf, pph, ppw, -1)
            freqs_visual_id = freqs_visual_id.reshape(1, 1, ppf * pph * ppw, -1)
            freqs = freqs * freqs_visual_id

        return freqs


class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim, heads=num_heads, kv_heads=num_heads, dim_head=dim // num_heads,
            qk_norm=qk_norm, eps=eps, bias=True, cross_attention_dim=None, out_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        self.attn2 = Attention(
            query_dim=dim, heads=num_heads, kv_heads=num_heads, dim_head=dim // num_heads,
            qk_norm=qk_norm, eps=eps, bias=True, cross_attention_dim=None, out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim, added_proj_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        batch_image_vae_seqlen=None,
        text_features_length=None,
        origin_hidden_states_seq_len: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        scale_shift_table = self.scale_shift_table.float()
        if temb.dim() == 4:
            # per-token temb: [1, total_seq_len, 6, dim]
            combined = scale_shift_table.unsqueeze(1) + temb.float()
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = combined.unbind(dim=2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            rotary_emb=rotary_emb,
            batch_image_vae_seqlen=batch_image_vae_seqlen,
            origin_hidden_states_seq_len=origin_hidden_states_seq_len,
            **kwargs,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            batch_image_vae_seqlen=batch_image_vae_seqlen,
            text_features_length=text_features_length,
            origin_hidden_states_seq_len=origin_hidden_states_seq_len,
            **kwargs,
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states


class WanTransformer3DModel(ModelMixin, ConfigMixin):
    """Wan transformer for video latents.

    Inference contract: the caller patch-embeds the latents (`patch_vae_latent`)
    and passes packed hidden states `[1, total_tokens, inner_dim]` together with
    the matching `rotary_emb` and per-sample `cu_seqlens` metadata.
    """

    _supports_gradient_checkpointing = False
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        use_src_id_rotary_emb: bool = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(
            attention_head_dim, patch_size, rope_max_seq_len, use_src_id_rotary_emb=use_src_id_rotary_emb
        )
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    def patch_vae_latent(self, hidden_states: torch.Tensor, source_id=None):
        """Patch-embed a VAE latent `[B,C,T,H,W]` into tokens, with its rotary emb."""
        rotary_emb = self.rope(hidden_states, source_id)
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # [B,C,T,H,W] -> [B,THW,C]
        return hidden_states, rotary_emb

    def patch_vae_embedding(self, hidden_states: torch.Tensor):
        """Patch-embed pre-packed VAE patches `[N,C,pt,ph,pw]` into `[N,inner_dim]`."""
        hidden_states = self.patch_embedding(hidden_states)
        return hidden_states.flatten(1)

    def prepare_inputs_for_sp(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        batch_image_vae_seqlen,
        text_features_length,
        timestep_proj_indices: torch.Tensor,
        temb: torch.Tensor,
    ):
        """Shard inputs across the Ulysses group and build cu_seqlens metadata.

        With Ulysses disabled, the slicing/padding steps are no-ops and the
        cu_seqlens describe the full (single-rank) sequence.
        """
        origin_hidden_states_seq_len = None
        split_hidden_states_seq_len = None

        if get_parallel_state().ulysses_enabled:
            origin_hidden_states_seq_len = hidden_states.shape[1]
            hidden_states = padding_tensor_for_seqeunce_parallel(hidden_states, dim=1)
            temb = padding_tensor_for_seqeunce_parallel(temb, dim=1)
            timestep_proj_indices = padding_tensor_for_seqeunce_parallel(timestep_proj_indices, dim=0)
            hidden_states = slice_input_tensor_scale_grad(hidden_states, dim=1)
            temb = slice_input_tensor_scale_grad(temb, dim=1)
            timestep_proj_indices = slice_input_tensor(timestep_proj_indices, dim=0)

        timestep_proj = timestep_proj[timestep_proj_indices].unsqueeze(0)

        device = hidden_states.device
        cu_seqlens_q_cache = torch.zeros(len(batch_image_vae_seqlen) + 1, dtype=torch.int32, device=device)
        cu_seqlens_q_cache[1:] = torch.tensor(
            batch_image_vae_seqlen, dtype=torch.int32, device=device
        ).cumsum(dim=0)
        max_seqlen_q_cache = max(batch_image_vae_seqlen)

        if get_parallel_state().ulysses_enabled:
            (
                cu_seqlens_k_cross_cache,
                cu_seqlens_q_cross_cache,
                max_seqlen_k_cross_cache,
                max_seqlen_q_cross_cache,
                split_hidden_states_seq_len,
            ) = gen_cu_seqlens_for_cross_attn(
                origin_hidden_states_seq_len, batch_image_vae_seqlen, text_features_length, device=device
            )
            encoder_hidden_states = encoder_hidden_states[
                :, cu_seqlens_k_cross_cache[0] : cu_seqlens_k_cross_cache[-1], :
            ]
            cu_seqlens_k_cross_cache = cu_seqlens_k_cross_cache - cu_seqlens_k_cross_cache[0]
        else:
            cu_seqlens_k_cross_cache = torch.zeros(
                len(text_features_length) + 1, dtype=torch.int32, device=device
            )
            cu_seqlens_k_cross_cache[1:] = torch.tensor(
                text_features_length, dtype=torch.int32, device=device
            ).cumsum(dim=0)
            max_seqlen_k_cross_cache = max(text_features_length)
            cu_seqlens_q_cross_cache = cu_seqlens_q_cache
            max_seqlen_q_cross_cache = max_seqlen_q_cache

        kwargs = {
            "origin_hidden_states_seq_len": origin_hidden_states_seq_len,
            "split_hidden_states_seq_len": split_hidden_states_seq_len,
            "cu_seqlens_q_cache": cu_seqlens_q_cache,
            "max_seqlen_q_cache": max_seqlen_q_cache,
            "cu_seqlens_k_cross_cache": cu_seqlens_k_cross_cache,
            "max_seqlen_k_cross_cache": max_seqlen_k_cross_cache,
            "cu_seqlens_q_cross_cache": cu_seqlens_q_cross_cache,
            "max_seqlen_q_cross_cache": max_seqlen_q_cross_cache,
        }
        return hidden_states, encoder_hidden_states, timestep_proj, temb, kwargs

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        batch_image_vae_seqlen,
        text_features_length,
        return_dict: bool = True,
    ):
        # hidden_states: [1, total_tokens, inner_dim], already patch-embedded.
        temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
            timestep, encoder_hidden_states, None
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # Expand the per-sample timestep embedding over each sample's token count.
        timestep_proj_indices = torch.repeat_interleave(
            torch.arange(len(batch_image_vae_seqlen), device=timestep_proj.device),
            torch.tensor(batch_image_vae_seqlen, device=timestep_proj.device),
        )
        temb = torch.cat(
            [temb[i : i + 1].expand(seq_len, -1) for i, seq_len in enumerate(batch_image_vae_seqlen)],
            dim=0,
        ).unsqueeze(0)  # [1, total_tokens, dim]

        # [1, seq, 1, head_dim] -> q/k layout
        rotary_emb = rotary_emb.transpose(1, 2)

        hidden_states, encoder_hidden_states, timestep_proj, temb, kwargs = self.prepare_inputs_for_sp(
            hidden_states, encoder_hidden_states, timestep_proj,
            batch_image_vae_seqlen, text_features_length, timestep_proj_indices, temb,
        )

        for block in self.blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states, timestep_proj, rotary_emb,
                batch_image_vae_seqlen, text_features_length, **kwargs,
            )

        # Output norm, projection.
        shift_table, scale_table = self.scale_shift_table.float().chunk(2, dim=1)
        shift = shift_table + temb.float()
        scale = scale_table + temb.float()
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        if get_parallel_state().ulysses_enabled:
            hidden_states = gather_outputs(
                hidden_states, gather_dim=1, padding_dim=1,
                unpad_dim_size=kwargs["origin_hidden_states_seq_len"],
            )

        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)
