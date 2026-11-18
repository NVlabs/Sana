# SPDX-License-Identifier: Apache-2.0
"""Ulysses sequence parallelism for the chunk-causal MiniMax H3 DiT.

``causal_model.py`` is single-rank: unlike the bidirectional model it inherited
no row arithmetic from upstream. This module keeps its full replicated input
embedding, slices that embedding at the SP boundary, and swaps in an attention
that exchanges sequence for heads, exactly as ``model_sp.py`` does for the
bidirectional model. The head exchange is that module's helper rather than a
second copy of it.

Both attention backends are sequence-parallel: the exchange wraps the kernel
call, so the cached varlen ``FlashAttention`` branch and the packed training
``FlexAttention`` branch (magi rectangles or a torch ``BlockMask``) each see a
full-sequence, head-sharded q/k/v and need no padding logic of their own.

The KV cache keeps the **post-exchange** k/v, i.e. ``[global_seq, local_heads,
head_dim]``: sequence-global, head-sharded. That is what makes
``utils/naive_cache.py`` need no change at all - every length it tracks stays a
global token count and it never inspects the head dims - and it is why the
routing tensors ``_prepare_cache_routing`` builds (``packed_query_indexes``,
``packed_past_key_value_indexes``, ``key_value_lens``, ``sample_lens``) are
passed through unsharded.

As in ``model_sp.py``, no row padding is introduced anywhere: the packer
already aligns the packed sequence to ``MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT``,
so ``sample_lens``/``q_ranges``/``k_ranges``/``attn_type_map`` and the
``BlockMask`` are consumed exactly as the caller built them. The state dict is
untouched: nothing here adds a parameter, buffer or submodule.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from dev.yanzuolu.common.distributed.unified_parallel import (
    Gather,
    Slice,
    gather_heads_scatter_seq,
    get_unified_parallel_group,
    get_unified_parallel_rank,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)
from dev.yanzuolu.utils.naive_cache import NaiveCache

from ..checkpointing import maybe_checkpoint
from .config import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
)
from .causal_model import (
    _CAUSAL_FLASH_ATTENTION,
    _CAUSAL_FLEX_ATTENTION,
    _CAUSAL_FORWARD_SUPPORTED_KWARGS,
    CausalMiniMaxH3Attention,
    CausalMiniMaxH3DiTModel,
)
from .model import (
    _BF16_DTYPE,
    _apply_qk_norm,
    _apply_rope_qk,
    _required_kwarg,
    _rope_cos_sin_cache,
)
from .model_sp import (
    _ulysses_scatter_heads,
    _unified_parallel_ctx,
    _validate_local_embedding_layout,
)


class CausalMiniMaxH3AttentionSP(CausalMiniMaxH3Attention):
    """CausalMiniMaxH3Attention with a sequence-to-heads Ulysses exchange.

    A pure ``forward`` override - no ``__init__``, no new attribute, and in
    particular ``layer_idx`` stays the one the parent set - so
    ``CausalMiniMaxH3DiTModelSP`` can install it by ``__class__`` assignment.
    """

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Any = None,
        q_ranges: Optional[torch.Tensor] = None,
        k_ranges: Optional[torch.Tensor] = None,
        attn_type_map: Optional[torch.Tensor] = None,
        attn_workloads: Any = None,
        sample_lens: Optional[torch.Tensor] = None,
        past_key_values: Optional[NaiveCache] = None,
        update_past_key_values: bool = False,
        key_value_lens: Optional[torch.Tensor] = None,
        packed_query_indexes: Optional[torch.Tensor] = None,
        packed_past_key_value_indexes: Optional[torch.Tensor] = None,
        capture_key_values: Optional[NaiveCache] = None,
    ) -> torch.Tensor:
        """x: [T_local, hidden] this rank's row shard -> [T_local, hidden]."""
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            return super().forward(
                x,
                rope_cache=rope_cache,
                attention_mask=attention_mask,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
                attn_workloads=attn_workloads,
                sample_lens=sample_lens,
                past_key_values=past_key_values,
                update_past_key_values=update_past_key_values,
                key_value_lens=key_value_lens,
                packed_query_indexes=packed_query_indexes,
                packed_past_key_value_indexes=packed_past_key_value_indexes,
                capture_key_values=capture_key_values,
            )

        total, n, d = x.shape[0], self.num_heads, self.head_dim

        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split(self.local_inner_dim, dim=-1)
        q = q.view(total, n, d)
        k = k.view(total, n, d)
        v = v.view(total, n, d)

        # RoPE before the exchange, on the rank-local rows, because rope_cache
        # is this rank's slice of img_position_ids (see model_sp.py; this is
        # where h3 diverges from wan). The fused qknorm+RoPE branch is skipped
        # for the same reason the parent skips it: model.py hard-disables it
        # via _SGL_KERNELS_AVAILABLE = False.
        cos_sin_cache, positions = rope_cache
        q, k = _apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        q, k = _apply_rope_qk(q, k, cos_sin_cache, positions)

        q, k, v = _ulysses_scatter_heads(q, k, v, get_unified_parallel_world_size())
        local_heads = q.shape[1]
        if capture_key_values is not None:
            capture_key_values.capture_selected_kvcache(self.layer_idx, k, v)


        if past_key_values is not None:  # cached rollout: varlen flash attention
            # k/v are sequence-global and head-sharded here, and every index
            # below is a global row index, so the merge is the parent's with
            # local_heads in place of n.
            if past_key_values.seq_lens(self.layer_idx) > 0:  # merge required
                persistent = past_key_values.merge_current_for_attention(
                    self.layer_idx, k, v
                )
                if persistent is not None:
                    k, v = persistent
                else:
                    past_keys = past_key_values.key_cache[self.layer_idx]
                    past_values = past_key_values.value_cache[self.layer_idx]
                    merged_len = packed_query_indexes.shape[0] + packed_past_key_value_indexes.shape[0]
                    merged_keys = past_keys.new_zeros(size=[merged_len, local_heads, d])
                    merged_values = past_values.new_zeros(size=[merged_len, local_heads, d])
                    merged_keys[packed_query_indexes] = k
                    merged_keys[packed_past_key_value_indexes] = past_keys
                    merged_values[packed_query_indexes] = v
                    merged_values[packed_past_key_value_indexes] = past_values
                    k, v = merged_keys, merged_values

            if update_past_key_values:
                past_key_values.update_kvcache(self.layer_idx, k, v)

            out = _CAUSAL_FLASH_ATTENTION(
                q,
                k,
                v,
                q_lens=sample_lens,
                k_lens=key_value_lens,
                softmax_scale=self.softmax_scale,
                causal=False,
            )
        else:  # packed training: flex (BlockMask) or flash_flex (rectangles)
            out = _CAUSAL_FLEX_ATTENTION(
                q,
                k,
                v,
                attention_mask,
                q_ranges,
                k_ranges,
                attn_type_map,
                attn_workloads,
                sample_lens,
            )

        out = gather_heads_scatter_seq(out.flatten(1), head_dim=1, seq_dim=0)
        out, _ = self.out_proj(out)
        return out


class CausalMiniMaxH3DiTModelSP(CausalMiniMaxH3DiTModel):
    """CausalMiniMaxH3DiTModel with the block stack sharded over packed rows.

    ``forward`` is a fork of the parent's, which is itself a fork of the
    bidirectional one; the differences are the explicit full-embedding-to-row
    ``Slice`` boundary, the sequence-parallel attention, and the output gather.
    The cache bookkeeping is untouched and stays global.
    """

    def __init__(
        self,
        config: Any,
        hf_config: dict[str, Any],
        quant_config: Any = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config, quant_config=quant_config)
        for block in self.blocks:
            if not isinstance(block.attn, CausalMiniMaxH3Attention):
                raise TypeError(f"Unsupported attention type: {type(block.attn).__name__}")
            block.attn.__class__ = CausalMiniMaxH3AttentionSP
        # As in model_sp.py, the text refiner is left alone: _embed refines the
        # whole prompt on every rank, so its attention sees no row shard.

    def _sequence_parallel_ctx(self) -> tuple[int, int]:
        """The context ``forward`` shards by, so ``build_rope_cache`` matches it."""
        return _unified_parallel_ctx()

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed causal forward over this rank's row shard.

        Returns the same global ``(video_logits, audio_logits)`` on every rank:
        the row shards are gathered before the output rows are selected.
        """
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            return super().forward(**kwargs)

        unexpected = sorted(set(kwargs) - _CAUSAL_FORWARD_SUPPORTED_KWARGS)
        if unexpected:
            raise TypeError(
                "CausalMiniMaxH3DiTModelSP.forward received unexpected kwargs: "
                f"{unexpected}; supported kwargs: "
                f"{sorted(_CAUSAL_FORWARD_SUPPORTED_KWARGS)}"
            )

        x = _required_kwarg(kwargs, "x")
        audio_x = _required_kwarg(kwargs, "audio_x")
        img_position_ids = _required_kwarg(kwargs, "img_position_ids")
        unique_timesteps = _required_kwarg(kwargs, "unique_timesteps")
        inverse_indices = (
            _required_kwarg(kwargs, "inverse_indices").view(-1).to(torch.long)
        )
        tap_outputs = kwargs.get("tap_outputs")
        tap_blocks = self._validate_taps(kwargs.get("tap_blocks"), tap_outputs)
        classify_mode = bool(kwargs.get("classify_mode", False))
        update_mask = _required_kwarg(kwargs, "update_mask")
        block_token_tags = kwargs.get("block_token_tags")
        token_tags = kwargs.get("token_tags")
        if block_token_tags is None:
            token_tags = _required_kwarg(kwargs, "token_tags").view(-1).to(torch.long)
        else:
            block_token_tags = block_token_tags.view(-1).to(torch.long)
            token_tags = None
        skip_mask_out_condition = bool(kwargs.get("skip_mask_out_condition", False))

        text_selected = _required_kwarg(kwargs, "prompt_embeds")

        img_pos = self._pos_ids(_required_kwarg(kwargs, "img_pos_info"), "img_pos_info")
        audio_pos = self._pos_ids(
            _required_kwarg(kwargs, "audio_pos_info"), "audio_pos_info"
        )
        text_pos = self._pos_ids(
            _required_kwarg(kwargs, "text_pos_info"),
            "text_pos_info",
        )
        infer_out_pos = self._pos_ids(
            _required_kwarg(kwargs, "img_pos_for_infer_output_info"),
            "img_pos_for_infer_output_info",
        )

        refiner_psp = _required_kwarg(kwargs, "refiner_packed_seq_params")
        refiner_cu = self._psp_field(
            refiner_psp, "refiner_packed_seq_params", "cu_seqlens_q"
        ).to(torch.int32)
        refiner_max = int(
            self._psp_field(refiner_psp, "refiner_packed_seq_params", "max_seqlen_q")
        )

        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        seq_len = int(x.shape[1])
        if token_tags is not None and token_tags.shape[0] != seq_len:
            raise ValueError(
                "token_tags must cover the full packed sequence "
                f"({seq_len}), got {token_tags.shape[0]}."
            )
        if inverse_indices.shape[0] != seq_len:
            raise ValueError(
                f"inverse_indices must be [{seq_len}], got {list(inverse_indices.shape)}"
            )
        device = x.device

        sp_ws = get_unified_parallel_world_size()
        sp_rank = get_unified_parallel_rank()
        # Same contract as model_sp.py: rows are never padded, so the world size
        # must divide the packer's alignment, and heads are padded rather than
        # required to divide.
        if MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT % sp_ws:
            raise ValueError(
                "packed sequence alignment "
                f"{MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT} not divisible by "
                f"unified-parallel world size {sp_ws}"
            )
        if seq_len % sp_ws:
            raise ValueError(
                f"packed seq_len {seq_len} not divisible by unified-parallel "
                f"world size {sp_ws}"
            )
        local_seq_len = seq_len // sp_ws
        row_start = sp_rank * local_seq_len
        row_stop = row_start + local_seq_len

        past_key_values: Optional[NaiveCache] = kwargs.get("past_key_values")
        update_past_key_values = bool(kwargs.get("update_past_key_values", False))
        sample_lens = _required_kwarg(kwargs, "sample_lens")
        if not torch.is_tensor(sample_lens):
            sample_lens = torch.tensor(sample_lens, dtype=torch.int32)
        sample_lens = sample_lens.view(-1).to(device=device, dtype=torch.int32)

        # RoPE is a pure function of the caller's absolute img_position_ids, so a
        # chunk carries its own positions and cached keys keep theirs. Under
        # sequence parallelism it is also row-local, since attention rotates
        # before the exchange.
        rope_cache = kwargs.get("rope_cache")
        if rope_cache is None:
            rope_freqs = self.rope(img_position_ids[:, row_start:row_stop]).to(device)
            rope_cache = (
                _rope_cos_sin_cache(rope_freqs, dtype=_BF16_DTYPE),
                torch.arange(local_seq_len, device=device, dtype=torch.long),
            )
        img_pos = img_pos.to(device)
        audio_pos = audio_pos.to(device)
        text_pos = text_pos.to(device)

        local_embedding_layout = kwargs.get("local_embedding_layout")
        if local_embedding_layout is not None:
            _validate_local_embedding_layout(
                local_embedding_layout, local_seq_len=local_seq_len
            )

        # Outside the checkpointed block stack, and over the FULL sequence on
        # every rank -- see model_sp.py for the measured cost of saving its
        # full-length internals instead of recomputing them.
        decoder_input, t_emb = maybe_checkpoint(
            self._embed,
            enabled=self.gradient_checkpointing,
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=text_selected,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos,
            audio_pos=audio_pos,
            text_pos=text_pos,
            refiner_cu_seqlens=refiner_cu.to(device),
            refiner_max_seqlen=refiner_max,
            row_start=0,
            row_stop=seq_len,
            device=device,
            refined_prompt_embeds_length=kwargs.get("refined_prompt_embeds_length"),
            local_embedding_layout=None,
        )
        if decoder_input.shape[0] % sp_ws:
            raise ValueError(
                f"full embedding length {decoder_input.shape[0]} not divisible by "
                f"unified-parallel world size {sp_ws}"
            )
        decoder_input = Slice.apply(
            get_unified_parallel_group(), decoder_input, 0, True
        )
        # request-step AdaLN input shared by all blocks
        adaln_input = nn.functional.silu(t_emb).to(_BF16_DTYPE)
        inverse_indices = inverse_indices.to(device)
        block_inverse = inverse_indices[row_start:row_stop]
        if block_token_tags is None:
            assert token_tags is not None
            token_tags = token_tags.to(device)
            block_token_tags = token_tags[row_start:row_stop].clamp(min=0)
        else:
            block_token_tags = block_token_tags.to(device)
            if block_token_tags.shape[0] != local_seq_len:
                raise ValueError(
                    "block_token_tags must cover the rank-local packed sequence "
                    f"({local_seq_len}), got {block_token_tags.shape[0]}."
                )
        block_combined = kwargs.get("block_combined_indices")
        if block_combined is None:
            block_combined = torch.add(
                block_token_tags,
                block_inverse,
                alpha=MINIMAX_H3_ADALN_MODALITY_NUM,
            )
        elif block_combined.shape[0] != local_seq_len:
            raise ValueError(
                "block_combined_indices must cover the rank-local packed "
                f"sequence ({local_seq_len}), got {block_combined.shape[0]}."
            )

        # Cache routing stays global: attention caches the post-exchange k/v,
        # which are sequence-global and only head-sharded.
        cache_kwargs = self._prepare_cache_routing(
            past_key_values=past_key_values,
            update_past_key_values=update_past_key_values,
            sample_lens=sample_lens,
            query_len=seq_len,
            device=device,
            q_ranges=kwargs.get("q_ranges"),
            attn_type_map=kwargs.get("attn_type_map"),
        )

        cache_kwargs["capture_key_values"] = kwargs.get("capture_key_values")
        hidden = self._checkpoint_block_stack(
            decoder_input,
            tap_blocks=tap_blocks,
            tap_outputs=tap_outputs,
            stop_after_last_tap=classify_mode and tap_blocks is not None,
            adaln_input=adaln_input,
            combined_indices=block_combined,
            rope_cache=rope_cache,
            adaln_params=None,
            attention_mask=kwargs.get("attention_mask"),
            q_ranges=kwargs.get("q_ranges"),
            k_ranges=kwargs.get("k_ranges"),
            attn_type_map=kwargs.get("attn_type_map"),
            attn_workloads=kwargs.get("attn_workloads"),
            **cache_kwargs,
        )

        if classify_mode:
            # Keep taps row-sharded for the discriminator's sequence-to-head
            # exchange. Its AdaLN uses the shared timestep embedding table.
            return tuple(tap_outputs or ()), t_emb

        video_logits, audio_logits = self.final_layer(
            hidden,
            adaln_input=adaln_input,
            inverse_indices=block_inverse,
        )
        # One collective for both heads, and before the row selection below:
        # infer_out_pos and audio_pos are global row indexes.
        video_width = video_logits.shape[-1]
        logits = Gather.apply(
            get_unified_parallel_group(),
            torch.cat((video_logits, audio_logits), dim=-1),
            0,
            True,
        )
        video_logits, audio_logits = logits.split(
            (video_width, logits.shape[-1] - video_width), dim=-1
        )

        video_logits = video_logits.index_select(0, infer_out_pos.to(device))
        audio_logits = audio_logits.index_select(0, audio_pos.to(device))
        if not skip_mask_out_condition:
            update_mask = update_mask.view(-1).to(device)
            if update_mask.shape[0] != video_logits.shape[0]:
                raise ValueError(
                    "update_mask length mismatch: "
                    f"{update_mask.shape[0]} != {video_logits.shape[0]}"
                )
            video_logits = video_logits * update_mask.unsqueeze(-1)
            # Audio has no condition rows in the supported tasks, so its derived
            # update mask is all ones. Honor an explicit mask when provided.
            update_audio_mask = kwargs.get("update_audio_mask")
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)
        return video_logits, audio_logits


__all__ = [
    "CausalMiniMaxH3AttentionSP",
    "CausalMiniMaxH3DiTModelSP",
]
