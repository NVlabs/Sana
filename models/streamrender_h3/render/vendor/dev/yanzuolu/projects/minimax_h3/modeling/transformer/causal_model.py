# SPDX-License-Identifier: Apache-2.0
"""Chunked causal MiniMax H3 DiT.

The bidirectional DiT in ``model.py`` attends densely over one packed sequence
of ``[text | media]`` rows. This module turns it into a chunk-causal student:
the packed sequence is cut into time-ordered chunks, a chunk sees itself in
full plus whatever earlier chunks the sink/window policy retains, and rollout
replays that visibility one chunk at a time against a KV cache.

Two paths, exactly as in ``projects/wan_t2v/modeling/causal_model.py``:

* **training** - one packed forward whose visibility comes from explicit
  rectangles. ``FlexAttention`` dispatches internally on
  ``FLEX_FLASH_ATTN_AVAILABLE``: magi ``flex_flash_attn_func`` consumes
  ``q_ranges``/``k_ranges``/``attn_type_map`` directly, and the torch
  ``flex_attention`` fallback consumes a ``BlockMask`` plus ``sample_lens``
  for its padding. Both sets of arguments are threaded through so either
  backend works; ltx2 only ever wired the magi one, which left its torch path
  dead, and that mistake is not repeated here.
* **cached inference** - one chunk at a time through varlen
  ``FlashAttention``, with the chunk's K/V merged against ``NaiveCache``.

The state dict is identical to the bidirectional model's: every subclass here
either adds no submodules at all or swaps one for a same-parameter subclass,
and the two attention wrappers are module-level singletons for the same reason
``model.py`` keeps ``_MINIMAX_H3_FLASH_ATTENTION`` at module level - so the
module tree, and therefore the state dict, is untouched.

Unlike wan, no RoPE re-basing is needed. H3 takes ``img_position_ids`` from the
caller and ``rope_cache`` is a pure function of them, so a chunk simply carries
its own absolute positions and the cached keys keep the ones they were rotated
with. wan's ``curr_rope`` bookkeeping has no counterpart here.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

from dev.yanzuolu.utils.flash_attn import FlashAttention
from dev.yanzuolu.utils.flex_attn import FlexAttention
from dev.yanzuolu.utils.naive_cache import NaiveCache

from ..checkpointing import maybe_checkpoint
from .config import MINIMAX_H3_ADALN_MODALITY_NUM
from .model import (
    _BF16_DTYPE,
    _apply_qk_norm,
    _apply_rope_qk,
    _modulate_gate,
    _modulate_scale_shift,
    _required_kwarg,
    _rope_cos_sin_cache,
    MiniMaxH3Attention,
    MiniMaxH3DiTBlock,
    MiniMaxH3DiTModel,
)

# Neither wrapper holds parameters or buffers, and a module-level instance keeps
# them out of the module tree, matching model.py's _MINIMAX_H3_FLASH_ATTENTION.
_CAUSAL_FLEX_ATTENTION = FlexAttention()
_CAUSAL_FLASH_ATTENTION = FlashAttention()


class CausalMiniMaxH3Attention(MiniMaxH3Attention):
    """MiniMaxH3Attention with chunk-causal routing and an optional KV cache.

    Adds ``layer_idx`` (a plain int, not a parameter) and replaces the parent's
    single varlen call with the training/inference split. The projection,
    q/k RMSNorm and RoPE steps are the parent's, minus the fused-kernel branch:
    ``model.py`` hard-disables it via ``_SGL_KERNELS_AVAILABLE = False``, so
    ``self._use_fused_qknorm_rope`` is always False in this build.
    """

    def __init__(self, *args, layer_idx: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_idx = layer_idx

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
        """x: [T, hidden] packed rows -> [T, hidden]."""
        total, n, d = x.shape[0], self.num_heads, self.head_dim

        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split(self.local_inner_dim, dim=-1)
        q = q.view(total, n, d)
        k = k.view(total, n, d)
        v = v.view(total, n, d)

        cos_sin_cache, positions = rope_cache
        q, k = _apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        q, k = _apply_rope_qk(q, k, cos_sin_cache, positions)

        if capture_key_values is not None:
            capture_key_values.capture_selected_kvcache(self.layer_idx, k, v)

        if past_key_values is not None:  # cached rollout: varlen flash attention
            if past_key_values.seq_lens(self.layer_idx) > 0:  # merge required
                past_keys = past_key_values.key_cache[self.layer_idx]
                past_values = past_key_values.value_cache[self.layer_idx]
                merged_len = packed_query_indexes.shape[0] + packed_past_key_value_indexes.shape[0]
                # Scatter rather than cat: the merged layout must interleave
                # per sample as [past_0, new_0, past_1, new_1, ...] while both
                # sources are contiguous-by-sample.
                merged_keys = past_keys.new_zeros(size=[merged_len, n, d])
                merged_values = past_values.new_zeros(size=[merged_len, n, d])
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

        out = out.reshape(total, n * d)
        out, _ = self.out_proj(out)
        return out


class CausalMiniMaxH3DiTBlock(MiniMaxH3DiTBlock):
    """MiniMaxH3DiTBlock whose attention is the causal one.

    The parent builds ``self.attn`` first and it is replaced here rather than
    re-running the whole ``__init__`` body: the causal attention registers the
    same parameters under the same names, so the swap is state-dict neutral,
    and under ``meta_init`` the discarded modules cost nothing.
    """

    def __init__(self, arch, quant_config, *, prefix: str, layer_idx: int) -> None:
        super().__init__(arch, quant_config, prefix=prefix)
        self.attn = CausalMiniMaxH3Attention(
            arch,
            quant_config,
            prefix=f"{prefix}.attn",
            layer_idx=layer_idx,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        adaln_params: tuple[torch.Tensor, ...] | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Identical to the parent, with the causal routing kwargs forwarded."""
        # Verbatim fork of MiniMaxH3DiTBlock.forward (model.py:1092-1139 as
        # vendored); re-check this body against it after any re-vendor.
        if adaln_params is None:
            adaln_params = self.adaln_proj(adaln_input)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_params

        residual = x
        h = self.norm1(x)
        h = _modulate_scale_shift(
            h, shift_msa, scale_msa, combined_indices, dtype=_BF16_DTYPE
        )
        h = self.attn(h, rope_cache=rope_cache, **attn_kwargs)
        x = _modulate_gate(residual, gate_msa, h, combined_indices, dtype=_BF16_DTYPE)

        residual = x
        h = self.norm2(x)
        h = _modulate_scale_shift(
            h, shift_mlp, scale_mlp, combined_indices, dtype=_BF16_DTYPE
        )
        h = self.mlp(h)
        return _modulate_gate(
            residual, gate_mlp, h, combined_indices, dtype=_BF16_DTYPE
        )


# The bidirectional contract minus ``packed_seq_params`` - its cu_seqlens and
# max_seqlen describe the dense document layout that the causal router replaces
# - plus the routing and cache arguments. ``refiner_packed_seq_params`` stays:
# the text refiner is unchanged and still runs over the whole prompt.
_CAUSAL_FORWARD_SUPPORTED_KWARGS = frozenset(
    {
        "x",
        "audio_x",
        "img_position_ids",
        "rope_cache",
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "update_audio_mask",
        "tap_blocks",
        "tap_outputs",
        "classify_mode",
        "token_tags",
        "block_token_tags",
        "block_combined_indices",
        "skip_mask_out_condition",
        "prompt_embeds",
        "refined_prompt_embeds_length",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "local_embedding_layout",
        "refiner_packed_seq_params",
        # causal routing
        "sample_lens",
        "attention_mask",
        "q_ranges",
        "k_ranges",
        "attn_type_map",
        "attn_workloads",
        "past_key_values",
        "update_past_key_values",
        "capture_key_values",
    }
)


class CausalMiniMaxH3DiTModel(MiniMaxH3DiTModel):
    """Chunk-causal MiniMax H3 DiT.

    Same parameters as ``MiniMaxH3DiTModel``, so a bidirectional checkpoint
    loads into it unchanged.

    ``forward`` is a fork of the parent's rather than an override of a smaller
    hook, because the parent keeps its packed forward in one method. The two
    real differences are the attention routing threaded into the block stack
    and the KV-cache bookkeeping in front of it; everything else is the parent's
    body with the Ulysses branches dropped, since this model is single-rank
    (``_ulysses_ctx`` is hard-wired to ``(1, 0)`` in this vendored build).
    """

    def __init__(self, config, hf_config: dict[str, Any], quant_config: Any = None) -> None:
        if config.arch_config.adaln_condition_dim:
            raise NotImplementedError(
                "per-token AdaLN conditioning is implemented for the bidirectional DiT only"
            )
        if config.arch_config.input_condition_channels:
            raise NotImplementedError(
                "per-token input conditioning is implemented for the bidirectional DiT only"
            )
        super().__init__(config=config, hf_config=hf_config, quant_config=quant_config)
        arch = config.arch_config
        # Rebuilt rather than built in place: the parent has no block-construction
        # hook, and the causal blocks carry the same parameters under the same
        # names, so this is state-dict neutral.
        self.blocks = nn.ModuleList(
            [
                CausalMiniMaxH3DiTBlock(
                    arch,
                    quant_config,
                    prefix=f"blocks.{index}",
                    layer_idx=index,
                )
                for index in range(arch.num_layers)
            ]
        )
        self._mark_missing_params_required()

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed causal forward.

        Training passes the routing arguments and leaves ``past_key_values``
        None; rollout passes one chunk plus the cache.
        ``refiner_packed_seq_params.cu_seqlens_q`` has B+1 entries describing
        exactly the B live text documents, ending at the total live text rows;
        padding is not described. ``text_pos_info.position_ids`` concatenates
        those per-sample text rows in document order. Returns
        ``(video_logits, audio_logits)`` for the rows selected by
        ``img_pos_for_infer_output_info`` and ``audio_pos_info``.
        """
        # Fork of MiniMaxH3DiTModel.forward (model.py:1657-1912 as vendored);
        # re-check this body against it after any re-vendor.
        unexpected = sorted(set(kwargs) - _CAUSAL_FORWARD_SUPPORTED_KWARGS)
        if unexpected:
            raise TypeError(
                "CausalMiniMaxH3DiTModel.forward received unexpected kwargs: "
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

        past_key_values: Optional[NaiveCache] = kwargs.get("past_key_values")
        update_past_key_values = bool(kwargs.get("update_past_key_values", False))
        sample_lens = _required_kwarg(kwargs, "sample_lens")
        if not torch.is_tensor(sample_lens):
            sample_lens = torch.tensor(sample_lens, dtype=torch.int32)
        sample_lens = sample_lens.view(-1).to(device=device, dtype=torch.int32)

        # RoPE is a pure function of the caller's absolute img_position_ids, so a
        # chunk carries its own positions and cached keys keep theirs. Serving
        # prepares this once per request; direct callers get the fallback.
        rope_cache = kwargs.get("rope_cache")
        if rope_cache is None:
            rope_freqs = self.rope(img_position_ids).to(device)
            rope_cache = (
                _rope_cos_sin_cache(rope_freqs, dtype=_BF16_DTYPE),
                torch.arange(seq_len, device=device, dtype=torch.long),
            )
        img_pos = img_pos.to(device)
        audio_pos = audio_pos.to(device)
        text_pos = text_pos.to(device)

        # Outside the checkpointed block stack -- see model_sp.py for the
        # measured cost of saving its full-length internals.
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
            local_embedding_layout=kwargs.get("local_embedding_layout"),
        )
        # request-step AdaLN input shared by all blocks
        adaln_input = nn.functional.silu(t_emb).to(_BF16_DTYPE)
        inverse_indices = inverse_indices.to(device)
        if block_token_tags is None:
            assert token_tags is not None
            block_token_tags = token_tags.to(device).clamp(min=0)
        else:
            block_token_tags = block_token_tags.to(device)
            if block_token_tags.shape[0] != seq_len:
                raise ValueError(
                    "block_token_tags must cover the packed sequence "
                    f"({seq_len}), got {block_token_tags.shape[0]}."
                )
        block_combined = kwargs.get("block_combined_indices")
        if block_combined is None:
            block_combined = torch.add(
                block_token_tags,
                inverse_indices,
                alpha=MINIMAX_H3_ADALN_MODALITY_NUM,
            )

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
            # Return graph-attached taps across the FSDP boundary. Mutating the
            # caller's tap_outputs list cannot carry them across that boundary.
            return tuple(tap_outputs or ()), t_emb

        video_logits, audio_logits = self.final_layer(
            hidden,
            adaln_input=adaln_input,
            inverse_indices=inverse_indices,
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

    def _prepare_cache_routing(
        self,
        *,
        past_key_values: Optional[NaiveCache],
        update_past_key_values: bool,
        sample_lens: torch.Tensor,
        query_len: int,
        device: torch.device,
        q_ranges: Optional[torch.Tensor] = None,
        attn_type_map: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Per-forward cache bookkeeping, done once for all layers.

        Builds the scatter indexes that interleave this chunk's K/V with the
        retained history into the per-sample packed layout varlen attention
        expects, and advances ``kvlens`` before any layer writes its cache.

        Also enforces the contract the cached path actually implements. It is
        plain varlen attention: every row of a document sees every other row of
        it, and there is no intra-document split structure at all - so the
        visibility specs the training mask encodes are rejected here rather
        than silently ignored.
        """
        if past_key_values is None:
            return {
                "past_key_values": None,
                "update_past_key_values": False,
                "sample_lens": sample_lens,
                "key_value_lens": None,
                "packed_query_indexes": None,
                "packed_past_key_value_indexes": None,
            }

        if attn_type_map is not None and bool((attn_type_map != 0).any()):
            raise ValueError(
                "attn_type_map asks for a causal diagonal, which the cached "
                "path cannot honor: cached varlen attention is fully visible "
                "inside a document and has no intra-document split structure, "
                "so it would be dense where training is lower-triangular."
            )
        if q_ranges is not None and int(q_ranges.shape[0]) != int(sample_lens.numel()):
            raise ValueError(
                f"a cached forward carries {int(q_ranges.shape[0])} attention "
                f"rectangles for {int(sample_lens.numel())} samples. Cached "
                "varlen attention has no intra-document split structure, so it "
                "requires exactly one split per sample."
            )
        if self.gradient_checkpointing and torch.is_grad_enabled():
            # maybe_checkpoint checkpoints on exactly this condition, and the
            # backward recompute would re-run the merge and update_kvcache
            # against an already-advanced cache.
            raise ValueError(
                "gradient checkpointing is not supported on the KV-cached "
                "path: the cached rollout is a no-grad path, so there is "
                "nothing to recompute, and the backward recompute would "
                "re-run the cache merge against an already-advanced cache."
            )

        bsz = past_key_values.batch_size
        if bsz != int(sample_lens.numel()):
            raise ValueError(
                "past_key_values.batch_size must match the number of sample_lens "
                f"entries, got {bsz} and {int(sample_lens.numel())}."
            )
        query_lens = sample_lens

        if past_key_values.seq_len > 0:  # merge required
            past_lens = torch.tensor(
                past_key_values.kvlens, dtype=torch.int32, device=device
            )
            past_len = int(past_lens.sum().item())

            # New rows land after their own sample's history: inclusive cumsum
            # of past lengths, repeated per query row.
            kv_cumsum = torch.cumsum(past_lens, dim=0)
            kv_offsets = torch.repeat_interleave(kv_cumsum, query_lens.to(torch.long))
            packed_query_indexes = (
                torch.arange(query_len, device=device) + kv_offsets
            ).to(torch.long)

            # History rows land before their own sample's new rows: exclusive
            # cumsum of query lengths, repeated per history row.
            q_cumsum = torch.cumsum(query_lens, dim=0)
            q_cumsum = torch.cat([q_cumsum.new_zeros(1), q_cumsum[:-1]], dim=0)
            q_offsets = torch.repeat_interleave(q_cumsum, past_lens.to(torch.long))
            packed_past_key_value_indexes = (
                torch.arange(past_len, device=device) + q_offsets
            ).to(torch.long)

            key_value_lens = query_lens + past_lens
        else:  # flash attention without history
            packed_query_indexes = None
            packed_past_key_value_indexes = None
            key_value_lens = query_lens

        if update_past_key_values:
            # Once per forward, before any layer calls update_kvcache.
            past_key_values.update_kvlens(query_lens.tolist())

        return {
            "past_key_values": past_key_values,
            "update_past_key_values": update_past_key_values,
            "sample_lens": query_lens,
            "key_value_lens": key_value_lens,
            "packed_query_indexes": packed_query_indexes,
            "packed_past_key_value_indexes": packed_past_key_value_indexes,
        }


EntryClass = CausalMiniMaxH3DiTModel

__all__ = [
    "CausalMiniMaxH3Attention",
    "CausalMiniMaxH3DiTBlock",
    "CausalMiniMaxH3DiTModel",
]
