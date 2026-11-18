# SPDX-License-Identifier: Apache-2.0
"""Ulysses sequence parallelism for the bidirectional MiniMax H3 DiT.

``model.py`` was ported from SGLang with its Ulysses control flow left intact
and only the collectives removed: ``_ulysses_ctx()`` is hard-wired to
``(1, 0)`` and the exchange points import from ``sglang...usp``. This module
re-supplies those collectives from this repo's
``common/distributed/unified_parallel``. It builds the full replicated input
embedding, slices that embedding at the SP boundary, and reuses the parent's
row arithmetic for ``inverse_indices`` and ``token_tags``. Only the source of
``(world_size, rank)`` changes; ``_ulysses_ctx`` is never called here.

Two deliberate divergences from ``projects/wan_t2v/modeling/model_sp.py``:

* **RoPE runs before the exchange, on the rank-local rows.** ``rope_cache`` is
  built from ``img_position_ids[:, row_start:row_stop]``, so the rotation
  belongs where those rows still live. wan rotates after the exchange because
  its ``freqs`` are global; copying that here would rotate every row of the
  gathered sequence with this rank's slice of the positions.
* **No sequence padding.** The packer already rounds the packed sequence up to
  ``MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT`` (64), so any world size dividing 64
  splits the rows evenly. wan's ``represented_len``/pad-document machinery has
  nothing to do here, and the packed attention metadata (``cu_seqlens``,
  ``max_seqlen``) stays exactly as the caller built it.

Heads are not padded either: the exchange needs a head count divisible by the
world size, which is the contract ``model.py``'s
``_validate_sequence_parallel_config`` already states, so
``_ulysses_scatter_heads`` enforces it rather than working around it. The state
dict is untouched: the model subclass adds no parameters, buffers or
submodules, and the attention subclass is a pure ``forward`` override, which is
what makes the ``__class__`` swap legal.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dev.yanzuolu.common.distributed.unified_parallel import (
    Gather,
    Slice,
    gather_heads_scatter_seq,
    gather_seq_scatter_heads_qkv,
    get_unified_parallel_group,
    get_unified_parallel_rank,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)

from ..checkpointing import maybe_checkpoint
from .config import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
)
from .model import (
    _BF16_DTYPE,
    _FORWARD_SUPPORTED_KWARGS,
    _apply_qk_norm,
    _apply_rope_qk,
    _minimax_h3_attention_core_bcg,
    _minimax_h3_attention_core_impl,
    _required_kwarg,
    _rope_cos_sin_cache,
    MiniMaxH3Attention,
    MiniMaxH3DiTModel,
)


def _unified_parallel_ctx() -> tuple[int, int]:
    """(world_size, rank) of this repo's unified-parallel group, or (1, 0).

    The seam ``MiniMaxH3DiTModel._sequence_parallel_ctx`` overrides in both SP
    models, so ``build_rope_cache`` slices by the same rank their forwards do
    instead of by the hard-stubbed ``_ulysses_ctx()``.
    """
    if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
        return 1, 0
    return get_unified_parallel_world_size(), get_unified_parallel_rank()


def _ulysses_scatter_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    up_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[T_local, n, d] row shards -> [T_global, n/up_size, d] head shards.

    ``n`` must divide ``up_size``, which is the contract
    ``_validate_sequence_parallel_config`` in ``model.py`` states. H3 has 56
    heads, so the head-divisible sizes are 1, 2, 4, 7, 8, 14, 28, 56; the rows
    are never padded either, so the world size must also divide
    ``MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT`` (64) and the usable sizes are
    1, 2, 4, 8.

    ``seq_dim=0`` because h3's packed sequence is 1-D thd rows with no batch
    dimension, unlike wan's bidirectional ``[B, S, ...]``.
    """
    total, n, d = q.shape
    if n % up_size:
        raise ValueError(
            f"attention heads {n} must be divisible by the unified-parallel "
            f"world size {up_size}. H3's 56 heads divide by 1, 2, 4, 7, 8, 14, "
            "28 and 56, and the packed sequence alignment 64 narrows that to "
            "1, 2, 4, 8."
        )
    inner_dim = n * d
    qkv = gather_seq_scatter_heads_qkv(
        torch.cat(
            [
                q.reshape(total, inner_dim),
                k.reshape(total, inner_dim),
                v.reshape(total, inner_dim),
            ],
            dim=-1,
        ),
        seq_dim=0,
    )
    q, k, v = qkv.split(inner_dim // up_size, dim=-1)
    local_heads = n // up_size
    global_total = q.shape[0]
    return (
        q.view(global_total, local_heads, d),
        k.view(global_total, local_heads, d),
        v.view(global_total, local_heads, d),
    )


def _validate_local_embedding_layout(
    layout: dict[str, Any],
    *,
    local_seq_len: int,
) -> None:
    """Reject a ``local_embedding_layout`` that is not this rank's.

    ``_embed``'s trusted-layout branch slices the text rows with two plain ints
    and writes them at local rows ``[0, text_rows)``, so a layout built on one
    rank and broadcast to the others is read without complaint and silently
    gives every rank the first rank's text rows.
    """
    text_start = int(layout["text_source_start"])
    text_stop = int(layout["text_source_stop"])
    if not 0 <= text_start <= text_stop:
        raise ValueError(
            "local_embedding_layout text slice must satisfy "
            f"0 <= text_source_start <= text_source_stop, got "
            f"[{text_start}, {text_stop})"
        )
    text_rows = text_stop - text_start
    if text_rows > local_seq_len:
        raise ValueError(
            f"local_embedding_layout carries {text_rows} text rows, more than "
            f"this rank's {local_seq_len} rows. The layout must be built per "
            "rank under sequence parallelism, against that rank's row window."
        )
    for name in ("img_row_ids", "audio_row_ids"):
        row_ids = layout[name]
        if not row_ids.numel():
            continue
        low = int(row_ids.min())
        high = int(row_ids.max())
        # Text owns the leading text_rows local rows, so the latent rows start
        # after them; a layout from another rank lands outside that window.
        if low < text_rows or high >= local_seq_len:
            raise ValueError(
                f"local_embedding_layout {name} spans [{low}, {high}], outside "
                f"this rank's latent rows [{text_rows}, {local_seq_len}). The "
                "layout must be built per rank under sequence parallelism, "
                "against that rank's row window."
            )


class MiniMaxH3AttentionSP(MiniMaxH3Attention):
    """MiniMaxH3Attention with a sequence-to-heads Ulysses exchange.

    Adds no attribute of its own - it is a pure ``forward`` override, which is
    what lets ``MiniMaxH3DiTModelSP`` install it by ``__class__`` assignment.
    """

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int,
        ulysses_active: bool = False,
    ) -> torch.Tensor:
        """x: [T_local, hidden] this rank's row shard -> [T_local, hidden].

        qkv projection, q/k RMSNorm and RoPE run on the local rows; the
        all-to-all then trades sequence for heads, so each rank attends the
        whole packed sequence with a slice of the heads and ``cu_seqlens``
        keeps its global packed-document semantics.
        """
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            return super().forward(
                x,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ulysses_active=ulysses_active,
            )

        total, n, d = x.shape[0], self.num_heads, self.head_dim
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split(self.local_inner_dim, dim=-1)
        q = q.view(total, n, d)
        k = k.view(total, n, d)
        v = v.view(total, n, d)

        # The parent's fused qknorm+RoPE branch is skipped: model.py hard-
        # disables it with _SGL_KERNELS_AVAILABLE = False, so
        # self._use_fused_qknorm_rope is always False in this build.
        cos_sin_cache, positions = rope_cache
        q, k = _apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        q, k = _apply_rope_qk(q, k, cos_sin_cache, positions)

        q, k, v = _ulysses_scatter_heads(q, k, v, get_unified_parallel_world_size())

        attention_core = (
            _minimax_h3_attention_core_bcg
            if self.bcg_breakpoint
            else _minimax_h3_attention_core_impl
        )
        out = attention_core(
            self,
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            # The exchange is done here with this repo's collectives; the
            # upstream sglang usp branch inside the core stays off.
            ulysses_active=False,
        )

        out = gather_heads_scatter_seq(out.flatten(1), head_dim=1, seq_dim=0)
        out, _ = self.out_proj(out)
        return out


class MiniMaxH3DiTModelSP(MiniMaxH3DiTModel):
    """MiniMaxH3DiTModel with the block stack sharded over packed rows.

    ``forward`` is a fork of the parent's for the same reason the causal model
    forks it: the parent keeps its packed forward in one method. It drops the
    branches that are statically dead in this build - the ring-degree guard
    and ``_resolve_attention_backend_once`` (both hard-stubbed), the batched
    block AdaLN and the output-column gathers (both require TP > 1, and
    ``get_tp_world_size`` hard-returns 1) - and replaces the removed sglang
    output all-gather with ``Gather``.
    """

    def __init__(
        self,
        config: Any,
        hf_config: dict[str, Any],
        quant_config: Any = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config, quant_config=quant_config)
        for block in self.blocks:
            if not isinstance(block.attn, MiniMaxH3Attention):
                raise TypeError(f"Unsupported attention type: {type(block.attn).__name__}")
            block.attn.__class__ = MiniMaxH3AttentionSP
        # The text refiner is deliberately left alone: _embed refines the whole
        # prompt on every rank, so its attention sees no row shard.

    def _sequence_parallel_ctx(self) -> tuple[int, int]:
        """The context ``forward`` shards by, so ``build_rope_cache`` matches it."""
        return _unified_parallel_ctx()

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed forward over this rank's row shard.

        Returns the same global ``(video_logits, audio_logits)`` on every rank:
        the row shards are gathered before the output rows are selected.
        """
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            return super().forward(**kwargs)

        unexpected = sorted(set(kwargs) - _FORWARD_SUPPORTED_KWARGS)
        if unexpected:
            raise TypeError(
                "MiniMaxH3DiTModelSP.forward received unexpected kwargs: "
                f"{unexpected}; supported kwargs: "
                f"{sorted(_FORWARD_SUPPORTED_KWARGS)}"
            )

        tap_outputs = kwargs.get("tap_outputs")
        tap_blocks = self._validate_taps(kwargs.get("tap_blocks"), tap_outputs)
        classify_mode = bool(kwargs.get("classify_mode", False))

        x = _required_kwarg(kwargs, "x")
        audio_x = _required_kwarg(kwargs, "audio_x")
        img_position_ids = _required_kwarg(kwargs, "img_position_ids")
        unique_timesteps = _required_kwarg(kwargs, "unique_timesteps")
        inverse_indices = (
            _required_kwarg(kwargs, "inverse_indices").view(-1).to(torch.long)
        )
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

        psp = _required_kwarg(kwargs, "packed_seq_params")
        cu_seqlens = self._psp_field(psp, "packed_seq_params", "cu_seqlens_q").to(
            torch.int32
        )
        raw_cu_seqlens_host = self._psp_optional_field(psp, "cu_seqlens_q_host")
        cu_seqlens_host = tuple(
            int(value)
            for value in (
                cu_seqlens.tolist()
                if raw_cu_seqlens_host is None
                else raw_cu_seqlens_host
            )
        )
        max_seqlen = int(self._psp_field(psp, "packed_seq_params", "max_seqlen_q"))
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
        # The packer aligns the packed sequence to
        # MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT, so no padding is needed as long
        # as the world size divides it. Heads must divide it too -- see
        # _ulysses_scatter_heads, which rejects an indivisible head count rather
        # than padding it.
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

        # RoPE is row-local before Ulysses exchanges sequence for heads inside
        # attention. Input embeddings are instead built in full below and sliced
        # at the explicit SP boundary. Serving normally prepares the request-static
        # cache once; direct model callers use this fallback.
        rope_cache = kwargs.get("rope_cache")
        if rope_cache is None:
            rope_freqs = self.rope(img_position_ids[:, row_start:row_stop]).to(device)
            rope_cache = (
                _rope_cos_sin_cache(rope_freqs, dtype=_BF16_DTYPE),
                torch.arange(
                    local_seq_len,
                    device=device,
                    dtype=torch.long,
                ),
            )
        img_pos = img_pos.to(device)
        audio_pos = audio_pos.to(device)
        text_pos = text_pos.to(device)

        local_embedding_layout = kwargs.get("local_embedding_layout")
        if local_embedding_layout is not None:
            _validate_local_embedding_layout(
                local_embedding_layout, local_seq_len=local_seq_len
            )

        # _embed sits outside the checkpointed block stack, and here it runs over
        # the FULL sequence on every rank -- the Slice below is what shards it.
        # Its two full-length internals, the [seq_len, hidden] zeros buffer and
        # the bf16 video rows, are 611.5 and 603.2 MiB at the smoke's shape and
        # were being saved for backward: DMD2 forwards this model three times per
        # FAKE step, so 3.56 GiB sat live for the whole backward (measured, job
        # 15440105: both sizes allocated 3x and freed 0x). Recomputing them costs
        # one more pass over projections and no attention.
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
        # Like the block-stack conditions below, each rank adds the input
        # condition to its own rows only.
        decoder_input = self._add_input_condition(
            decoder_input, kwargs, row_start=row_start, row_stop=row_stop, device=device
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
        # Sharded exactly like the decoder rows: each rank keeps the condition
        # rows inside its own [row_start, row_stop).
        condition_kwargs = self._adaln_condition_kwargs(
            kwargs, row_start=row_start, row_stop=row_stop, device=device
        )

        hidden = self._checkpoint_block_stack(
            decoder_input,
            tap_blocks=tap_blocks,
            tap_outputs=tap_outputs,
            stop_after_last_tap=classify_mode and tap_blocks is not None,
            adaln_input=adaln_input,
            combined_indices=block_combined,
            rope_cache=rope_cache,
            cu_seqlens=cu_seqlens.to(device),
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            # MiniMaxH3AttentionSP owns the exchange; this flag only selects the
            # unvendored sglang usp path inside the attention core.
            ulysses_active=False,
            adaln_params=None,
            **condition_kwargs,
        )

        if classify_mode:
            # Returning before Gather leaves the taps row-sharded, exactly as the
            # discriminator's _ulysses_scatter_kv expects.
            #
            # They leave through the RETURN VALUE, and the `tap_outputs` list is
            # an internal detail of whoever calls this. A caller on the far side
            # of an FSDP boundary cannot use that list at all: FSDP2's
            # pre_forward tree_flattens and tree_unflattens args/kwargs, which
            # rebuilds a list into a NEW object, so the callee appends to the
            # copy and the caller's list stays empty. It only rebuilds when some
            # input tensor requires grad, which is why the FAKE phase (detached
            # latents, early return) worked and GEN (generator output, carries
            # grad) came back with zero taps -- job 15441216.
            #
            # t_emb rides along because the discriminator's AdaLN input is
            # derived from it, and reaching it through a forward hook on
            # time_embedder was the other half of that same reach-across.
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
            # Audio has no condition rows in the supported tasks, so its
            # derived update mask is all ones. Honor an explicit mask when
            # provided.
            update_audio_mask = kwargs.get("update_audio_mask")
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)
        return video_logits, audio_logits


__all__ = [
    "MiniMaxH3AttentionSP",
    "MiniMaxH3DiTModelSP",
]
