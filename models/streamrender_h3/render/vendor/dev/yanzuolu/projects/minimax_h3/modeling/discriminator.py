# SPDX-License-Identifier: Apache-2.0
"""Cross-attention discriminator head for MiniMax H3 DMD2 training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from dev.yanzuolu.common.distributed.unified_parallel import (
    Gather,
    SeqAllToAll,
    get_unified_parallel_group,
    get_unified_parallel_rank,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)
from dev.yanzuolu.utils.flash_attn import FlashAttention

from .checkpointing import gradient_checkpointing
from .discriminator_output import MiniMaxH3DiscriminatorOutput
from .transformer.config import (
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
    MiniMaxH3DiTArchConfig,
)
from .transformer.model import (
    _BF16_DTYPE,
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MiniMaxH3AdalnProj,
    MiniMaxH3MLP,
    RowParallelLinear,
    _apply_qk_norm,
    _modulate_gate,
    _modulate_scale_shift,
    _norm,
)

_MINIMAX_H3_DISCRIMINATOR_FLASH_ATTENTION = FlashAttention()



def _ulysses_scatter_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    up_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """[T_local, n, d] row shards -> [T_global, n/up_size, d] head shards."""
    n = k.shape[1]
    if n % up_size:
        raise ValueError(
            f"attention heads {n} must be divisible by the unified-parallel "
            f"world size {up_size}"
        )
    if MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT % up_size:
        raise ValueError(
            "MiniMax H3 packed sequence alignment "
            f"{MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT} must be divisible by "
            f"unified-parallel world size {up_size}"
        )
    group = get_unified_parallel_group()
    assert group is not None
    return (
        SeqAllToAll.apply(group, k, 1, 0, False),
        SeqAllToAll.apply(group, v, 1, 0, False),
    )


class MiniMaxH3DiscriminatorCrossAttention(nn.Module):
    """H3-width cross-attention from replicated learned queries to packed tap rows."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: Any = None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_heads = arch.num_attention_heads
        self.head_dim = arch.attention_head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.q_proj = ColumnParallelLinear(
            arch.hidden_size,
            self.inner_dim,
            bias=False,
            gather_output=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.kv_proj = MergedColumnParallelLinear(
            arch.hidden_size,
            [self.inner_dim, self.inner_dim],
            bias=False,
            gather_output=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
        )
        self.q_norm = _norm(self.head_dim, eps=arch.qk_norm_eps)
        self.k_norm = _norm(self.head_dim, eps=arch.qk_norm_eps)
        self.out_proj = RowParallelLinear(
            self.inner_dim,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        if not any(parameter.is_meta for parameter in self.parameters()):
            self.reset_parameters()

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight, generator=generator)
        nn.init.xavier_uniform_(self.kv_proj.weight, generator=generator)
        nn.init.xavier_uniform_(self.out_proj.weight, generator=generator)
        nn.init.ones_(self.q_norm.weight)
        nn.init.ones_(self.k_norm.weight)

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        *,
        q_lens: torch.Tensor,
        k_lens: torch.Tensor,
        kv_indices: torch.Tensor | None = None,
        sample_q_lens: torch.Tensor | None = None,
        sample_k_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend packed ``query`` documents to the matching packed ``kv`` documents.

        ``kv_indices`` addresses rows in the original global packed sequence.
        Projection therefore runs on every local tap row before Ulysses changes
        the layout from sequence shards to head shards, and only then are the
        requested document rows packed for varlen attention.

        With sample lengths, local queries precede sample queries. Both read
        the same projected, normalized KV buffer with their own boundaries.
        """
        if q_lens.numel() != k_lens.numel():
            raise ValueError(
                f"q_lens has {q_lens.numel()} documents but k_lens has "
                f"{k_lens.numel()}"
            )
        if (sample_q_lens is None) != (sample_k_lens is None):
            raise ValueError("sample query and key lengths must be provided together")

        q_total = query.shape[0]
        k_total = kv.shape[0]
        q, _ = self.q_proj(query)
        kv, _ = self.kv_proj(kv)
        k, v = kv.split(self.inner_dim, dim=-1)
        q = q.view(q_total, self.num_heads, self.head_dim)
        k = k.view(k_total, self.num_heads, self.head_dim)
        v = v.view(k_total, self.num_heads, self.head_dim)

        up_size = (
            get_unified_parallel_world_size()
            if is_unified_parallel_initialized()
            else 1
        )
        if up_size > 1:
            k, v = _ulysses_scatter_kv(k, v, up_size)

        if kv_indices is not None:
            indices = kv_indices.to(device=k.device, dtype=torch.long)
            k = k.index_select(0, indices)
            v = v.index_select(0, indices)
            expected_k_rows = int(k_lens.to(torch.long).sum().item())
            if k.shape[0] != expected_k_rows or v.shape[0] != expected_k_rows:
                raise ValueError(
                    f"kv_indices selects {k.shape[0]} key rows but k_lens "
                    f"describes {expected_k_rows}"
                )

        if up_size > 1:
            local_heads = self.num_heads // up_size
            head_start = get_unified_parallel_rank() * local_heads
            q = q[:, head_start : head_start + local_heads]

        q, k = _apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        if sample_q_lens is None:
            out = _MINIMAX_H3_DISCRIMINATOR_FLASH_ATTENTION(
                q,
                k,
                v,
                q_lens=q_lens.to(dtype=torch.int32),
                k_lens=k_lens.to(dtype=torch.int32),
                dropout_p=0.0,
                softmax_scale=self.softmax_scale,
                causal=False,
            )
        else:
            local_query_rows = int(q_lens.sum().item())
            local_out = _MINIMAX_H3_DISCRIMINATOR_FLASH_ATTENTION(
                q[:local_query_rows],
                k,
                v,
                q_lens=q_lens.to(dtype=torch.int32),
                k_lens=k_lens.to(dtype=torch.int32),
                dropout_p=0.0,
                softmax_scale=self.softmax_scale,
                causal=False,
            )
            sample_out = _MINIMAX_H3_DISCRIMINATOR_FLASH_ATTENTION(
                q[local_query_rows:],
                k,
                v,
                q_lens=sample_q_lens.to(dtype=torch.int32),
                k_lens=sample_k_lens.to(dtype=torch.int32),
                dropout_p=0.0,
                softmax_scale=self.softmax_scale,
                causal=False,
            )
            out = torch.cat((local_out, sample_out), dim=0)
        if up_size > 1:
            group = get_unified_parallel_group()
            assert group is not None
            out = Gather.apply(group, out, 1, True)

        out = out.reshape(q_total, self.inner_dim)
        out, _ = self.out_proj(out)
        return out


class MiniMaxH3DiscriminatorBlock(nn.Module):
    """One AdaLN-Zero H3 block with learned-query cross-attention and no RoPE."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: Any = None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.kv_norm = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3DiscriminatorCrossAttention(
            arch,
            quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = MiniMaxH3MLP(arch, quant_config, prefix=f"{prefix}.mlp")
        # Every discriminator row is a query token, so unlike the trunk block
        # (text/video/audio) there is only one modality to modulate.
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            6 * arch.hidden_size,
            quant_config,
            prefix=f"{prefix}.adaln_proj",
            expand_ratio=6,
            modality_num=1,
        )
        self._enable_gradient_checkpointing = False
        if not any(parameter.is_meta for parameter in self.parameters()):
            self.reset_parameters()

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        self.attn.reset_parameters(generator=generator)
        nn.init.xavier_uniform_(self.mlp.fc1.weight, generator=generator)
        nn.init.xavier_uniform_(self.mlp.fc2.weight, generator=generator)
        nn.init.ones_(self.norm1.weight)
        nn.init.ones_(self.kv_norm.weight)
        nn.init.ones_(self.norm2.weight)
        # The feature path must be LIVE at init. With the gates ALSO zero
        # (AdaLN-Zero), every block outputs exactly its residual -- the learned
        # queries, which carry no sample information -- so the head input is
        # bit-identical for the real and the fake pass, the +-sigmoid(0)
        # grad_outs cancel exactly on the zeroed logit weight, and that zero
        # weight blocks every upstream gradient: an exact fixed point (trial
        # 15446442 sat at logits == 0.0 for 80 steps; reproduced in a toy
        # simulation of this block's dataflow). LTX2 avoids it by zeroing only
        # the output layer of an otherwise randomly-initialized conv stack.
        # Same principle here: weight = 0 keeps the start timestep-independent
        # and fifty-fifty, and the bias opens the gates (chunk(6) order is
        # shift_msa | scale_msa | gate_msa | shift_mlp | scale_mlp | gate_mlp),
        # so the block starts as a plain pre-norm cross-attention block and the
        # timestep modulation ramps in from the zero weight.
        hidden = self.adaln_proj.hidden_size
        nn.init.zeros_(self.adaln_proj.linear.weight)
        with torch.no_grad():
            bias = self.adaln_proj.linear.bias
            bias.zero_()
            bias[2 * hidden : 3 * hidden].fill_(1.0)
            bias[5 * hidden : 6 * hidden].fill_(1.0)

    def set_gradient_checkpointing(self, enable: bool) -> None:
        self._enable_gradient_checkpointing = enable

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        q_lens: torch.Tensor,
        k_lens: torch.Tensor,
        kv_indices: torch.Tensor | None = None,
        sample_q_lens: torch.Tensor | None = None,
        sample_k_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return gradient_checkpointing(
            self._forward,
            query,
            kv,
            use_reentrant=False,
            enabled=self._enable_gradient_checkpointing and self.training and torch.is_grad_enabled(),
            adaln_input=adaln_input,
            combined_indices=combined_indices,
            q_lens=q_lens,
            k_lens=k_lens,
            kv_indices=kv_indices,
            sample_q_lens=sample_q_lens,
            sample_k_lens=sample_k_lens,
        )

    def _forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        q_lens: torch.Tensor,
        k_lens: torch.Tensor,
        kv_indices: torch.Tensor | None = None,
        sample_q_lens: torch.Tensor | None = None,
        sample_k_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        adaln_params = self.adaln_proj(adaln_input)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            adaln_params
        )

        residual = query
        h = self.norm1(query)
        h = _modulate_scale_shift(
            h,
            shift_msa,
            scale_msa,
            combined_indices,
            dtype=_BF16_DTYPE,
        )
        h = self.attn(
            h,
            self.kv_norm(kv),
            q_lens=q_lens,
            k_lens=k_lens,
            kv_indices=kv_indices,
            sample_q_lens=sample_q_lens,
            sample_k_lens=sample_k_lens,
        )
        query = _modulate_gate(
            residual,
            gate_msa,
            h,
            combined_indices,
            dtype=_BF16_DTYPE,
        )

        residual = query
        h = self.norm2(query)
        h = _modulate_scale_shift(
            h,
            shift_mlp,
            scale_mlp,
            combined_indices,
            dtype=_BF16_DTYPE,
        )
        h = self.mlp(h)
        return _modulate_gate(
            residual,
            gate_mlp,
            h,
            combined_indices,
            dtype=_BF16_DTYPE,
        )


class MiniMaxH3DMD2Discriminator(nn.Module):
    """Learned-query discriminator returning one scalar per sample per tap."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        num_queries: int,
        num_taps: int = 4,
        quant_config: Any = None,
        prefix: str = "discriminator",
    ) -> None:
        super().__init__()
        if num_queries <= 0:
            raise ValueError(f"num_queries must be positive, got {num_queries}")
        if num_taps <= 0:
            raise ValueError(f"num_taps must be positive, got {num_taps}")
        self.hidden_size = arch.hidden_size
        self.num_queries = num_queries
        self.num_taps = num_taps
        self.queries = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        num_queries,
                        arch.hidden_size,
                        dtype=_BF16_DTYPE,
                    )
                )
                for _ in range(self.num_taps)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3DiscriminatorBlock(
                    arch,
                    quant_config,
                    prefix=f"{prefix}.blocks.{index}",
                )
                for index in range(self.num_taps)
            ]
        )
        head_dim = self.num_queries * arch.hidden_size
        self.final_norm = nn.ModuleList(
            [
                nn.LayerNorm(
                    head_dim,
                    eps=arch.final_norm_eps,
                    dtype=_BF16_DTYPE,
                )
                for _ in range(self.num_taps)
            ]
        )
        self.logit = nn.ModuleList(
            [
                nn.Linear(
                    head_dim,
                    1,
                    dtype=_BF16_DTYPE,
                )
                for _ in range(self.num_taps)
            ]
        )
        if not any(parameter.is_meta for parameter in self.parameters()):
            self.reset_parameters()

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Signature matches minimax_h3's other models, which is what NativeGradientCheckpointing drives."""
        for block in self.blocks:
            block.set_gradient_checkpointing(enabled)

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        for block in self.blocks:
            block.reset_parameters(generator=generator)
        for query in self.queries:
            nn.init.normal_(query, mean=0.0, std=0.02, generator=generator)
        for final_norm, logit in zip(self.final_norm, self.logit, strict=True):
            nn.init.ones_(final_norm.weight)
            nn.init.zeros_(final_norm.bias)
            # Zero logits give log(2) per adversarial term and zero initial
            # gradient into the generator. Open block gates provide distinct
            # real and fake features so the classifier weight can learn.
            nn.init.zeros_(logit.weight)
            if logit.bias is not None:
                nn.init.zeros_(logit.bias)

    def forward(
        self,
        tap_hidden_states: Sequence[torch.Tensor],
        *,
        adaln_input: torch.Tensor,
        k_lens: torch.Tensor,
        live_documents: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits shaped ``[B, num_taps]`` for packed tap hidden tensors."""
        if len(tap_hidden_states) != self.num_taps:
            raise ValueError(
                "MiniMax H3 discriminator requires exactly "
                f"{self.num_taps} taps, got "
                f"{len(tap_hidden_states)}"
            )
        num_documents = k_lens.numel()
        if live_documents.numel() != num_documents:
            raise ValueError(
                f"live_documents has {live_documents.numel()} entries but k_lens "
                f"describes {num_documents} documents"
            )
        batch_size = adaln_input.shape[0]

        q_lens = torch.full_like(k_lens, self.num_queries, dtype=torch.int32)
        if bool(((k_lens == 0) & (q_lens > 0)).any()):
            raise ValueError("attention documents with queries must have at least one key")
        document_sample_indices = live_documents.to(torch.long).cumsum(dim=0) - 1
        combined_indices = document_sample_indices.repeat_interleave(self.num_queries)
        logits = []
        for query, block, final_norm, logit, tap_hidden in zip(
            self.queries,
            self.blocks,
            self.final_norm,
            self.logit,
            tap_hidden_states,
            strict=True,
        ):
            packed_query = (
                query.unsqueeze(0)
                .expand(num_documents, -1, -1)
                .reshape(num_documents * self.num_queries, self.hidden_size)
            )
            hidden = block(
                packed_query,
                tap_hidden,
                adaln_input=adaln_input,
                combined_indices=combined_indices,
                q_lens=q_lens,
                k_lens=k_lens,
            )
            live_hidden = hidden.view(
                num_documents, self.num_queries, self.hidden_size
            )[live_documents]
            if live_hidden.shape[0] != batch_size:
                raise ValueError(
                    f"live_documents selects {live_hidden.shape[0]} samples but "
                    f"adaln_input has {batch_size} rows"
                )
            logits.append(logit(final_norm(live_hidden.flatten(1))).squeeze(-1))

        return torch.stack(logits, dim=1)


class MiniMaxH3DMD2DiscriminatorV2(MiniMaxH3DMD2Discriminator):
    """Configurable chunk-wise head over packed trunk taps.

    Every tap keeps its own learned queries and discriminator block. Their
    query hidden states are combined by one joint head, producing one logit for
    each target-media chunk. Explicit chunk ranges select target rows from a
    causal prefix layout; the default layout retains native AV suffix parsing.
    """

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        num_queries: int,
        video_chunk_size: int,
        num_taps: int = 4,
        quant_config: Any = None,
        prefix: str = "discriminator",
    ) -> None:
        if (
            not isinstance(video_chunk_size, int)
            or isinstance(video_chunk_size, bool)
            or video_chunk_size <= 0
        ):
            raise ValueError(
                "video_chunk_size must be a positive integer other than bool"
            )
        super().__init__(
            arch,
            num_queries=num_queries,
            num_taps=num_taps,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.video_chunk_size = video_chunk_size

        # V1 owns one norm and projection per tap. V2 reuses only its queries
        # and blocks, then registers a stable joint-head checkpoint structure.
        del self.final_norm
        del self.logit
        joint_head_dim = self.num_taps * self.num_queries * self.hidden_size
        self.joint_final_norm = nn.LayerNorm(
            joint_head_dim,
            eps=arch.final_norm_eps,
            dtype=_BF16_DTYPE,
        )
        self.joint_logit = nn.Linear(
            joint_head_dim,
            1,
            dtype=_BF16_DTYPE,
        )
        if not any(parameter.is_meta for parameter in self.parameters()):
            self._reset_joint_head_parameters()

    def _reset_joint_head_parameters(self) -> None:
        nn.init.ones_(self.joint_final_norm.weight)
        nn.init.zeros_(self.joint_final_norm.bias)
        nn.init.zeros_(self.joint_logit.weight)
        if self.joint_logit.bias is not None:
            nn.init.zeros_(self.joint_logit.bias)

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        # MiniMaxH3DMD2Discriminator.__init__ dispatches virtually before the
        # V2 joint head exists. Initialize its temporary V1 head normally, then
        # let V2 __init__ replace and initialize that head exactly once.
        if not hasattr(self, "joint_final_norm"):
            super().reset_parameters(generator=generator)
            return
        for block in self.blocks:
            block.reset_parameters(generator=generator)
        for query in self.queries:
            nn.init.normal_(query, mean=0.0, std=0.02, generator=generator)
        self._reset_joint_head_parameters()

    def _chunk_layout(
        self,
        *,
        k_lens: torch.Tensor,
        live_documents: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        target_audio_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return packed chunk rows, chunk key lengths, and chunk sample ids."""
        if (
            k_lens.dim() != 1
            or k_lens.dtype == torch.bool
            or k_lens.is_floating_point()
        ):
            raise ValueError("k_lens must be a one-dimensional integer tensor")
        if bool((k_lens <= 0).any()):
            raise ValueError("k_lens must describe only non-empty documents")
        if live_documents.dim() != 1 or live_documents.dtype != torch.bool:
            raise ValueError("live_documents must be a one-dimensional bool tensor")
        if live_documents.numel() != k_lens.numel():
            raise ValueError(
                f"live_documents has {live_documents.numel()} entries but k_lens "
                f"describes {k_lens.numel()} documents"
            )
        if token_tags.dim() != 1:
            raise ValueError("token_tags must be one-dimensional")
        if position_ids.dim() == 3 and position_ids.shape[0] == 1:
            positions = position_ids[0]
        elif position_ids.dim() == 2:
            positions = position_ids
        else:
            raise ValueError(
                "position_ids must have shape [S, 3] or [1, S, 3], got "
                f"{list(position_ids.shape)}"
            )
        if positions.shape != (token_tags.numel(), 3):
            raise ValueError(
                f"position_ids must cover {token_tags.numel()} rows with three "
                f"coordinates, got {list(positions.shape)}"
            )
        total_rows = int(k_lens.to(torch.long).sum().item())
        if total_rows != token_tags.numel():
            raise ValueError(
                f"k_lens covers {total_rows} rows but token_tags has "
                f"{token_tags.numel()} rows"
            )
        if (
            target_audio_rows.dim() != 1
            or target_audio_rows.dtype == torch.bool
            or target_audio_rows.is_floating_point()
        ):
            raise ValueError(
                "target_audio_rows must be a one-dimensional integer tensor"
            )
        if target_audio_rows.numel() == 0:
            raise ValueError("target_audio_rows must select target audio")
        if target_audio_rows.device != token_tags.device:
            raise ValueError("target_audio_rows and token_tags must share a device")
        target_audio_rows = target_audio_rows.to(torch.long)
        if bool(
            ((target_audio_rows < 0) | (target_audio_rows >= total_rows)).any()
        ):
            raise ValueError("target_audio_rows contains an out-of-range row")
        if target_audio_rows.numel() > 1 and not bool(
            (target_audio_rows[1:] > target_audio_rows[:-1]).all()
        ):
            raise ValueError("target_audio_rows must be strictly increasing")
        chunk_rows: list[torch.Tensor] = []
        chunk_lens: list[int] = []
        chunk_sample_indices: list[int] = []
        document_start = 0
        sample_index = 0
        selected_audio_rows = 0
        for document_len, is_live in zip(
            k_lens.to(torch.long).tolist(), live_documents.tolist(), strict=True
        ):
            document_stop = document_start + document_len
            if not is_live:
                document_start = document_stop
                continue

            document_tags = token_tags[document_start:document_stop]
            target_video_tag = document_tags[-1]
            non_video = torch.nonzero(
                document_tags != target_video_tag, as_tuple=False
            ).flatten()
            video_start = (
                int(non_video[-1].item()) + 1 if non_video.numel() else 0
            )
            if video_start == document_len:
                raise ValueError(
                    f"live document {sample_index} must end in target VIDEO rows"
                )
            global_video_start = document_start + video_start
            document_audio_rows = target_audio_rows[
                (target_audio_rows >= document_start)
                & (target_audio_rows < document_stop)
            ]
            if document_audio_rows.numel() == 0:
                raise ValueError(
                    f"live document {sample_index} has no selected target AUDIO rows"
                )
            expected_audio_rows = torch.arange(
                int(document_audio_rows[0].item()),
                global_video_start,
                dtype=torch.long,
                device=target_audio_rows.device,
            )
            if not torch.equal(document_audio_rows, expected_audio_rows):
                raise ValueError(
                    f"live document {sample_index} target_audio_rows must select "
                    "the complete target AUDIO block immediately before target VIDEO"
                )
            target_audio_tags = token_tags.index_select(0, document_audio_rows)
            target_audio_tag = target_audio_tags[0]
            if not bool((target_audio_tags == target_audio_tag).all()):
                raise ValueError(
                    f"live document {sample_index} target_audio_rows must share "
                    "one token tag"
                )
            if bool(target_audio_tag == target_video_tag):
                raise ValueError(
                    f"live document {sample_index} target AUDIO and VIDEO tags "
                    "must differ"
                )
            selected_audio_rows += document_audio_rows.numel()
            video_positions = positions[global_video_start:document_stop]
            video_times = video_positions[:, 0]
            latent_starts = torch.cat(
                (
                    torch.zeros(1, dtype=torch.long, device=video_times.device),
                    torch.nonzero(
                        video_times[1:] != video_times[:-1], as_tuple=False
                    ).flatten()
                    + 1,
                )
            )
            latent_stops = torch.cat(
                (
                    latent_starts[1:],
                    latent_starts.new_tensor([video_times.numel()]),
                )
            )
            frame_rows_by_latent = latent_stops - latent_starts
            frame_rows = int(frame_rows_by_latent[0].item())
            if frame_rows <= 0 or not bool(
                (frame_rows_by_latent == frame_rows).all()
            ):
                raise ValueError(
                    f"live document {sample_index} target VIDEO rows do not form "
                    "equal-sized latent frames"
                )
            latent_t = int(latent_starts.numel())
            video_grid = video_positions.view(latent_t, frame_rows, 3)
            if not torch.equal(
                video_grid[:, :, 1:],
                video_grid[:1, :, 1:].expand(latent_t, -1, -1),
            ):
                raise ValueError(
                    f"live document {sample_index} target VIDEO spatial layout "
                    "changes between latent frames"
                )
            latent_times = video_times.index_select(0, latent_starts)
            if latent_t > 1 and not bool(
                (latent_times[1:] > latent_times[:-1]).all()
            ):
                raise ValueError(
                    f"live document {sample_index} target VIDEO temporal positions "
                    "must be strictly increasing"
                )

            audio_positions = positions.index_select(0, document_audio_rows)
            flat_audio_times = audio_positions[:, 0]
            channel_starts = torch.cat(
                (
                    torch.zeros(
                        1, dtype=torch.long, device=flat_audio_times.device
                    ),
                    torch.nonzero(
                        flat_audio_times[1:] <= flat_audio_times[:-1],
                        as_tuple=False,
                    ).flatten()
                    + 1,
                )
            )
            channel_stops = torch.cat(
                (
                    channel_starts[1:],
                    channel_starts.new_tensor([flat_audio_times.numel()]),
                )
            )
            channel_lengths = channel_stops - channel_starts
            audio_t = int(channel_lengths[0].item())
            if audio_t == 0 or not bool((channel_lengths == audio_t).all()):
                raise ValueError(
                    f"live document {sample_index} target AUDIO channel segments "
                    "must have equal positive lengths"
                )
            num_audio_channels = int(channel_starts.numel())
            channel_positions = audio_positions.view(
                num_audio_channels, audio_t, 3
            )
            audio_times = channel_positions[0, :, 0].contiguous()
            if audio_t > 1 and not bool((audio_times[1:] > audio_times[:-1]).all()):
                raise ValueError(
                    f"live document {sample_index} target AUDIO temporal positions "
                    "must be strictly increasing within each channel"
                )
            if not torch.equal(
                channel_positions[:, :, 0],
                audio_times.unsqueeze(0).expand(num_audio_channels, -1),
            ):
                raise ValueError(
                    f"live document {sample_index} target AUDIO channels must "
                    "share one temporal position vector"
                )
            if not torch.equal(
                channel_positions[:, :, 1:],
                channel_positions[:, :1, 1:].expand(-1, audio_t, -1),
            ):
                raise ValueError(
                    f"live document {sample_index} target AUDIO channel positions "
                    "must be constant over time"
                )
            if audio_times[0] != latent_times[0]:
                raise ValueError(
                    f"live document {sample_index} target AUDIO and VIDEO must "
                    "share a temporal origin"
                )

            chunk_boundaries = latent_times[
                self.video_chunk_size :: self.video_chunk_size
            ].contiguous()
            audio_chunk_indices = torch.bucketize(
                audio_times, chunk_boundaries, right=True
            )
            for chunk_index, latent_start in enumerate(
                range(0, latent_t, self.video_chunk_size)
            ):
                latent_stop = min(
                    latent_start + self.video_chunk_size, latent_t
                )
                audio_indices = torch.nonzero(
                    audio_chunk_indices == chunk_index, as_tuple=False
                ).flatten()
                if audio_indices.numel() == 0:
                    raise ValueError(
                        f"live document {sample_index} video chunk {chunk_index} "
                        "has no target AUDIO rows"
                    )
                chunk_audio_rows = torch.cat(
                    [
                        document_audio_rows.index_select(
                            0, audio_indices + channel_index * audio_t
                        )
                        for channel_index in range(num_audio_channels)
                    ]
                )
                chunk_video_start = global_video_start + latent_start * frame_rows
                chunk_video_stop = global_video_start + latent_stop * frame_rows
                chunk_video_rows = torch.arange(
                    chunk_video_start,
                    chunk_video_stop,
                    dtype=torch.long,
                    device=token_tags.device,
                )
                rows = torch.cat((chunk_audio_rows, chunk_video_rows))
                chunk_rows.append(rows)
                chunk_lens.append(rows.numel())
                chunk_sample_indices.append(sample_index)

            sample_index += 1
            document_start = document_stop

        if not chunk_rows:
            raise ValueError("live_documents selects no target media documents")
        if selected_audio_rows != target_audio_rows.numel():
            raise ValueError(
                "target_audio_rows contains rows outside live target documents"
            )
        return (
            torch.cat(chunk_rows),
            k_lens.new_tensor(chunk_lens, dtype=torch.int32),
            k_lens.new_tensor(chunk_sample_indices, dtype=torch.long),
        )

    @staticmethod
    def _explicit_chunk_layout(
        *,
        k_lens: torch.Tensor,
        live_documents: torch.Tensor,
        total_rows: int,
        chunk_ranges: torch.Tensor,
        chunk_sample_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand disjoint target ranges contained in their live sample documents."""
        if k_lens.ndim != 1 or k_lens.dtype == torch.bool or k_lens.is_floating_point():
            raise ValueError("k_lens must be a one-dimensional integer tensor")
        if bool((k_lens <= 0).any()) or int(k_lens.sum().item()) != total_rows:
            raise ValueError("k_lens must describe nonempty documents covering all tap rows")
        if live_documents.shape != k_lens.shape or live_documents.dtype != torch.bool:
            raise ValueError("live_documents must be a bool vector matching k_lens")
        if chunk_ranges.ndim != 2 or chunk_ranges.shape[1] != 2:
            raise ValueError("chunk_ranges must have shape [N_chunk, 2]")
        if chunk_ranges.dtype == torch.bool or chunk_ranges.is_floating_point():
            raise ValueError("chunk_ranges must be an integer tensor")
        if chunk_sample_indices.shape != chunk_ranges.shape[:1]:
            raise ValueError("chunk_sample_indices must have one entry per chunk")
        if chunk_sample_indices.dtype == torch.bool or chunk_sample_indices.is_floating_point():
            raise ValueError("chunk_sample_indices must be an integer tensor")
        ranges = chunk_ranges.to(torch.long)
        samples = chunk_sample_indices.to(device=ranges.device, dtype=torch.long)
        lengths = k_lens.to(device=ranges.device, dtype=torch.long)
        live = live_documents.to(ranges.device)
        document_stops = lengths.cumsum(0)
        sample_starts = (document_stops - lengths)[live]
        sample_stops = document_stops[live]
        if ranges.shape[0] == 0 or not torch.equal(
            torch.unique_consecutive(samples),
            torch.arange(sample_starts.numel(), device=ranges.device),
        ):
            raise ValueError("chunk groups must cover every live sample in sample-major order")
        if bool((ranges[:, 1] <= ranges[:, 0]).any()) or bool(
            (ranges[1:, 0] < ranges[:-1, 1]).any()
        ):
            raise ValueError("chunk ranges must be nonempty, ordered and disjoint")
        if bool((ranges[:, 0] < sample_starts[samples]).any()) or bool(
            (ranges[:, 1] > sample_stops[samples]).any()
        ):
            raise ValueError("each chunk range must lie inside its corresponding live sample")
        rows = torch.cat([
            torch.arange(start, stop, device=ranges.device)
            for start, stop in ranges.tolist()
        ])
        return rows, (ranges[:, 1] - ranges[:, 0]).to(torch.int32), samples

    def _resolve_chunk_layout(
        self,
        *,
        k_lens: torch.Tensor,
        live_documents: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        target_audio_rows: torch.Tensor,
        chunk_ranges: torch.Tensor | None,
        chunk_sample_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if chunk_ranges is None:
            if chunk_sample_indices is not None:
                raise ValueError("chunk_sample_indices requires explicit chunk_ranges")
            return self._chunk_layout(
                k_lens=k_lens,
                live_documents=live_documents,
                token_tags=token_tags,
                position_ids=position_ids,
                target_audio_rows=target_audio_rows,
            )
        if chunk_sample_indices is None:
            raise ValueError("explicit chunk_ranges requires chunk_sample_indices")
        return self._explicit_chunk_layout(
            k_lens=k_lens,
            live_documents=live_documents,
            total_rows=token_tags.numel(),
            chunk_ranges=chunk_ranges,
            chunk_sample_indices=chunk_sample_indices,
        )

    def forward(
        self,
        tap_hidden_states: Sequence[torch.Tensor],
        *,
        adaln_input: torch.Tensor,
        k_lens: torch.Tensor,
        live_documents: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        target_audio_rows: torch.Tensor,
        chunk_ranges: torch.Tensor | None = None,
        chunk_sample_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return sample-major, chunk-major logits shaped ``[N_chunk, 1]``."""
        if len(tap_hidden_states) != self.num_taps:
            raise ValueError(
                "MiniMax H3 discriminator requires exactly "
                f"{self.num_taps} taps, got {len(tap_hidden_states)}"
            )

        up_size = (
            get_unified_parallel_world_size()
            if is_unified_parallel_initialized()
            else 1
        )

        global_rows = token_tags.numel()
        chunk_rows, chunk_k_lens, chunk_sample_indices = self._resolve_chunk_layout(
            k_lens=k_lens,
            live_documents=live_documents,
            token_tags=token_tags,
            position_ids=position_ids,
            target_audio_rows=target_audio_rows,
            chunk_ranges=chunk_ranges,
            chunk_sample_indices=chunk_sample_indices,
        )
        batch_size = adaln_input.shape[0]
        num_samples = int(live_documents.to(torch.long).sum().item())
        if num_samples != batch_size:
            raise ValueError(
                f"live_documents selects {num_samples} samples but adaln_input "
                f"has {batch_size} rows"
            )

        num_chunks = chunk_k_lens.numel()
        q_lens = torch.full_like(
            chunk_k_lens, self.num_queries, dtype=torch.int32
        )
        combined_indices = chunk_sample_indices.repeat_interleave(self.num_queries)
        tap_query_hidden_states = []
        for query, block, tap_hidden in zip(
            self.queries,
            self.blocks,
            tap_hidden_states,
            strict=True,
        ):
            if tap_hidden.shape[0] * up_size != global_rows:
                raise ValueError(
                    f"tap hidden state has {tap_hidden.shape[0]} local rows at "
                    f"up_size={up_size}, but metadata describes {global_rows} "
                    "global rows"
                )
            packed_query = (
                query.unsqueeze(0)
                .expand(num_chunks, -1, -1)
                .reshape(num_chunks * self.num_queries, self.hidden_size)
            )
            hidden = block(
                packed_query,
                tap_hidden,
                adaln_input=adaln_input,
                combined_indices=combined_indices,
                q_lens=q_lens,
                k_lens=chunk_k_lens,
                kv_indices=chunk_rows,
            )
            tap_query_hidden_states.append(
                hidden.view(num_chunks, self.num_queries, self.hidden_size)
            )

        # [N_chunk, num_taps, num_queries, hidden] preserves tap-major then
        # query-major order when flattened for the joint classifier.
        joint_hidden = torch.stack(tap_query_hidden_states, dim=1).flatten(1)
        return self.joint_logit(self.joint_final_norm(joint_hidden))


class MiniMaxH3DMD2DiscriminatorV3(MiniMaxH3DMD2DiscriminatorV2):
    """Local and whole-target queries share each tap's block and projected KV.

    Target chunks are packed once in sample order. Local queries attend one
    chunk, while global queries attend all retained chunks of the same sample.
    Chunk filtering happens before either attention branch reads target KV.
    """

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        num_queries: int,
        video_chunk_size: int,
        num_global_queries: int = 1,
        num_taps: int = 4,
        quant_config: Any = None,
        prefix: str = "discriminator",
    ) -> None:
        if (
            not isinstance(num_global_queries, int)
            or isinstance(num_global_queries, bool)
            or num_global_queries <= 0
        ):
            raise ValueError("num_global_queries must be a positive integer other than bool")
        super().__init__(
            arch,
            num_queries=num_queries,
            video_chunk_size=video_chunk_size,
            num_taps=num_taps,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.num_global_queries = num_global_queries
        self.global_queries = nn.ParameterList([
            nn.Parameter(torch.empty(num_global_queries, arch.hidden_size, dtype=_BF16_DTYPE))
            for _ in range(self.num_taps)
        ])
        global_head_dim = self.num_taps * self.num_global_queries * self.hidden_size
        self.global_final_norm = nn.LayerNorm(
            global_head_dim,
            eps=arch.final_norm_eps,
            dtype=_BF16_DTYPE,
        )
        self.global_logit = nn.Linear(global_head_dim, 1, dtype=_BF16_DTYPE)
        if not any(parameter.is_meta for parameter in self.parameters()):
            self._reset_global_parameters()

    def _reset_global_parameters(self, generator: torch.Generator | None = None) -> None:
        for query in self.global_queries:
            nn.init.normal_(query, mean=0.0, std=0.02, generator=generator)
        nn.init.ones_(self.global_final_norm.weight)
        nn.init.zeros_(self.global_final_norm.bias)
        nn.init.zeros_(self.global_logit.weight)
        nn.init.zeros_(self.global_logit.bias)

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        super().reset_parameters(generator=generator)
        if hasattr(self, "global_queries"):
            self._reset_global_parameters(generator=generator)

    def materialize_missing_global_parameters(
        self, generator: torch.Generator | None = None
    ) -> None:
        """Initialize missing global weights after loading a complete V2 head."""
        global_prefixes = ("global_queries.", "global_final_norm.", "global_logit.")
        if any(
            parameter.is_meta
            for name, parameter in self.named_parameters()
            if not name.startswith(global_prefixes)
        ):
            raise ValueError("partial V3 initialization requires complete local discriminator weights")
        for index, query in enumerate(self.global_queries):
            if query.is_meta:
                query = nn.Parameter(
                    torch.empty_like(query, device="cpu"), requires_grad=query.requires_grad
                )
                self.global_queries[index] = query
                nn.init.normal_(query, mean=0.0, std=0.02, generator=generator)
        for module, name, initializer in (
            (self.global_final_norm, "weight", nn.init.ones_),
            (self.global_final_norm, "bias", nn.init.zeros_),
            (self.global_logit, "weight", nn.init.zeros_),
            (self.global_logit, "bias", nn.init.zeros_),
        ):
            parameter = getattr(module, name)
            if parameter.is_meta:
                parameter = nn.Parameter(
                    torch.empty_like(parameter, device="cpu"),
                    requires_grad=parameter.requires_grad,
                )
                setattr(module, name, parameter)
                initializer(parameter)

    def forward(
        self,
        tap_hidden_states: Sequence[torch.Tensor],
        *,
        adaln_input: torch.Tensor,
        k_lens: torch.Tensor,
        live_documents: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        target_audio_rows: torch.Tensor,
        chunk_ranges: torch.Tensor | None = None,
        chunk_sample_indices: torch.Tensor | None = None,
        chunk_keep_mask: torch.Tensor | None = None,
    ) -> MiniMaxH3DiscriminatorOutput:
        if len(tap_hidden_states) != self.num_taps:
            raise ValueError(f"V3 requires {self.num_taps} taps, got {len(tap_hidden_states)}")
        chunk_rows, chunk_k_lens, chunk_sample_indices = self._resolve_chunk_layout(
            k_lens=k_lens,
            live_documents=live_documents,
            token_tags=token_tags,
            position_ids=position_ids,
            target_audio_rows=target_audio_rows,
            chunk_ranges=chunk_ranges,
            chunk_sample_indices=chunk_sample_indices,
        )
        batch_size = adaln_input.shape[0]
        if int(live_documents.to(torch.long).sum().item()) != batch_size:
            raise ValueError("live_documents and adaln_input must describe the same samples")
        if chunk_keep_mask is not None:
            if chunk_keep_mask.dtype != torch.bool or chunk_keep_mask.shape != chunk_k_lens.shape:
                raise ValueError("chunk_keep_mask must be a bool vector over the original chunks")
            keep = chunk_keep_mask.to(device=chunk_rows.device)
            row_keep = torch.repeat_interleave(keep, chunk_k_lens.to(torch.long))
            chunk_rows = chunk_rows[row_keep]
            chunk_k_lens = chunk_k_lens[keep]
            chunk_sample_indices = chunk_sample_indices[keep]
        sample_indices = torch.arange(batch_size, device=chunk_sample_indices.device)
        if not torch.equal(torch.unique_consecutive(chunk_sample_indices), sample_indices):
            raise ValueError("every sample must retain at least one target chunk")

        sample_k_lens = torch.zeros(
            batch_size, device=chunk_k_lens.device, dtype=chunk_k_lens.dtype
        ).scatter_add_(0, chunk_sample_indices, chunk_k_lens)
        q_lens = torch.full_like(chunk_k_lens, self.num_queries, dtype=torch.int32)
        sample_q_lens = torch.full_like(
            sample_k_lens, self.num_global_queries, dtype=torch.int32
        )
        combined_indices = torch.cat((
            chunk_sample_indices.repeat_interleave(self.num_queries),
            sample_indices.repeat_interleave(self.num_global_queries),
        ))
        num_chunks = chunk_k_lens.numel()
        local_query_rows = num_chunks * self.num_queries
        up_size = get_unified_parallel_world_size() if is_unified_parallel_initialized() else 1
        local_hidden_states = []
        sample_hidden_states = []
        for query, global_query, block, tap_hidden in zip(
            self.queries, self.global_queries, self.blocks, tap_hidden_states, strict=True
        ):
            if tap_hidden.shape[0] * up_size != token_tags.numel():
                raise ValueError("tap row shards must cover the global packed metadata")
            packed_query = torch.cat((
                query.unsqueeze(0).expand(num_chunks, -1, -1).reshape(
                    local_query_rows, self.hidden_size
                ),
                global_query.unsqueeze(0).expand(batch_size, -1, -1).reshape(
                    batch_size * self.num_global_queries, self.hidden_size
                ),
            ))
            hidden = block(
                packed_query,
                tap_hidden,
                adaln_input=adaln_input,
                combined_indices=combined_indices,
                q_lens=q_lens,
                k_lens=chunk_k_lens,
                kv_indices=chunk_rows,
                sample_q_lens=sample_q_lens,
                sample_k_lens=sample_k_lens,
            )
            local_hidden_states.append(
                hidden[:local_query_rows].view(num_chunks, self.num_queries, self.hidden_size)
            )
            sample_hidden_states.append(
                hidden[local_query_rows:].view(batch_size, self.num_global_queries, self.hidden_size)
            )
        local_hidden = torch.stack(local_hidden_states, dim=1).flatten(1)
        sample_hidden = torch.stack(sample_hidden_states, dim=1).flatten(1)
        return MiniMaxH3DiscriminatorOutput(
            self.joint_logit(self.joint_final_norm(local_hidden)),
            self.global_logit(self.global_final_norm(sample_hidden)),
            chunk_sample_indices,
        )


__all__ = [
    "MiniMaxH3DMD2Discriminator",
    "MiniMaxH3DMD2DiscriminatorV2",
    "MiniMaxH3DMD2DiscriminatorV3",
    "MiniMaxH3DiscriminatorBlock",
    "MiniMaxH3DiscriminatorCrossAttention",
]
