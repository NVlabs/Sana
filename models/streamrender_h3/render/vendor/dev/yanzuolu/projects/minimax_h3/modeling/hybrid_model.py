# SPDX-License-Identifier: Apache-2.0
"""Single-parameter-tree dense or chunk-causal MiniMax-H3 DiT.

The attention mask slot carries a typed dense marker.  Dense requests use the
released MiniMax-H3 varlen FlashAttention wrapper with one isolated document per
sample.  Every other request delegates to the existing causal attention path,
including packed FlexAttention training and KV-cached rollout.

The hybrid models only change the Python class of attention modules already
constructed by the causal models.  They add no parameters, buffers, or
submodules.  Blocks remain ``CausalMiniMaxH3DiTBlock`` and state-dict names stay
identical to the causal X0 wrapper.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Final, Optional

import torch

from dev.yanzuolu.common.distributed.unified_parallel import (
    gather_heads_scatter_seq,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)
from dev.yanzuolu.utils.naive_cache import NaiveCache

from dev.yanzuolu.projects.minimax_h3.modeling.transformer.causal_model import (
    CausalMiniMaxH3Attention,
    CausalMiniMaxH3DiTModel,
)
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.causal_model_sp import (
    CausalMiniMaxH3AttentionSP,
    CausalMiniMaxH3DiTModelSP,
)
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.config import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.model import (
    _MINIMAX_H3_FLASH_ATTENTION,
    _apply_qk_norm,
    _apply_rope_qk,
)
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.model_sp import _ulysses_scatter_heads
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.x0_model import MiniMaxH3X0Model


class MiniMaxH3DenseRouting:
    """Host-side document lengths for one dense packed forward."""

    __slots__ = ("_sample_lens", "_kernel_lens")

    def __init__(self, sample_lens: Sequence[int] | torch.Tensor) -> None:
        lengths = _host_sample_lens(sample_lens)
        if not lengths:
            raise ValueError("MiniMaxH3DenseRouting needs at least one document")
        if any(length <= 0 for length in lengths):
            raise ValueError(
                "every sample_lens entry must be positive, got " f"{lengths}"
            )
        self._sample_lens = lengths
        self._kernel_lens = torch.tensor(lengths, dtype=torch.int32, device="cpu")

    @property
    def sample_lens(self) -> tuple[int, ...]:
        return self._sample_lens

    @property
    def total_rows(self) -> int:
        return sum(self._sample_lens)

    def kernel_lengths(self, *, total_rows: int, origin: str) -> torch.Tensor:
        if self.total_rows != int(total_rows):
            raise ValueError(
                f"{origin} dense routing describes {self.total_rows} rows but "
                f"attention received {int(total_rows)} rows"
            )
        return self._kernel_lens

    def __repr__(self) -> str:
        return (
            f"MiniMaxH3DenseRouting(documents={len(self._sample_lens)}, "
            f"total_rows={self.total_rows})"
        )


class MiniMaxH3DenseAttentionSentinel:
    """Singleton marker selecting dense attention for one forward."""

    __slots__ = ()

    _instance: ClassVar[Optional["MiniMaxH3DenseAttentionSentinel"]] = None

    def __new__(cls) -> "MiniMaxH3DenseAttentionSentinel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def bind(
        self, sample_lens: Sequence[int] | torch.Tensor
    ) -> MiniMaxH3DenseRouting:
        return MiniMaxH3DenseRouting(sample_lens)

    def __repr__(self) -> str:
        return "MINIMAX_H3_DENSE_ATTENTION"

    def __reduce__(self) -> tuple[Any, tuple[()]]:
        return (type(self), ())


MINIMAX_H3_DENSE_ATTENTION: Final[MiniMaxH3DenseAttentionSentinel] = (
    MiniMaxH3DenseAttentionSentinel()
)


def _host_sample_lens(sample_lens: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
    if torch.is_tensor(sample_lens):
        if sample_lens.dim() != 1:
            raise ValueError(
                f"sample_lens must be one-dimensional, got {list(sample_lens.shape)}"
            )
        if sample_lens.device.type != "cpu":
            raise ValueError(
                "dense routing must be bound from CPU sample_lens so every layer "
                "can reuse the same host metadata"
            )
        return tuple(int(value) for value in sample_lens.tolist())
    return tuple(int(value) for value in sample_lens)


def minimax_h3_attention_is_dense(attention_mask: Any) -> bool:
    """Return whether ``attention_mask`` requests dense attention."""
    return attention_mask is MINIMAX_H3_DENSE_ATTENTION or isinstance(
        attention_mask, MiniMaxH3DenseRouting
    )


def _reject_active_cache(
    past_key_values: Optional[NaiveCache],
    update_past_key_values: bool,
    *,
    origin: str,
) -> None:
    if past_key_values is None and not update_past_key_values:
        return
    raise ValueError(f"{origin} cannot combine dense attention with an active KV cache")


def _dense_kernel_lengths(
    attention_mask: Any,
    sample_lens: Optional[torch.Tensor],
    *,
    total_rows: int,
    origin: str,
) -> torch.Tensor:
    if isinstance(attention_mask, MiniMaxH3DenseRouting):
        return attention_mask.kernel_lengths(total_rows=total_rows, origin=origin)

    if sample_lens is None:
        raise ValueError(
            f"{origin} received the dense sentinel without sample_lens. Bind the "
            "sentinel to the packed document lengths"
        )
    lengths = sample_lens.view(-1)
    if lengths.dtype != torch.int32:
        lengths = lengths.to(torch.int32)
    if lengths.device.type == "cpu":
        host = [int(value) for value in lengths.tolist()]
        if any(value <= 0 for value in host):
            raise ValueError(f"{origin} received non-positive sample_lens {host}")
        if sum(host) != int(total_rows):
            raise ValueError(
                f"{origin} sample_lens sum to {sum(host)} but attention received "
                f"{int(total_rows)} rows"
            )
    return lengths


def _project_qkv_with_rope(
    attention: CausalMiniMaxH3Attention,
    x: torch.Tensor,
    rope_cache: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total, num_heads, head_dim = x.shape[0], attention.num_heads, attention.head_dim
    qkv, _ = attention.qkv_proj(x)
    q, k, v = qkv.split(attention.local_inner_dim, dim=-1)
    q = q.view(total, num_heads, head_dim)
    k = k.view(total, num_heads, head_dim)
    v = v.view(total, num_heads, head_dim)

    cos_sin_cache, positions = rope_cache
    q, k = _apply_qk_norm(
        q, k, attention.q_norm, attention.k_norm, attention.head_dim
    )
    q, k = _apply_rope_qk(q, k, cos_sin_cache, positions)
    return q, k, v


def _dense_varlen_attention(
    attention: CausalMiniMaxH3Attention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attention_mask: Any,
    sample_lens: Optional[torch.Tensor],
    origin: str,
) -> torch.Tensor:
    lengths = _dense_kernel_lengths(
        attention_mask,
        sample_lens,
        total_rows=int(q.shape[0]),
        origin=origin,
    )
    return _MINIMAX_H3_FLASH_ATTENTION(
        q,
        k,
        v,
        q_lens=lengths,
        k_lens=lengths,
        dropout_p=0.0,
        softmax_scale=attention.softmax_scale,
        causal=False,
    )


def _dense_local_forward(
    attention: CausalMiniMaxH3Attention,
    x: torch.Tensor,
    *,
    rope_cache: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Any,
    sample_lens: Optional[torch.Tensor],
    origin: str,
) -> torch.Tensor:
    total, num_heads, head_dim = x.shape[0], attention.num_heads, attention.head_dim
    q, k, v = _project_qkv_with_rope(attention, x, rope_cache)
    out = _dense_varlen_attention(
        attention,
        q,
        k,
        v,
        attention_mask=attention_mask,
        sample_lens=sample_lens,
        origin=origin,
    )
    out = out.reshape(total, num_heads * head_dim)
    out, _ = attention.out_proj(out)
    return out


class MiniMaxH3HybridAttention(CausalMiniMaxH3Attention):
    """Causal MiniMax-H3 attention with typed dense dispatch."""

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
    ) -> torch.Tensor:
        if not minimax_h3_attention_is_dense(attention_mask):
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
            )

        origin = f"{type(self).__name__}.forward"
        _reject_active_cache(past_key_values, update_past_key_values, origin=origin)
        return _dense_local_forward(
            self,
            x,
            rope_cache=rope_cache,
            attention_mask=attention_mask,
            sample_lens=sample_lens,
            origin=origin,
        )


class MiniMaxH3HybridAttentionSP(CausalMiniMaxH3AttentionSP):
    """Hybrid attention using the causal model's Ulysses exchange."""

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
    ) -> torch.Tensor:
        if not minimax_h3_attention_is_dense(attention_mask):
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
            )

        origin = f"{type(self).__name__}.forward"
        _reject_active_cache(past_key_values, update_past_key_values, origin=origin)
        if (
            not is_unified_parallel_initialized()
            or get_unified_parallel_world_size() <= 1
        ):
            return _dense_local_forward(
                self,
                x,
                rope_cache=rope_cache,
                attention_mask=attention_mask,
                sample_lens=sample_lens,
                origin=origin,
            )

        q, k, v = _project_qkv_with_rope(self, x, rope_cache)
        q, k, v = _ulysses_scatter_heads(
            q, k, v, get_unified_parallel_world_size()
        )
        out = _dense_varlen_attention(
            self,
            q,
            k,
            v,
            attention_mask=attention_mask,
            sample_lens=sample_lens,
            origin=origin,
        )
        out = gather_heads_scatter_seq(out.flatten(1), head_dim=1, seq_dim=0)
        out, _ = self.out_proj(out)
        return out


class MiniMaxH3HybridDiTModel(CausalMiniMaxH3DiTModel):
    """Causal DiT whose existing attention modules also serve dense forwards."""

    def __init__(
        self,
        config: Any,
        hf_config: dict[str, Any],
        quant_config: Any = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config, quant_config=quant_config)
        for block in self.blocks:
            if type(block.attn) is not CausalMiniMaxH3Attention:
                raise TypeError(
                    "expected CausalMiniMaxH3Attention, got "
                    f"{type(block.attn).__name__}"
                )
            block.attn.__class__ = MiniMaxH3HybridAttention


class MiniMaxH3HybridDiTModelSP(CausalMiniMaxH3DiTModelSP):
    """Hybrid MiniMax-H3 DiT with Ulysses sequence parallelism."""

    def __init__(
        self,
        config: Any,
        hf_config: dict[str, Any],
        quant_config: Any = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config, quant_config=quant_config)
        for block in self.blocks:
            if type(block.attn) is not CausalMiniMaxH3AttentionSP:
                raise TypeError(
                    "expected CausalMiniMaxH3AttentionSP, got "
                    f"{type(block.attn).__name__}"
                )
            block.attn.__class__ = MiniMaxH3HybridAttentionSP


class MiniMaxH3HybridX0DiT(MiniMaxH3X0Model):
    """Flat-kwargs X0 wrapper over the hybrid MiniMax-H3 DiT."""

    _minimax_h3_hybrid_abi = True
    _DIT_CLASS: type[CausalMiniMaxH3DiTModel] = MiniMaxH3HybridDiTModel

    def __init__(self, **arch_kwargs: Any) -> None:
        arch = MiniMaxH3DiTArchConfig(**arch_kwargs)
        super().__init__(
            type(self)._DIT_CLASS(
                MiniMaxH3DiTConfig(arch_config=arch),
                hf_config={},
            )
        )


class MiniMaxH3HybridX0DiTSP(MiniMaxH3HybridX0DiT):
    """Public sequence-parallel X0 wrapper for mixed MiniMax-H3 trials."""

    _DIT_CLASS: type[CausalMiniMaxH3DiTModel] = MiniMaxH3HybridDiTModelSP


EntryClass = MiniMaxH3HybridX0DiT

__all__ = [
    "EntryClass",
    "MINIMAX_H3_DENSE_ATTENTION",
    "MiniMaxH3DenseAttentionSentinel",
    "MiniMaxH3DenseRouting",
    "MiniMaxH3HybridAttention",
    "MiniMaxH3HybridAttentionSP",
    "MiniMaxH3HybridDiTModel",
    "MiniMaxH3HybridDiTModelSP",
    "MiniMaxH3HybridX0DiT",
    "MiniMaxH3HybridX0DiTSP",
    "minimax_h3_attention_is_dense",
]
