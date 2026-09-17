# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
#
# `HyperFlowSolAttnProcessor.__call__` is derived from `MiniMaxH3AttnProcessor.__call__` in diffusers
# (`models/transformers/transformer_minimax_h3.py`), Copyright 2025 The MiniMax Team and The HuggingFace Team,
# Apache License 2.0. See THIRD_PARTY_NOTICES.md.
#
# Sparse attention uses NVIDIA Sol-Attn (Li et al., 2026):
#   https://arxiv.org/abs/2607.24027
#   https://github.com/NVlabs/Sana/tree/main/techniques/sparse_backends
"""Optional NVIDIA Sol-Attn sparse attention for the HyperFlow 8-step loop.

Sol-Attn is NVIDIA's open-source sparse attention kernel (Apache-2.0, ``NVlabs/Sana`` →
``techniques/sparse_backends``). This module ships neither the kernel nor a variant of it — only the recipe we
validated on MiniMax-H3 with the HyperFlow adapter and a processor that applies it:

* the first ``dense_steps`` denoising steps run dense attention (the coarse structure is decided there);
* blocks listed in ``dense_layers`` run dense attention at every step;
* everything else runs ``sol_attn(q, k, v, tau=..., thresh_type=..., ...)``.

The processor is the official ``MiniMaxH3AttnProcessor`` with the ``dispatch_attention_fn`` call behind that switch;
projections, QK-norm and RoPE are untouched. Dense attention goes through diffusers' own dispatcher, so
``transformer.set_attention_backend(...)`` keeps working for the dense share.

Install the kernel from source (bf16, head_dim 128, non-causal, CUDA):

    pip install --no-build-isolation \\
        "git+https://github.com/NVlabs/Sana.git@9bfca5c4bf35774a1d44c27b0c3c91041fb8dad0#subdirectory=techniques/sparse_backends"
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3AttnProcessor,
    MiniMaxH3TransformerBlock,
    _apply_rotary_emb,
)

from .blocks import current_step
from .lora import find_transformers

logger = logging.getLogger(__name__)

SOL_ATTN_HEAD_DIM = 128


@dataclass(frozen=True)
class SolAttnRecipe:
    """The kernel parameters (their native names) plus our two scheduling knobs."""

    dense_steps: int = 2
    dense_layers: tuple[int, ...] = (0, 1)
    tau: float = 1.0
    thresh_type: str = "diag"
    sink_tokens: int = 0
    sink_start: int | None = None  # the kernel's default: `None` places the sink tokens at the end of the sequence
    kv_splits: int | str = "auto"

    def __post_init__(self) -> None:
        if self.dense_steps < 0:
            raise ValueError("`dense_steps` must be >= 0.")
        if self.tau < 0:
            raise ValueError("`tau` must be >= 0.")
        if self.thresh_type not in ("diag", "exact"):
            raise ValueError("`thresh_type` must be 'diag' or 'exact'.")
        if self.kv_splits != "auto" and int(self.kv_splits) not in (1, 2, 4):
            raise ValueError("`kv_splits` must be 'auto', 1, 2 or 4.")

    def kernel_kwargs(self, query: torch.Tensor, accepted: frozenset[str]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "tau": self.tau,
            "thresh_type": self.thresh_type,
            "sink_tokens": self.sink_tokens,
            "sink_start": self.sink_start,
            "kv_splits": resolve_kv_splits(query, self.kv_splits),
        }
        return {key: value for key, value in kwargs.items() if key in accepted}


def resolve_kv_splits(query: torch.Tensor, kv_splits: int | str) -> int:
    """``auto`` → 4 on SM90 for sequences >= 65536 when the CuTe path is importable, else 1."""
    if kv_splits != "auto":
        return int(kv_splits)
    if query.is_cuda and tuple(torch.cuda.get_device_capability(query.device)) == (9, 0) and query.shape[1] >= 65536:
        try:
            import cuda.bindings.driver  # noqa: F401
            import cutlass.cute  # noqa: F401
        except ImportError:
            return 1
        return 4
    return 1


_sol_attn = None
_sol_params: frozenset[str] = frozenset()
_sol_available: bool | None = None
_warned_missing = False


def sol_attn_available() -> bool:
    global _sol_available
    if _sol_available is None:
        try:
            import sol_attn  # noqa: F401
        except ImportError:
            _sol_available = False
        else:
            _sol_available = True
    return _sol_available


def _kernel():
    global _sol_attn, _sol_params
    if _sol_attn is None:
        try:
            from sol_attn import sol_attn
        except ImportError as error:
            raise RuntimeError(
                "Sol-Attn is not installed. See `hyperflow_h3.sol_attn` for the install command, or skip "
                "`enable_sol_attention`."
            ) from error
        _sol_attn = sol_attn
        _sol_params = frozenset(inspect.signature(sol_attn).parameters)
    return _sol_attn, _sol_params


class HyperFlowSolAttnProcessor(MiniMaxH3AttnProcessor):
    """``MiniMaxH3AttnProcessor`` whose attention call is dense or Sol-Attn depending on step and layer."""

    def __init__(self, layer_index: int, recipe: SolAttnRecipe):
        super().__init__()
        self.layer_index = int(layer_index)
        self.recipe = recipe
        self.calls_sparse = 0
        self.calls_dense = 0

    def use_sparse(self, step: int | None) -> bool:
        if step is None:
            # Outside a HyperFlow loop (e.g. the official blocks with HyperFlow disabled) attention stays dense.
            return False
        return step >= self.recipe.dense_steps and self.layer_index not in self.recipe.dense_layers

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        if self.use_sparse(current_step()) and attention_mask is None and self._kernel_ready():
            self.calls_sparse += 1
            hidden_states = self._sparse(query, key, value)
        else:
            self.calls_dense += 1
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    @staticmethod
    def _kernel_ready() -> bool:
        global _warned_missing
        if sol_attn_available():
            return True
        if not _warned_missing:
            _warned_missing = True
            logger.warning("Sol-Attn kernel not installed; the Sol-Attn processor is running dense attention.")
        return False

    def _sparse(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # Tensors are `[batch, sequence, heads, head_dim]`, the layout both diffusers and the kernel use.
        if query.dtype != torch.bfloat16:
            raise TypeError(f"Sol-Attn needs bfloat16 activations, got {query.dtype}.")
        if query.shape[-1] != SOL_ATTN_HEAD_DIM:
            raise ValueError(f"Sol-Attn needs head_dim={SOL_ATTN_HEAD_DIM}, got {query.shape[-1]}.")
        kernel, accepted = _kernel()
        if self._parallel_config is None:
            return self._run_kernel(kernel, accepted, query, key, value)
        return _ulysses_sparse(self, kernel, accepted, query, key, value, self._parallel_config)

    def _run_kernel(self, kernel, accepted: frozenset[str], query, key, value) -> torch.Tensor:
        # Under context parallel this runs on the exchanged tensors, so `kv_splits="auto"` sees the full sequence.
        kwargs = self.recipe.kernel_kwargs(query, accepted)
        return kernel(query.contiguous(), key.contiguous(), value.contiguous(), **kwargs)


def _ulysses_sparse(
    processor: HyperFlowSolAttnProcessor,
    kernel,
    accepted: frozenset[str],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    parallel_config: Any,
) -> torch.Tensor:
    """Run the kernel inside diffusers' Ulysses exchange.

    Context parallel hands each rank a shard of the sequence with every head. The exchange (all-to-all) turns that into
    the full sequence for a subset of heads, which is exactly what one GPU sees per head; Sol-Attn routes blocks per
    head, so the result is the same as on one GPU. The output is exchanged back to sequence shards. Ring attention is
    not possible: it needs the log-sum-exp of each partial result, which the kernel does not return.
    """
    _check_parallel_config(parallel_config)
    from diffusers.models.attention_dispatch import _templated_context_parallel_attention

    def forward_op(ctx, q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa, return_lse, **_):
        if return_lse:
            raise ValueError("Sol-Attn does not return the log-sum-exp.")
        return processor._run_kernel(kernel, accepted, q, k, v)

    def backward_op(*args, **kwargs):
        raise NotImplementedError("Sol-Attn is inference-only.")

    return _templated_context_parallel_attention(
        query,
        key,
        value,
        None,  # attn_mask: `_sparse` is only entered without one
        0.0,  # dropout_p
        False,  # is_causal
        None,  # scale: the kernel's default
        False,  # enable_gqa
        False,  # return_lse
        forward_op=forward_op,
        backward_op=backward_op,
        _parallel_config=parallel_config,
    )


def _check_parallel_config(parallel_config: Any) -> None:
    """Sol-Attn combines with Ulysses context parallel only."""
    if parallel_config is None:
        return
    cp_config = getattr(parallel_config, "context_parallel_config", None)
    if cp_config is not None and getattr(cp_config, "ring_degree", 1) > 1:
        raise RuntimeError(_RING_INCOMPATIBLE)


def _blocks(transformer: Any) -> Iterable[tuple[int, MiniMaxH3TransformerBlock]]:
    return enumerate(transformer.transformer_blocks)


_RING_INCOMPATIBLE = (
    "Sol-Attn supports Ulysses context parallel only: ring attention needs the log-sum-exp of each partial result, "
    "which the Sol-Attn kernel does not return. Use `ContextParallelConfig(ulysses_degree=N)`."
)


def enable_sol_attention(
    target: Any,
    *,
    dense_steps: int = 2,
    dense_layers: Iterable[int] = (0, 1),
    tau: float = 1.0,
    thresh_type: str = "diag",
    sink_tokens: int = 0,
    sink_start: int | None = None,
    kv_splits: int | str = "auto",
    require_kernel: bool = True,
) -> SolAttnRecipe:
    """Install the Sol-Attn processor on every DiT block of the pipeline's MiniMax-H3 transformer(s).

    Defaults are the recipe validated with the HyperFlow adapter: dense for the first 2 of 8 steps and for blocks
    0–1, ``tau=1.0``, ``thresh_type="diag"``, no sink tokens. The token refiner (text stream) is never touched: it is
    short and runs once per step.

    The dense share honours the transformer's attention backend (``set_attention_backend``); call that before or after
    this function. Under Ulysses context parallel (``enable_parallelism``) the sparse share runs inside diffusers'
    sequence exchange, per head on the full sequence, exactly as on one GPU; ring attention is refused. Set
    ``require_kernel=False`` to install the processor on a machine without the kernel (every call then falls back to
    dense with a warning at first use — useful for tests).
    """
    if require_kernel and not sol_attn_available():
        raise RuntimeError(
            "Sol-Attn is not installed. Install it from NVlabs/Sana (`techniques/sparse_backends`, see "
            "`hyperflow_h3.sol_attn`) or pass `require_kernel=False`."
        )
    recipe = SolAttnRecipe(
        dense_steps=dense_steps,
        dense_layers=tuple(int(i) for i in dense_layers),
        tau=tau,
        thresh_type=thresh_type,
        sink_tokens=sink_tokens,
        sink_start=sink_start,
        kv_splits=kv_splits,
    )
    for name, transformer in find_transformers(target).items():
        for index, block in _blocks(transformer):
            previous = block.attn.processor
            _check_parallel_config(getattr(previous, "_parallel_config", None))
            processor = HyperFlowSolAttnProcessor(index, recipe)
            # Keep the backend the user selected with `set_attention_backend`.
            processor._attention_backend = getattr(previous, "_attention_backend", None)
            processor._parallel_config = getattr(previous, "_parallel_config", None)
            block.attn.set_processor(processor)
        logger.info("Sol-Attn enabled on %s: %s", name, asdict(recipe))
    return recipe


def disable_sol_attention(target: Any) -> None:
    """Restore the official dense processor on every DiT block."""
    for transformer in find_transformers(target).values():
        for _, block in _blocks(transformer):
            previous = block.attn.processor
            if isinstance(previous, HyperFlowSolAttnProcessor):
                processor = MiniMaxH3AttnProcessor()
                processor._attention_backend = previous._attention_backend
                processor._parallel_config = previous._parallel_config
                block.attn.set_processor(processor)


__all__ = [
    "SOL_ATTN_HEAD_DIM",
    "HyperFlowSolAttnProcessor",
    "SolAttnRecipe",
    "disable_sol_attention",
    "enable_sol_attention",
    "resolve_kv_splits",
    "sol_attn_available",
]
