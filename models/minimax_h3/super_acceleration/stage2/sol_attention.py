"""Sol-Attn policy adapter for the three-step LTX 2.5 Stage 2 refiner.

The adapter intentionally changes only video self-attention (``attn1``):
transformer layer 0 stays dense and layers 1 through 47 use Sol-Attn.  On the
four-way sequence-parallel path, every attention function is an All2AllAttention
wrapper.  We preserve that wrapper and replace only its local-head
``original_attention`` callable, so the all-to-all sequence/head exchange still
surrounds the sparse kernel.  Sol-Engine performs architecture dispatch and
selects its in-tree SM100 kernel on GB200.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Sequence


STAGE2_TAUS = (1.0, 1.25, 1.5)
STAGE2_TRANSFORMER_LAYERS = 48


def _transformer_blocks(transformer: Any) -> Sequence[Any]:
    """Resolve raw and four-way sequence-parallel LTX transformer layouts.

    ``DiffusionStage._build_transformer`` returns an X0Model.  In the dense
    case its velocity model exposes ``transformer_blocks`` directly; with the
    sequence-parallel builder the raw model lives one level deeper at
    ``velocity_model.model``.  Keeping model unwrapping separate from attention
    patching makes the policy independently contract-testable.
    """

    current = transformer
    visited: set[int] = set()
    for _ in range(4):
        if id(current) in visited:
            break
        visited.add(id(current))

        blocks = getattr(current, "transformer_blocks", None)
        if blocks is not None:
            return blocks

        if hasattr(current, "velocity_model"):
            current = current.velocity_model
        elif hasattr(current, "model"):
            current = current.model
        else:
            break

    raise TypeError(
        "could not resolve LTX transformer_blocks from the supplied model; "
        "expected a raw transformer, X0Model, or sequence-parallel X0Model"
    )


def _eager_block(block: Any) -> Any:
    """Return the underlying block when ``torch.compile`` installed a wrapper."""

    return getattr(block, "_orig_mod", block)


class Stage2SolAttention:
    """Apply the fixed three-step Stage 2 Sol-Attn policy.

    Call :meth:`begin_denoise` immediately before each measured or warm-up
    denoise invocation.  Layer 0 remains completely untouched.  Arrival at
    layer 1 proves that dense layer 0 completed and advances the schedule once
    per step; layers 1 through 47 consume the selected tau.
    """

    def __init__(
        self,
        transformer: Any,
        *,
        isolate_sol_from_compile: bool = False,
    ) -> None:
        blocks = _transformer_blocks(transformer)
        if len(blocks) != STAGE2_TRANSFORMER_LAYERS:
            raise ValueError(
                "the fixed Stage 2 Sol-Attn policy requires exactly "
                f"{STAGE2_TRANSFORMER_LAYERS} transformer layers, got "
                f"{len(blocks)}"
            )

        self._step = -1
        self._dense_calls = 0
        self._sol_calls = 0
        self._isolate_sol_from_compile = isolate_sol_from_compile
        self._blocks = tuple(_eager_block(block) for block in blocks)
        self._all2all_wrappers = tuple(
            block.attn1.attention_function for block in self._blocks
        )
        for layer_index, all2all in enumerate(self._all2all_wrappers):
            if not hasattr(all2all, "original_attention"):
                raise TypeError(
                    "four-way sequence parallel requires attn1.attention_function "
                    "to be an All2AllAttention-like wrapper exposing "
                    f"original_attention; layer {layer_index} does not"
                )
            if not callable(all2all.original_attention):
                raise TypeError(
                    f"layer {layer_index} All2AllAttention.original_attention "
                    "is not callable"
                )

        sol_call = self._call
        if isolate_sol_from_compile:
            # Keep the stateful tau schedule and CuTe call on the eager side of
            # an explicit graph boundary.  Dynamo can still compile projections,
            # norms, cross-attention, and MLP regions around this callable.
            import torch

            sol_call = torch.compiler.disable(sol_call)

        # Layer 0's outer All2AllAttention and its dense inner callable are both
        # deliberately unchanged.  Only local-head attention in layers 1..47
        # is redirected through Sol-Attn.
        for layer_index in range(1, STAGE2_TRANSFORMER_LAYERS):
            all2all = self._all2all_wrappers[layer_index]
            dense = all2all.original_attention
            all2all.original_attention = partial(
                sol_call, layer_index, dense
            )

    def begin_denoise(self) -> None:
        """Reset the per-sample schedule and in-tree backend counters."""

        from techniques.sparse_backends.sol_attn_backend import reset_sol_attn_state

        self._step = -1
        self._dense_calls = 0
        self._sol_calls = 0
        reset_sol_attn_state()

    @staticmethod
    def _dense_bthd(dense: Callable[..., Any], q: Any, k: Any, v: Any) -> Any:
        batch, q_tokens, heads, head_dim = q.shape
        out = dense(
            q.reshape(batch, q_tokens, heads * head_dim),
            k.reshape(batch, k.shape[1], heads * head_dim),
            v.reshape(batch, v.shape[1], heads * head_dim),
            heads,
        )
        return out.view(batch, q_tokens, heads, head_dim)

    def _call(
        self,
        layer_index: int,
        dense: Callable[..., Any],
        q: Any,
        k: Any,
        v: Any,
        heads: int,
    ) -> Any:
        if layer_index == 1:
            next_step = self._step + 1
            if next_step >= len(STAGE2_TAUS):
                raise RuntimeError(
                    "Stage 2 executed more than the required three denoise "
                    "steps; call begin_denoise() before a new sample"
                )
            self._step = next_step
            # Layer 0 is intentionally not wrapped.  Reaching layer 1 once per
            # step is the invariant proving one dense layer-0 call completed.
            self._dense_calls += 1

        if not 0 <= self._step < len(STAGE2_TAUS):
            raise RuntimeError(
                "Sol-Attn layer executed before layer 1 selected the Stage 2 "
                "tau; verify transformer layer ordering"
            )

        from techniques.sparse_backends.sol_attn_backend import _run_sol_attn_bthd

        batch, tokens, hidden = q.shape
        if hidden % heads:
            raise ValueError(
                f"attention hidden size {hidden} is not divisible by {heads} heads"
            )
        head_dim = hidden // heads
        q_bthd = q.view(batch, tokens, heads, head_dim).contiguous()
        k_bthd = k.view(batch, k.shape[1], heads, head_dim).contiguous()
        v_bthd = v.view(batch, v.shape[1], heads, head_dim).contiguous()
        out = _run_sol_attn_bthd(
            q_bthd,
            k_bthd,
            v_bthd,
            tau=STAGE2_TAUS[self._step],
            thresh_type="diag",
            kv_splits="auto",
            dense_fn=partial(self._dense_bthd, dense),
        )
        self._sol_calls += 1
        return out.reshape(batch, tokens, hidden)

    def stats(self) -> dict[str, Any]:
        """Return adapter and backend counters for per-rank validation."""

        from techniques.sparse_backends.sol_attn_backend import get_sol_attn_stats

        return {
            "backend": "sol",
            "architecture": "sm100-auto-dispatch",
            "stage2_taus": list(STAGE2_TAUS),
            "thresh_type": "diag",
            "kv_splits": "auto",
            "dense_layers": [0],
            "sol_layers": list(range(1, STAGE2_TRANSFORMER_LAYERS)),
            "cross_attention": "dense",
            "compile_boundary": (
                "eager_inner_sol_callable"
                if self._isolate_sol_from_compile
                else "none"
            ),
            "completed_steps": self._step + 1,
            "dense_calls": self._dense_calls,
            "dense_calls_source": "inferred_from_layer1_entry_after_dense_layer0",
            "sol_calls": self._sol_calls,
            "kernel": get_sol_attn_stats(),
        }


class Stage2DenseAttention:
    """Instrument the unchanged native dense video self-attention path."""

    def __init__(
        self,
        transformer: Any,
        *,
        isolate_sol_from_compile: bool = False,
        compiled_transformer: bool = False,
    ) -> None:
        if isolate_sol_from_compile:
            raise ValueError("Dense attention instrumentation has no Sol compile boundary")
        self._compiled_transformer = bool(compiled_transformer)
        blocks = _transformer_blocks(transformer)
        if len(blocks) != STAGE2_TRANSFORMER_LAYERS:
            raise ValueError(
                "the fixed Stage 2 Dense policy requires exactly "
                f"{STAGE2_TRANSFORMER_LAYERS} transformer layers, got {len(blocks)}"
            )
        self._step = -1
        self._dense_calls = 0
        self._all2all_wrappers = tuple(
            _eager_block(block).attn1.attention_function for block in blocks
        )
        for layer_index, all2all in enumerate(self._all2all_wrappers):
            if not hasattr(all2all, "original_attention") or not callable(
                all2all.original_attention
            ):
                raise TypeError(
                    "Dense video self-attention requires an All2AllAttention-like "
                    f"wrapper at layer {layer_index}"
                )
            dense = all2all.original_attention
            all2all.original_attention = partial(self._call, layer_index, dense)

    def begin_denoise(self) -> None:
        self._step = -1
        self._dense_calls = 0

    def _call(
        self,
        layer_index: int,
        dense: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if layer_index == 0:
            next_step = self._step + 1
            if next_step >= len(STAGE2_TAUS):
                raise RuntimeError(
                    "Dense Stage 2 executed more than the required three denoise steps; "
                    "call begin_denoise() before a new sample"
                )
            self._step = next_step
        if self._step < 0:
            raise RuntimeError("Dense attention layer executed before layer 0")
        self._dense_calls += 1
        return dense(*args, **kwargs)

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "dense",
            "architecture": "ltx_native_dense",
            "stage2_taus": list(STAGE2_TAUS),
            "dense_layers": list(range(STAGE2_TRANSFORMER_LAYERS)),
            "sol_layers": [],
            "cross_attention": "dense",
            "compile_boundary": (
                "compiled_transformer_blocks" if self._compiled_transformer else "none"
            ),
            "completed_steps": self._step + 1,
            "dense_calls": self._dense_calls,
            "sol_calls": 0,
            "kernel": {"backend": "dense", "kernel_calls": 0},
        }
