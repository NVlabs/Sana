"""Stage-2 Sol-Attn adapter for the distilled LTX-2.5 pipeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager

STAGE2_TAUS = (1.0, 1.25, 1.5)
LAYERS_PER_FORWARD = 48


def route_for_call(call_index: int) -> tuple[float | None, int, int]:
    """Return ``(tau, forward, layer)``; ``tau=None`` selects dense attention."""

    forward_index, layer_index = divmod(call_index, LAYERS_PER_FORWARD)
    if forward_index >= len(STAGE2_TAUS):
        raise RuntimeError(f"unexpected Stage-2 video attention call {call_index}")
    tau = None if layer_index == 0 else STAGE2_TAUS[forward_index]
    return tau, forward_index, layer_index


def _disable_torch_compile(function):
    import torch

    return torch.compiler.disable(function)


class LTX25Stage2SolAttention:
    """Keep non-video calls and layer 0 dense; route the remaining calls to Sol."""

    def __init__(self) -> None:
        from ltx_core.model.transformer.attention import AttentionFunction

        self._dense = AttentionFunction.AUTOMATIC.to_callable()
        self.reset()

    @property
    def label(self) -> str:
        return f"LTX25-Stage2-Sol({self._dense.label})"

    def reset(self) -> None:
        from techniques.sparse_backends.sol_attn_backend import reset_sol_attn_state

        self.video_calls = 0
        self.sol_calls = 0
        self.dense_video_calls = 0
        self.tau_calls: Counter[float] = Counter()
        self.enabled = False
        reset_sol_attn_state()

    @contextmanager
    def stage2(self, enabled: bool) -> Iterator[None]:
        previous = self.enabled
        self.enabled = enabled
        try:
            yield
        finally:
            self.enabled = previous

    @staticmethod
    def _is_video_self(q, k, v, heads: int) -> bool:
        return q.shape == k.shape == v.shape and q.shape[-1] // heads == 128

    def _dense_bthd(self, q, k, v):
        batch, tokens, heads, head_dim = q.shape
        output = self._dense(
            q.reshape(batch, tokens, heads * head_dim),
            k.reshape(batch, tokens, heads * head_dim),
            v.reshape(batch, tokens, heads * head_dim),
            heads,
        )
        return output.view(batch, tokens, heads, head_dim)

    @_disable_torch_compile
    def __call__(self, q, k, v, heads: int):
        if not self.enabled or not self._is_video_self(q, k, v, heads):
            return self._dense(q, k, v, heads)

        call_index = self.video_calls
        self.video_calls += 1
        tau, _forward_index, _layer_index = route_for_call(call_index)
        if tau is None:
            self.dense_video_calls += 1
            return self._dense(q, k, v, heads)

        from techniques.sparse_backends.sol_attn_backend import _run_sol_attn_bthd

        batch, tokens, channels = q.shape
        head_dim = channels // heads
        q_bthd = q.view(batch, tokens, heads, head_dim).contiguous()
        k_bthd = k.view(batch, tokens, heads, head_dim).contiguous()
        v_bthd = v.view(batch, tokens, heads, head_dim).contiguous()
        output = _run_sol_attn_bthd(
            q_bthd,
            k_bthd,
            v_bthd,
            tau=tau,
            thresh_type="diag",
            kv_splits=1,
            dense_fn=self._dense_bthd,
        )
        self.sol_calls += 1
        self.tau_calls[tau] += 1
        return output.reshape(batch, tokens, channels)

    def stats(self) -> dict[str, object]:
        from techniques.sparse_backends.sol_attn_backend import get_sol_attn_stats

        return {
            "label": self.label,
            "video_calls": self.video_calls,
            "sol_calls": self.sol_calls,
            "dense_video_calls": self.dense_video_calls,
            "tau_calls": {str(tau): self.tau_calls[tau] for tau in STAGE2_TAUS},
            "kernel": get_sol_attn_stats(),
        }
