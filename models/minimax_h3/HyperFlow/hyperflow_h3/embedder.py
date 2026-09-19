# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
#
# Two-time (t, r) conditioning follows AnyFlow (Gu et al., 2026):
#   https://arxiv.org/abs/2605.13724
#   https://github.com/NVlabs/AnyFlow
"""Two-time conditioning as a drop-in replacement for ``transformer.time_embedder``."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn


class TwoTimeEmbedder(nn.Module):
    r"""Interval-conditioned timestep embedding: ``emb_t(t) + gate * (emb_r(r) - emb_t(t))``.

    The official ``MiniMaxH3Transformer3DModel.forward`` embeds its distinct timesteps with two lines::

        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))

    HyperFlow conditions every step on the interval it integrates, ``(t, r)``, not on the point ``t``. Replacing
    ``transformer.time_embedder`` with this module is the whole model-side change: ``base`` is the official
    ``TimestepEmbedding`` (with the adapter's LoRA on it), ``endpoint`` is an identically shaped copy of it (with its
    own LoRA) that embeds ``r``, and ``gate`` blends the two. The endpoint timesteps are set by the HyperFlow denoise
    block right before each forward; the module refuses to run without them so the official loop cannot silently
    produce single-time outputs from a two-time adapter.

    ``passthrough=True`` turns the module back into the official embedder (``r`` ignored). [`disable_hyperflow`]
    uses it together with disabling the LoRA layers, which is the exact base model again.

    Args:
        base: The transformer's ``time_embedder``.
        endpoint: A deep copy of it, made before any LoRA is injected.
        time_proj: The transformer's parameter-free sinusoidal projection (``Timesteps``), copied.
        gate: Blend factor from the adapter's metadata (``hyperflow_gate``).
    """

    def __init__(self, base: nn.Module, endpoint: nn.Module, time_proj: nn.Module, gate: float):
        super().__init__()
        self.base = base
        self.endpoint = endpoint
        self.time_proj = time_proj
        self.gate = float(gate)
        self.passthrough = False
        self._endpoint_timesteps: torch.Tensor | None = None

    @classmethod
    def wrap(cls, transformer: nn.Module, gate: float) -> TwoTimeEmbedder:
        """Replace ``transformer.time_embedder`` in place and return the wrapper (idempotent)."""
        current = transformer.time_embedder
        if isinstance(current, cls):
            return current
        endpoint = copy.deepcopy(current)
        time_proj = copy.deepcopy(transformer.time_proj)
        wrapper = cls(current, endpoint, time_proj, gate)
        transformer.time_embedder = wrapper
        return wrapper

    # The official forward reads `time_embedder.linear_1.weight.dtype`; keep the attribute reachable.
    @property
    def linear_1(self) -> nn.Module:
        return self.base.linear_1

    @property
    def linear_2(self) -> nn.Module:
        return self.base.linear_2

    def set_endpoint(self, endpoint_timesteps: torch.Tensor | None) -> None:
        """Set (or clear, with ``None``) the endpoint of each distinct timestep of the next forward."""
        self._endpoint_timesteps = endpoint_timesteps

    @contextmanager
    def endpoint_context(self, endpoint_timesteps: torch.Tensor) -> Iterator[None]:
        """Set the endpoints for one forward and clear them afterwards, even if the forward raises."""
        self.set_endpoint(endpoint_timesteps)
        try:
            yield
        finally:
            self.set_endpoint(None)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        t_emb = self.base(sample)
        if self.passthrough:
            return t_emb
        if self._endpoint_timesteps is None:
            raise RuntimeError(
                "HyperFlow needs the endpoint timesteps of this forward, but none were set. Run the transformer "
                "through the HyperFlow denoise blocks (`hyperflow_blocks(...)`), or call `disable_hyperflow(...)` "
                "to use it as the plain base model."
            )
        endpoint = self._endpoint_timesteps.to(device=sample.device)
        if endpoint.shape[0] != sample.shape[0]:
            raise ValueError(
                f"Got {sample.shape[0]} distinct timesteps but {endpoint.shape[0]} endpoints; every timestep needs "
                "exactly one endpoint. Build the plan with `build_row_time_pairs`."
            )
        r_proj = self.time_proj(endpoint).to(sample.dtype)
        r_emb = self.endpoint(r_proj)
        return t_emb + self.gate * (r_emb - t_emb)


__all__ = ["TwoTimeEmbedder"]
