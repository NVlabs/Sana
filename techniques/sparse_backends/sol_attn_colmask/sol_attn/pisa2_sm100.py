"""Stable public wrapper around the evidence-bound SM100 colmask runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def make_pisa2_sm100(
    T: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    global_threshold: torch.Tensor,
    scale: float,
    *,
    is_causal: bool = False,
    trace_route_masks: bool = False,
    guard_elements: int = 0,
):
    """Build the evidence-bound colmask runner for one prepared input."""

    from experiments.pisa2.native_bf16_claude50_colmask_full45_runner import (
        make_native_bf16_claude50_colmask_full45_runner,
    )

    return make_native_bf16_claude50_colmask_full45_runner(
        T,
        q,
        k,
        v,
        kc,
        vc,
        global_threshold,
        scale,
        is_causal=is_causal,
        trace_route_masks=trace_route_masks,
        guard_elements=guard_elements,
    )


__all__ = ["make_pisa2_sm100"]
