"""Collective FirstBlockCache for the SGLang MiniMax-H3 block stack."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

import torch
import torch.distributed as dist

from .adapter import StepContext, emit_event


CacheAction = Literal["compute", "reuse"]


def enabled() -> bool:
    return os.getenv("H3_FIRSTBLOCKCACHE", "0") == "1"


@dataclass(frozen=True)
class FirstBlockCacheConfig:
    threshold: float

    @classmethod
    def from_env(cls) -> "FirstBlockCacheConfig":
        config = cls(float(os.getenv("H3_CACHE_THRESHOLD", "0.08")))
        if config.threshold <= 0:
            raise ValueError("H3_CACHE_THRESHOLD must be positive")
        return config


@torch.no_grad()
def _relative_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    if current.shape != previous.shape:
        raise ValueError(
            f"MiniMax-H3 FirstBlockCache shape changed: {current.shape} != "
            f"{previous.shape}"
        )
    totals = torch.stack(
        (
            (current - previous).abs().sum(dtype=torch.float32),
            previous.abs().sum(dtype=torch.float32),
        )
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return float((totals[0] / totals[1].clamp_min(1.0e-8)).cpu().item())


class _Controller:
    def __init__(self, config: FirstBlockCacheConfig) -> None:
        self.config = config
        self.request_key: tuple[str, int] | None = None
        self.previous_head_residual: torch.Tensor | None = None
        self.tail_residual: torch.Tensor | None = None
        self.pending_head_output: torch.Tensor | None = None
        self.calls = 0
        self.compute = 0
        self.reuse = 0

    def _finish_request(self) -> None:
        if self.request_key is None or not self.calls:
            return
        emit_event(
            "firstblockcache_request_summary",
            request_epoch=self.request_key[0],
            request_index=self.request_key[1],
            calls=self.calls,
            compute=self.compute,
            reuse=self.reuse,
            reuse_rate=self.reuse / self.calls,
            threshold=self.config.threshold,
        )

    def _begin_request(self, context: StepContext) -> None:
        self._finish_request()
        self.request_key = (context.request_epoch, context.request_index)
        self.previous_head_residual = None
        self.tail_residual = None
        self.pending_head_output = None
        self.calls = 0
        self.compute = 0
        self.reuse = 0
        emit_event(
            "firstblockcache_request_start",
            request_epoch=context.request_epoch,
            request_index=context.request_index,
            threshold=self.config.threshold,
        )

    @torch.no_grad()
    def after_head(
        self,
        head_input: torch.Tensor,
        head_output: torch.Tensor,
        *,
        context: StepContext,
    ) -> tuple[torch.Tensor, CacheAction]:
        request_key = (context.request_epoch, context.request_index)
        if self.request_key != request_key:
            self._begin_request(context)

        head_residual = (head_output - head_input).detach()
        relative_l1: float | None = None
        shape_changed = (
            self.previous_head_residual is not None
            and self.previous_head_residual.shape != head_residual.shape
        )
        if shape_changed:
            emit_event(
                "firstblockcache_shape_reset",
                request_epoch=context.request_epoch,
                request_index=context.request_index,
                step_index=context.step_index,
                previous_shape=list(self.previous_head_residual.shape),
                current_shape=list(head_residual.shape),
            )
            self.previous_head_residual = None
            self.tail_residual = None
            self.pending_head_output = None
        must_compute = (
            self.previous_head_residual is None or self.tail_residual is None
        )
        reason = "shape_change" if shape_changed else "initialize"
        if not must_compute:
            relative_l1 = _relative_l1(head_residual, self.previous_head_residual)
            must_compute = relative_l1 > self.config.threshold
            reason = "threshold" if must_compute else "below_threshold"

        self.calls += 1
        if must_compute:
            self.compute += 1
            self.previous_head_residual = head_residual.clone()
            self.pending_head_output = head_output.detach().clone()
            hidden = head_output
            action: CacheAction = "compute"
        else:
            self.reuse += 1
            self.pending_head_output = None
            hidden = head_output + self.tail_residual
            action = "reuse"

        emit_event(
            "firstblockcache_decision",
            request_epoch=context.request_epoch,
            request_index=context.request_index,
            step_index=context.step_index,
            action=action,
            reason=reason,
            relative_l1=relative_l1,
            threshold=self.config.threshold,
        )
        return hidden, action

    @torch.no_grad()
    def after_tail(self, hidden: torch.Tensor, *, context: StepContext) -> torch.Tensor:
        if self.pending_head_output is None:
            raise RuntimeError("MiniMax-H3 FirstBlockCache has no head-block output")
        self.tail_residual = (hidden - self.pending_head_output).detach()
        self.pending_head_output = None
        emit_event(
            "firstblockcache_refresh",
            request_epoch=context.request_epoch,
            request_index=context.request_index,
            step_index=context.step_index,
        )
        return hidden


_CONTROLLER: _Controller | None = None
_CONFIG: FirstBlockCacheConfig | None = None


def _controller() -> _Controller:
    global _CONFIG, _CONTROLLER
    if not enabled():
        raise RuntimeError("MiniMax-H3 FirstBlockCache is not enabled")
    config = FirstBlockCacheConfig.from_env()
    if _CONTROLLER is None or config != _CONFIG:
        _CONFIG = config
        _CONTROLLER = _Controller(config)
    return _CONTROLLER


@torch.no_grad()
def after_head(
    head_input: torch.Tensor,
    head_output: torch.Tensor,
    *,
    context: StepContext,
) -> tuple[torch.Tensor, CacheAction]:
    return _controller().after_head(head_input, head_output, context=context)


@torch.no_grad()
def after_tail(hidden: torch.Tensor, *, context: StepContext) -> torch.Tensor:
    return _controller().after_tail(hidden, context=context)


__all__ = [
    "FirstBlockCacheConfig",
    "after_head",
    "after_tail",
    "enabled",
]
