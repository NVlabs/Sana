"""Collective EasyCache controller for the SGLang MiniMax-H3 block stack."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

import torch
import torch.distributed as dist

from .adapter import StepContext, emit_event


CacheAction = Literal["compute", "reuse"]


def enabled() -> bool:
    return os.getenv("H3_EASYCACHE", "0") == "1"


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@torch.no_grad()
def _distributed_means(values: list[torch.Tensor]) -> list[float]:
    sums = torch.stack(
        [value.abs().sum(dtype=torch.float32) for value in values]
    )
    counts = torch.tensor(
        [value.numel() for value in values],
        device=sums.device,
        dtype=torch.float32,
    )
    totals = torch.stack((sums, counts))
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    means = totals[0] / totals[1].clamp_min(1.0)
    return [float(value) for value in means.cpu().tolist()]


@dataclass(frozen=True)
class EasyCacheConfig:
    threshold: float
    retain_steps: int
    cooldown_steps: int
    max_hits: int
    num_forwards: int

    @classmethod
    def from_env(cls) -> "EasyCacheConfig":
        config = cls(
            threshold=_env_float("H3_EASYCACHE_THRESHOLD", 0.30),
            retain_steps=_env_int("H3_EASYCACHE_RETAIN_STEPS", 6),
            cooldown_steps=_env_int("H3_EASYCACHE_COOLDOWN_STEPS", 4),
            max_hits=_env_int("H3_EASYCACHE_MAX_HITS", 2),
            num_forwards=_env_int("H3_EASYCACHE_NUM_FORWARDS", 49),
        )
        if config.threshold <= 0:
            raise ValueError("H3_EASYCACHE_THRESHOLD must be positive")
        if min(config.retain_steps, config.cooldown_steps, config.max_hits) < 0:
            raise ValueError("MiniMax-H3 EasyCache step counts must be non-negative")
        if config.num_forwards <= 0:
            raise ValueError("H3_EASYCACHE_NUM_FORWARDS must be positive")
        return config


class _Controller:
    def __init__(self, config: EasyCacheConfig) -> None:
        self.config = config
        self.request_key: tuple[str, int] | None = None
        self.last_step: int | None = None
        self.previous_step_input: torch.Tensor | None = None
        self.last_full_input: torch.Tensor | None = None
        self.last_full_output: torch.Tensor | None = None
        self.residual: torch.Tensor | None = None
        self.pending_input: torch.Tensor | None = None
        self.k: float | None = None
        self.accumulator = 0.0
        self.hits = 0
        self.calls = 0
        self.compute = 0
        self.reuse = 0

    def _finish_request(self) -> None:
        if self.request_key is None or not self.calls:
            return
        emit_event(
            "easycache_request_summary",
            request_epoch=self.request_key[0],
            request_index=self.request_key[1],
            calls=self.calls,
            compute=self.compute,
            reuse=self.reuse,
            reuse_rate=self.reuse / self.calls,
            threshold=self.config.threshold,
            retain_steps=self.config.retain_steps,
            cooldown_steps=self.config.cooldown_steps,
            max_hits=self.config.max_hits,
        )

    def _begin_request(self, context: StepContext) -> None:
        self._finish_request()
        self.request_key = (context.request_epoch, context.request_index)
        self.last_step = None
        self.previous_step_input = None
        self.last_full_input = None
        self.last_full_output = None
        self.residual = None
        self.pending_input = None
        self.k = None
        self.accumulator = 0.0
        self.hits = 0
        self.calls = 0
        self.compute = 0
        self.reuse = 0
        emit_event(
            "easycache_request_start",
            request_epoch=context.request_epoch,
            request_index=context.request_index,
            threshold=self.config.threshold,
            retain_steps=self.config.retain_steps,
            cooldown_steps=self.config.cooldown_steps,
            max_hits=self.config.max_hits,
            num_forwards=self.config.num_forwards,
        )

    @torch.no_grad()
    def before_blocks(
        self,
        hidden: torch.Tensor,
        *,
        context: StepContext,
    ) -> tuple[torch.Tensor, CacheAction]:
        request_key = (context.request_epoch, context.request_index)
        if self.request_key != request_key or (
            self.last_step is not None and context.step_index <= self.last_step
        ):
            self._begin_request(context)
        self.last_step = context.step_index

        step = context.step_index
        reason: str
        estimate: float | None = None
        input_change: float | None = None
        output_norm: float | None = None
        must_compute = step < self.config.retain_steps
        if must_compute:
            reason = "warmup"
            self.accumulator = 0.0
        elif step >= self.config.num_forwards - self.config.cooldown_steps:
            must_compute = True
            reason = "cooldown"
            self.accumulator = 0.0
        elif (
            self.previous_step_input is None
            or self.last_full_output is None
            or self.residual is None
            or self.k is None
        ):
            must_compute = True
            reason = "initialize"
        elif self.residual.shape != hidden.shape:
            must_compute = True
            reason = "shape_change"
        elif self.config.max_hits > 0 and self.hits >= self.config.max_hits:
            must_compute = True
            reason = "hit_cap"
        else:
            input_change, output_norm = _distributed_means(
                [hidden - self.previous_step_input, self.last_full_output]
            )
            estimate = self.k * input_change / max(output_norm, 1.0e-8)
            self.accumulator += estimate
            must_compute = self.accumulator >= self.config.threshold
            reason = "threshold" if must_compute else "below_threshold"
            if must_compute:
                self.accumulator = 0.0

        current_input = hidden.detach().clone()
        self.previous_step_input = current_input
        self.calls += 1
        if must_compute:
            self.compute += 1
            self.pending_input = current_input
            action: CacheAction = "compute"
        else:
            self.reuse += 1
            self.hits += 1
            self.pending_input = None
            hidden = hidden + self.residual
            action = "reuse"

        emit_event(
            "easycache_decision",
            request_epoch=context.request_epoch,
            request_index=context.request_index,
            step_index=step,
            action=action,
            reason=reason,
            predicted_relative_change=estimate,
            input_change=input_change,
            last_full_output_norm=output_norm,
            accumulator=self.accumulator,
            k=self.k,
            continuous_hits=self.hits,
        )
        return hidden, action

    @torch.no_grad()
    def after_blocks(self, hidden: torch.Tensor, *, context: StepContext) -> torch.Tensor:
        current_input = self.pending_input
        if current_input is None:
            raise RuntimeError("MiniMax-H3 EasyCache has no pending block input")

        input_change: float | None = None
        output_change: float | None = None
        if self.last_full_input is not None and self.last_full_output is not None:
            input_change, output_change = _distributed_means(
                [
                    current_input - self.last_full_input,
                    hidden - self.last_full_output,
                ]
            )
            self.k = output_change / max(input_change, 1.0e-8)

        self.last_full_input = current_input
        self.last_full_output = hidden.detach().clone()
        self.residual = (hidden - current_input).detach()
        self.pending_input = None
        self.accumulator = 0.0
        self.hits = 0
        emit_event(
            "easycache_refresh",
            request_epoch=context.request_epoch,
            request_index=context.request_index,
            step_index=context.step_index,
            k=self.k,
            input_change=input_change,
            output_change=output_change,
        )
        return hidden


_CONTROLLER: _Controller | None = None
_CONFIG: EasyCacheConfig | None = None


def _controller() -> _Controller:
    global _CONFIG, _CONTROLLER
    if not enabled():
        raise RuntimeError("MiniMax-H3 EasyCache is not enabled")
    config = EasyCacheConfig.from_env()
    if _CONTROLLER is None or config != _CONFIG:
        _CONFIG = config
        _CONTROLLER = _Controller(config)
    return _CONTROLLER


@torch.no_grad()
def before_blocks(
    hidden: torch.Tensor,
    *,
    context: StepContext,
) -> tuple[torch.Tensor, CacheAction]:
    return _controller().before_blocks(hidden, context=context)


@torch.no_grad()
def after_blocks(hidden: torch.Tensor, *, context: StepContext) -> torch.Tensor:
    return _controller().after_blocks(hidden, context=context)


__all__ = [
    "EasyCacheConfig",
    "after_blocks",
    "before_blocks",
    "enabled",
]
