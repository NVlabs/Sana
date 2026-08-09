"""TeaCache residual reuse for the packed MiniMax-H3 main DiT."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.distributed as dist

from adapter import StepContext, emit_event


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _coefficients() -> tuple[float, ...]:
    raw = os.getenv("H3_TEACACHE_COEFFICIENTS", "1.0,0.0")
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("H3_TEACACHE_COEFFICIENTS must not be empty")
    return values


def enabled() -> bool:
    return os.getenv("H3_TEACACHE_ENABLED", "0") == "1"


@dataclass(frozen=True)
class TeaCacheConfig:
    threshold: float
    retain_steps: int
    cooldown_steps: int
    coefficients: tuple[float, ...]

    @classmethod
    def from_env(cls) -> "TeaCacheConfig":
        config = cls(
            threshold=_env_float("H3_TEACACHE_THRESHOLD", 0.10),
            retain_steps=_env_int("H3_TEACACHE_RETAIN_STEPS", 5),
            cooldown_steps=_env_int("H3_TEACACHE_COOLDOWN_STEPS", 1),
            coefficients=_coefficients(),
        )
        if config.threshold <= 0:
            raise ValueError("H3_TEACACHE_THRESHOLD must be positive")
        if min(config.retain_steps, config.cooldown_steps) < 0:
            raise ValueError("MiniMax-H3 TeaCache step counts must be non-negative")
        return config


def _polyval(coefficients: tuple[float, ...], value: float) -> float:
    result = 0.0
    for coefficient in coefficients:
        result = result * value + coefficient
    return result


@torch.no_grad()
def _relative_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    if current.shape != previous.shape:
        raise ValueError(
            f"MiniMax-H3 TeaCache probe shape changed: {current.shape} != {previous.shape}"
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


class _TeaCacheController:
    def __init__(self, config: TeaCacheConfig) -> None:
        self.config = config
        self.request_epoch: str | None = None
        self.previous_modulated_input: torch.Tensor | None = None
        self.residual: torch.Tensor | None = None
        self.pending_input: torch.Tensor | None = None
        self.accumulator = 0.0
        self.calls = 0
        self.compute = 0
        self.reuse = 0

    def _emit_summary(self) -> None:
        if self.request_epoch is None or self.calls == 0:
            return
        emit_event(
            "teacache_generation_summary",
            request_epoch=self.request_epoch,
            calls=self.calls,
            compute=self.compute,
            reuse=self.reuse,
            reuse_rate=self.reuse / self.calls,
            threshold=self.config.threshold,
            coefficients=self.config.coefficients,
        )

    def _begin_generation(self, request_epoch: str) -> None:
        self._emit_summary()
        self.request_epoch = request_epoch
        self.previous_modulated_input = None
        self.residual = None
        self.pending_input = None
        self.accumulator = 0.0
        self.calls = 0
        self.compute = 0
        self.reuse = 0
        emit_event(
            "teacache_generation_start",
            request_epoch=request_epoch,
            threshold=self.config.threshold,
            retain_steps=self.config.retain_steps,
            cooldown_steps=self.config.cooldown_steps,
            coefficients=self.config.coefficients,
        )

    @torch.no_grad()
    def before_blocks(
        self,
        hidden: torch.Tensor,
        modulated_input: torch.Tensor,
        *,
        context: StepContext,
    ) -> tuple[torch.Tensor, str]:
        if self.request_epoch != context.request_epoch:
            self._begin_generation(context.request_epoch)

        num_forwards = _env_int("H3_TEACACHE_NUM_FORWARDS", 49)
        step = context.step_index
        relative_l1: float | None = None
        rescaled_l1: float | None = None
        reason = "threshold"
        must_compute = step < self.config.retain_steps
        if must_compute:
            reason = "warmup"
            self.accumulator = 0.0
        elif step >= num_forwards - self.config.cooldown_steps:
            must_compute = True
            reason = "cooldown"
            self.accumulator = 0.0
        elif self.previous_modulated_input is None or self.residual is None:
            must_compute = True
            reason = "initialize"
        else:
            relative_l1 = _relative_l1(modulated_input, self.previous_modulated_input)
            rescaled_l1 = _polyval(self.config.coefficients, relative_l1)
            self.accumulator += rescaled_l1
            must_compute = self.accumulator >= self.config.threshold
            if must_compute:
                self.accumulator = 0.0
            else:
                reason = "below_threshold"

        self.previous_modulated_input = modulated_input.detach().clone()
        self.calls += 1
        if must_compute:
            self.compute += 1
            self.pending_input = hidden.detach().clone()
            action = "compute"
        else:
            self.reuse += 1
            self.pending_input = None
            hidden = hidden + self.residual
            action = "reuse"

        emit_event(
            "teacache_decision",
            request_epoch=context.request_epoch,
            step_index=step,
            action=action,
            reason=reason,
            relative_l1=relative_l1,
            rescaled_l1=rescaled_l1,
            accumulator=self.accumulator,
        )
        return hidden, action

    @torch.no_grad()
    def after_blocks(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.pending_input is None:
            raise RuntimeError("MiniMax-H3 TeaCache is missing its block input")
        self.residual = (hidden - self.pending_input).detach()
        self.pending_input = None
        return hidden


_CONTROLLER: _TeaCacheController | None = None
_CONFIG: TeaCacheConfig | None = None


def _controller() -> _TeaCacheController | None:
    global _CONFIG, _CONTROLLER
    if not enabled():
        return None
    config = TeaCacheConfig.from_env()
    if _CONTROLLER is None or config != _CONFIG:
        _CONFIG = config
        _CONTROLLER = _TeaCacheController(config)
    return _CONTROLLER


@torch.no_grad()
def before_blocks(
    hidden: torch.Tensor,
    modulated_input: torch.Tensor,
    *,
    context: StepContext,
) -> tuple[torch.Tensor, str | None]:
    controller = _controller()
    if controller is None:
        return hidden, None
    return controller.before_blocks(hidden, modulated_input, context=context)


@torch.no_grad()
def after_blocks(hidden: torch.Tensor) -> torch.Tensor:
    controller = _controller()
    if controller is None:
        return hidden
    return controller.after_blocks(hidden)


__all__ = ["TeaCacheConfig", "after_blocks", "before_blocks", "enabled"]
