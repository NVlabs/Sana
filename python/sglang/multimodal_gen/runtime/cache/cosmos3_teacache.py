# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 model-level TeaCache residual replay.

Cosmos3 already caches UND text K/V per request. This hook targets the
per-denoising-step GEN pathway: on a cache hit it skips the GEN decoder layer
stack and replays the cached residual, then still runs norm/projection and
unpatchify.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed import get_sp_group, get_sp_world_size
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, value, default)
        return default


def _env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return tuple(float(v) for v in value.split(",") if v.strip() != "")
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, value, default)
        return default


def _poly1d(coefficients: tuple[float, ...], x: float) -> float:
    """Evaluate a polynomial (highest-degree-first) at x via Horner's method."""
    value = 0.0
    for coefficient in coefficients:
        value = value * x + coefficient
    return value


@dataclass(frozen=True)
class Cosmos3TeaCacheConfig:
    enabled: bool = True
    threshold: float = 0.04
    start_step: int = 5
    end_step: int | None = None
    max_continuous_hits: int = 1
    periodic_recompute_steps: int = 0
    detach_on_store: bool = True
    clone_on_hit: bool = False
    log_decisions: bool = False
    # Polynomial (highest-degree-first, Horner) rescale of the per-step rel-L1
    # distance before accumulation. Identity (1, 0) => no rescale (uncalibrated).
    coefficients: tuple[float, ...] = (1.0, 0.0)


@dataclass
class Cosmos3TeaCacheStats:
    calls: int = 0
    computes: int = 0
    hits: int = 0
    disabled: int = 0
    threshold_recomputes: int = 0
    periodic_recomputes: int = 0
    missing_recomputes: int = 0
    boundary_recomputes: int = 0
    skipped_steps: list[int] = field(default_factory=list)
    computed_steps: list[int] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        eligible = self.computes + self.hits
        return 0.0 if eligible == 0 else self.hits / eligible

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "computes": self.computes,
            "hits": self.hits,
            "disabled": self.disabled,
            "hit_rate": round(self.hit_rate, 4),
            "threshold_recomputes": self.threshold_recomputes,
            "periodic_recomputes": self.periodic_recomputes,
            "missing_recomputes": self.missing_recomputes,
            "boundary_recomputes": self.boundary_recomputes,
            "skipped_steps": sorted(set(self.skipped_steps)),
            "computed_steps": sorted(set(self.computed_steps)),
        }


@dataclass
class _Entry:
    previous_feature: torch.Tensor | None = None
    residual: torch.Tensor | None = None
    accumulated_distance: float = 0.0
    last_compute_step: int | None = None
    continuous_hits: int = 0


@dataclass(frozen=True)
class Cosmos3TeaCacheDecision:
    should_skip: bool
    key: tuple[Any, ...]
    reason: str
    hidden_gen: torch.Tensor | None = None


class Cosmos3TeaCacheCoordinator:
    def __init__(self, config: Cosmos3TeaCacheConfig) -> None:
        self.config = config
        self._entries: dict[tuple[Any, ...], _Entry] = {}
        self._stats: dict[str, Cosmos3TeaCacheStats] = {}

    def reset(self) -> None:
        self._entries.clear()

    def reset_stats(self) -> None:
        self._stats.clear()

    def stats_summary(self) -> dict[str, Any]:
        return {key: stats.as_dict() for key, stats in sorted(self._stats.items())}

    def _key(
        self,
        *,
        cache_key: str,
        hidden_gen: torch.Tensor,
        noisy_frame_mask_present: bool,
        sequence_shard_enabled: bool,
    ) -> tuple[Any, ...]:
        return (
            cache_key,
            tuple(hidden_gen.shape),
            str(hidden_gen.dtype),
            bool(noisy_frame_mask_present),
            bool(sequence_shard_enabled),
        )

    def _eligible(
        self,
        *,
        step: int,
        num_inference_steps: int | None,
    ) -> bool:
        if not self.config.enabled:
            return False
        if step < self.config.start_step:
            return False
        if num_inference_steps is not None and step >= num_inference_steps - 1:
            return False
        if self.config.end_step is not None and step >= self.config.end_step:
            return False
        return True

    @staticmethod
    def _feature(hidden_gen: torch.Tensor) -> torch.Tensor:
        return hidden_gen.detach()

    @staticmethod
    def _rel_l1(feature: torch.Tensor, previous_feature: torch.Tensor) -> float:
        diff_mean = (feature.float() - previous_feature.float()).abs().mean()
        prev_mean = previous_feature.float().abs().mean().clamp_min(1e-6)

        sp_size = get_sp_world_size()
        sp_group = get_sp_group() if sp_size > 1 else None
        if sp_group is not None and dist.is_available() and dist.is_initialized():
            dist.all_reduce(
                diff_mean, op=dist.ReduceOp.AVG, group=sp_group.device_group
            )
            dist.all_reduce(
                prev_mean, op=dist.ReduceOp.AVG, group=sp_group.device_group
            )

        return float((diff_mean / prev_mean).detach().cpu().item())

    def lookup(
        self,
        *,
        hidden_gen: torch.Tensor,
        cache_key: str,
        step: int,
        num_inference_steps: int | None,
        noisy_frame_mask_present: bool = False,
        sequence_shard_enabled: bool = False,
    ) -> Cosmos3TeaCacheDecision:
        stats = self._stats.setdefault(cache_key, Cosmos3TeaCacheStats())
        stats.calls += 1

        key = self._key(
            cache_key=cache_key,
            hidden_gen=hidden_gen,
            noisy_frame_mask_present=noisy_frame_mask_present,
            sequence_shard_enabled=sequence_shard_enabled,
        )

        if not self._eligible(step=step, num_inference_steps=num_inference_steps):
            stats.disabled += 1
            stats.boundary_recomputes += 1
            stats.computed_steps.append(step)
            return Cosmos3TeaCacheDecision(False, key, "disabled_or_boundary")

        entry = self._entries.setdefault(key, _Entry())
        if entry.residual is None or entry.previous_feature is None:
            stats.computes += 1
            stats.missing_recomputes += 1
            stats.computed_steps.append(step)
            return Cosmos3TeaCacheDecision(False, key, "missing_cache")

        if (
            self.config.periodic_recompute_steps > 0
            and entry.last_compute_step is not None
        ):
            if step - entry.last_compute_step >= self.config.periodic_recompute_steps:
                entry.continuous_hits = 0
                stats.computes += 1
                stats.periodic_recomputes += 1
                stats.computed_steps.append(step)
                return Cosmos3TeaCacheDecision(False, key, "periodic_recompute")

        if (
            self.config.max_continuous_hits >= 0
            and entry.continuous_hits >= self.config.max_continuous_hits
        ):
            entry.continuous_hits = 0
            stats.computes += 1
            stats.periodic_recomputes += 1
            stats.computed_steps.append(step)
            return Cosmos3TeaCacheDecision(False, key, "continuous_hit_cap")

        feature = self._feature(hidden_gen)
        if entry.previous_feature.shape != feature.shape:
            entry.continuous_hits = 0
            stats.computes += 1
            stats.missing_recomputes += 1
            stats.computed_steps.append(step)
            return Cosmos3TeaCacheDecision(False, key, "feature_shape_changed")

        rel_l1 = self._rel_l1(feature, entry.previous_feature)
        # Calibration: rescale the per-step distance by a polynomial. Identity
        # coefficients (1, 0) leave rel_l1 unchanged (uncalibrated behavior).
        rel_l1 = max(0.0, _poly1d(self.config.coefficients, rel_l1))
        accumulated = entry.accumulated_distance + rel_l1
        if accumulated >= self.config.threshold:
            if self.config.log_decisions:
                logger.info(
                    "Cosmos3 TeaCache recompute step=%s key=%s rel_l1=%.6f "
                    "accum=%.6f threshold=%.6f",
                    step,
                    cache_key,
                    rel_l1,
                    accumulated,
                    self.config.threshold,
                )
            entry.accumulated_distance = 0.0
            entry.continuous_hits = 0
            stats.computes += 1
            stats.threshold_recomputes += 1
            stats.computed_steps.append(step)
            return Cosmos3TeaCacheDecision(False, key, f"threshold:{accumulated:.6f}")

        entry.accumulated_distance = accumulated
        entry.continuous_hits += 1
        stats.hits += 1
        stats.skipped_steps.append(step)
        residual = (
            entry.residual.clone() if self.config.clone_on_hit else entry.residual
        )
        if self.config.log_decisions:
            logger.info(
                "Cosmos3 TeaCache hit step=%s key=%s rel_l1=%.6f accum=%.6f",
                step,
                cache_key,
                rel_l1,
                accumulated,
            )
        return Cosmos3TeaCacheDecision(
            True, key, "cache_hit", hidden_gen=hidden_gen + residual
        )

    def store(
        self,
        decision: Cosmos3TeaCacheDecision | None,
        *,
        original_hidden_gen: torch.Tensor,
        hidden_gen: torch.Tensor,
        step: int,
    ) -> None:
        if decision is None or decision.should_skip:
            return
        entry = self._entries.setdefault(decision.key, _Entry())
        residual = hidden_gen - original_hidden_gen
        if self.config.detach_on_store:
            residual = residual.detach()
        entry.residual = residual
        entry.previous_feature = self._feature(original_hidden_gen)
        entry.last_compute_step = step
        entry.continuous_hits = 0


def cosmos3_teacache_config_from_env() -> Cosmos3TeaCacheConfig:
    return Cosmos3TeaCacheConfig(
        enabled=_env_flag("SGLANG_COSMOS3_TEACACHE_ENABLED", False),
        threshold=_env_float("SGLANG_COSMOS3_TEACACHE_THRESH", 0.04),
        start_step=_env_int("SGLANG_COSMOS3_TEACACHE_START", 5),
        end_step=(
            None
            if os.environ.get("SGLANG_COSMOS3_TEACACHE_END", "") == ""
            else _env_int("SGLANG_COSMOS3_TEACACHE_END", -1)
        ),
        max_continuous_hits=_env_int("SGLANG_COSMOS3_TEACACHE_MAX_CONTINUOUS_HITS", 1),
        periodic_recompute_steps=_env_int(
            "SGLANG_COSMOS3_TEACACHE_PERIODIC_RECOMPUTE_STEPS", 0
        ),
        detach_on_store=_env_flag("SGLANG_COSMOS3_TEACACHE_DETACH_ON_STORE", True),
        clone_on_hit=_env_flag("SGLANG_COSMOS3_TEACACHE_CLONE_ON_HIT", False),
        log_decisions=_env_flag("SGLANG_COSMOS3_TEACACHE_LOG_DECISIONS", False),
        coefficients=_env_float_tuple(
            "SGLANG_COSMOS3_TEACACHE_COEFFICIENTS", (1.0, 0.0)
        ),
    )


__all__ = [
    "Cosmos3TeaCacheConfig",
    "Cosmos3TeaCacheCoordinator",
    "Cosmos3TeaCacheDecision",
    "cosmos3_teacache_config_from_env",
]
