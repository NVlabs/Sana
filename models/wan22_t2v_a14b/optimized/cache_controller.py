"""Experiment-local Wan2.2 A14B cache families on the frozen CP4 substrate.

The module is imported only when ``WAN22_CACHE_METHOD`` selects one of the
three families allowed by the cache_ca workflow.  Context parallelism is
installed before this adapter.  Diffusers therefore leaves its sequence-split
hook on the original first transformer block; this adapter always executes
that block and conditionally replays or forecasts the residual of blocks 1-39.

Keeping block 0 fresh has two useful properties: the CP split/gather contract
is unchanged, and every cache hit still receives current latent, timestep, and
text conditioning before the cached tail residual is applied.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


ALLOWED_METHODS = {"teacache", "easycache", "taylorseer"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw in (None, "") else float(raw)


def _env_floats(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return list(default)
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one coefficient")
    return values


def configured_method() -> str:
    method = os.environ.get("WAN22_CACHE_METHOD", "").strip().lower()
    if method in {"", "off", "none", "0", "false"}:
        return ""
    if method not in ALLOWED_METHODS:
        allowed = ", ".join(sorted(ALLOWED_METHODS))
        raise ValueError(f"unsupported WAN22_CACHE_METHOD={method!r}; expected {allowed} or off")
    return method


def _polyval(coefficients: list[float], value: float) -> float:
    result = 0.0
    for coefficient in coefficients:
        result = result * value + coefficient
    return result


def _distributed_mean(value: torch.Tensor) -> float:
    scalar = value.detach().float()
    if scalar.numel() != 1:
        scalar = scalar.mean()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
        scalar /= dist.get_world_size()
    return float(scalar.item())


def _relative_l1(current: torch.Tensor, previous: torch.Tensor, distributed: bool = False) -> float:
    numerator = (current.float() - previous.float()).abs().mean()
    denominator = previous.float().abs().mean().clamp_min(1e-8)
    if distributed and dist.is_available() and dist.is_initialized():
        pair = torch.stack([numerator, denominator])
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
        pair /= dist.get_world_size()
        numerator, denominator = pair[0], pair[1]
    return float((numerator / denominator.clamp_min(1e-8)).item())


class WanCacheController:
    """Prompt-local policy shared by both experts and both CFG branches."""

    expected_branches = 2

    def __init__(self, method: str, num_steps: int):
        if method not in ALLOWED_METHODS:
            raise ValueError(f"unsupported cache family: {method}")
        self.method = method
        self.num_steps = int(num_steps)

        defaults_start = {"teacache": 5, "easycache": 5, "taylorseer": 4}
        defaults_tail = {"teacache": 3, "easycache": 3, "taylorseer": 3}
        self.start_step = _env_int("WAN22_CACHE_START_STEP", defaults_start[method])
        self.tail_steps = _env_int("WAN22_CACHE_TAIL_STEPS", defaults_tail[method])
        self.max_reuse = _env_int("WAN22_CACHE_MAX_REUSE", 1)

        # Public Wan 14B TeaCache retention-step polynomial.  Each expert keeps
        # an independent accumulator because the MoE switch changes dynamics.
        self.tea_threshold = _env_float("WAN22_TEACACHE_THRESHOLD", 0.05)
        self.tea_coefficients = _env_floats(
            "WAN22_TEACACHE_COEFFICIENTS",
            [-3.03318725e5, 4.90537029e4, -2.65530556e3, 5.87365115e1, -3.15583525e-1],
        )
        self.tea_periodic = _env_int("WAN22_TEACACHE_PERIODIC", 0)

        self.easy_threshold = _env_float("WAN22_EASYCACHE_THRESHOLD", 0.30)

        self.taylor_order = _env_int("WAN22_TAYLOR_ORDER", 1)
        if self.taylor_order not in {1, 2}:
            raise ValueError("WAN22_TAYLOR_ORDER must be 1 or 2")
        self.taylor_interval = _env_int("WAN22_TAYLOR_INTERVAL", 3)
        if self.taylor_interval < 2:
            raise ValueError("WAN22_TAYLOR_INTERVAL must be at least 2")
        self.taylor_error_threshold = _env_float("WAN22_TAYLOR_ERROR_THRESHOLD", math.inf)
        self.taylor_damping = _env_float("WAN22_TAYLOR_DAMPING", 1.0)

        self.probe_tokens = max(1, _env_int("WAN22_CACHE_PROBE_TOKENS", 64))
        self.probe_channels = max(1, _env_int("WAN22_CACHE_PROBE_CHANNELS", 128))

        self.payloads: dict[tuple[str, int], torch.Tensor] = {}
        self.histories: dict[tuple[str, int], list[tuple[int, torch.Tensor, torch.Tensor]]] = {}
        self.experts: dict[str, dict[str, Any]] = {}
        self.trace: list[dict[str, Any]] = []
        self.compute_steps = 0
        self.reuse_steps = 0
        self.branch_compute = 0
        self.branch_reuse = 0
        self.hit_pattern: list[str] = []

        self.step = -1
        self.expert = ""
        self.branch_counter = 0
        self.current_decision: str | None = None
        self.current_reason = ""
        self.current_indicator: float | None = None
        self.current_predicted_error: float | None = None
        self.current_forecast_error: float | None = None
        self.current_input_probe: torch.Tensor | None = None
        self.current_signal: torch.Tensor | None = None
        self.current_branch0_output_probe: torch.Tensor | None = None

    def _probe(self, value: torch.Tensor) -> torch.Tensor:
        detached = value.detach()
        if detached.ndim >= 3:
            sequence = detached.shape[-2]
            stride = max(1, sequence // self.probe_tokens)
            detached = detached[..., ::stride, : self.probe_channels]
            detached = detached[..., : self.probe_tokens, :]
        return detached.float().clone()

    def _expert_state(self, expert: str) -> dict[str, Any]:
        return self.experts.setdefault(
            expert,
            {
                "accumulated": 0.0,
                "consecutive_reuse": 0,
                "steps_since_compute": 0,
                "last_compute_step": None,
                "previous_signal": None,
                "previous_input_probe": None,
                "last_compute_input_probe": None,
                "last_compute_output_probe": None,
                "last_output_norm": None,
                "k": None,
                "force_next": False,
                "forecast_error": None,
            },
        )

    def _payloads_ready(self, expert: str) -> bool:
        for branch in range(self.expected_branches):
            key = (expert, branch)
            if self.method == "taylorseer":
                if len(self.histories.get(key) or []) < self.taylor_order + 1:
                    return False
            elif key not in self.payloads:
                return False
        return True

    def _common_force_reason(self, state: dict[str, Any]) -> str | None:
        if self.step < self.start_step:
            return "warmup_guard"
        if self.step >= self.num_steps - self.tail_steps:
            return "tail_guard"
        if not self._payloads_ready(self.expert):
            return "payload_seed"
        if self.max_reuse > 0 and state["consecutive_reuse"] >= self.max_reuse:
            return "max_reuse_guard"
        return None

    def begin_call(
        self,
        expert: str,
        fresh_prefix_output: torch.Tensor,
        timestep_projection: torch.Tensor,
    ) -> int:
        """Return the CFG branch and decide once, on the conditional branch."""
        if self.branch_counter == 0:
            self.step += 1
            self.expert = str(expert)
            self.current_decision = None
            self.current_reason = ""
            self.current_indicator = None
            self.current_predicted_error = None
            self.current_forecast_error = None
            self.current_branch0_output_probe = None
            self.current_signal = timestep_projection.detach().float().clone()
            self.current_input_probe = self._probe(fresh_prefix_output) if self.method == "easycache" else None

            if self.method == "teacache":
                self._decide_teacache()
            elif self.method == "easycache":
                self._decide_easycache()
            else:
                self._decide_taylorseer()
        elif str(expert) != self.expert:
            raise RuntimeError(
                f"Wan cache expert changed within one CFG pair: {self.expert!r} -> {expert!r}"
            )

        branch = self.branch_counter
        self.branch_counter += 1
        if branch >= self.expected_branches:
            raise RuntimeError("Wan cache observed more than two CFG calls in one denoising step")
        return branch

    def _set_decision(self, reuse: bool, reason: str) -> None:
        self.current_decision = "reuse" if reuse else "compute"
        self.current_reason = reason

    def _decide_teacache(self) -> None:
        state = self._expert_state(self.expert)
        force_reason = self._common_force_reason(state)
        previous = state["previous_signal"]
        if force_reason is None and previous is None:
            force_reason = "signal_seed"
        if force_reason is None and self.tea_periodic > 0 and state["steps_since_compute"] >= self.tea_periodic:
            force_reason = "periodic_refresh"

        assert self.current_signal is not None
        if force_reason is not None:
            state["accumulated"] = 0.0
            self._set_decision(False, force_reason)
        else:
            relative = _relative_l1(self.current_signal, previous)
            indicator = _polyval(self.tea_coefficients, relative)
            state["accumulated"] += indicator
            self.current_indicator = indicator
            reuse = state["accumulated"] < self.tea_threshold
            if not reuse:
                state["accumulated"] = 0.0
            self._set_decision(reuse, "accumulated_signal_below_threshold" if reuse else "signal_refresh")
        state["previous_signal"] = self.current_signal

    def _decide_easycache(self) -> None:
        state = self._expert_state(self.expert)
        force_reason = self._common_force_reason(state)
        previous_input = state["previous_input_probe"]
        if force_reason is None and (
            previous_input is None or state["k"] is None or state["last_output_norm"] is None
        ):
            force_reason = "online_error_seed"

        assert self.current_input_probe is not None
        if force_reason is not None:
            state["accumulated"] = 0.0
            self._set_decision(False, force_reason)
        else:
            raw_change = _distributed_mean((self.current_input_probe - previous_input).abs().mean())
            predicted = float(state["k"]) * raw_change / max(float(state["last_output_norm"]), 1e-8)
            state["accumulated"] += predicted
            self.current_predicted_error = predicted
            reuse = state["accumulated"] < self.easy_threshold
            if not reuse:
                state["accumulated"] = 0.0
            self._set_decision(reuse, "online_error_below_threshold" if reuse else "online_error_refresh")
        state["previous_input_probe"] = self.current_input_probe

    def _decide_taylorseer(self) -> None:
        state = self._expert_state(self.expert)
        force_reason = self._common_force_reason(state)
        if force_reason is None and state["force_next"]:
            force_reason = "forecast_error_guard"
            state["force_next"] = False
        last_compute_step = state["last_compute_step"]
        if force_reason is None and last_compute_step is None:
            force_reason = "history_seed"
        if force_reason is None and self.step - int(last_compute_step) >= self.taylor_interval:
            force_reason = "refresh_cadence"
        self._set_decision(force_reason is None, "taylor_forecast" if force_reason is None else force_reason)

    def _forecast_value(self, history: list[tuple[int, Any, Any]], value_index: int) -> Any:
        latest_step, latest = history[0][0], history[0][value_index]
        if len(history) < 2:
            return latest
        previous_step, previous = history[1][0], history[1][value_index]
        spacing = max(latest_step - previous_step, 1)
        u = float(self.step - latest_step) / float(spacing)
        forecast = latest + (latest - previous) * (u * self.taylor_damping)
        if self.taylor_order >= 2 and len(history) >= 3:
            older_step, older = history[2][0], history[2][value_index]
            previous_spacing = max(previous_step - older_step, 1)
            if previous_spacing == spacing:
                second_difference = latest - 2 * previous + older
                forecast = forecast + second_difference * (u * (u + 1.0) / 2.0) * self.taylor_damping
        return forecast

    def cached_residual(self, expert: str, branch: int) -> torch.Tensor | None:
        if self.current_decision != "reuse":
            return None
        key = (str(expert), int(branch))
        if self.method == "taylorseer":
            history = self.histories.get(key) or []
            if not history:
                self._set_decision(False, "missing_branch_history")
                return None
            self.branch_reuse += 1
            return self._forecast_value(history, 1)
        payload = self.payloads.get(key)
        if payload is None:
            self._set_decision(False, "missing_branch_payload")
            return None
        self.branch_reuse += 1
        return payload

    def record_residual(self, expert: str, branch: int, residual: torch.Tensor) -> None:
        key = (str(expert), int(branch))
        detached = residual.detach()
        probe = self._probe(detached)
        if branch == 0:
            self.current_branch0_output_probe = probe

        if self.method != "taylorseer":
            self.payloads[key] = detached
            self.branch_compute += 1
            return

        history = self.histories.setdefault(key, [])
        if branch == 0 and history:
            predicted_probe = self._forecast_value(history, 2)
            error = _relative_l1(probe, predicted_probe, distributed=True)
            self.current_forecast_error = error
            state = self._expert_state(str(expert))
            state["forecast_error"] = error
            if math.isfinite(self.taylor_error_threshold) and error > self.taylor_error_threshold:
                state["force_next"] = True
        history.insert(0, (self.step, detached, probe))
        del history[self.taylor_order + 1 :]
        self.branch_compute += 1

    def finish_call(self, branch: int) -> None:
        if branch != self.expected_branches - 1:
            return
        if self.current_decision is None:
            raise RuntimeError("Wan cache decision missing at end of CFG pair")
        state = self._expert_state(self.expert)
        reused = self.current_decision == "reuse"
        if reused:
            self.reuse_steps += 1
            state["consecutive_reuse"] += 1
            state["steps_since_compute"] += 1
            self.hit_pattern.append("R")
        else:
            self.compute_steps += 1
            state["consecutive_reuse"] = 0
            state["steps_since_compute"] = 0
            state["last_compute_step"] = self.step
            self.hit_pattern.append("C")

        if self.method == "easycache" and not reused:
            assert self.current_input_probe is not None
            assert self.current_branch0_output_probe is not None
            previous_input = state["last_compute_input_probe"]
            previous_output = state["last_compute_output_probe"]
            if previous_input is not None and previous_output is not None:
                input_change = _distributed_mean((self.current_input_probe - previous_input).abs().mean())
                output_change = _distributed_mean((self.current_branch0_output_probe - previous_output).abs().mean())
                state["k"] = output_change / max(input_change, 1e-8)
            state["last_compute_input_probe"] = self.current_input_probe
            state["last_compute_output_probe"] = self.current_branch0_output_probe
            state["last_output_norm"] = _distributed_mean(self.current_branch0_output_probe.abs().mean())

        self.trace.append(
            {
                "step": self.step,
                "expert": self.expert,
                "decision": self.current_decision,
                "reason": self.current_reason,
                "branches": self.branch_counter,
                "indicator": self.current_indicator,
                "accumulated": float(state["accumulated"]),
                "predicted_error": self.current_predicted_error,
                "online_k": state["k"],
                "forecast_error": self.current_forecast_error,
            }
        )
        self.branch_counter = 0

    def finalize(self) -> dict[str, Any]:
        if self.branch_counter != 0:
            raise RuntimeError("Wan cache finished with an incomplete CFG pair")
        total = self.compute_steps + self.reuse_steps
        payload = {
            "teacache": "per_expert_per_cfg_blocks_1_39_residual",
            "easycache": "per_expert_per_cfg_runtime_adaptive_blocks_1_39_transform_vector",
            "taylorseer": f"per_expert_per_cfg_order_{self.taylor_order}_forecasted_blocks_1_39_residual",
        }[self.method]
        signal = {
            "teacache": "wan_timestep_projection_polynomial_rescaled_relative_l1",
            "easycache": "online_fresh_block0_change_scaled_by_measured_tail_transform_change",
            "taylorseer": "finite_difference_tail_residual_history",
        }[self.method]
        return {
            "schema_version": 1,
            "method": self.method,
            "signal_source": signal,
            "reuse_payload": payload,
            "refresh_rule": {
                "teacache": "accumulated polynomial-rescaled timestep-projection delta threshold",
                "easycache": "accumulated online-predicted transform-vector relative error threshold",
                "taylorseer": "expert-local forecast interval plus optional measured forecast-error guard",
            }[self.method],
            "off_path": (
                "WAN22_CACHE_METHOD unset/off skips this module; the CP4 model executes all 40 original blocks"
            ),
            "placement": {
                "fresh_prefix_blocks": 1,
                "cached_tail_blocks": 39,
                "total_blocks": 40,
                "context_parallel_contract": "original block0 split hook and proj_out gather hook preserved",
            },
            "parameters": {
                "start_step": self.start_step,
                "tail_steps": self.tail_steps,
                "max_reuse": self.max_reuse,
                "teacache_threshold": self.tea_threshold,
                "teacache_coefficients": self.tea_coefficients,
                "teacache_periodic": self.tea_periodic,
                "easycache_threshold": self.easy_threshold,
                "taylor_order": self.taylor_order,
                "taylor_interval": self.taylor_interval,
                "taylor_error_threshold": (
                    self.taylor_error_threshold if math.isfinite(self.taylor_error_threshold) else None
                ),
                "taylor_damping": self.taylor_damping,
                "probe_tokens_per_rank": self.probe_tokens,
                "probe_channels": self.probe_channels,
            },
            "total_steps": total,
            "compute_steps": self.compute_steps,
            "reuse_steps": self.reuse_steps,
            "hit_rate": (self.reuse_steps / total) if total else 0.0,
            "branch_compute": self.branch_compute,
            "branch_reuse": self.branch_reuse,
            "hit_pattern": "".join(self.hit_pattern),
            "experts": {
                name: {
                    "last_compute_step": state["last_compute_step"],
                    "forecast_error": state["forecast_error"],
                }
                for name, state in self.experts.items()
            },
            "trace": self.trace,
        }


class CachedTailStack(nn.Module):
    """Execute CP-sharding block 0, then compute/replay the remaining residual."""

    def __init__(self, blocks: nn.ModuleList, expert: str):
        super().__init__()
        if len(blocks) < 2:
            raise ValueError("Wan cache adapter requires at least two transformer blocks")
        self.original_blocks = blocks
        self.expert = str(expert)
        self.controller: WanCacheController | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        controller = self.controller
        if controller is None:
            for block in self.original_blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, temb, rotary_emb)
            return hidden_states

        # Diffusers registered the CP input split hook on this exact module
        # before the proxy was installed. Its output is therefore rank-local.
        fresh_prefix = self.original_blocks[0](hidden_states, encoder_hidden_states, temb, rotary_emb)
        branch = controller.begin_call(self.expert, fresh_prefix, temb)
        residual = controller.cached_residual(self.expert, branch)
        if residual is None:
            tail_output = fresh_prefix
            for block in self.original_blocks[1:]:
                tail_output = block(tail_output, encoder_hidden_states, temb, rotary_emb)
            residual = tail_output - fresh_prefix
            controller.record_residual(self.expert, branch, residual)
        else:
            tail_output = fresh_prefix + residual.type_as(fresh_prefix)
        controller.finish_call(branch)
        return tail_output


class WanCacheRuntime:
    """Install expert-local proxies once and swap prompt-local controllers."""

    def __init__(self, pipe: Any, method: str, num_steps: int):
        self.method = method
        self.num_steps = int(num_steps)
        self.proxies: list[CachedTailStack] = []
        for expert, model in (("high_noise", pipe.transformer), ("low_noise", pipe.transformer_2)):
            if model is None:
                continue
            original_blocks = model.blocks
            proxy = CachedTailStack(original_blocks, expert)
            model.blocks = nn.ModuleList([proxy])
            self.proxies.append(proxy)

    def new_controller(self) -> WanCacheController:
        controller = WanCacheController(self.method, self.num_steps)
        for proxy in self.proxies:
            proxy.controller = controller
        return controller

    def clear_controller(self) -> None:
        for proxy in self.proxies:
            proxy.controller = None


def install_cache(pipe: Any, num_steps: int) -> WanCacheRuntime | None:
    method = configured_method()
    if not method:
        return None
    return WanCacheRuntime(pipe, method=method, num_steps=num_steps)
