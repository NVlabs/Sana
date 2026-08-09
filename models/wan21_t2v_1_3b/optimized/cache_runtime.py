"""Experiment-local cache seams for the Wan2.2 TI2V Diffusers runtime.

The module is deliberately inert unless ``WAN22_CACHE_FAMILY`` names one of
``teacache``, ``easycache``, or ``taylorseer``.  This keeps the disabled path
identical to the materialized baseline while allowing config manifests to
select and fully describe one cache family at a time.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return default if value in (None, "") else value


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name, "1" if default else "0")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _poly(coefficients: list[float], x: float) -> float:
    value = 0.0
    for coefficient in coefficients:
        value = value * x + coefficient
    return value


def _mean_abs_delta(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    return (current - previous).abs().mean()


def _sample(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output.sample


class _RuntimeBase:
    family = ""

    def __init__(self) -> None:
        self._generations: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None

    def _archive_current(self) -> None:
        if self._current is not None and self._current.get("calls", 0):
            self._generations.append(self._current)

    def _new_generation(self, tag: str) -> None:
        self._archive_current()
        self._current = {
            "tag": tag,
            "calls": 0,
            "compute": 0,
            "reuse": 0,
            "cond_compute_steps": [],
            "cond_reuse_steps": [],
            "uncond_compute_steps": [],
            "uncond_reuse_steps": [],
            "trace": [],
        }

    def _record(
        self,
        *,
        step: int,
        branch: str,
        compute: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        assert self._current is not None
        action = "compute" if compute else "reuse"
        self._current["calls"] += 1
        self._current[action] += 1
        self._current[f"{branch}_{action}_steps"].append(step)
        row = {"step": step, "branch": branch, "action": action}
        if details:
            row.update(details)
        self._current["trace"].append(row)

    def summary(self) -> dict[str, Any]:
        generations = list(self._generations)
        if self._current is not None and self._current.get("calls", 0):
            generations.append(self._current)
        return {
            "family": self.family,
            "variant": self.variant,
            "parameters": self.parameters,
            "signal_source": self.signal_source,
            "reuse_payload": self.reuse_payload,
            "refresh_rule": self.refresh_rule,
            "off_path": "WAN22_CACHE_FAMILY unset/off: no wrapper or cache hook is installed",
            "generations": generations,
            "totals": {
                "calls": sum(g["calls"] for g in generations),
                "compute": sum(g["compute"] for g in generations),
                "reuse": sum(g["reuse"] for g in generations),
            },
        }

    def describe(self) -> str:
        return f"{self.family}/{self.variant} {self.parameters}"


class TeaCacheRuntime(_RuntimeBase):
    """Wan block-residual TeaCache with a timestep-projection signal."""

    family = "teacache"
    variant = "wan_timestep_projection_block_residual"
    signal_source = "first token of Wan timestep_proj (all T2V token timesteps are equal)"
    reuse_payload = "branch-local residual across the complete transformer block stack"
    refresh_rule = "polynomial-rescaled relative-L1 signal distance accumulated to threshold"

    def __init__(self, transformer: torch.nn.Module, num_steps: int) -> None:
        super().__init__()
        self.transformer = transformer
        self.num_steps = num_steps
        self.threshold = _env_float("WAN22_TEACACHE_THRESHOLD", 0.12)
        self.start_step = _env_int("WAN22_TEACACHE_START_STEP", 2)
        self.end_step = _env_int("WAN22_TEACACHE_END_STEP", num_steps - 2)
        self.max_hits = _env_int("WAN22_TEACACHE_MAX_HITS", 0)
        self.periodic = _env_int("WAN22_TEACACHE_PERIODIC", 0)
        raw_coefficients = _env("WAN22_TEACACHE_COEFFICIENTS", "1.0,0.0")
        self.coefficients = [float(x.strip()) for x in raw_coefficients.split(",") if x.strip()]
        if self.threshold <= 0 or not self.coefficients:
            raise ValueError("TeaCache requires a positive threshold and at least one coefficient")
        self.parameters = {
            "threshold": self.threshold,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "coefficients": self.coefficients,
            "max_continuous_hits": self.max_hits,
            "periodic_recompute": self.periodic,
        }
        self._call_index = 0
        self._states: dict[str, dict[str, Any]] = {}
        self._install()

    def begin_generation(self, tag: str) -> None:
        self._new_generation(tag)
        self._call_index = 0
        self._states = {
            branch: {"previous_signal": None, "acc": 0.0, "hits": 0, "since": 0, "residual": None}
            for branch in ("cond", "uncond")
        }

    def _decide(self, signal: torch.Tensor, branch: str, step: int) -> tuple[bool, dict[str, Any]]:
        state = self._states[branch]
        force_reason = None
        if step < self.start_step:
            force_reason = "warmup"
        elif step >= self.end_step:
            force_reason = "cooldown"
        elif state["previous_signal"] is None or state["residual"] is None:
            force_reason = "initialize"
        elif self.periodic > 0 and state["since"] >= self.periodic:
            force_reason = "periodic"
        elif self.max_hits > 0 and state["hits"] >= self.max_hits:
            force_reason = "hit_cap"

        relative_l1 = None
        indicator = None
        compute = force_reason is not None
        if not compute:
            previous = state["previous_signal"]
            relative_l1 = float(
                (_mean_abs_delta(signal, previous) / previous.abs().mean().clamp_min(1e-8)).item()
            )
            indicator = _poly(self.coefficients, relative_l1)
            state["acc"] += indicator
            compute = state["acc"] >= self.threshold
            if compute:
                force_reason = "threshold"

        # Clone the tiny first-token signal so a view does not retain the very
        # large per-token modulation tensor after this forward.
        state["previous_signal"] = signal.detach().clone()
        details = {
            "relative_l1": relative_l1,
            "indicator": indicator,
            "accumulator": float(state["acc"]),
            "reason": force_reason or "below_threshold",
        }
        return compute, details

    def _install(self) -> None:
        transformer = self.transformer

        def wrapped(
            hidden_states: torch.Tensor,
            timestep: torch.LongTensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_image: torch.Tensor | None = None,
            return_dict: bool = True,
            attention_kwargs: dict[str, Any] | None = None,
        ):
            del attention_kwargs  # Baseline evaluation has no LoRA attention kwargs.
            branch = "cond" if self._call_index % 2 == 0 else "uncond"
            step = self._call_index // 2
            self._call_index += 1
            if self._current is None:
                self.begin_generation("implicit")

            batch_size, _channels, num_frames, height, width = hidden_states.shape
            p_t, p_h, p_w = transformer.config.patch_size
            post_patch_num_frames = num_frames // p_t
            post_patch_height = height // p_h
            post_patch_width = width // p_w

            rotary_emb = transformer.rope(hidden_states)
            hidden_states = transformer.patch_embedding(hidden_states)
            hidden_states = hidden_states.flatten(2).transpose(1, 2)

            if timestep.ndim == 2:
                timestep_seq_len = timestep.shape[1]
                timestep = timestep.flatten()
            else:
                timestep_seq_len = None

            temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = transformer.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                timestep_seq_len=timestep_seq_len,
            )
            if timestep_seq_len is not None:
                timestep_proj = timestep_proj.unflatten(2, (6, -1))
                signal = timestep_proj[:, 0]
            else:
                timestep_proj = timestep_proj.unflatten(1, (6, -1))
                signal = timestep_proj

            if encoder_hidden_states_image is not None:
                encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

            compute, details = self._decide(signal, branch, step)
            state = self._states[branch]
            if compute:
                block_input = hidden_states
                if torch.is_grad_enabled() and transformer.gradient_checkpointing:
                    for block in transformer.blocks:
                        hidden_states = transformer._gradient_checkpointing_func(
                            block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                        )
                else:
                    for block in transformer.blocks:
                        hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                state["residual"] = (hidden_states - block_input).detach()
                state["acc"] = 0.0
                state["hits"] = 0
                state["since"] = 0
            else:
                hidden_states = hidden_states + state["residual"]
                state["hits"] += 1
                state["since"] += 1
            self._record(step=step, branch=branch, compute=compute, details=details)

            if temb.ndim == 3:
                shift, scale = (
                    transformer.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)
                ).chunk(2, dim=2)
                shift, scale = shift.squeeze(2), scale.squeeze(2)
            else:
                shift, scale = (transformer.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
            shift, scale = shift.to(hidden_states.device), scale.to(hidden_states.device)
            hidden_states = (
                transformer.norm_out(hidden_states.float()) * (1 + scale) + shift
            ).type_as(hidden_states)
            hidden_states = transformer.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(
                batch_size,
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
                p_t,
                p_h,
                p_w,
                -1,
            )
            hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
            output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
            if not return_dict:
                return (output,)
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            return Transformer2DModelOutput(sample=output)

        transformer.forward = wrapped


class EasyCacheRuntime(_RuntimeBase):
    """Official EasyCache transform-vector controller adapted to Diffusers Wan."""

    family = "easycache"
    variant = "wan_runtime_adaptive_transform_vector"
    signal_source = "step-to-step mean absolute change of the raw latent input"
    reuse_payload = "branch-local transformer output minus raw latent input"
    refresh_rule = "online output/input change factor times relative input drift, accumulated to threshold"

    def __init__(self, transformer: torch.nn.Module, num_steps: int) -> None:
        super().__init__()
        self.transformer = transformer
        self.num_steps = num_steps
        self.threshold = _env_float("WAN22_EASYCACHE_THRESHOLD", 0.05)
        self.retain_steps = _env_int("WAN22_EASYCACHE_RETAIN_STEPS", 7)
        self.cooldown_steps = _env_int("WAN22_EASYCACHE_COOLDOWN_STEPS", 1)
        if self.threshold <= 0:
            raise ValueError("EasyCache requires a positive threshold")
        self.parameters = {
            "threshold": self.threshold,
            "retain_steps": self.retain_steps,
            "cooldown_steps": self.cooldown_steps,
        }
        self._call_index = 0
        self._pair_compute = True
        self._cond: dict[str, Any] = {}
        self._uncond: dict[str, Any] = {}
        self._install()

    def begin_generation(self, tag: str) -> None:
        self._new_generation(tag)
        self._call_index = 0
        self._pair_compute = True
        self._cond = {
            "previous_step_input": None,
            "last_full_input": None,
            "last_full_output": None,
            "residual": None,
            "k": None,
            "acc": 0.0,
        }
        self._uncond = {"residual": None}

    def _cond_decision(self, current_input: torch.Tensor, step: int) -> tuple[bool, dict[str, Any]]:
        state = self._cond
        reason = None
        estimate = None
        force = step < self.retain_steps or step >= self.num_steps - self.cooldown_steps
        if force:
            reason = "warmup" if step < self.retain_steps else "cooldown"
            state["acc"] = 0.0
            compute = True
        elif state["previous_step_input"] is None or state["residual"] is None or state["k"] is None:
            reason = "initialize"
            compute = True
        else:
            input_change = _mean_abs_delta(current_input, state["previous_step_input"])
            output_norm = state["last_full_output"].abs().mean().clamp_min(1e-8)
            estimate = float((state["k"] * input_change / output_norm).item())
            state["acc"] += estimate
            compute = state["acc"] >= self.threshold
            if compute:
                reason = "threshold"
                state["acc"] = 0.0
            else:
                reason = "below_threshold"
        state["previous_step_input"] = current_input.detach()
        return compute, {
            "predicted_relative_change": estimate,
            "accumulator": float(state["acc"]),
            "k": state["k"],
            "reason": reason,
        }

    def _install(self) -> None:
        transformer = self.transformer
        original_forward = transformer.forward

        def wrapped(
            hidden_states: torch.Tensor,
            timestep: torch.LongTensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_image: torch.Tensor | None = None,
            return_dict: bool = True,
            attention_kwargs: dict[str, Any] | None = None,
        ):
            branch = "cond" if self._call_index % 2 == 0 else "uncond"
            step = self._call_index // 2
            self._call_index += 1
            if self._current is None:
                self.begin_generation("implicit")

            if branch == "cond":
                compute, details = self._cond_decision(hidden_states, step)
                self._pair_compute = compute
            else:
                compute = self._pair_compute or self._uncond["residual"] is None
                details = {
                    "predicted_relative_change": None,
                    "accumulator": float(self._cond["acc"]),
                    "k": self._cond["k"],
                    "reason": "follow_cond_pair" if self._uncond["residual"] is not None else "initialize",
                }

            branch_state = self._cond if branch == "cond" else self._uncond
            if not compute:
                sample = hidden_states + branch_state["residual"]
                self._record(step=step, branch=branch, compute=False, details=details)
                if return_dict:
                    from diffusers.models.modeling_outputs import Transformer2DModelOutput

                    return Transformer2DModelOutput(sample=sample)
                return (sample,)

            output = original_forward(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
            )
            sample = _sample(output)
            if branch == "cond":
                previous_input = self._cond["last_full_input"]
                previous_output = self._cond["last_full_output"]
                if previous_input is not None and previous_output is not None:
                    input_change = _mean_abs_delta(hidden_states, previous_input).clamp_min(1e-8)
                    output_change = _mean_abs_delta(sample, previous_output)
                    self._cond["k"] = float((output_change / input_change).item())
                    details["k_after_refresh"] = self._cond["k"]
                self._cond["last_full_input"] = hidden_states.detach()
                self._cond["last_full_output"] = sample.detach()
            branch_state["residual"] = (sample - hidden_states).detach()
            self._record(step=step, branch=branch, compute=True, details=details)
            return output

        transformer.forward = wrapped


class TaylorSeerRuntime(_RuntimeBase):
    """Diffusers-native TaylorSeer with branch-local hook state."""

    family = "taylorseer"
    variant = "diffusers_native_lite_projection_forecast"
    signal_source = "finite-difference history of branch-local projected transformer outputs"
    reuse_payload = "Taylor factors for proj_out; transformer blocks are skipped on forecast steps"
    refresh_rule = "full refresh at a native fixed Taylor interval with warmup and cooldown"

    def __init__(self, transformer: torch.nn.Module, num_steps: int) -> None:
        super().__init__()
        self.transformer = transformer
        self.num_steps = num_steps
        self.interval = _env_int("WAN22_TAYLOR_INTERVAL", 3)
        self.warmup = _env_int("WAN22_TAYLOR_WARMUP", 3)
        self.cooldown_start = _env_int("WAN22_TAYLOR_COOLDOWN_START", num_steps - 2)
        self.order = _env_int("WAN22_TAYLOR_ORDER", 1)
        self.lite = _env_bool("WAN22_TAYLOR_LITE", True)
        dtype_name = str(_env("WAN22_TAYLOR_DTYPE", "bfloat16")).lower()
        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}.get(dtype_name)
        if dtype is None or self.interval <= 0 or self.order < 0:
            raise ValueError("Invalid TaylorSeer interval, order, or factor dtype")
        if not self.lite:
            raise ValueError("This experiment uses TaylorSeer lite mode to keep per-layer factor memory bounded")
        self.parameters = {
            "cache_interval": self.interval,
            "disable_cache_before_step": self.warmup,
            "disable_cache_after_step": self.cooldown_start,
            "max_order": self.order,
            "taylor_factors_dtype": dtype_name,
            "use_lite_mode": self.lite,
        }
        from diffusers.hooks import TaylorSeerCacheConfig

        config = TaylorSeerCacheConfig(
            cache_interval=self.interval,
            disable_cache_before_step=self.warmup,
            disable_cache_after_step=self.cooldown_start,
            max_order=self.order,
            taylor_factors_dtype=dtype,
            use_lite_mode=True,
        )
        transformer.enable_cache(config)
        self._call_index = 0
        self._install_counter()

    def begin_generation(self, tag: str) -> None:
        self._new_generation(tag)
        self._call_index = 0
        self.transformer._reset_stateful_cache(recurse=True)

    def _native_compute(self, step: int) -> tuple[bool, str]:
        if step < self.warmup:
            return True, "warmup"
        if step >= self.cooldown_start:
            return True, "cooldown"
        # Mirrors TaylorSeerCacheHook._measure_should_compute exactly.
        compute = (step - self.warmup - 1) % self.interval == 0
        return compute, "interval_refresh" if compute else "taylor_forecast"

    def _install_counter(self) -> None:
        original_forward = self.transformer.forward

        def wrapped(*args, **kwargs):
            branch = "cond" if self._call_index % 2 == 0 else "uncond"
            step = self._call_index // 2
            self._call_index += 1
            if self._current is None:
                self.begin_generation("implicit")
            compute, reason = self._native_compute(step)
            output = original_forward(*args, **kwargs)
            self._record(
                step=step,
                branch=branch,
                compute=compute,
                details={"reason": reason, "order": self.order, "interval": self.interval},
            )
            return output

        self.transformer.forward = wrapped


def maybe_enable_cache(pipe: Any, num_steps: int) -> _RuntimeBase | None:
    """Enable exactly one requested cache family, or leave the pipeline untouched."""

    family = str(_env("WAN22_CACHE_FAMILY", "off")).strip().lower()
    if family in {"", "0", "none", "off", "false"}:
        return None
    if family == "teacache":
        return TeaCacheRuntime(pipe.transformer, num_steps)
    if family == "easycache":
        return EasyCacheRuntime(pipe.transformer, num_steps)
    if family == "taylorseer":
        return TaylorSeerRuntime(pipe.transformer, num_steps)
    raise ValueError(f"Unsupported WAN22_CACHE_FAMILY={family!r}")
