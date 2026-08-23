"""Few-step Stage-1 cache controls for the pinned SGLang MiniMax-H3 loop.

EasyCache and TeaCache are deliberately installed around ``_forward_dit``.
That keeps their Python control logic outside the compiled DiT while every
computed step still enters the same compiled transformer.  FirstBlockCache is
provided by SGLang's native Cache-DiT integration; this module records its
request-local telemetry and installs the H3 out-of-place gate safety shim that
Cache-DiT requires.

The H3 scheduler mutates both its packed input buffer and returned velocity
storage in place.  Consequently every value retained across steps is cloned.
Reusing a detached view here silently turns deltas into zero or corrupts the
cached prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch


PINNED_SGLANG_COMMIT = "12eadf86f12aec2e6f81a6e38b61b964a4c6b529"
CACHE_MODES = ("none", "easy", "tea", "fb")
ALLOWED_NFE = (4, 6, 8, 12, 16, 49)

# These are the measured 49-NFE H3 starting points.  Few-step schedules make
# much larger moves, so the outer-step thresholds scale inversely with NFE.
_HISTORICAL_NFE = 49
_BASE_THRESHOLDS = {"easy": 0.50, "tea": 0.10, "fb": 0.09}

# A second true forward is required to initialize the Easy/Tea change model.
# The cap keeps the resulting integer schedules near 2.5x.  At four NFE the
# closest faithful schedule is two true forwards (2.0x), not a fabricated 2.5x.
_MAX_CONTINUOUS_HITS = {4: 2, 6: 4, 8: 3, 12: 2, 16: 2, 49: 2}

_H3_GATE_PATCH_STATE: dict[str, Any] | None = None


def _native_fb_rdt() -> float:
    """Return an explicitly configured, real Cache-DiT threshold.

    Cache-DiT treats ``RDT >= 1`` as a sentinel which bypasses residual
    measurement and unconditionally reuses the cache.  That is not a valid
    content-adaptive FirstBlockCache configuration for these experiments, so
    reject it before the model is loaded or compiled.
    """

    raw = os.environ.get("SGLANG_CACHE_DIT_RDT")
    if raw is None or not raw.strip():
        raise RuntimeError(
            "SGLANG_CACHE_DIT_RDT must be set explicitly for cache_mode=fb"
        )
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"SGLANG_CACHE_DIT_RDT must be a finite float in [0,1), got {raw!r}"
        ) from exc
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise RuntimeError(
            f"SGLANG_CACHE_DIT_RDT must be finite and in [0,1), got {raw!r}"
        )
    return value


def _native_fb_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value


def _native_fb_config() -> dict[str, Any]:
    taylorseer = _native_fb_int("SGLANG_CACHE_DIT_TAYLORSEER", 0, minimum=0)
    if taylorseer not in {0, 1}:
        raise RuntimeError("SGLANG_CACHE_DIT_TAYLORSEER must be exactly 0 or 1")
    return {
        "Fn": _native_fb_int("SGLANG_CACHE_DIT_FN", 1, minimum=1),
        "Bn": _native_fb_int("SGLANG_CACHE_DIT_BN", 0, minimum=0),
        "warmup": _native_fb_int("SGLANG_CACHE_DIT_WARMUP", 1, minimum=0),
        "RDT": _native_fb_rdt(),
        "MC": _native_fb_int("SGLANG_CACHE_DIT_MC", 2, minimum=1),
        "TaylorSeer": bool(taylorseer),
        "SCM": os.environ.get("SGLANG_CACHE_DIT_SCM_PRESET", "none"),
    }


def _install_h3_out_of_place_gate_patch() -> dict[str, Any]:
    """Keep Cache-DiT's saved H3 hidden state from being mutated in place.

    The pinned H3 CUDA/BF16 fast path calls ``indexed_gate_bf16_`` and mutates
    ``x``.  Cache-DiT Pattern_3 retains ``original_hidden_states`` as an alias
    of that tensor, so the mutation corrupts the residual later stored for
    reuse.  Replacing the small dispatcher (not the container source) with the
    exact upstream out-of-place fallback preserves the original tensor.
    """

    global _H3_GATE_PATCH_STATE
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as h3_module

    current = getattr(h3_module, "_modulate_gate", None)
    if not callable(current):
        raise RuntimeError("pinned MiniMax-H3 _modulate_gate is unavailable")
    if _H3_GATE_PATCH_STATE is not None:
        if current is not _H3_GATE_PATCH_STATE["replacement"]:
            raise RuntimeError("MiniMax-H3 _modulate_gate changed after cache patch")
        return dict(_H3_GATE_PATCH_STATE["audit"])
    if getattr(current, "_h3_cache_dit_out_of_place", False):
        raise RuntimeError("unexpected pre-installed MiniMax-H3 gate patch")

    original = current

    def _modulate_gate_out_of_place(
        x: torch.Tensor,
        gate: torch.Tensor,
        other: torch.Tensor,
        indices: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # This is the pinned upstream non-fused branch.  Addition allocates new
        # storage even when the requested dtype already equals x.dtype.
        return (x + gate.index_select(0, indices) * other).to(dtype)

    _modulate_gate_out_of_place.__name__ = "_modulate_gate"
    _modulate_gate_out_of_place.__qualname__ = "_modulate_gate"
    _modulate_gate_out_of_place.__module__ = h3_module.__name__
    _modulate_gate_out_of_place._h3_cache_dit_out_of_place = True
    _modulate_gate_out_of_place._h3_cache_dit_original = original
    h3_module._modulate_gate = _modulate_gate_out_of_place

    audit = {
        "installed": True,
        "module": h3_module.__name__,
        "symbol": "_modulate_gate",
        "original_callable": f"{original.__module__}.{original.__qualname__}",
        "replacement": "out_of_place_index_select_mul_add_to_dtype",
        "indexed_gate_bf16_inplace_disabled": True,
        "cache_dit_alias_preserved": True,
    }
    _H3_GATE_PATCH_STATE = {
        "module": h3_module,
        "replacement": _modulate_gate_out_of_place,
        "audit": audit,
    }
    return dict(audit)


def _assert_h3_out_of_place_gate_patch() -> None:
    if _H3_GATE_PATCH_STATE is None:
        raise RuntimeError("MiniMax-H3 out-of-place cache gate patch is not installed")
    module = _H3_GATE_PATCH_STATE["module"]
    replacement = _H3_GATE_PATCH_STATE["replacement"]
    if getattr(module, "_modulate_gate", None) is not replacement:
        raise RuntimeError("MiniMax-H3 out-of-place cache gate patch was replaced")


def cache_parameters(cache_mode: str, nfe: int) -> dict[str, Any]:
    if cache_mode not in CACHE_MODES:
        raise ValueError(f"unknown cache mode {cache_mode!r}")
    if nfe not in ALLOWED_NFE:
        raise ValueError(f"unsupported NFE {nfe}; expected one of {ALLOWED_NFE}")
    if cache_mode == "none":
        return {
            "threshold": None,
            "warmup_steps": 0,
            "cooldown_steps": 0,
            "max_continuous_cached_steps": 0,
        }
    if cache_mode == "fb":
        # Native Cache-DiT is mounted once per persistent service and only its
        # request context (not DBCacheConfig) is refreshed when NFE changes.
        # Its process-wide threshold is supplied by the diagnostic/production
        # launcher and validated here; it must never be the RDT>=1 sentinel.
        native = _native_fb_config()
        return {
            "threshold": native["RDT"],
            "threshold_reference_nfe": None,
            "threshold_reference_value": None,
            "warmup_steps": native["warmup"],
            "cooldown_steps": 0,
            "max_continuous_cached_steps": native["MC"],
            "front_blocks": native["Fn"],
            "back_blocks": native["Bn"],
            "taylorseer": native["TaylorSeer"],
            "scm_preset": native["SCM"],
            "target_denoise_speedup": None,
            "four_nfe_integer_limit": 2.0 if nfe == 4 else None,
            "process_fixed_native_cache_dit_config": True,
        }
    return {
        "threshold": _BASE_THRESHOLDS[cache_mode] * _HISTORICAL_NFE / nfe,
        "threshold_reference_nfe": _HISTORICAL_NFE,
        "threshold_reference_value": _BASE_THRESHOLDS[cache_mode],
        "warmup_steps": 1,
        "cooldown_steps": 0,
        "max_continuous_cached_steps": _MAX_CONTINUOUS_HITS[nfe],
        "target_denoise_speedup": 2.5,
        "four_nfe_integer_limit": 2.0 if nfe == 4 else None,
    }


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(payload), sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def _relative_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    numerator = (current - previous).abs().mean()
    denominator = previous.abs().mean().clamp_min(1e-8)
    return float((numerator / denominator).item())


def _target_video_input(call_kwargs: Mapping[str, Any]) -> torch.Tensor:
    packed = call_kwargs.get("x")
    output_info = call_kwargs.get("img_pos_for_infer_output_info")
    if not torch.is_tensor(packed) or packed.ndim != 3 or int(packed.shape[0]) != 1:
        raise RuntimeError("MiniMax-H3 cache expected packed x with shape [1,S,96]")
    if not isinstance(output_info, Mapping):
        raise RuntimeError("MiniMax-H3 cache has no target-video position metadata")
    positions = output_info.get("position_ids")
    if not torch.is_tensor(positions) or positions.ndim != 1:
        raise RuntimeError("MiniMax-H3 target-video positions are unavailable")
    return packed[0].index_select(0, positions).detach().clone()


def _target_audio_input(call_kwargs: Mapping[str, Any]) -> torch.Tensor:
    """Clone the current audio rows in the same order returned by H3.

    The packed audio buffer is mutated in place after every Euler update, just
    like the video buffer.  Cached audio therefore has to be represented as a
    residual relative to these current rows; replaying an old absolute audio
    prediction would jump back to the previous step's trajectory.
    """

    packed = call_kwargs.get("audio_x")
    output_info = call_kwargs.get("audio_pos_info")
    if not torch.is_tensor(packed) or packed.ndim != 3 or int(packed.shape[0]) != 1:
        raise RuntimeError("MiniMax-H3 cache expected packed audio_x with shape [1,S,32]")
    if not isinstance(output_info, Mapping):
        raise RuntimeError("MiniMax-H3 cache has no audio position metadata")
    positions = output_info.get("position_ids")
    if not torch.is_tensor(positions) or positions.ndim != 1:
        raise RuntimeError("MiniMax-H3 audio positions are unavailable")
    return packed[0].index_select(0, positions).detach().clone()


def _tea_signal(call_kwargs: Mapping[str, Any], current_input: torch.Tensor) -> torch.Tensor:
    """Cheap timestep-modulated proxy for the first-block TeaCache signal.

    SGLang's stage hook sees patchified target rows before the transformer has
    projected them to block width.  Multiplying by the exact target timestep
    preserves both latent and timestep movement without entering block zero.
    The identity calibration is explicit in telemetry; H3 has no fitted Tea
    polynomial.
    """

    unique = call_kwargs.get("unique_timesteps")
    inverse = call_kwargs.get("inverse_indices")
    output_info = call_kwargs.get("img_pos_for_infer_output_info")
    if not torch.is_tensor(unique) or not torch.is_tensor(inverse):
        return current_input
    positions = output_info.get("position_ids") if isinstance(output_info, Mapping) else None
    if not torch.is_tensor(positions) or int(positions.numel()) == 0:
        return current_input
    slot = inverse.index_select(0, positions[:1]).to(torch.long)
    timestep = unique.index_select(0, slot).reshape(1, 1).to(current_input)
    return current_input * (1.0 + timestep)


@dataclass
class _OuterStepCacheState:
    method: str
    nfe: int
    threshold: float
    max_continuous_hits: int
    accumulator: float = 0.0
    consecutive_hits: int = 0
    previous_input: torch.Tensor | None = None
    last_computed_input: torch.Tensor | None = None
    previous_output: torch.Tensor | None = None
    previous_signal: torch.Tensor | None = None
    sensitivity: float | None = None
    residual_video: torch.Tensor | None = None
    residual_audio: torch.Tensor | None = None
    computed_steps: int = 0
    cached_steps: int = 0
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def decide(
        self,
        *,
        step_index: int,
        current_input: torch.Tensor,
        signal: torch.Tensor,
    ) -> tuple[bool, str, float | None]:
        if self.residual_video is None or self.residual_audio is None:
            return True, "initialize", None
        if self.method == "easy":
            if self.previous_input is None or self.sensitivity is None:
                self.previous_input = current_input.detach().clone()
                return True, "initialize_sensitivity", None
            input_change = (current_input - self.previous_input).abs().mean()
            output_scale = self.previous_output.abs().mean().clamp_min(1e-8)
            indicator = float((self.sensitivity * input_change / output_scale).item())
            self.previous_input = current_input.detach().clone()
        elif self.method == "tea":
            if self.previous_signal is None:
                self.previous_signal = signal.detach().clone()
                return True, "initialize_signal", None
            indicator = _relative_l1(signal, self.previous_signal)
            self.previous_signal = signal.detach().clone()
        else:
            raise RuntimeError(f"outer cache cannot execute method {self.method!r}")
        self.accumulator += indicator
        if self.accumulator >= self.threshold:
            self.accumulator = 0.0
            self.consecutive_hits = 0
            return True, "threshold", indicator
        if self.consecutive_hits >= self.max_continuous_hits:
            self.accumulator = 0.0
            self.consecutive_hits = 0
            return True, "max_continuous", indicator
        return False, "reuse", indicator

    def after_compute(
        self,
        *,
        current_input: torch.Tensor,
        current_audio_input: torch.Tensor,
        video_output: torch.Tensor,
        audio_output: torch.Tensor,
    ) -> None:
        # Compute EasyCache sensitivity before moving either computed history.
        if self.method == "easy" and self.last_computed_input is not None:
            input_change = (
                current_input - self.last_computed_input
            ).abs().mean().clamp_min(1e-8)
            output_change = (video_output - self.previous_output).abs().mean()
            self.sensitivity = float((output_change / input_change).item())
        self.last_computed_input = current_input.detach().clone()
        self.previous_input = current_input.detach().clone()
        self.previous_output = video_output.detach().clone()
        self.residual_video = (video_output - current_input).detach().clone()
        if tuple(audio_output.shape) != tuple(current_audio_input.shape):
            raise RuntimeError(
                "MiniMax-H3 audio output/input shapes disagree: "
                f"{tuple(audio_output.shape)} != {tuple(current_audio_input.shape)}"
            )
        self.residual_audio = (
            audio_output - current_audio_input
        ).detach().clone()
        self.computed_steps += 1
        self.consecutive_hits = 0

    def reuse(
        self, *, current_input: torch.Tensor, current_audio_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.cached_steps += 1
        self.consecutive_hits += 1
        # Both results are disposable: the Euler update writes into their
        # storage later in the same step.
        return (
            (current_input + self.residual_video).clone(),
            (current_audio_input + self.residual_audio).clone(),
        )


def _native_fb_telemetry(transformer: Any, nfe: int) -> dict[str, Any]:
    candidates = [transformer]
    for name in ("model", "_orig_mod", "_sglang_cache_dit_adapter"):
        candidate = getattr(transformer, name, None)
        if candidate is not None:
            candidates.append(candidate)
    manager = None
    for candidate in candidates:
        manager = getattr(candidate, "_context_manager", None)
        if manager is not None:
            break
    if manager is None:
        return {
            "telemetry_available": False,
            "scheduled_steps": nfe,
            "head_block_forwards": nfe,
        }
    payload: dict[str, Any] = {
        "telemetry_available": True,
        "scheduled_steps": nfe,
        "head_block_forwards": nfe,
    }
    for key, method_name in (
        ("cached_steps_raw", "get_cached_steps"),
        ("residual_diffs", "get_residual_diffs"),
        ("accumulated_cached_steps", "get_accumulated_cached_steps"),
    ):
        method = getattr(manager, method_name, None)
        if callable(method):
            try:
                value = method()
                if torch.is_tensor(value):
                    value = value.detach().cpu().tolist()
                elif isinstance(value, tuple):
                    value = list(value)
                payload[key] = value
            except Exception as exc:  # telemetry must not invalidate a video
                payload[f"{key}_error"] = type(exc).__name__
    cached_raw = payload.get("cached_steps_raw")
    if isinstance(cached_raw, bool):
        cached_count = int(cached_raw)
    elif isinstance(cached_raw, int):
        cached_count = int(cached_raw)
    elif isinstance(cached_raw, list):
        cached_count = len(cached_raw)
    else:
        accumulated = payload.get("accumulated_cached_steps")
        cached_count = int(accumulated) if isinstance(accumulated, int) else None
    payload["cached_steps"] = cached_count
    payload["full_stack_forwards"] = (
        None if cached_count is None else nfe - cached_count
    )
    return payload


_INSTALL_STATE: dict[str, Any] | None = None


def install_stage1_cache_overlay(
    *,
    cache_mode: str,
    telemetry_path: str,
    width: int,
    height: int,
    allowed_nfe: tuple[int, ...] = ALLOWED_NFE,
) -> dict[str, Any]:
    """Install a process-fixed cache arm around the current H3 denoise stage."""

    global _INSTALL_STATE
    if cache_mode not in CACHE_MODES:
        raise ValueError(f"cache_mode must be one of {CACHE_MODES}, got {cache_mode!r}")
    if not telemetry_path:
        raise ValueError("cache telemetry path is required")
    if tuple(allowed_nfe) != ALLOWED_NFE:
        raise ValueError(f"grid NFE set is pinned to {ALLOWED_NFE}")
    native_config = _native_fb_config() if cache_mode == "fb" else None
    requested = {
        "cache_mode": cache_mode,
        "telemetry_path": str(telemetry_path),
        "width": int(width),
        "height": int(height),
        "allowed_nfe": list(allowed_nfe),
        "native_cache_dit_config": native_config,
    }
    if _INSTALL_STATE is not None:
        if _INSTALL_STATE["config"] != requested:
            raise RuntimeError(f"a different Stage-1 cache overlay is active: {_INSTALL_STATE}")
        if cache_mode == "fb":
            _assert_h3_out_of_place_gate_patch()
        return dict(_INSTALL_STATE)

    native_requested = os.environ.get("SGLANG_CACHE_DIT_ENABLED", "0") == "1"
    if native_requested != (cache_mode == "fb"):
        raise RuntimeError(
            "SGLANG_CACHE_DIT_ENABLED must be 1 exactly for cache_mode=fb and 0 otherwise"
        )
    gate_patch = (
        _install_h3_out_of_place_gate_patch() if cache_mode == "fb" else None
    )

    from sglang.multimodal_gen.runtime.pipelines import minimax_h3_pipeline
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
        denoising as denoising_module,
    )

    current_stage = minimax_h3_pipeline.MiniMaxH3DenoisingStage
    if getattr(current_stage, "_h3_stage1_cache_overlay", False):
        raise RuntimeError("unexpected pre-installed Stage-1 cache overlay")

    class MiniMaxH3Stage1CacheDenoisingStage(current_stage):
        def __init__(self, transformer: Any, pipeline: Any = None) -> None:
            super().__init__(transformer=transformer, pipeline=pipeline)
            self._stage1_outer_cache: _OuterStepCacheState | None = None
            self._stage1_active_nfe: int | None = None
            self._stage1_request_index = 0

        def _run_full_loop(self, batch: Any, server_args: Any) -> None:
            if cache_mode == "fb":
                _assert_h3_out_of_place_gate_patch()
                if _native_fb_config() != native_config:
                    raise RuntimeError(
                        "native Cache-DiT environment changed after overlay installation"
                    )
            ctx = denoising_module._resolve_full_loop_context(batch)
            if ctx.plan is None or str(ctx.plan.task) != "fl2va":
                raise NotImplementedError("Stage-1 grid accepts only task='fl2va'")
            nfe = len(list(ctx.sigmas["video"])) - 1
            if nfe not in allowed_nfe:
                raise ValueError(f"Stage-1 grid NFE {nfe} is not in {allowed_nfe}")
            self._stage1_active_nfe = nfe
            self._stage1_outer_cache = None
            torch.cuda.synchronize()
            started_ns = time.perf_counter_ns()
            status = "ok"
            error_type = None
            try:
                result = super()._run_full_loop(batch, server_args)
            except BaseException as exc:
                status = "error"
                error_type = type(exc).__name__
                raise
            finally:
                torch.cuda.synchronize()
                denoise_s = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
                state = self._stage1_outer_cache
                if cache_mode in {"easy", "tea"} and state is not None:
                    cache_stats: dict[str, Any] = {
                        "scheduled_steps": nfe,
                        "computed_forwards": state.computed_steps,
                        "cached_steps": state.cached_steps,
                        "decisions": state.decisions,
                        "final_sensitivity": state.sensitivity,
                    }
                elif cache_mode == "fb":
                    cache_stats = _native_fb_telemetry(self.transformer, nfe)
                else:
                    cache_stats = {
                        "scheduled_steps": nfe,
                        "computed_forwards": nfe,
                        "cached_steps": 0,
                    }
                params = cache_parameters(cache_mode, nfe)
                sampling = batch.sampling_params
                stage1_grid = None
                if isinstance(getattr(batch, "extra", None), Mapping):
                    candidate = batch.extra.get("minimax_h3_stage1_grid")
                    if isinstance(candidate, Mapping):
                        stage1_grid = dict(candidate)
                record = {
                    "schema_version": 1,
                    "request_index": self._stage1_request_index,
                    "status": status,
                    "error_type": error_type,
                    "cache_mode": cache_mode,
                    "nfe": nfe,
                    "width": width,
                    "height": height,
                    "denoise_total_s": denoise_s,
                    "seed": getattr(sampling, "seed", None),
                    "output_file_name": getattr(sampling, "output_file_name", None),
                    "is_warmup": bool(getattr(batch, "is_warmup", False)),
                    "parameters": params,
                    "cache_safety": (
                        {
                            "h3_out_of_place_gate_patch": gate_patch,
                            "rdt_sentinel_rejected": True,
                        }
                        if cache_mode == "fb"
                        else None
                    ),
                    # This is populated by the inner Stage-1 grid overlay after
                    # it has inspected the live pipeline.  Persist it beside
                    # cache telemetry so every retained request proves whether
                    # a LoRA was actually merged, or that the Teacher pipeline
                    # had no adapter state at all.
                    "model_binding": stage1_grid,
                    **cache_stats,
                }
                _append_jsonl(Path(telemetry_path), record)
                if isinstance(getattr(batch, "extra", None), Mapping):
                    batch.extra["minimax_h3_stage1_cache"] = record
                self._stage1_request_index += 1
                self._stage1_active_nfe = None
            return result

        def _forward_dit(
            self,
            model: Any,
            call_kwargs: dict[str, Any],
            step_index: int,
            *,
            batch: Any,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if cache_mode not in {"easy", "tea"}:
                return super()._forward_dit(
                    model, call_kwargs, step_index, batch=batch
                )
            nfe = self._stage1_active_nfe
            if nfe is None:
                raise RuntimeError("Stage-1 cache forward is outside an active request")
            params = cache_parameters(cache_mode, nfe)
            if step_index == 0:
                self._stage1_outer_cache = _OuterStepCacheState(
                    method=cache_mode,
                    nfe=nfe,
                    threshold=float(params["threshold"]),
                    max_continuous_hits=int(params["max_continuous_cached_steps"]),
                )
            state = self._stage1_outer_cache
            if state is None or state.nfe != nfe:
                raise RuntimeError("Stage-1 cache state was not initialized")
            current_input = _target_video_input(call_kwargs)
            current_audio_input = _target_audio_input(call_kwargs)
            signal = (
                _tea_signal(call_kwargs, current_input)
                if cache_mode == "tea"
                else current_input
            )
            compute, reason, indicator = state.decide(
                step_index=step_index,
                current_input=current_input,
                signal=signal,
            )
            decision = {
                "step": step_index,
                "action": "compute" if compute else "reuse",
                "reason": reason,
                "indicator": indicator,
                "accumulator": state.accumulator,
            }
            state.decisions.append(decision)
            if not compute:
                return state.reuse(
                    current_input=current_input,
                    current_audio_input=current_audio_input,
                )
            video, audio = super()._forward_dit(
                model, call_kwargs, step_index, batch=batch
            )
            # Cache before the scheduler mutates returned velocity storage.
            state.after_compute(
                current_input=current_input,
                current_audio_input=current_audio_input,
                video_output=video.detach().clone(),
                audio_output=audio.detach().clone(),
            )
            return video, audio

    MiniMaxH3Stage1CacheDenoisingStage.__name__ = (
        "MiniMaxH3Stage1CacheDenoisingStage"
    )
    MiniMaxH3Stage1CacheDenoisingStage.__qualname__ = (
        "MiniMaxH3Stage1CacheDenoisingStage"
    )
    MiniMaxH3Stage1CacheDenoisingStage._h3_stage1_cache_overlay = True
    MiniMaxH3Stage1CacheDenoisingStage._h3_stage1_cache_stock_stage = current_stage
    minimax_h3_pipeline.MiniMaxH3DenoisingStage = MiniMaxH3Stage1CacheDenoisingStage

    _INSTALL_STATE = {
        "installed": True,
        "name": f"sglang_minimax_h3_stage1_{cache_mode}_cache_v1",
        "pinned_sglang_commit": PINNED_SGLANG_COMMIT,
        "config": requested,
        "computed_steps_remain_torch_compiled": True,
        "tea_polynomial": [1.0, 0.0] if cache_mode == "tea" else None,
        "tea_signal": (
            "target_patch_rows_scaled_by_exact_target_timestep"
            if cache_mode == "tea"
            else None
        ),
        "native_cache_dit": cache_mode == "fb",
        "native_cache_dit_config": native_config,
        "h3_out_of_place_gate_patch": gate_patch,
        "parameter_table": {
            str(nfe): cache_parameters(cache_mode, nfe) for nfe in allowed_nfe
        },
    }
    return dict(_INSTALL_STATE)


__all__ = [
    "ALLOWED_NFE",
    "CACHE_MODES",
    "PINNED_SGLANG_COMMIT",
    "cache_parameters",
    "install_stage1_cache_overlay",
]
