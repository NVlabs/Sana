"""Student-only LightX2V LoRA overlay for pinned SGLang MiniMax-H3.

This reuses the reviewed key normalization, fused-name mapping, and exact
single-TP LoRA support from the 2+3 experiment.  It supports the validated
dynamic path and the more efficient single-service deployment path where the
adapter is merged once at startup.  It does not replace the denoising loop.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Mapping

import torch

from sglang_h3_lowres_sweep_overlay import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    install_lowres_sweep_overlay,
)
from sglang_h3_hybrid_overlay import (
    EXPECTED_MAPPED_LORA_LAYERS,
    EXPECTED_RAW_LORA_PAIRS,
    EXPECTED_MAPPED_LORA_TENSORS,
    _install_dynamic_merged_lora_support,
    _install_lora_key_normalizer,
    _install_minimax_lora_name_mapping,
)


PINNED_SGLANG_COMMIT = "12eadf86f12aec2e6f81a6e38b61b964a4c6b529"
_INSTALL_STATE: dict[str, Any] | None = None


def _integer_environment(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    return value


def _append_jsonl(path: str, payload: Mapping[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    encoded = (json.dumps(dict(payload), sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def _validate_lora_coverage(
    pipeline: Any,
    *,
    lora_path: str,
    lora_nickname: str,
    lora_scale: float,
    merge_mode: str,
) -> dict[str, Any]:
    if str(pipeline.lora_path) != lora_path:
        raise RuntimeError(f"pipeline LoRA path {pipeline.lora_path!r} != {lora_path!r}")
    if str(pipeline.lora_nickname) != lora_nickname:
        raise RuntimeError(
            f"pipeline LoRA nickname {pipeline.lora_nickname!r} != {lora_nickname!r}"
        )
    if str(pipeline.server_args.lora_merge_mode) != merge_mode:
        raise RuntimeError(
            f"pipeline merge mode {pipeline.server_args.lora_merge_mode!r} != {merge_mode!r}"
        )
    if not math.isclose(
        float(pipeline.server_args.lora_scale), lora_scale, abs_tol=1e-12
    ):
        raise RuntimeError(
            f"pipeline LoRA scale {pipeline.server_args.lora_scale} != {lora_scale}"
        )
    adapter = pipeline.lora_adapters.get(lora_nickname)
    if not isinstance(adapter, Mapping) or len(adapter) != EXPECTED_MAPPED_LORA_TENSORS:
        raise RuntimeError(
            f"mapped LoRA tensor count is "
            f"{None if not isinstance(adapter, Mapping) else len(adapter)}, "
            f"expected {EXPECTED_MAPPED_LORA_TENSORS}"
        )
    matched = [
        (name, layer)
        for name, layer in pipeline.lora_layers.items()
        if name + ".lora_A" in adapter and name + ".lora_B" in adapter
    ]
    if len(matched) != EXPECTED_MAPPED_LORA_LAYERS:
        raise RuntimeError(
            f"mapped LoRA covers {len(matched)} wrappers; "
            f"expected {EXPECTED_MAPPED_LORA_LAYERS}"
        )
    for name, layer in matched:
        if layer.lora_A is None or layer.lora_B is None:
            raise RuntimeError(f"LoRA wrapper {name} has no loaded A/B tensors")
        rank = int(layer.lora_A.shape[-2])
        if rank != 128 or int(layer.lora_rank) != rank or int(layer.lora_alpha) != rank:
            raise RuntimeError(
                f"LoRA wrapper {name} rank metadata={layer.lora_rank}/{layer.lora_alpha}, "
                f"tensor rank={rank}; expected 128/128"
            )
        if not math.isclose(float(layer.strength), lora_scale, abs_tol=1e-12):
            raise RuntimeError(
                f"LoRA wrapper {name} strength={layer.strength}; expected {lora_scale}"
            )
        if merge_mode == "dynamic":
            if layer.merged or layer.disable_lora:
                raise RuntimeError(f"dynamic LoRA wrapper {name} is not active")
        elif not layer.merged:
            raise RuntimeError(f"startup-merge LoRA wrapper {name} is not merged")
    pipeline_merged = bool(pipeline.is_lora_merged.get("transformer", False))
    if pipeline_merged != (merge_mode == "merge"):
        raise RuntimeError(
            f"pipeline merged state {pipeline_merged} disagrees with mode {merge_mode}"
        )
    return {
        "mapped_tensors": len(adapter),
        "mapped_layers": len(matched),
        "wrapped_layers_total": len(pipeline.lora_layers),
        "merge_mode": merge_mode,
        "merged_layers": sum(int(layer.merged) for _, layer in matched),
        "active_dynamic_layers": sum(
            int(not layer.merged and not layer.disable_lora) for _, layer in matched
        ),
    }


def install_firstframe_student_lora_overlay(
    *,
    lora_path: str,
    lora_nickname: str = "lx2v_4s_v01_544p",
    lora_scale: float = 0.0625,
    merge_mode: str = "dynamic",
    compile_enabled: bool = False,
    forward_evaluations: int = 4,
    distilled_nfe: int = 4,
    recommended_nfe: tuple[int, ...] = (4,),
    video_shift: float = 12.0,
    audio_shift: float = 3.0,
    allow_offgrid_steps: bool = False,
) -> dict[str, Any]:
    """Install one fail-closed fixed-canvas FL2VA student experiment overlay."""

    global _INSTALL_STATE
    if not lora_path:
        raise ValueError("student overlay requires a LoRA path")
    if not math.isclose(float(lora_scale), 8.0 / 128.0, abs_tol=1e-12):
        raise ValueError("student LoRA scale must be alpha/rank = 8/128")
    if merge_mode not in {"dynamic", "merge"}:
        raise ValueError(f"student merge_mode must be dynamic or merge, got {merge_mode!r}")
    if (
        isinstance(forward_evaluations, bool)
        or not isinstance(forward_evaluations, int)
        or forward_evaluations < 1
    ):
        raise ValueError("student forward_evaluations must be a positive integer")
    if distilled_nfe not in {4, 8}:
        raise ValueError("student distilled_nfe must be 4 or 8")
    expected_recommended = (4,) if distilled_nfe == 4 else (8, 4)
    if tuple(recommended_nfe) != expected_recommended:
        raise ValueError(
            f"distilled_nfe={distilled_nfe} requires recommended_nfe="
            f"{expected_recommended}"
        )
    if forward_evaluations not in recommended_nfe and not allow_offgrid_steps:
        raise RuntimeError(
            f"{forward_evaluations} forwards are outside the adapter's recommended "
            f"NFE set {recommended_nfe}; set allow_offgrid_steps=True to acknowledge"
        )
    if forward_evaluations in recommended_nfe and allow_offgrid_steps:
        raise ValueError("allow_offgrid_steps must be false for a recommended NFE")
    expected_sigma_points = forward_evaluations + 1
    expected_video_sigmas = [
        video_shift * ((forward_evaluations - index) / forward_evaluations)
        / (
            1.0
            + (video_shift - 1.0)
            * ((forward_evaluations - index) / forward_evaluations)
        )
        for index in range(forward_evaluations)
    ] + [0.0]
    expected_audio_sigmas = [
        audio_shift * ((forward_evaluations - index) / forward_evaluations)
        / (
            1.0
            + (audio_shift - 1.0)
            * ((forward_evaluations - index) / forward_evaluations)
        )
        for index in range(forward_evaluations)
    ] + [0.0]
    student_width = _integer_environment("H3_FF_STUDENT_WIDTH", DEFAULT_WIDTH)
    student_height = _integer_environment("H3_FF_STUDENT_HEIGHT", DEFAULT_HEIGHT)
    expected_latent_height = student_height // 16
    expected_latent_width = student_width // 16
    denoise_timing_file = os.environ.get("H3_FF_DENOISE_TIMING_FILE", "").strip()
    requested = {
        "lora_path": str(lora_path),
        "lora_nickname": str(lora_nickname),
        "lora_scale": float(lora_scale),
        "merge_mode": merge_mode,
        "compile_enabled": bool(compile_enabled),
        "forward_evaluations": int(forward_evaluations),
        "sigma_points": expected_sigma_points,
        "distilled_nfe": int(distilled_nfe),
        "recommended_nfe": list(recommended_nfe),
        "video_shift": float(video_shift),
        "audio_shift": float(audio_shift),
        "off_native_distillation_grid": forward_evaluations != distilled_nfe,
        "officially_recommended_nfe": forward_evaluations in recommended_nfe,
        "offgrid_acknowledged": bool(allow_offgrid_steps),
        "student_width": student_width,
        "student_height": student_height,
        "denoise_timing_file": denoise_timing_file or None,
    }
    if _INSTALL_STATE is not None:
        if _INSTALL_STATE["config"] != requested:
            raise RuntimeError(
                f"a different first-frame student overlay is active: {_INSTALL_STATE}"
            )
        return dict(_INSTALL_STATE)

    compile_flag = os.environ.get("H3_FF_COMPILE", "0")
    if compile_flag not in {"0", "1"} or (compile_flag == "1") != bool(
        compile_enabled
    ):
        raise RuntimeError(
            "H3_FF_COMPILE must be 0/1 and agree with compile_enabled"
        )

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    resolution = install_lowres_sweep_overlay(
        width=student_width,
        height=student_height,
    )
    _install_lora_key_normalizer()
    _install_minimax_lora_name_mapping()
    _install_dynamic_merged_lora_support(compile_enabled=compile_enabled)

    from sglang.multimodal_gen.runtime.pipelines import minimax_h3_pipeline
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
        denoising as denoising_module,
    )

    current_stage = minimax_h3_pipeline.MiniMaxH3DenoisingStage
    if getattr(current_stage, "_h3_firstframe_student_overlay", False):
        raise RuntimeError("unexpected pre-installed first-frame student stage")

    class MiniMaxH3FirstFrameStudentDenoisingStage(current_stage):
        def __init__(self, transformer: Any, pipeline: Any = None) -> None:
            super().__init__(transformer=transformer, pipeline=pipeline)
            if pipeline is None:
                raise RuntimeError("student LoRA stage requires its owning pipeline")
            self._ff_student_pipeline = pipeline

        def _run_full_loop(self, batch: Any, server_args: Any) -> None:
            ctx = denoising_module._resolve_full_loop_context(batch)
            if ctx.plan is None or str(ctx.plan.task) != "fl2va":
                raise NotImplementedError("student overlay accepts only task='fl2va'")
            if (ctx.latent_h, ctx.latent_w) != (
                expected_latent_height,
                expected_latent_width,
            ):
                raise ValueError(
                    f"student latent grid {ctx.latent_h}x{ctx.latent_w}; expected "
                    f"{expected_latent_height}x{expected_latent_width} for "
                    f"{student_width}x{student_height} pixels"
                )
            video_sigmas = list(ctx.sigmas["video"])
            audio_sigmas = list(ctx.sigmas["audio"])
            if len(video_sigmas) != expected_sigma_points or len(audio_sigmas) != expected_sigma_points:
                raise ValueError(
                    f"student request must have {expected_sigma_points} sigma points/"
                    f"{forward_evaluations} forwards, got "
                    f"video={len(video_sigmas)} audio={len(audio_sigmas)}"
                )
            for name, actual, expected in (
                ("video", video_sigmas, expected_video_sigmas),
                ("audio", audio_sigmas, expected_audio_sigmas),
            ):
                if any(
                    not math.isclose(
                        float(actual_value),
                        float(expected_value),
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                    for actual_value, expected_value in zip(actual, expected)
                ):
                    raise ValueError(
                        f"student {name} sigma grid {actual} does not match "
                        f"the expected {expected}"
                    )
            runtime_compile = bool(getattr(server_args, "enable_torch_compile", False))
            if runtime_compile != bool(compile_enabled):
                raise RuntimeError(
                    f"runtime compile={runtime_compile} disagrees with overlay={compile_enabled}"
                )

            pipeline = self._ff_student_pipeline
            coverage = _validate_lora_coverage(
                pipeline,
                lora_path=str(lora_path),
                lora_nickname=str(lora_nickname),
                lora_scale=float(lora_scale),
                merge_mode=merge_mode,
            )
            if not isinstance(getattr(batch, "extra", None), Mapping):
                raise RuntimeError("MiniMax-H3 batch.extra is unavailable")
            batch.extra["minimax_h3_firstframe_student_lora"] = {
                "effective_layers": EXPECTED_MAPPED_LORA_LAYERS,
                "coverage": coverage,
                "sigma_points": len(video_sigmas),
                "latent_grid": [ctx.latent_h, ctx.latent_w],
                "pixel_canvas": [student_width, student_height],
            }
            torch.cuda.synchronize()
            started_ns = time.perf_counter_ns()
            try:
                result = super()._run_full_loop(batch, server_args)
            except BaseException as exc:
                torch.cuda.synchronize()
                total_s = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
                if denoise_timing_file:
                    _append_jsonl(
                        denoise_timing_file,
                        {
                            "avg_forward": total_s / forward_evaluations,
                            "height": student_height,
                            "latent_height": ctx.latent_h,
                            "latent_width": ctx.latent_w,
                            "nfe": forward_evaluations,
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "total": total_s,
                            "unit": "seconds",
                            "width": student_width,
                        },
                    )
                raise
            torch.cuda.synchronize()
            total_s = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
            batch.extra["minimax_h3_firstframe_student_lora"]["denoise_timing"] = {
                "total": total_s,
                "avg_forward": total_s / forward_evaluations,
                "unit": "seconds",
            }
            if denoise_timing_file:
                _append_jsonl(
                    denoise_timing_file,
                    {
                        "avg_forward": total_s / forward_evaluations,
                        "height": student_height,
                        "latent_height": ctx.latent_h,
                        "latent_width": ctx.latent_w,
                        "nfe": forward_evaluations,
                        "status": "ok",
                        "total": total_s,
                        "unit": "seconds",
                        "width": student_width,
                    },
                )
            return result

    MiniMaxH3FirstFrameStudentDenoisingStage.__name__ = (
        "MiniMaxH3FirstFrameStudentDenoisingStage"
    )
    MiniMaxH3FirstFrameStudentDenoisingStage.__qualname__ = (
        "MiniMaxH3FirstFrameStudentDenoisingStage"
    )
    MiniMaxH3FirstFrameStudentDenoisingStage._h3_firstframe_student_overlay = True
    MiniMaxH3FirstFrameStudentDenoisingStage._h3_firstframe_stock_stage = current_stage
    minimax_h3_pipeline.MiniMaxH3DenoisingStage = (
        MiniMaxH3FirstFrameStudentDenoisingStage
    )

    _INSTALL_STATE = {
        "installed": True,
        "name": (
            f"sglang_minimax_h3_firstframe_{student_width}x{student_height}_"
            f"{lora_nickname}_{forward_evaluations}nfe_v2"
        ),
        "pinned_sglang_commit": PINNED_SGLANG_COMMIT,
        "config": requested,
        "resolution_overlay": resolution,
        "expected_raw_lora_pairs": EXPECTED_RAW_LORA_PAIRS,
        "expected_mapped_lora_layers": EXPECTED_MAPPED_LORA_LAYERS,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    return dict(_INSTALL_STATE)


__all__ = ["install_firstframe_student_lora_overlay"]
