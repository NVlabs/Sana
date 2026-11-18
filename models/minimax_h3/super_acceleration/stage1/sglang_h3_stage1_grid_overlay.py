"""Dynamic-NFE, fixed-canvas Stage-1 model overlay for the GB200 grid."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch

from sglang_h3_lowres_sweep_overlay import install_lowres_sweep_overlay
from sglang_h3_hybrid_overlay import (
    EXPECTED_MAPPED_LORA_LAYERS,
    _install_dynamic_merged_lora_support,
    _install_lora_key_normalizer,
    _install_minimax_lora_name_mapping,
)
from sglang_h3_firstframe_lora_overlay import _validate_lora_coverage
from sglang_h3_stage1_cache_overlay import (
    ALLOWED_NFE,
    CACHE_MODES,
    install_stage1_cache_overlay,
)


PINNED_SGLANG_COMMIT = "12eadf86f12aec2e6f81a6e38b61b964a4c6b529"
_INSTALL_STATE: dict[str, Any] | None = None


def _expected_sigmas(nfe: int, shift: float) -> list[float]:
    return [
        shift * ((nfe - index) / nfe)
        / (1.0 + (shift - 1.0) * ((nfe - index) / nfe))
        for index in range(nfe)
    ] + [0.0]


def _validate_teacher_lora_absence(pipeline: Any) -> dict[str, Any]:
    """Fail closed if a nominal Teacher process contains any LoRA state."""

    lora_path = getattr(pipeline, "lora_path", None)
    lora_nickname = getattr(pipeline, "lora_nickname", None)
    adapters = getattr(pipeline, "lora_adapters", None)
    merged = getattr(pipeline, "is_lora_merged", None)
    if lora_path not in {None, ""}:
        raise RuntimeError(f"Teacher unexpectedly has LoRA path {lora_path!r}")
    # SGLang may retain an inert default nickname even when lora_path is None;
    # absence is established by no path, no adapter tensors and no merged state.
    if isinstance(adapters, Mapping) and len(adapters) != 0:
        raise RuntimeError(f"Teacher unexpectedly has {len(adapters)} LoRA adapters")
    if isinstance(merged, Mapping) and any(bool(value) for value in merged.values()):
        raise RuntimeError(f"Teacher unexpectedly has merged LoRA state {dict(merged)}")
    return {
        "lora_path": lora_path,
        "lora_nickname": lora_nickname,
        "adapter_count": 0 if not isinstance(adapters, Mapping) else len(adapters),
        "any_component_merged": (
            False
            if not isinstance(merged, Mapping)
            else any(bool(value) for value in merged.values())
        ),
    }


def install_stage1_grid_overlay(
    *,
    model_profile: str,
    width: int,
    height: int,
    cache_mode: str,
    telemetry_path: str,
    compile_enabled: bool,
    lora_path: str | None,
    lora_nickname: str | None,
    lora_scale: float | None,
    lora_merge_mode: str,
    video_shift: float,
    audio_shift: float,
    distilled_nfe: int | None,
) -> dict[str, Any]:
    """Install one process profile while admitting all five requested NFEs."""

    global _INSTALL_STATE
    if cache_mode not in CACHE_MODES:
        raise ValueError(f"cache_mode must be one of {CACHE_MODES}")
    if lora_merge_mode != "merge":
        raise ValueError("the production grid requires startup-merged LoRA weights")
    has_lora = lora_path is not None
    if has_lora != (lora_nickname is not None and lora_scale is not None):
        raise ValueError("LoRA path, nickname and scale must be supplied together")
    if has_lora and (not Path(str(lora_path)).is_file()):
        raise FileNotFoundError(f"LoRA is unavailable: {lora_path}")
    if not has_lora and distilled_nfe is not None:
        raise ValueError("teacher profile must not declare a distilled NFE")
    if has_lora and distilled_nfe not in {4, 8}:
        raise ValueError("LoRA distilled NFE must be 4 or 8")
    requested = {
        "model_profile": model_profile,
        "width": int(width),
        "height": int(height),
        "cache_mode": cache_mode,
        "telemetry_path": str(telemetry_path),
        "compile_enabled": bool(compile_enabled),
        "lora_path": lora_path,
        "lora_nickname": lora_nickname,
        "lora_scale": lora_scale,
        "lora_merge_mode": lora_merge_mode,
        "video_shift": float(video_shift),
        "audio_shift": float(audio_shift),
        "distilled_nfe": distilled_nfe,
        "allowed_nfe": list(ALLOWED_NFE),
    }
    if _INSTALL_STATE is not None:
        if _INSTALL_STATE["config"] != requested:
            raise RuntimeError(f"a different Stage-1 grid overlay is active: {_INSTALL_STATE}")
        return dict(_INSTALL_STATE)

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    resolution = install_lowres_sweep_overlay(width=width, height=height)
    if has_lora:
        _install_lora_key_normalizer()
        _install_minimax_lora_name_mapping()
        _install_dynamic_merged_lora_support(compile_enabled=compile_enabled)

    from sglang.multimodal_gen.runtime.pipelines import minimax_h3_pipeline
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
        denoising as denoising_module,
    )

    current_stage = minimax_h3_pipeline.MiniMaxH3DenoisingStage
    if getattr(current_stage, "_h3_stage1_grid_overlay", False):
        raise RuntimeError("unexpected pre-installed Stage-1 grid overlay")

    expected_latent_height = height // 16
    expected_latent_width = width // 16

    class MiniMaxH3Stage1GridDenoisingStage(current_stage):
        def __init__(self, transformer: Any, pipeline: Any = None) -> None:
            super().__init__(transformer=transformer, pipeline=pipeline)
            if pipeline is None:
                raise RuntimeError("Stage-1 grid requires its owning pipeline")
            self._stage1_grid_pipeline = pipeline
            self._stage1_lora_coverage: dict[str, Any] | None = None

        def _run_full_loop(self, batch: Any, server_args: Any) -> None:
            ctx = denoising_module._resolve_full_loop_context(batch)
            if ctx.plan is None or str(ctx.plan.task) != "fl2va":
                raise NotImplementedError("Stage-1 grid accepts only first-frame FL2VA")
            if (ctx.latent_h, ctx.latent_w) != (
                expected_latent_height,
                expected_latent_width,
            ):
                raise ValueError(
                    f"latent grid {ctx.latent_h}x{ctx.latent_w}; expected "
                    f"{expected_latent_height}x{expected_latent_width} for {width}x{height}"
                )
            video_sigmas = list(ctx.sigmas["video"])
            audio_sigmas = list(ctx.sigmas["audio"])
            nfe = len(video_sigmas) - 1
            if nfe not in ALLOWED_NFE or len(audio_sigmas) != nfe + 1:
                raise ValueError(
                    f"request has video/audio sigma lengths "
                    f"{len(video_sigmas)}/{len(audio_sigmas)}; NFE must be in {ALLOWED_NFE}"
                )
            for name, actual, expected in (
                ("video", video_sigmas, _expected_sigmas(nfe, video_shift)),
                ("audio", audio_sigmas, _expected_sigmas(nfe, audio_shift)),
            ):
                if any(
                    not math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=1e-6)
                    for got, want in zip(actual, expected)
                ):
                    raise ValueError(
                        f"{name} sigma grid {actual} does not match shift={video_shift if name == 'video' else audio_shift}"
                    )
            runtime_compile = bool(getattr(server_args, "enable_torch_compile", False))
            if runtime_compile != compile_enabled:
                raise RuntimeError(
                    f"runtime compile={runtime_compile} disagrees with overlay={compile_enabled}"
                )
            coverage = None
            teacher_audit = None
            if has_lora:
                if self._stage1_lora_coverage is None:
                    self._stage1_lora_coverage = _validate_lora_coverage(
                        self._stage1_grid_pipeline,
                        lora_path=str(lora_path),
                        lora_nickname=str(lora_nickname),
                        lora_scale=float(lora_scale),
                        merge_mode=lora_merge_mode,
                    )
                coverage = self._stage1_lora_coverage
            else:
                teacher_audit = _validate_teacher_lora_absence(
                    self._stage1_grid_pipeline
                )
            if not isinstance(getattr(batch, "extra", None), Mapping):
                raise RuntimeError("MiniMax-H3 batch.extra is unavailable")
            batch.extra["minimax_h3_stage1_grid"] = {
                "model_profile": model_profile,
                "pixel_canvas": [width, height],
                "latent_grid": [ctx.latent_h, ctx.latent_w],
                "nfe": nfe,
                "off_native_distillation_grid": (
                    None if distilled_nfe is None else nfe != distilled_nfe
                ),
                "offgrid_acknowledged_by_sweep": (
                    None if distilled_nfe is None else nfe != distilled_nfe
                ),
                "lora_applied": has_lora,
                "lora_path": lora_path,
                "lora_nickname": lora_nickname,
                "lora_scale": lora_scale,
                "lora_coverage": coverage,
                "teacher_lora_audit": teacher_audit,
            }
            return super()._run_full_loop(batch, server_args)

    MiniMaxH3Stage1GridDenoisingStage.__name__ = "MiniMaxH3Stage1GridDenoisingStage"
    MiniMaxH3Stage1GridDenoisingStage.__qualname__ = "MiniMaxH3Stage1GridDenoisingStage"
    MiniMaxH3Stage1GridDenoisingStage._h3_stage1_grid_overlay = True
    MiniMaxH3Stage1GridDenoisingStage._h3_stage1_grid_stock_stage = current_stage
    minimax_h3_pipeline.MiniMaxH3DenoisingStage = MiniMaxH3Stage1GridDenoisingStage

    cache = install_stage1_cache_overlay(
        cache_mode=cache_mode,
        telemetry_path=telemetry_path,
        width=width,
        height=height,
    )
    _INSTALL_STATE = {
        "installed": True,
        "name": f"sglang_minimax_h3_stage1_grid_{model_profile}_{width}x{height}_{cache_mode}_v1",
        "pinned_sglang_commit": PINNED_SGLANG_COMMIT,
        "config": requested,
        "resolution_overlay": resolution,
        "cache_overlay": cache,
        "expected_lora_layers": EXPECTED_MAPPED_LORA_LAYERS if has_lora else 0,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    return dict(_INSTALL_STATE)


__all__ = ["install_stage1_grid_overlay"]
