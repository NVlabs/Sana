"""Pinned SGLang MiniMax-H3 1080p-teacher/480p-student overlay.

This module is deliberately narrow.  It implements the reviewed ``t2_l3_480``
trajectory on SGLang commit ``12eadf86``:

* two dense 1920x1088 teacher evaluations with LoRA disabled;
* a clean/noise resolution handoff at the existing sigma node; and
* three dense 864x480 student evaluations with one dynamic LightX2V LoRA.

The installer must run at module scope in the benchmark driver.  SGLang starts
workers with multiprocessing ``spawn``, so a patch installed only below a main
guard would not exist in the GPU workers.

This is an experiment overlay, not a general MiniMax-H3 pipeline.  It fails
closed for non-T2VA requests, conditional inputs, other step counts, other
canvases, merged LoRA, or checkpoint/source-layout drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from sglang_h3_resolution_overlay import install_1080p_overlay


PINNED_SGLANG_COMMIT = "12eadf86f12aec2e6f81a6e38b61b964a4c6b529"
HIGH_PIXEL_HEIGHT = 1088
HIGH_PIXEL_WIDTH = 1920
HIGH_LATENT_HEIGHT = 68
HIGH_LATENT_WIDTH = 120
LOW_PIXEL_HEIGHT = 480
LOW_PIXEL_WIDTH = 864
LOW_LATENT_HEIGHT = 30
LOW_LATENT_WIDTH = 54
VIDEO_CHANNELS = 24
EXPECTED_RAW_LORA_TENSORS = 624
EXPECTED_RAW_LORA_PAIRS = 312
EXPECTED_MAPPED_LORA_TENSORS = 416
EXPECTED_MAPPED_LORA_LAYERS = 208


@dataclass(frozen=True)
class HybridOverlayConfig:
    high_short_edge: int
    low_short_edge: int
    teacher_steps: int
    student_steps: int
    lora_path: str
    lora_nickname: str
    lora_scale: float
    video_flow_shift: float
    audio_flow_shift: float
    telemetry_path: str | None
    torch_compile: bool
    compile_mode: str


_INSTALL_STATE: dict[str, Any] | None = None
_NORMALIZE_STATS: dict[str, Any] | None = None


# External Diffusers/LightX2V names -> native SGLang MiniMax-H3 names.  Q/K/V
# are stacked because SGLang owns one MergedColumnParallelLinear qkv_proj.
_LORA_PARAM_NAMES_MAPPING: dict[str, str | tuple[str, int, int]] = {
    r"^transformer_blocks\.(\d+)\.attn\.to_q(\.lora_[AB])$": (
        r"blocks.\1.attn.qkv_proj\2",
        0,
        3,
    ),
    r"^transformer_blocks\.(\d+)\.attn\.to_k(\.lora_[AB])$": (
        r"blocks.\1.attn.qkv_proj\2",
        1,
        3,
    ),
    r"^transformer_blocks\.(\d+)\.attn\.to_v(\.lora_[AB])$": (
        r"blocks.\1.attn.qkv_proj\2",
        2,
        3,
    ),
    r"^transformer_blocks\.(\d+)\.attn\.to_out\.0(\.lora_[AB])$": (
        r"blocks.\1.attn.out_proj\2"
    ),
    r"^transformer_blocks\.(\d+)\.ff\.net\.0\.proj(\.lora_[AB])$": (
        r"blocks.\1.mlp.fc1\2"
    ),
    r"^transformer_blocks\.(\d+)\.ff\.net\.2(\.lora_[AB])$": (
        r"blocks.\1.mlp.fc2\2"
    ),
    r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_q(\.lora_[AB])$": (
        r"token_refiner.blocks.\1.attn.qkv_proj\2",
        0,
        3,
    ),
    r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_k(\.lora_[AB])$": (
        r"token_refiner.blocks.\1.attn.qkv_proj\2",
        1,
        3,
    ),
    r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_v(\.lora_[AB])$": (
        r"token_refiner.blocks.\1.attn.qkv_proj\2",
        2,
        3,
    ),
    r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_out\.0(\.lora_[AB])$": (
        r"token_refiner.blocks.\1.attn.out_proj\2"
    ),
    r"^token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.0\.proj(\.lora_[AB])$": (
        r"token_refiner.blocks.\1.mlp.fc1\2"
    ),
    r"^token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.2(\.lora_[AB])$": (
        r"token_refiner.blocks.\1.mlp.fc2\2"
    ),
}


def _validate_config(config: HybridOverlayConfig) -> None:
    if config.high_short_edge != 1080 or config.low_short_edge != 480:
        raise ValueError("hybrid overlay is fixed to semantic 1080p -> 480p")
    if config.teacher_steps != 2 or config.student_steps != 3:
        raise ValueError("hybrid overlay is fixed to two teacher + three student steps")
    if not config.lora_path:
        raise ValueError("hybrid overlay requires a startup-loaded LoRA path")
    if not config.lora_nickname:
        raise ValueError("hybrid overlay requires a non-empty LoRA nickname")
    for name, value in (
        ("lora_scale", config.lora_scale),
        ("video_flow_shift", config.video_flow_shift),
        ("audio_flow_shift", config.audio_flow_shift),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    if not math.isclose(config.lora_scale, 8.0 / 128.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "this checkpoint omits adapter alpha; lora_scale must model alpha/rank "
            "as 8/128 == 0.0625"
        )
    if (
        config.torch_compile
        and config.compile_mode != "max-autotune-no-cudagraphs"
    ):
        raise ValueError(
            "hybrid compile mode is pinned to 'max-autotune-no-cudagraphs', "
            f"got {config.compile_mode!r}"
        )


def _strict_compile_environment() -> tuple[bool, str]:
    raw = os.environ.get("H3_HYBRID_TORCH_COMPILE")
    if raw not in {"0", "1"}:
        raise RuntimeError(
            "H3_HYBRID_TORCH_COMPILE must be explicitly set to 0 (eager smoke) "
            "or 1 (compiled formal run) before installing the overlay"
        )
    enabled = raw == "1"
    legacy_vae = os.environ.get("H3_HYBRID_COMPILE_VAE")
    if legacy_vae is not None and legacy_vae != raw:
        raise RuntimeError(
            f"H3_HYBRID_COMPILE_VAE={legacy_vae!r} disagrees with "
            f"H3_HYBRID_TORCH_COMPILE={raw!r}"
        )
    mode = os.environ.get(
        "SGLANG_VAE_TORCH_COMPILE_MODE",
        "max-autotune-no-cudagraphs",
    )
    dit_mode = os.environ.get("SGLANG_TORCH_COMPILE_MODE", mode)
    if enabled and dit_mode != mode:
        raise RuntimeError(
            f"DiT/VAE compile modes disagree: {dit_mode!r} vs {mode!r}"
        )
    return enabled, mode


def _strict_bool_environment(name: str, *, default: str) -> bool:
    raw = os.environ.get(name, default)
    if raw not in {"0", "1"}:
        raise RuntimeError(f"{name} must be exactly 0 or 1, got {raw!r}")
    return raw == "1"


def _slice_merged_lora_b_tp1(
    layer: Any,
    B: torch.Tensor,
    original_slice: Any,
) -> torch.Tensor:
    """Slice stacked QKV B or preserve a fused-FFN 2-D B at TP=1."""

    if int(getattr(layer.base_layer, "tp_size", 1)) != 1:
        raise RuntimeError("H3 merged LoRA B overlay requires tp_size=1")
    if B.ndim == 2:
        return B
    if B.ndim == 3:
        return original_slice(layer, B)
    raise ValueError(
        f"merged LoRA B must be 2-D fused FFN or 3-D stacked QKV, got {tuple(B.shape)}"
    )


def _strip_lightx2v_default_suffix(name: str) -> str:
    for kind in ("lora_A", "lora_B"):
        suffix = f".{kind}.default.weight"
        if name.endswith(suffix):
            return name[: -len(suffix)] + f".{kind}.weight"
    return name


def _validate_external_lora_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    keys = tuple(state_dict)
    if len(keys) != EXPECTED_RAW_LORA_TENSORS:
        raise ValueError(
            f"LightX2V checkpoint has {len(keys)} tensors; "
            f"expected {EXPECTED_RAW_LORA_TENSORS}"
        )
    a_suffix = ".lora_A.weight"
    b_suffix = ".lora_B.weight"
    a_names = {name[: -len(a_suffix)] for name in keys if name.endswith(a_suffix)}
    b_names = {name[: -len(b_suffix)] for name in keys if name.endswith(b_suffix)}
    unsupported = [
        name for name in keys if not name.endswith((a_suffix, b_suffix))
    ]
    if unsupported:
        raise ValueError(f"unsupported LightX2V keys after normalization: {unsupported[:3]}")
    if a_names != b_names or len(a_names) != EXPECTED_RAW_LORA_PAIRS:
        raise ValueError(
            "LightX2V A/B coverage mismatch: "
            f"A={len(a_names)} B={len(b_names)} paired={len(a_names & b_names)}; "
            f"expected {EXPECTED_RAW_LORA_PAIRS} pairs"
        )
    allowed_targets = (
        ".attn.to_q",
        ".attn.to_k",
        ".attn.to_v",
        ".attn.to_out.0",
        ".ff.net.0.proj",
        ".ff.net.2",
    )
    bad_targets = sorted(name for name in a_names if not name.endswith(allowed_targets))
    if bad_targets:
        raise ValueError(f"unexpected LightX2V target modules: {bad_targets[:3]}")
    return {
        "raw_tensors": len(keys),
        "raw_pairs": len(a_names),
        "raw_a": len(a_names),
        "raw_b": len(b_names),
    }


def _install_lora_key_normalizer() -> None:
    from sglang.multimodal_gen.runtime.pipelines_core import lora_pipeline

    current = lora_pipeline.normalize_lora_state_dict
    if getattr(current, "_h3_hybrid_overlay", False):
        return
    original = current

    def normalize_lightx2v(
        state_dict: Mapping[str, torch.Tensor],
        logger: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        global _NORMALIZE_STATS
        normalized = original(state_dict, logger=logger)
        remapped: dict[str, torch.Tensor] = {}
        for name, tensor in normalized.items():
            target = _strip_lightx2v_default_suffix(name)
            if ".default." in target:
                raise ValueError(f"unhandled LoRA adapter namespace in key {name!r}")
            if target in remapped:
                raise ValueError(f"duplicate normalized LoRA key {target!r}")
            remapped[target] = tensor
        _NORMALIZE_STATS = _validate_external_lora_keys(remapped)
        return remapped

    normalize_lightx2v._h3_hybrid_overlay = True  # type: ignore[attr-defined]
    normalize_lightx2v._h3_hybrid_original = original  # type: ignore[attr-defined]
    lora_pipeline.normalize_lora_state_dict = normalize_lightx2v


def _install_minimax_lora_name_mapping() -> None:
    from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
        MiniMaxH3DiTArchConfig,
    )
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        MiniMaxH3DiTModel,
    )

    existing = dict(getattr(MiniMaxH3DiTModel, "param_names_mapping", {}) or {})
    for pattern, replacement in _LORA_PARAM_NAMES_MAPPING.items():
        conflict = existing.get(pattern)
        if conflict is not None and conflict != replacement:
            raise RuntimeError(f"conflicting pinned MiniMax-H3 mapping for {pattern!r}")
    MiniMaxH3DiTModel.param_names_mapping = {
        **_LORA_PARAM_NAMES_MAPPING,
        **existing,
    }

    original_post_init = MiniMaxH3DiTArchConfig.__post_init__
    if getattr(original_post_init, "_h3_hybrid_overlay", False):
        return

    def post_init_with_lora_mapping(self: Any) -> None:
        original_post_init(self)
        current = dict(self.param_names_mapping or {})
        for pattern, replacement in _LORA_PARAM_NAMES_MAPPING.items():
            conflict = current.get(pattern)
            if conflict is not None and conflict != replacement:
                raise RuntimeError(
                    f"conflicting runtime MiniMax-H3 mapping for {pattern!r}"
                )
        self.param_names_mapping = {**_LORA_PARAM_NAMES_MAPPING, **current}

    post_init_with_lora_mapping._h3_hybrid_overlay = True  # type: ignore[attr-defined]
    MiniMaxH3DiTArchConfig.__post_init__ = post_init_with_lora_mapping


def _install_low_resolution_delivery_validator() -> None:
    """Keep H3's strict AV probe while admitting the intentional 480p tail.

    The queued/canonical request must remain the 1920x1088 teacher workload;
    changing its resolved plan would also change latent preparation.  Only the
    final file-size expectation is replaced after the worker has produced the
    864x480 student result.  Frame count, FPS, audio, codec, pixel format, and
    AV-drift checks continue to use SGLang's pinned probe unchanged.
    """

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        video_adapter,
    )

    cls = video_adapter.MiniMaxH3VideoModelAdapter
    current = cls.validate_final_outputs_sync
    if getattr(current, "_h3_hybrid_overlay", False):
        return

    def validate_hybrid_final_outputs_sync(
        self: Any,
        output_paths: list[str],
        batch: Any,
    ) -> dict[str, str]:
        expected_outputs = int(getattr(batch, "num_outputs_per_prompt", 1))
        if len(output_paths) != expected_outputs:
            raise RuntimeError(
                "MiniMax H3 hybrid video generation produced "
                f"{len(output_paths)} output files, expected {expected_outputs}"
            )
        shape = self._resolved_shape(batch)
        if shape is None:
            raise RuntimeError("MiniMax H3 hybrid delivery requires a resolved shape")
        expected_frame_count = int(shape.get("frame_count") or 0)
        teacher_size = (
            int(shape.get("width") or 0),
            int(shape.get("height") or 0),
        )
        if expected_frame_count != 243:
            raise RuntimeError(
                f"hybrid resolved frame count {expected_frame_count}; expected 243"
            )
        if teacher_size != (HIGH_PIXEL_WIDTH, HIGH_PIXEL_HEIGHT):
            raise RuntimeError(
                f"hybrid resolved teacher size {teacher_size}; expected "
                f"{(HIGH_PIXEL_WIDTH, HIGH_PIXEL_HEIGHT)}"
            )

        final_media_fields: dict[str, str] = {}
        for output_index, output_path in enumerate(output_paths):
            media_fields = video_adapter._probe_minimax_h3_output_fields(
                output_path,
                expected_frame_count=expected_frame_count,
                expected_size=(LOW_PIXEL_WIDTH, LOW_PIXEL_HEIGHT),
            )
            if output_index == 0:
                final_media_fields = media_fields
            elif media_fields != final_media_fields:
                raise RuntimeError(
                    "generated MiniMax H3 hybrid outputs have inconsistent media "
                    f"metadata: output 0={final_media_fields}, output "
                    f"{output_index}={media_fields}"
                )
        return final_media_fields

    validate_hybrid_final_outputs_sync._h3_hybrid_overlay = True  # type: ignore[attr-defined]
    validate_hybrid_final_outputs_sync._h3_hybrid_original = current  # type: ignore[attr-defined]
    cls.validate_final_outputs_sync = validate_hybrid_final_outputs_sync


def _install_dynamic_merged_lora_support(*, compile_enabled: bool) -> None:
    """Fix the pinned dynamic wrapper for stacked QKV and 2-D fused FFN LoRA.

    The stock loader stacks three Q/K/V A/B tensors to ``[3,...]``.  The
    pinned MergedColumn dynamic forward assumes a 2-D matmul, while its B
    slicer assumes 3-D weights; consequently neither stacked QKV nor the 2-D
    fused fc1 checkpoint is executable.  TP is fixed to one in this benchmark,
    allowing a compact exact dynamic implementation.
    """

    from torch.distributed.tensor import DTensor
    from sglang.multimodal_gen.runtime.layers.lora import linear as lora_linear

    cls = lora_linear.MergedColumnParallelLinearWithLoRA
    installed_mode = getattr(cls, "_h3_hybrid_dynamic_support", None)
    if installed_mode is not None:
        if bool(installed_mode["compiled"]) != bool(compile_enabled):
            raise RuntimeError(
                f"dynamic merged LoRA was already installed with {installed_mode}"
            )
        return

    original_set_lora_weights = lora_linear.BaseLayerWithLoRA.set_lora_weights
    original_slice_merged_b = cls.slice_lora_b_weights

    def slice_merged_b_tp1(self: Any, B: torch.Tensor) -> torch.Tensor:
        """Handle both stacked QKV B and fused-FFN 2-D B at TP=1.

        Pinned SGLang's merged-column slicer assumes every B is stacked 3-D.
        LightX2V's fused ``fc1`` is already represented as one 2-D output
        matrix, so indexing it as ``B[:, start:end, :]`` raises at startup
        merge.  No slice is necessary at TP=1.
        """

        return _slice_merged_lora_b_tp1(self, B, original_slice_merged_b)

    slice_merged_b_tp1._h3_hybrid_overlay = True  # type: ignore[attr-defined]
    cls.slice_lora_b_weights = slice_merged_b_tp1

    def set_lora_weights_with_stacked_rank(
        self: Any,
        A: torch.Tensor,
        B: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if A.ndim == 3:
            if B.ndim != 3 or int(A.shape[0]) != int(B.shape[0]):
                raise ValueError(
                    f"stacked merged LoRA needs matching A/B groups, got "
                    f"A={tuple(A.shape)} B={tuple(B.shape)}"
                )
            # Stock infers A.shape[0] == number of fused projections.  The
            # actual rank is A.shape[-2].  Keep alpha/rank == 1 here because
            # the external alpha=8/rank=128 is represented by strength=.0625.
            self.lora_rank = int(A.shape[-2])
            self.lora_alpha = int(A.shape[-2])
        original_set_lora_weights(self, A, B, *args, **kwargs)

    set_lora_weights_with_stacked_rank._h3_hybrid_overlay = True  # type: ignore[attr-defined]
    lora_linear.BaseLayerWithLoRA.set_lora_weights = set_lora_weights_with_stacked_rank

    def merged_dynamic_forward(self: Any, input_: torch.Tensor):
        if self.merged or self.disable_lora:
            return self.base_layer(input_)
        if int(getattr(self.base_layer, "tp_size", 1)) != 1:
            raise RuntimeError("H3 stacked dynamic LoRA overlay requires tp_size=1")

        lora_A = self.lora_A
        lora_B = self.lora_B
        if isinstance(lora_B, DTensor):
            lora_B = lora_B.to_local()
            lora_A = lora_A.to_local()
        if lora_A is None or lora_B is None:
            raise RuntimeError("active dynamic LoRA layer has no A/B tensors")

        output, output_bias = self.base_layer(input_)
        input_lora = input_.to(dtype=lora_A.dtype)
        flat_input = input_lora.reshape(-1, int(input_lora.shape[-1]))
        if lora_A.ndim == 2 and lora_B.ndim == 2:
            delta = flat_input @ lora_A.T @ lora_B.T
        elif lora_A.ndim == 3 and lora_B.ndim == 3:
            if (
                int(lora_A.shape[0]) != int(lora_B.shape[0])
                or int(lora_A.shape[-2]) != int(lora_B.shape[-1])
            ):
                raise ValueError(
                    f"bad stacked merged LoRA shapes A={tuple(lora_A.shape)} "
                    f"B={tuple(lora_B.shape)}"
                )
            hidden = torch.matmul(flat_input.unsqueeze(0), lora_A.transpose(-1, -2))
            grouped = torch.matmul(hidden, lora_B.transpose(-1, -2))
            delta = grouped.transpose(0, 1).reshape(int(flat_input.shape[0]), -1)
        else:
            raise ValueError(
                f"merged LoRA A/B ranks differ: A={tuple(lora_A.shape)} "
                f"B={tuple(lora_B.shape)}"
            )
        delta = delta.reshape(*input_lora.shape[:-1], int(delta.shape[-1]))
        if int(delta.shape[-1]) != int(output.shape[-1]):
            raise ValueError(
                f"merged LoRA output width {int(delta.shape[-1])} != "
                f"base output width {int(output.shape[-1])}"
            )
        if self.lora_alpha != self.lora_rank:
            delta = delta * (self.lora_alpha / self.lora_rank)
        delta = delta * self.strength
        return output + delta.to(dtype=output.dtype), output_bias

    # Formal runs compile high/low sequence lengths into separate hot graphs.
    # Smoke runs stay genuinely eager so they diagnose logic without paying or
    # hiding any compilation work.
    cls.forward = (
        torch.compile(
            merged_dynamic_forward,
            mode=os.environ.get(
                "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
            ),
            fullgraph=False,
            dynamic=None,
        )
        if compile_enabled
        else merged_dynamic_forward
    )
    cls._h3_hybrid_dynamic_support = {
        "installed": True,
        "compiled": bool(compile_enabled),
    }


def _expected_shifted_sigmas(points: int, shift: float) -> list[float]:
    base = torch.linspace(1.0, 0.0, points, dtype=torch.float32)
    return (shift * base / (1.0 + (shift - 1.0) * base)).tolist()


def _validate_schedule(
    actual: list[float], *, points: int, shift: float, name: str
) -> None:
    expected = _expected_shifted_sigmas(points, shift)
    if len(actual) != points:
        raise ValueError(f"{name} schedule has {len(actual)} points; expected {points}")
    if not all(left > right for left, right in zip(actual, actual[1:])):
        raise ValueError(f"{name} sigmas are not strictly decreasing: {actual}")
    if not torch.allclose(
        torch.tensor(actual, dtype=torch.float32),
        torch.tensor(expected, dtype=torch.float32),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(
            f"{name} sigma grid is not the pinned {points}-point shift={shift:g} grid: "
            f"actual={actual} expected={expected}"
        )


def _resolution_handoff(
    *,
    video_vae: Any,
    x0_rows: torch.Tensor,
    eps_rows: torch.Tensor,
    latent_t: int,
    sigma_resume: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    # Keep all tile collectives, exact temporal padding/token-drop behavior,
    # global variance measurement, and phase telemetry in the reviewed helper.
    # Every Ulysses rank calls it with the same replicated target rows.
    from sglang_h3_handoff import LATENT_FRAMES, four_gpu_vae_handoff

    if latent_t != LATENT_FRAMES:
        raise ValueError(
            f"hybrid handoff latent_t={latent_t}; pinned 243-frame request "
            f"requires {LATENT_FRAMES}"
        )
    return four_gpu_vae_handoff(
        video_vae,
        x0_rows,
        eps_rows,
        sigma_resume=sigma_resume,
        decode_dtype=torch.float16,
        parallel_resize=True,
        parallel_spectral=True,
        profile_phases=_strict_bool_environment(
            "H3_HYBRID_PROFILE_HANDOFF_PHASES", default="0"
        ),
    )


def _matched_lora_layers(pipeline: Any, config: HybridOverlayConfig) -> dict[str, Any]:
    if str(pipeline.lora_path) != config.lora_path:
        raise RuntimeError(
            f"pipeline LoRA path {pipeline.lora_path!r} != overlay {config.lora_path!r}"
        )
    if str(pipeline.lora_nickname) != config.lora_nickname:
        raise RuntimeError(
            f"pipeline LoRA nickname {pipeline.lora_nickname!r} != "
            f"overlay {config.lora_nickname!r}"
        )
    if str(pipeline.server_args.lora_merge_mode) != "dynamic":
        raise RuntimeError("hybrid overlay requires lora_merge_mode='dynamic'")
    if not math.isclose(
        float(pipeline.server_args.lora_scale),
        config.lora_scale,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"pipeline LoRA scale {pipeline.server_args.lora_scale} != "
            f"overlay {config.lora_scale}"
        )
    adapter = pipeline.lora_adapters.get(config.lora_nickname)
    if not isinstance(adapter, Mapping):
        raise RuntimeError(f"LoRA adapter {config.lora_nickname!r} was not loaded")
    if len(adapter) != EXPECTED_MAPPED_LORA_TENSORS:
        raise RuntimeError(
            f"mapped LoRA has {len(adapter)} tensors; "
            f"expected {EXPECTED_MAPPED_LORA_TENSORS}"
        )

    matched: list[tuple[str, Any]] = []
    for name, layer in pipeline.lora_layers.items():
        if name + ".lora_A" in adapter and name + ".lora_B" in adapter:
            matched.append((name, layer))
    if len(matched) != EXPECTED_MAPPED_LORA_LAYERS:
        raise RuntimeError(
            f"mapped LoRA covers {len(matched)} SGLang wrappers; "
            f"expected {EXPECTED_MAPPED_LORA_LAYERS}"
        )
    for name, layer in matched:
        if layer.merged:
            raise RuntimeError(f"LoRA layer {name} is merged; dynamic mode is required")
        if layer.lora_A is None or layer.lora_B is None:
            raise RuntimeError(f"LoRA layer {name} has no loaded A/B tensors")
        rank = int(layer.lora_A.shape[-2])
        if rank != 128:
            raise RuntimeError(f"LoRA layer {name} rank={rank}; expected 128")
        # Correct/verify the fused QKV rank after the stock loader.  The patched
        # set_lora_weights normally establishes this before this check.
        if int(layer.lora_rank) != rank or int(layer.lora_alpha) != rank:
            raise RuntimeError(
                f"LoRA layer {name} metadata rank/alpha="
                f"{layer.lora_rank}/{layer.lora_alpha}; expected {rank}/{rank}"
            )
        if not math.isclose(float(layer.strength), config.lora_scale, abs_tol=1e-12):
            raise RuntimeError(
                f"LoRA layer {name} strength={layer.strength}; "
                f"expected {config.lora_scale}"
            )
    return {
        "raw": dict(_NORMALIZE_STATS or {}),
        "mapped_tensors": len(adapter),
        "mapped_layers": len(matched),
        "wrapped_layers_total": len(pipeline.lora_layers),
    }


def _active_matched_lora_count(pipeline: Any, nickname: str) -> int:
    adapter = pipeline.lora_adapters[nickname]
    return sum(
        1
        for name, layer in pipeline.lora_layers.items()
        if name + ".lora_A" in adapter
        and name + ".lora_B" in adapter
        and not layer.merged
        and not layer.disable_lora
    )


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _world_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _append_telemetry(path: str | None, record: Mapping[str, Any]) -> None:
    if path is None or _world_rank() != 0:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def _validate_published_batch(batch: Any, *, latent_t: int, audio_t: int) -> dict[str, Any]:
    expected_video = (1, VIDEO_CHANNELS, latent_t, LOW_LATENT_HEIGHT, LOW_LATENT_WIDTH)
    expected_audio = (2, 32, audio_t)
    if not isinstance(batch.latents, torch.Tensor) or tuple(batch.latents.shape) != expected_video:
        raise RuntimeError(
            f"published hybrid video latent {getattr(batch.latents, 'shape', None)} "
            f"!= {expected_video}"
        )
    if not isinstance(batch.audio_latents, torch.Tensor) or tuple(batch.audio_latents.shape) != expected_audio:
        raise RuntimeError(
            f"published hybrid audio latent {getattr(batch.audio_latents, 'shape', None)} "
            f"!= {expected_audio}"
        )
    if tuple(batch.raw_latent_shape) != expected_video:
        raise RuntimeError(f"raw_latent_shape {batch.raw_latent_shape!r} != {expected_video}")
    if int(batch.height) != LOW_PIXEL_HEIGHT or int(batch.width) != LOW_PIXEL_WIDTH:
        raise RuntimeError(
            f"published canvas {batch.height}x{batch.width} != "
            f"{LOW_PIXEL_HEIGHT}x{LOW_PIXEL_WIDTH}"
        )
    return {
        "video_latent_shape": list(expected_video),
        "audio_latent_shape": list(expected_audio),
        "decoded_pixel_shape_expected": [LOW_PIXEL_HEIGHT, LOW_PIXEL_WIDTH],
    }


def validate_decoded_hybrid_output(output: torch.Tensor, *, frames: int) -> dict[str, Any]:
    """Validate a decoded SGLang tensor before the runner saves it."""

    expected = (1, 3, int(frames), LOW_PIXEL_HEIGHT, LOW_PIXEL_WIDTH)
    if not isinstance(output, torch.Tensor) or tuple(output.shape) != expected:
        raise ValueError(f"decoded hybrid output shape {getattr(output, 'shape', None)} != {expected}")
    if not torch.isfinite(output).all():
        raise ValueError("decoded hybrid output contains non-finite values")
    return {"shape": list(expected), "dtype": str(output.dtype), "finite": True}


def _build_hybrid_stage_class(config: HybridOverlayConfig):
    from functools import partial

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
        MiniMaxH3DenoiseBranch,
        minimax_h3_denoise_loop,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
        denoising as denoising_module,
    )
    from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import maybe_nvtx_range

    stock_cls = denoising_module.MiniMaxH3DenoisingStage

    class MiniMaxH3HybridDenoisingStage(stock_cls):
        """Two high teacher evals, exact resolution splice, three low LoRA evals."""

        def __init__(self, transformer: Any, pipeline: Any = None) -> None:
            super().__init__(transformer=transformer, pipeline=pipeline)
            if pipeline is None:
                raise RuntimeError("hybrid denoising stage requires its owning pipeline")
            self._hybrid_pipeline = pipeline
            self._hybrid_video_vae = pipeline.get_module("video_vae")

        def _run_full_loop(self, batch: Any, server_args: Any) -> None:
            ctx = denoising_module._resolve_full_loop_context(batch)
            if ctx.plan is None or str(ctx.plan.task) != "t2va":
                raise NotImplementedError("hybrid overlay currently supports only task='t2va'")
            if not torch.cuda.is_available():
                raise RuntimeError("MiniMax-H3 hybrid denoise requires CUDA")
            if (ctx.latent_h, ctx.latent_w) != (
                HIGH_LATENT_HEIGHT,
                HIGH_LATENT_WIDTH,
            ):
                raise ValueError(
                    f"hybrid head grid {ctx.latent_h}x{ctx.latent_w} != "
                    f"{HIGH_LATENT_HEIGHT}x{HIGH_LATENT_WIDTH}"
                )

            device = torch.device("cuda")
            sigmas_video = [float(value) for value in ctx.sigmas["video"]]
            sigmas_audio = [float(value) for value in ctx.sigmas["audio"]]
            points = config.teacher_steps + config.student_steps + 1
            _validate_schedule(
                sigmas_video,
                points=points,
                shift=config.video_flow_shift,
                name="video",
            )
            _validate_schedule(
                sigmas_audio,
                points=points,
                shift=config.audio_flow_shift,
                name="audio",
            )
            runtime_compile = bool(
                getattr(server_args, "enable_torch_compile", False)
            )
            if runtime_compile != config.torch_compile:
                raise RuntimeError(
                    f"SGLang enable_torch_compile={runtime_compile} disagrees with "
                    f"H3_HYBRID_TORCH_COMPILE={int(config.torch_compile)}"
                )
            if config.torch_compile:
                self._maybe_enable_cache_dit_and_torch_compile(points - 1, batch)

            if self._hybrid_video_vae.training:
                self._hybrid_video_vae.eval()
            if config.torch_compile:
                from sglang_h3_handoff import compile_local_vae_cores

                vae_compile = compile_local_vae_cores(
                    self._hybrid_video_vae,
                    mode=config.compile_mode,
                    fullgraph=False,
                    dynamic=False,
                    stack_tiling=True,
                )
            else:
                unexpected = getattr(
                    self._hybrid_video_vae, "_h3_handoff_compile", None
                )
                if unexpected is not None:
                    raise RuntimeError(
                        "eager smoke found a pre-compiled handoff VAE: "
                        f"{unexpected}"
                    )
                # Stack complete tiles without compiling their local cores;
                # this retains the same four-rank quality/parallel algorithm.
                self._hybrid_video_vae.stack_tiling = True
                vae_compile = {
                    "installed": False,
                    "scope": "eager_local_encoder_decoder",
                    "stack_tiling": True,
                }

            denoising_module._assemble_condition_rows(ctx)
            if ctx.include_cond or ctx.cond_rows is not None or ctx.audio_ref_rows is not None:
                raise NotImplementedError("hybrid overlay does not resize condition anchors")
            emb = ctx.embeddings["positive"]
            packed_high = denoising_module._build_packed_layout(ctx, emb)
            tags_high = packed_high["token_tags"]
            tags_high[packed_high["text_pos"].view(-1)] = (
                emb["text_token_tags"].view(-1).to(torch.long)
            )

            sampling = batch.sampling_params
            imgvid_noise_aug, audio_noise_aug = denoising_module.minimax_h3_condition_noise_aug(
                sampling
            )
            denoising_module._apply_condition_noise_aug(
                ctx,
                sampling=sampling,
                imgvid_noise_aug=imgvid_noise_aug,
                audio_noise_aug=audio_noise_aug,
            )

            coverage = _matched_lora_layers(self._hybrid_pipeline, config)
            self._hybrid_pipeline.deactivate_lora_weights(target="transformer")
            head_active = _active_matched_lora_count(
                self._hybrid_pipeline, config.lora_nickname
            )
            if head_active != 0:
                raise RuntimeError(f"teacher head still has {head_active} active LoRA layers")

            placement_managed = self._component_residency_manager is not None
            if placement_managed:
                self._manage_dit_use_site(self.transformer, "transformer", batch)
            started_total = time.perf_counter()
            telemetry: dict[str, Any] = {
                "kind": "sglang_minimax_h3_t2_l3_480",
                "rank": _world_rank(),
                "schedule_video": sigmas_video,
                "schedule_audio": sigmas_audio,
                "lora": coverage,
                "head_lora_active_layers": head_active,
                "torch_compile": config.torch_compile,
                "compile_mode": config.compile_mode,
                "vae_compile": vae_compile,
            }
            try:
                model = denoising_module._resolve_denoise_model(
                    self.transformer,
                    device,
                    placement_managed=placement_managed,
                )
                positive_high = MiniMaxH3DenoiseBranch(
                    packed=packed_high,
                    text_embeddings=emb["hidden_states"],
                    token_tags=tags_high,
                    device=device,
                )
                denoising_module._precompute_refined_prompt_embeds(
                    model, positive_high, device=device
                )
                denoising_module._precompute_rope_cache(
                    model, positive_high, device=device
                )
                initial_video, initial_audio = denoising_module._expand_initial_rows(
                    ctx, positive_high
                )
                capture: dict[str, torch.Tensor] = {}

                with (
                    maybe_nvtx_range("denoising_loop", self.current_use_nvtx),
                    self.progress_bar(
                        total=points - 1,
                        batch=batch,
                        desc="minimax_h3 hybrid 2+3",
                    ) as progress_bar,
                ):

                    def on_step(_step: int, _video_rows: torch.Tensor, _audio_rows: torch.Tensor) -> None:
                        progress_bar.update()
                        if not batch.is_warmup:
                            self.step_profile()

                    def head_forward(
                        forward_model: Any,
                        call_kwargs: dict[str, Any],
                        step: int,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        if step == config.teacher_steps - 1:
                            capture["x_before"] = call_kwargs["x"][0].index_select(
                                0, positive_high.img_target_seq_idx
                            ).float().clone()
                        video_velocity, audio_velocity = self._forward_dit(
                            forward_model,
                            call_kwargs,
                            step,
                            batch=batch,
                        )
                        if step == config.teacher_steps - 1:
                            capture["velocity"] = video_velocity.float().clone()
                        return video_velocity, audio_velocity

                    _sync_cuda()
                    head_started = time.perf_counter()
                    high_video_rows, high_audio_rows = minimax_h3_denoise_loop(
                        model=model,
                        model_forward=head_forward,
                        positive=positive_high,
                        initial_video_rows=initial_video,
                        initial_audio_rows=initial_audio,
                        keyframe_cond_rows=None,
                        audio_ref_rows=None,
                        sigmas_video=sigmas_video[: config.teacher_steps + 1],
                        sigmas_audio=sigmas_audio[: config.teacher_steps + 1],
                        device=device,
                        imgvid_cond_noise_aug_for_inference=float(imgvid_noise_aug),
                        audio_cond_noise_aug_for_inference=float(audio_noise_aug),
                        on_step=on_step,
                        step_profiler=partial(self._profile_denoising_step, batch=batch),
                    )
                    _sync_cuda()
                    telemetry["teacher_head_s"] = time.perf_counter() - head_started

                    if set(capture) != {"x_before", "velocity"}:
                        raise RuntimeError(f"teacher capture is incomplete: {sorted(capture)}")
                    sigma_prev = sigmas_video[config.teacher_steps - 1]
                    sigma_resume = sigmas_video[config.teacher_steps]
                    x0_rows = capture["x_before"] + sigma_prev * capture["velocity"]
                    resumed_target = high_video_rows[positive_high.video_target_slice]
                    if tuple(resumed_target.shape) != tuple(x0_rows.shape):
                        raise RuntimeError(
                            f"teacher row shape mismatch x={tuple(resumed_target.shape)} "
                            f"x0={tuple(x0_rows.shape)}"
                        )
                    if sigma_resume <= 0.0:
                        raise RuntimeError("resolution handoff requires a positive resume sigma")
                    eps_rows = (
                        resumed_target - (1.0 - sigma_resume) * x0_rows
                    ) / sigma_resume

                    _sync_cuda()
                    handoff_started = time.perf_counter()
                    low_initial_video, handoff_stats = _resolution_handoff(
                        video_vae=self._hybrid_video_vae,
                        x0_rows=x0_rows,
                        eps_rows=eps_rows,
                        latent_t=ctx.latent_t,
                        sigma_resume=sigma_resume,
                    )
                    _sync_cuda()
                    telemetry["handoff_s"] = time.perf_counter() - handoff_started
                    telemetry["handoff"] = handoff_stats

                    ctx.latent_h = LOW_LATENT_HEIGHT
                    ctx.latent_w = LOW_LATENT_WIDTH
                    packed_low = denoising_module._build_packed_layout(ctx, emb)
                    tags_low = packed_low["token_tags"]
                    tags_low[packed_low["text_pos"].view(-1)] = (
                        emb["text_token_tags"].view(-1).to(torch.long)
                    )
                    positive_low = MiniMaxH3DenoiseBranch(
                        packed=packed_low,
                        text_embeddings=emb["hidden_states"],
                        token_tags=tags_low,
                        device=device,
                    )
                    # Activate the student before text refinement.  The adapter
                    # covers eight token-refiner fused wrappers, so reusing the
                    # teacher's base-model refined embedding would silently make
                    # those eight mapped wrappers ineffective.
                    self._hybrid_pipeline.set_lora(
                        config.lora_nickname,
                        None,
                        target="transformer",
                        strength=config.lora_scale,
                        merge_mode="dynamic",
                    )
                    tail_active = _active_matched_lora_count(
                        self._hybrid_pipeline, config.lora_nickname
                    )
                    if tail_active != EXPECTED_MAPPED_LORA_LAYERS:
                        raise RuntimeError(
                            f"student tail has {tail_active} active LoRA layers; "
                            f"expected {EXPECTED_MAPPED_LORA_LAYERS}"
                        )
                    telemetry["tail_lora_active_layers"] = tail_active
                    denoising_module._precompute_refined_prompt_embeds(
                        model, positive_low, device=device
                    )
                    denoising_module._precompute_rope_cache(
                        model, positive_low, device=device
                    )
                    telemetry["student_refiner_after_lora_activation"] = True

                    def tail_forward(
                        forward_model: Any,
                        call_kwargs: dict[str, Any],
                        local_step: int,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        return self._forward_dit(
                            forward_model,
                            call_kwargs,
                            local_step + config.teacher_steps,
                            batch=batch,
                        )

                    def tail_profiler(local_step: int):
                        return self._profile_denoising_step(
                            local_step + config.teacher_steps,
                            batch=batch,
                        )

                    _sync_cuda()
                    tail_started = time.perf_counter()
                    low_video_rows, low_audio_rows = minimax_h3_denoise_loop(
                        model=model,
                        model_forward=tail_forward,
                        positive=positive_low,
                        initial_video_rows=low_initial_video,
                        initial_audio_rows=high_audio_rows,
                        keyframe_cond_rows=None,
                        audio_ref_rows=None,
                        sigmas_video=sigmas_video[config.teacher_steps :],
                        sigmas_audio=sigmas_audio[config.teacher_steps :],
                        device=device,
                        imgvid_cond_noise_aug_for_inference=float(imgvid_noise_aug),
                        audio_cond_noise_aug_for_inference=float(audio_noise_aug),
                        on_step=on_step,
                        step_profiler=tail_profiler,
                    )
                    _sync_cuda()
                    telemetry["student_tail_s"] = time.perf_counter() - tail_started
            finally:
                self._finish_active_component_use()

            denoising_module._publish_full_loop_outputs(
                ctx,
                batch=batch,
                positive=positive_low,
                video_rows=low_video_rows,
                audio_rows=low_audio_rows,
            )
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
                MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
            )

            state = batch.extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY]
            state["latent_h"] = LOW_LATENT_HEIGHT
            state["latent_w"] = LOW_LATENT_WIDTH
            state["initial_video_rows"] = low_initial_video
            batch.raw_latent_shape = (
                1,
                VIDEO_CHANNELS,
                ctx.latent_t,
                LOW_LATENT_HEIGHT,
                LOW_LATENT_WIDTH,
            )
            batch.height = LOW_PIXEL_HEIGHT
            batch.width = LOW_PIXEL_WIDTH
            output_validation = _validate_published_batch(
                batch,
                latent_t=ctx.latent_t,
                audio_t=ctx.audio_t,
            )
            telemetry["output"] = output_validation
            telemetry["denoise_and_handoff_total_s"] = time.perf_counter() - started_total
            batch.extra["minimax_h3_hybrid_telemetry"] = telemetry
            _append_telemetry(config.telemetry_path, telemetry)

    MiniMaxH3HybridDenoisingStage.__name__ = "MiniMaxH3HybridDenoisingStage"
    MiniMaxH3HybridDenoisingStage.__qualname__ = "MiniMaxH3HybridDenoisingStage"
    return MiniMaxH3HybridDenoisingStage


def install_hybrid_overlay(
    *,
    high_short_edge: int = 1080,
    low_short_edge: int = 480,
    teacher_steps: int = 2,
    student_steps: int = 3,
    lora_path: str,
    lora_nickname: str,
    lora_scale: float = 0.0625,
    video_flow_shift: float = 12.0,
    audio_flow_shift: float = 3.0,
    telemetry_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Install the process-local pinned hybrid implementation.

    Call this before ``DiffGenerator.from_pretrained`` and in every spawned
    process (normally by invoking it at driver module scope).
    """

    global _INSTALL_STATE
    # The pinned VAE keeps its input/output projection matmuls in float32.  This
    # deployment target favors throughput over strict FP32 mantissa accuracy,
    # so let GB200 Tensor Cores use fast float32 matmul in every spawned worker.
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    compile_enabled, compile_mode = _strict_compile_environment()
    config = HybridOverlayConfig(
        high_short_edge=int(high_short_edge),
        low_short_edge=int(low_short_edge),
        teacher_steps=int(teacher_steps),
        student_steps=int(student_steps),
        lora_path=str(lora_path),
        lora_nickname=str(lora_nickname),
        lora_scale=float(lora_scale),
        video_flow_shift=float(video_flow_shift),
        audio_flow_shift=float(audio_flow_shift),
        telemetry_path=(None if telemetry_path is None else str(telemetry_path)),
        torch_compile=compile_enabled,
        compile_mode=compile_mode,
    )
    _validate_config(config)
    if _INSTALL_STATE is not None:
        if _INSTALL_STATE["config"] != asdict(config):
            raise RuntimeError(
                f"a different MiniMax-H3 hybrid overlay is already installed: "
                f"{_INSTALL_STATE['config']}"
            )
        return dict(_INSTALL_STATE)

    resolution_overlay = install_1080p_overlay(config.high_short_edge)
    _install_low_resolution_delivery_validator()
    _install_lora_key_normalizer()
    _install_minimax_lora_name_mapping()
    _install_dynamic_merged_lora_support(compile_enabled=config.torch_compile)

    from sglang.multimodal_gen.runtime.pipelines import minimax_h3_pipeline

    current_stage = minimax_h3_pipeline.MiniMaxH3DenoisingStage
    if getattr(current_stage, "_h3_hybrid_overlay", False):
        raise RuntimeError("unexpected pre-installed hybrid denoising stage")
    hybrid_stage = _build_hybrid_stage_class(config)
    hybrid_stage._h3_hybrid_overlay = True
    hybrid_stage._h3_hybrid_stock_stage = current_stage
    minimax_h3_pipeline.MiniMaxH3DenoisingStage = hybrid_stage

    _INSTALL_STATE = {
        "installed": True,
        "name": "sglang_minimax_h3_t2_l3_480_overlay_v1",
        "pinned_sglang_commit": PINNED_SGLANG_COMMIT,
        "config": asdict(config),
        "resolution_overlay": resolution_overlay,
        "high_internal_pixels": [HIGH_PIXEL_HEIGHT, HIGH_PIXEL_WIDTH],
        "low_output_pixels": [LOW_PIXEL_HEIGHT, LOW_PIXEL_WIDTH],
        "expected_raw_lora_pairs": EXPECTED_RAW_LORA_PAIRS,
        "expected_mapped_lora_layers": EXPECTED_MAPPED_LORA_LAYERS,
        "output_is_low_resolution": True,
        "torch_compile": config.torch_compile,
        "compile_mode": config.compile_mode,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "compile_scope": {
            "dit": "SGLang native" if config.torch_compile else "eager",
            "dynamic_merged_lora": (
                "torch.compile" if config.torch_compile else "eager"
            ),
            "vae": (
                "local encoder/decoder only; collectives eager"
                if config.torch_compile
                else "eager"
            ),
        },
    }
    return dict(_INSTALL_STATE)


__all__ = [
    "EXPECTED_MAPPED_LORA_LAYERS",
    "EXPECTED_RAW_LORA_PAIRS",
    "HIGH_PIXEL_HEIGHT",
    "HIGH_PIXEL_WIDTH",
    "LOW_PIXEL_HEIGHT",
    "LOW_PIXEL_WIDTH",
    "PINNED_SGLANG_COMMIT",
    "install_hybrid_overlay",
    "validate_decoded_hybrid_output",
]
