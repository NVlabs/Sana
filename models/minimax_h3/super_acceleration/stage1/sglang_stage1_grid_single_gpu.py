#!/usr/bin/env python3
"""One persistent one-GB200 service for a Stage-1 grid cell family.

A process fixes model profile, canvas, and cache method, then evaluates all
five NFE values over the same sixteen first-frame prompts.  This amortizes
model loading and compile warmup across 80 retained videos.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any


PINNED_IMAGE = "docker://lmsysorg/sglang@sha256:71145ca99ebc458265e93cebd00b52bb9f419f052e7d0de09a54fa0f72fed888"
PINNED_SGLANG_COMMIT = "12eadf86f12aec2e6f81a6e38b61b964a4c6b529"
PINNED_MODEL_REVISION = "bfc8ed0353f5a9733be73e6b2c98ec0948195b86"
PINNED_SGLANG_MODEL_SHA256 = (
    "5f87319969c446685ee93d422fc34a7c040defb238eff2274d664f2f8310e997"
)
FPS = 24
FRAME_COUNT = 124
ALLOWED_STEPS = (4, 6, 8, 12, 16, 49)

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "teacher": {
        "lora": False,
        "filename": None,
        "expected_bytes": None,
        "nickname": None,
        "scale": None,
        "alpha": None,
        "rank": None,
        "distilled_nfe": None,
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "training_resolution": "native_teacher",
    },
    "lx2v_4s_v01_544p": {
        "lora": True,
        "filename": "minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
        "expected_bytes": 1_383_677_888,
        "nickname": "lx2v_4s_v01_544p",
        "scale": 8.0 / 128.0,
        "alpha": 8,
        "rank": 128,
        "distilled_nfe": 4,
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "training_resolution": "544p_mixed_aspect_ratio",
    },
    "lx2v_8s_v10_544p": {
        "lora": True,
        "filename": "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
        "expected_bytes": 1_383_677_768,
        "nickname": "lx2v_8s_v10_544p",
        "scale": 8.0 / 128.0,
        "alpha": 8,
        "rank": 128,
        "distilled_nfe": 8,
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "training_resolution": "544p_mixed_aspect_ratio",
    },
    "lx2v_4s_v10_768p": {
        "lora": True,
        "filename": "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
        "expected_bytes": 1_383_677_808,
        "nickname": "lx2v_4s_v10_768p",
        "scale": 1.0,
        "alpha": 128,
        "rank": 128,
        "distilled_nfe": 4,
        "video_shift": 6.0,
        "audio_shift": 3.0,
        "training_resolution": "768p",
    },
}
RESOLUTIONS = ((896, 512), (768, 448), (672, 384), (576, 320))
CACHE_MODES = ("none", "easy", "tea", "fb")


def _strict_bool_env(name: str, default: str) -> bool:
    raw = os.environ.get(name, default)
    if raw not in {"0", "1"}:
        raise RuntimeError(f"{name} must be exactly 0 or 1, got {raw!r}")
    return raw == "1"


PROCESS_ACTIVE = _strict_bool_env("H3_GRID_ACTIVE", "0")
PROCESS_PROFILE_NAME = os.environ.get("H3_GRID_MODEL_PROFILE", "teacher")
PROCESS_CACHE_MODE = os.environ.get("H3_GRID_CACHE_MODE", "none")
try:
    PROCESS_WIDTH = int(os.environ.get("H3_GRID_WIDTH", "896"))
    PROCESS_HEIGHT = int(os.environ.get("H3_GRID_HEIGHT", "512"))
except ValueError as exc:
    raise RuntimeError("H3_GRID_WIDTH/HEIGHT must be integers") from exc
PROCESS_COMPILE = _strict_bool_env("H3_GRID_COMPILE", "1")
PROCESS_TELEMETRY = os.environ.get("H3_GRID_TELEMETRY", "")
PROCESS_LORA_PATH = os.environ.get("H3_GRID_LORA_PATH", "").strip() or None
PROCESS_PROFILE = MODEL_PROFILES.get(PROCESS_PROFILE_NAME)
PROCESS_OVERLAY: dict[str, Any] | None = None

if PROCESS_ACTIVE:
    if PROCESS_PROFILE is None:
        raise RuntimeError(f"unknown model profile {PROCESS_PROFILE_NAME!r}")
    if (PROCESS_WIDTH, PROCESS_HEIGHT) not in RESOLUTIONS:
        raise RuntimeError(f"unsupported grid canvas {PROCESS_WIDTH}x{PROCESS_HEIGHT}")
    if PROCESS_CACHE_MODE not in CACHE_MODES:
        raise RuntimeError(f"unsupported cache mode {PROCESS_CACHE_MODE!r}")
    if not PROCESS_TELEMETRY:
        raise RuntimeError("H3_GRID_TELEMETRY is required")
    if bool(PROCESS_LORA_PATH) != bool(PROCESS_PROFILE["lora"]):
        raise RuntimeError("H3_GRID_LORA_PATH disagrees with the model profile")

    from sglang_h3_stage1_grid_overlay import install_stage1_grid_overlay

    stage = install_stage1_grid_overlay(
        model_profile=PROCESS_PROFILE_NAME,
        width=PROCESS_WIDTH,
        height=PROCESS_HEIGHT,
        cache_mode=PROCESS_CACHE_MODE,
        telemetry_path=PROCESS_TELEMETRY,
        compile_enabled=PROCESS_COMPILE,
        lora_path=PROCESS_LORA_PATH,
        lora_nickname=PROCESS_PROFILE["nickname"],
        lora_scale=PROCESS_PROFILE["scale"],
        lora_merge_mode="merge",
        video_shift=float(PROCESS_PROFILE["video_shift"]),
        audio_shift=float(PROCESS_PROFILE["audio_shift"]),
        distilled_nfe=PROCESS_PROFILE["distilled_nfe"],
    )
    from sglang_h3_single_gpu_compile_overlay import (
        install_single_gpu_vae_compile_overlay,
    )

    vae_compile = install_single_gpu_vae_compile_overlay(
        arm="teacher" if PROCESS_PROFILE_NAME == "teacher" else "student",
        enabled=PROCESS_COMPILE,
        mode=os.environ.get(
            "SGLANG_VAE_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
        ),
    )
    # Importing this helper is safe because H3_FF_ARM is intentionally unset;
    # the legacy runner therefore installs no competing process overlay.
    from sglang_firstframe_single_gpu import _install_fail_closed_zmq_bind

    zmq = _install_fail_closed_zmq_bind()
    PROCESS_OVERLAY = {"stage": stage, "vae_compile": vae_compile, "zmq": zmq}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_int_list(raw: str, *, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be a comma-separated integer list") from exc
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{label} must be non-empty and contain no duplicates")
    return values


def _load_manifest(path: Path, indices: tuple[int, ...]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("conditioning") != "first_frame_only":
        raise ValueError("manifest conditioning must be first_frame_only")
    if payload.get("reference_conditions") is not False:
        raise ValueError("manifest must disable reference conditions")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("manifest items must be a list")
    selected = []
    for index in indices:
        if index < 0 or index >= len(items):
            raise ValueError(f"prompt index {index} is outside the manifest")
        item = dict(items[index])
        if int(item.get("index", -1)) != index:
            raise ValueError(f"manifest item {index} has a mismatched index")
        if item.get("first_frame_role") != "first_frame":
            raise ValueError(f"manifest item {index} is not a first frame")
        if item.get("reference_conditions") != []:
            raise ValueError(f"manifest item {index} contains references")
        if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
            raise ValueError(f"manifest item {index} has no prompt")
        selected.append(item)
    return selected


def _prepare_images(
    *,
    items: list[dict[str, Any]],
    source_asset_root: Path,
    output_dir: Path,
    width: int,
    height: int,
) -> dict[int, Path]:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared: dict[int, Path] = {}
    for item in items:
        index = int(item["index"])
        source_image = item.get("source_image")
        recorded = source_image if source_image is not None else item.get("student_image")
        if not isinstance(recorded, str) or not Path(recorded).name:
            raise ValueError(f"manifest item {index} has no source/student image")
        name = Path(recorded).name
        if source_image is not None:
            source = Path(recorded)
            if not source.is_absolute():
                source = source_asset_root / source
        else:
            # Preserve the established grid-manifest layout.
            source = source_asset_root / "firstframes_896x512" / name
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"source first frame is unavailable: {source}")
        target = output_dir / name
        if not target.is_file() or target.stat().st_size <= 0:
            with Image.open(source) as image:
                image.convert("RGB").resize(
                    (width, height), Image.Resampling.LANCZOS
                ).save(target)
        with Image.open(target) as image:
            if tuple(image.size) != (width, height):
                raise ValueError(f"prepared image {target} has size {image.size}")
        prepared[index] = target.resolve()
    return prepared


def _validate_runtime(lora_path: Path | None) -> dict[str, Any]:
    import torch
    import triton

    if torch.__version__ != os.environ.get("H3_EXPECTED_TORCH", "2.11.0+cu130"):
        raise RuntimeError(f"unexpected torch {torch.__version__}")
    if triton.__version__ != os.environ.get("H3_EXPECTED_TRITON", "3.6.0"):
        raise RuntimeError(f"unexpected Triton {triton.__version__}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"grid worker sees {torch.cuda.device_count()} GPUs, expected 1")
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability != (10, 0):
        raise RuntimeError(f"GPU capability is {capability}, expected SM100")
    module = "sglang.multimodal_gen.runtime.models.dits.minimax_h3"
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"pinned SGLang module is unavailable: {module}")
    model_source = Path(spec.origin)
    source_sha = hashlib.sha256(model_source.read_bytes()).hexdigest()
    if source_sha != PINNED_SGLANG_MODEL_SHA256:
        raise RuntimeError(f"unexpected H3 source SHA-256 {source_sha}")
    if PROCESS_PROFILE["lora"]:
        if lora_path is None or not lora_path.is_file():
            raise FileNotFoundError(f"LoRA unavailable: {lora_path}")
        if lora_path.name != PROCESS_PROFILE["filename"]:
            raise RuntimeError(f"wrong LoRA filename {lora_path.name}")
        if lora_path.stat().st_size != PROCESS_PROFILE["expected_bytes"]:
            raise RuntimeError(f"wrong LoRA byte size {lora_path.stat().st_size}")
    elif lora_path is not None:
        raise RuntimeError(f"Teacher must not receive a LoRA path: {lora_path}")
    return {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(capability),
        "visible_gpu_count": torch.cuda.device_count(),
        "upstream_model_source": str(model_source),
        "upstream_model_sha256": source_sha,
        "lora_bytes": None if lora_path is None else lora_path.stat().st_size,
    }


def _sampling_params(
    *,
    item: dict[str, Any],
    image_path: Path,
    nfe: int,
    output_dir: Path,
    output_name: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "prompt": item["prompt"],
        "task": "fl2va",
        "conditions": [
            {
                "type": "image",
                "uri": str(image_path),
                "role": "keyframe",
                "frame_index": 0,
            }
        ],
        "target": {
            "short_edge": 512,
            "aspect_ratio": f"{PROCESS_WIDTH}:{PROCESS_HEIGHT}",
            "duration_seconds": 5.0,
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": nfe + 1,
        "flow_shift": float(PROCESS_PROFILE["video_shift"]),
        "audio_flow_shift": float(PROCESS_PROFILE["audio_shift"]),
        "seed": int(seed),
        "output_path": str(output_dir),
        "output_file_name": output_name,
        "save_output": True,
        "return_file_paths_only": True,
    }


def _run_request(generator: Any, request: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    result = generator.generate(sampling_params_kwargs=request)
    outer = time.perf_counter() - started
    if result is None or isinstance(result, list):
        raise RuntimeError(f"expected one generation result, got {result!r}")
    metrics = dict(result.metrics or {})
    if "total_duration_s" in metrics:
        inference = float(metrics["total_duration_s"])
    elif "total_duration_ms" in metrics:
        inference = float(metrics["total_duration_ms"]) / 1000.0
    else:
        inference = float(result.generation_time)
    return {
        "inference_time_s": inference,
        "outer_generate_wall_s": outer,
        "generation_wall_s_reported": float(result.generation_time),
        "peak_memory_mb": _json_safe(result.peak_memory_mb),
        "output_file": result.output_file_path,
        "metrics": _json_safe(metrics),
    }


def _probe_media(path: Path, ffprobe: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"generated video is unavailable: {path}")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_frames,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout).get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"ffprobe found {len(streams)} streams in {path}")
    return dict(streams[0])


def _last_telemetry(path: Path, before: int) -> tuple[int, dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    if len(lines) != before + 1:
        raise RuntimeError(
            f"cache telemetry advanced from {before} to {len(lines)}, expected one row"
        )
    return len(lines), dict(json.loads(lines[-1]))


def _flatten_residual_diffs(value: Any, *, location: str = "residual_diffs") -> list[float]:
    if isinstance(value, bool):
        raise RuntimeError(f"{location} contains a boolean")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise RuntimeError(
                f"{location} contains invalid Cache-DiT residual diff {value!r}"
            )
        return [number]
    if isinstance(value, dict):
        flattened: list[float] = []
        for key, item in value.items():
            flattened.extend(
                _flatten_residual_diffs(item, location=f"{location}[{key!r}]")
            )
        return flattened
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for index, item in enumerate(value):
            flattened.extend(
                _flatten_residual_diffs(item, location=f"{location}[{index}]")
            )
        return flattened
    raise RuntimeError(
        f"{location} must contain only numeric residual diffs, got {type(value).__name__}"
    )


def _validate_retained_fb_residuals(
    telemetry: dict[str, Any],
) -> dict[str, float | int | None]:
    if "residual_diffs_error" in telemetry:
        raise RuntimeError(
            f"Cache-DiT residual telemetry failed: {telemetry['residual_diffs_error']}"
        )
    if "residual_diffs" not in telemetry:
        raise RuntimeError("retained FBCache request has no residual_diffs telemetry")
    values = _flatten_residual_diffs(telemetry["residual_diffs"])
    if not values:
        # A zero-threshold correctness control can legitimately perform no
        # reuse, in which case Cache-DiT reports no accepted residual diffs.
        # Empty telemetry is invalid only if a cache hit actually occurred.
        if int(telemetry.get("cached_steps", -1)) != 0:
            raise RuntimeError(
                "retained FBCache request reused cache with no residual diffs"
            )
        return {
            "residual_diff_count": 0,
            "residual_diff_min": None,
            "residual_diff_max": None,
        }
    return {
        "residual_diff_count": len(values),
        "residual_diff_min": min(values),
        "residual_diff_max": max(values),
    }


def _cache_counts(
    telemetry: dict[str, Any], nfe: int, *, retained: bool = False
) -> dict[str, Any]:
    scheduled = int(telemetry.get("scheduled_steps", nfe))
    if scheduled != nfe:
        raise RuntimeError(f"cache telemetry scheduled {scheduled} steps, expected {nfe}")
    compute_value = (
        telemetry.get("full_stack_forwards")
        if PROCESS_CACHE_MODE == "fb"
        else telemetry.get("computed_forwards")
    )
    reuse_value = telemetry.get("cached_steps")
    if PROCESS_CACHE_MODE == "fb" and telemetry.get("telemetry_available") is not True:
        raise RuntimeError("native FirstBlockCache telemetry is unavailable")
    if not isinstance(compute_value, int) or not isinstance(reuse_value, int):
        raise RuntimeError(
            f"cache telemetry has non-integer compute/reuse counts: {telemetry}"
        )
    compute = int(compute_value)
    reuse = int(reuse_value)
    if compute < 0 or reuse < 0 or compute + reuse != nfe:
        raise RuntimeError(
            f"cache telemetry compute={compute} reuse={reuse} does not sum to NFE={nfe}"
        )
    result: dict[str, Any] = {"calls": nfe, "compute": compute, "reuse": reuse}
    if PROCESS_CACHE_MODE == "fb":
        if int(telemetry.get("head_block_forwards", -1)) != nfe:
            raise RuntimeError(
                f"FBCache head block count is not the scheduled NFE {nfe}"
            )
        if retained:
            result.update(_validate_retained_fb_residuals(telemetry))
    return result


def _validate_model_binding(telemetry: dict[str, Any]) -> dict[str, Any]:
    binding = telemetry.get("model_binding")
    if not isinstance(binding, dict):
        raise RuntimeError("request telemetry has no live model-binding audit")
    if binding.get("model_profile") != PROCESS_PROFILE_NAME:
        raise RuntimeError(
            f"runtime model profile {binding.get('model_profile')!r} != "
            f"{PROCESS_PROFILE_NAME!r}"
        )
    applied = binding.get("lora_applied")
    coverage = binding.get("lora_coverage")
    teacher_audit = binding.get("teacher_lora_audit")
    if PROCESS_PROFILE["lora"]:
        if applied is not True or not isinstance(coverage, dict):
            raise RuntimeError(f"LoRA profile is not active in the live pipeline: {binding}")
        expected_layers = int(PROCESS_OVERLAY["stage"]["expected_lora_layers"])
        if (
            int(coverage.get("mapped_layers", -1)) != expected_layers
            or int(coverage.get("merged_layers", -1)) != expected_layers
            or int(coverage.get("active_dynamic_layers", -1)) != 0
            or coverage.get("merge_mode") != "merge"
        ):
            raise RuntimeError(f"live LoRA coverage is incomplete: {coverage}")
        if binding.get("lora_path") != PROCESS_LORA_PATH:
            raise RuntimeError("live LoRA path disagrees with the process contract")
        if binding.get("lora_nickname") != PROCESS_PROFILE["nickname"]:
            raise RuntimeError("live LoRA nickname disagrees with the process contract")
        if teacher_audit is not None:
            raise RuntimeError("LoRA profile unexpectedly contains a Teacher audit")
    else:
        if applied is not False or coverage is not None or not isinstance(teacher_audit, dict):
            raise RuntimeError(f"Teacher model-binding audit failed: {binding}")
        if (
            teacher_audit.get("lora_path") is not None
            or int(teacher_audit.get("adapter_count", -1)) != 0
            or teacher_audit.get("any_component_merged") is not False
        ):
            raise RuntimeError(f"Teacher contains LoRA state: {teacher_audit}")
    return binding


def _load_completed_record(
    *,
    record_path: Path,
    output: Path,
    item: dict[str, Any],
    image_path: Path,
    nfe: int,
) -> dict[str, Any] | None:
    """Return a resumable record only when the complete cell contract matches.

    A non-empty MP4 plus an arbitrary JSON file is not sufficient: a previous
    interrupted or differently configured sweep may have left both behind.
    Invalid pairs are regenerated in place by the normal request path.
    """

    if not output.is_file() or output.stat().st_size <= 0 or not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    index = int(item["index"])
    prompt_id = str(item.get("id") or f"prompt_{index:02d}")
    expected_sha = hashlib.sha256(item["prompt"].encode()).hexdigest()
    expected = {
        "schema_version": 2,
        "model_profile": PROCESS_PROFILE_NAME,
        "cache_mode": PROCESS_CACHE_MODE,
        "width": PROCESS_WIDTH,
        "height": PROCESS_HEIGHT,
        "nfe": nfe,
        "prompt_index": index,
        "prompt_id": prompt_id,
        "prompt_sha256": expected_sha,
        "seed": int(item["seed"]),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        return None
    if Path(str(record.get("first_frame", ""))).resolve() != image_path.resolve():
        return None
    media = record.get("media")
    if not isinstance(media, dict):
        return None
    if Path(str(media.get("path", ""))).resolve() != output.resolve():
        return None
    probe = media.get("probe")
    if not isinstance(probe, dict):
        return None
    try:
        shape = (int(probe["width"]), int(probe["height"]))
        frames = int(probe["nb_read_frames"])
        cache_counts = record["cache_telemetry"]
        calls = int(cache_counts["calls"])
        compute = int(cache_counts["compute"])
        reuse = int(cache_counts["reuse"])
    except (KeyError, TypeError, ValueError):
        return None
    if shape != (PROCESS_WIDTH, PROCESS_HEIGHT) or frames != FRAME_COUNT:
        return None
    if calls != nfe or compute < 0 or reuse < 0 or compute + reuse != nfe:
        return None
    denoise = record.get("denoise_time_s")
    if not isinstance(denoise, (int, float)) or isinstance(denoise, bool) or denoise <= 0:
        return None
    telemetry = record.get("telemetry")
    if not isinstance(telemetry, dict):
        return None
    try:
        validated_counts = _cache_counts(telemetry, nfe, retained=True)
        live_binding = _validate_model_binding(telemetry)
    except (KeyError, TypeError, ValueError, RuntimeError):
        return None
    if any(
        int(cache_counts.get(key, -1)) != int(validated_counts[key])
        for key in ("calls", "compute", "reuse")
    ):
        return None
    if record.get("model_binding") != live_binding:
        return None
    return record


def _validate_ports(generator: Any, ports: tuple[int, int, int, int]) -> dict[str, Any]:
    args = generator.server_args
    actual = (int(args.port), int(args.scheduler_port), int(args.master_port), int(args.nccl_port))
    if actual != ports or args.strict_ports is not True:
        raise RuntimeError(f"SGLang ports {actual}/strict={args.strict_ports} != {ports}/True")
    return {
        "http": actual[0],
        "scheduler": actual[1],
        "master": actual[2],
        "nccl": actual[3],
        "strict": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-asset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-subfolder", default="FL2VA")
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--lora-path", type=Path)
    parser.add_argument("--prompt-indices", default=",".join(str(i) for i in range(16)))
    parser.add_argument("--steps", default=",".join(str(i) for i in ALLOWED_STEPS))
    parser.add_argument("--http-port", type=int, required=True)
    parser.add_argument("--scheduler-port", type=int, required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--nccl-port", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, default=16)
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    if not PROCESS_ACTIVE:
        parser.error("H3_GRID_ACTIVE=1 is required")
    if args.model_revision != PINNED_MODEL_REVISION:
        parser.error(f"model revision is pinned to {PINNED_MODEL_REVISION}")
    prompt_indices = _parse_int_list(args.prompt_indices, label="prompt indices")
    steps = _parse_int_list(args.steps, label="steps")
    if any(step not in ALLOWED_STEPS for step in steps):
        parser.error(f"steps must be selected from {ALLOWED_STEPS}")
    minimum_warmups = max(2, len(prompt_indices))
    if args.warmup_requests < minimum_warmups:
        parser.error(
            f"at least {minimum_warmups} warmups are required: two heat VAE "
            "encode/decode and every distinct prompt shape must enter the compiled DiT"
        )
    ports = (args.http_port, args.scheduler_port, args.master_port, args.nccl_port)
    if len(set(ports)) != 4 or any(not 1024 <= port <= 65535 for port in ports):
        parser.error("service ports must be distinct values in [1024,65535]")
    args.manifest = args.manifest.resolve()
    args.source_asset_root = args.source_asset_root.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    args.lora_path = None if args.lora_path is None else args.lora_path.resolve()
    if args.lora_path != (None if PROCESS_LORA_PATH is None else Path(PROCESS_LORA_PATH).resolve()):
        parser.error("--lora-path disagrees with H3_GRID_LORA_PATH")

    items = _load_manifest(args.manifest, prompt_indices)
    images = _prepare_images(
        items=items,
        source_asset_root=args.source_asset_root,
        output_dir=args.out / "_inputs",
        width=PROCESS_WIDTH,
        height=PROCESS_HEIGHT,
    )
    runtime = _validate_runtime(args.lora_path)
    telemetry_path = Path(PROCESS_TELEMETRY)
    telemetry_before = (
        len(telemetry_path.read_text(encoding="utf-8").splitlines())
        if telemetry_path.is_file()
        else 0
    )

    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

    generator_kwargs: dict[str, Any] = {
        "local_mode": True,
        "model_path": args.model_path,
        "model_subfolder": args.model_subfolder,
        "model_variant": "fl2va",
        "revision": args.model_revision,
        "num_gpus": 1,
        "tp_size": 1,
        "ulysses_degree": 1,
        "ring_degree": 1,
        "enable_cfg_parallel": False,
        "performance_mode": "speed",
        "use_fsdp_inference": False,
        "layerwise_offload_components": [],
        "dit_cpu_offload": False,
        "vae_cpu_offload": False,
        "enable_torch_compile": PROCESS_COMPILE,
        "regional_compile": False,
        "enable_breakable_cuda_graph": False,
        "offload_during_compile": False,
        "warmup_mode": "off",
        "warmup": False,
        "server_warmup": False,
        "port": ports[0],
        "scheduler_port": ports[1],
        "master_port": ports[2],
        "nccl_port": ports[3],
        "strict_ports": True,
    }
    if PROCESS_PROFILE["lora"]:
        generator_kwargs.update(
            {
                "lora_path": str(args.lora_path),
                "lora_nickname": str(PROCESS_PROFILE["nickname"]),
                "lora_scale": float(PROCESS_PROFILE["scale"]),
                "lora_merge_mode": "merge",
                "lora_target_modules": ["qkv_proj", "out_proj", "mlp.fc1", "mlp.fc2"],
            }
        )

    _write_json(
        args.out / "service_config.json",
        {
            "schema_version": 2,
            "status": "loading",
            "model_profile": PROCESS_PROFILE_NAME,
            "model": PROCESS_PROFILE,
            "cache_mode": PROCESS_CACHE_MODE,
            "width": PROCESS_WIDTH,
            "height": PROCESS_HEIGHT,
            "steps": list(steps),
            "prompt_indices": list(prompt_indices),
            "expected_retained_videos": len(steps) * len(prompt_indices),
            "compile": PROCESS_COMPILE,
            "overlay": PROCESS_OVERLAY,
        },
    )
    load_started = time.perf_counter()
    generator = DiffGenerator.from_pretrained(**generator_kwargs)
    load_s = time.perf_counter() - load_started
    service_ports = _validate_ports(generator, ports)
    warmups: list[dict[str, Any]] = []
    completed_records: list[dict[str, Any]] = []
    try:
        warmup_dir = args.out / "_warmup"
        warmup_dir.mkdir(parents=True, exist_ok=True)
        warmup_nfe = steps[0]
        for ordinal in range(args.warmup_requests):
            warmup_item = items[ordinal % len(items)]
            output_name = f"warmup_{ordinal:02d}.mp4"
            request = _sampling_params(
                item=warmup_item,
                image_path=images[int(warmup_item["index"])],
                nfe=warmup_nfe,
                output_dir=warmup_dir,
                output_name=output_name,
                seed=int(warmup_item["seed"]) + 100_000 + ordinal,
            )
            record = _run_request(generator, request)
            telemetry_before, telemetry = _last_telemetry(telemetry_path, telemetry_before)
            record.update(
                {
                    "excluded": True,
                    "phase": "warmup",
                    "model_profile": PROCESS_PROFILE_NAME,
                    "cache_mode": PROCESS_CACHE_MODE,
                    "width": PROCESS_WIDTH,
                    "height": PROCESS_HEIGHT,
                    "nfe": warmup_nfe,
                    "prompt_index": int(warmup_item["index"]),
                    "prompt_id": str(
                        warmup_item.get("id")
                        or f"prompt_{int(warmup_item['index']):02d}"
                    ),
                    "prompt_sha256": hashlib.sha256(
                        warmup_item["prompt"].encode()
                    ).hexdigest(),
                    "seed": int(warmup_item["seed"]) + 100_000 + ordinal,
                    "first_frame": str(images[int(warmup_item["index"])]),
                    "telemetry": telemetry,
                    "denoise_time_s": float(telemetry["denoise_total_s"]),
                    "cache_telemetry": _cache_counts(telemetry, warmup_nfe),
                    "model_binding": _validate_model_binding(telemetry),
                }
            )
            warmups.append(record)

        for nfe in steps:
            for item in items:
                index = int(item["index"])
                prompt_id = str(item.get("id") or f"prompt_{index:02d}")
                video_dir = args.out / "videos" / prompt_id
                record_dir = args.out / "records" / prompt_id
                video_dir.mkdir(parents=True, exist_ok=True)
                record_dir.mkdir(parents=True, exist_ok=True)
                output = video_dir / f"nfe_{nfe}.mp4"
                record_path = record_dir / f"nfe_{nfe}.json"
                completed = _load_completed_record(
                    record_path=record_path,
                    output=output,
                    item=item,
                    image_path=images[index],
                    nfe=nfe,
                )
                if completed is not None:
                    completed_records.append(completed)
                    continue
                request = _sampling_params(
                    item=item,
                    image_path=images[index],
                    nfe=nfe,
                    output_dir=video_dir,
                    output_name=output.name,
                    seed=int(item["seed"]),
                )
                record = _run_request(generator, request)
                telemetry_before, telemetry = _last_telemetry(telemetry_path, telemetry_before)
                probe = _probe_media(output, args.ffprobe)
                actual_shape = (int(probe["width"]), int(probe["height"]))
                if actual_shape != (PROCESS_WIDTH, PROCESS_HEIGHT):
                    raise RuntimeError(f"output {output} has shape {actual_shape}")
                if int(probe.get("nb_read_frames") or 0) != FRAME_COUNT:
                    raise RuntimeError(f"output {output} has {probe.get('nb_read_frames')} frames")
                record.update(
                    {
                        "schema_version": 2,
                        "model_profile": PROCESS_PROFILE_NAME,
                        "cache_mode": PROCESS_CACHE_MODE,
                        "width": PROCESS_WIDTH,
                        "height": PROCESS_HEIGHT,
                        "nfe": nfe,
                        "prompt_index": index,
                        "prompt_id": prompt_id,
                        "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
                        "seed": int(item["seed"]),
                        "first_frame": str(images[index]),
                        "media": {"path": str(output), "probe": probe},
                        "telemetry": telemetry,
                        "denoise_time_s": float(telemetry["denoise_total_s"]),
                        "cache_telemetry": _cache_counts(
                            telemetry, nfe, retained=True
                        ),
                        "model_binding": _validate_model_binding(telemetry),
                    }
                )
                _write_json(record_path, record)
                completed_records.append(record)
                _write_json(
                    args.out / "progress.json",
                    {
                        "status": "running",
                        "completed": len(completed_records),
                        "expected": len(steps) * len(items),
                        "last": {"nfe": nfe, "prompt_index": index},
                    },
                )
    finally:
        generator.shutdown()

    expected = len(steps) * len(items)
    unique = {(int(row["nfe"]), int(row["prompt_index"])) for row in completed_records}
    if len(completed_records) != expected or len(unique) != expected:
        raise RuntimeError(
            f"service retained {len(completed_records)} rows/{len(unique)} unique, expected {expected}"
        )
    summary = {
        "schema_version": 2,
        "status": "complete",
        "kind": "sglang_minimax_h3_stage1_grid_single_gb200_compiled_hot",
        "framework": {
            "name": "SGLang multimodal_gen",
            "commit": PINNED_SGLANG_COMMIT,
            "container": os.environ.get("H3_CONTAINER_IMAGE", PINNED_IMAGE),
            "overlay": PROCESS_OVERLAY,
        },
        "runtime": runtime,
        "model_profile": PROCESS_PROFILE_NAME,
        "model": {**PROCESS_PROFILE, "lora_path": None if args.lora_path is None else str(args.lora_path)},
        "cache_mode": PROCESS_CACHE_MODE,
        "workload": {
            "task": "fl2va",
            "conditioning": "first_frame_only",
            "reference_conditions": False,
            "width": PROCESS_WIDTH,
            "height": PROCESS_HEIGHT,
            "frames": FRAME_COUNT,
            "fps": FPS,
            "duration_seconds": 5.0,
            "steps": list(steps),
            "prompt_indices": list(prompt_indices),
            "retained_videos": expected,
        },
        "topology": {
            "num_gpus_per_service": 1,
            "tp_size": 1,
            "ulysses_degree": 1,
            "torch_compile": PROCESS_COMPILE,
            "compile_mode": os.environ.get("SGLANG_TORCH_COMPILE_MODE"),
            "vae_compile_mode": os.environ.get("SGLANG_VAE_TORCH_COMPILE_MODE"),
            "lora_merge_mode": "merge" if PROCESS_PROFILE["lora"] else None,
            "ports": service_ports,
        },
        "timing": {"model_load_s_excluded": load_s, "warmups_excluded": warmups},
        "records": completed_records,
    }
    _write_json(args.out / "benchmark.json", summary)
    _write_json(args.out / "progress.json", {"status": "complete", "completed": expected, "expected": expected})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
