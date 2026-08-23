#!/usr/bin/env python3
"""Single-GB200 SGLang FL2VA teacher/student hot-inference benchmark.

One invocation owns one visible GPU and one persistent SGLang generator.  The
teacher arm is native 1344x768 with 50 sigma points (49 forwards).  The student
arm is 896x512 with a pinned LightX2V four- or eight-NFE LoRA profile.
Both are first-frame-only FL2VA requests lasting five seconds (124 frames).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
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
PINNED_SGLANG_SCHEDULER_SHA256 = (
    "a3bb7f479cc45b58846fab4e80e191b447273e11db1ea0744a534d961cd90861"
)
PINNED_SGLANG_ZMQ_COMMON_SHA256 = (
    "dfc7c9d3ac8d8bb029de76b3a9d574ed3772f17400575435bcc01526037627d6"
)
LORA_PROFILES: dict[str, dict[str, Any]] = {
    "lx2v_4s_v01_544p": {
        "nickname": "lx2v_4s_v01_544p",
        "filename": "minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
        "expected_bytes": 1_383_677_888,
        "source_revision": "050494d5fe05bd1b1140b8565ea51dc33a5085a5",
        "training_resolution": "544p_mixed_aspect_ratio",
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "distilled_nfe": 4,
        "recommended_nfe": (4,),
        "rank": 128,
        "alpha": 8,
        "scale": 8.0 / 128.0,
    },
    "lx2v_8s_v10_544p": {
        "nickname": "lx2v_8s_v10_544p",
        "filename": "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
        "expected_bytes": 1_383_677_768,
        "source_revision": "5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1",
        "training_resolution": "544p_mixed_aspect_ratio",
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "distilled_nfe": 8,
        "recommended_nfe": (8, 4),
        "rank": 128,
        "alpha": 8,
        "scale": 8.0 / 128.0,
    },
}
PROCESS_LORA_PROFILE_NAME = os.environ.get(
    "H3_FF_LORA_PROFILE", "lx2v_4s_v01_544p"
)
if PROCESS_LORA_PROFILE_NAME not in LORA_PROFILES:
    raise RuntimeError(
        f"unknown H3_FF_LORA_PROFILE={PROCESS_LORA_PROFILE_NAME!r}; "
        f"expected one of {sorted(LORA_PROFILES)}"
    )
PROCESS_LORA_PROFILE = LORA_PROFILES[PROCESS_LORA_PROFILE_NAME]
FPS = 24
FRAME_COUNT = 124


def _make_fail_closed_scheduler_zmq_bind(
    original: Any, expected_scheduler_port: int
) -> Any:
    """Forbid SGLang's unsafe local-mode alternate-port retry."""

    def fail_closed_scheduler_zmq_bind(
        context: Any,
        socket_type: Any,
        endpoint: str,
        bind: bool,
        max_bind_retries: int = 10,
        same_port: bool = False,
    ) -> tuple[Any, str]:
        del max_bind_retries, same_port
        if not bind or not endpoint.startswith("tcp://"):
            raise RuntimeError(
                f"unexpected Scheduler ZMQ operation: bind={bind}, endpoint={endpoint}"
            )
        try:
            actual_requested_port = int(endpoint.rsplit(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"invalid Scheduler TCP endpoint: {endpoint}") from exc
        if actual_requested_port != expected_scheduler_port:
            raise RuntimeError(
                f"unexpected Scheduler endpoint {endpoint}; expected port "
                f"{expected_scheduler_port}"
            )
        socket, actual_endpoint = original(
            context,
            socket_type,
            endpoint,
            True,
            max_bind_retries=1,
            same_port=True,
        )
        if actual_endpoint != endpoint:
            raise RuntimeError(
                f"Scheduler ZMQ bind moved from {endpoint} to {actual_endpoint}"
            )
        return socket, actual_endpoint

    fail_closed_scheduler_zmq_bind._h3_fail_closed = True  # type: ignore[attr-defined]
    fail_closed_scheduler_zmq_bind._h3_scheduler_port = (  # type: ignore[attr-defined]
        expected_scheduler_port
    )
    return fail_closed_scheduler_zmq_bind


def _install_fail_closed_zmq_bind() -> dict[str, Any]:
    scheduler_port_value = os.environ.get("H3_SCHEDULER_PORT")
    if scheduler_port_value is None:
        raise RuntimeError("H3_SCHEDULER_PORT is required")
    expected_scheduler_port = int(scheduler_port_value)
    from sglang.multimodal_gen.runtime.managers import scheduler as scheduler_module
    from sglang.multimodal_gen.runtime.utils import common as common_utils

    scheduler_path = Path(str(scheduler_module.__file__)).resolve(strict=True)
    common_path = Path(str(common_utils.__file__)).resolve(strict=True)
    scheduler_sha256 = hashlib.sha256(scheduler_path.read_bytes()).hexdigest()
    common_sha256 = hashlib.sha256(common_path.read_bytes()).hexdigest()
    if scheduler_sha256 != PINNED_SGLANG_SCHEDULER_SHA256:
        raise RuntimeError(f"unexpected Scheduler source SHA-256 {scheduler_sha256}")
    if common_sha256 != PINNED_SGLANG_ZMQ_COMMON_SHA256:
        raise RuntimeError(f"unexpected ZMQ helper source SHA-256 {common_sha256}")
    helper_signature = inspect.signature(common_utils.get_zmq_socket)
    parameters = list(helper_signature.parameters.values())
    expected_names = [
        "context",
        "socket_type",
        "endpoint",
        "bind",
        "max_bind_retries",
        "same_port",
    ]
    if [parameter.name for parameter in parameters] != expected_names:
        raise RuntimeError(f"unexpected get_zmq_socket signature {helper_signature}")
    if any(
        parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in parameters
    ):
        raise RuntimeError(f"unexpected get_zmq_socket signature {helper_signature}")
    if any(
        parameter.default is not inspect.Parameter.empty
        for parameter in parameters[:4]
    ) or [parameter.default for parameter in parameters[4:]] != [10, False]:
        raise RuntimeError(f"unexpected get_zmq_socket defaults {helper_signature}")
    original = scheduler_module.get_zmq_socket
    if getattr(original, "_h3_fail_closed", False):
        if getattr(original, "_h3_scheduler_port", None) != expected_scheduler_port:
            raise RuntimeError("Scheduler ZMQ wrapper targets a different port")
    else:
        if original is not common_utils.get_zmq_socket:
            raise RuntimeError("Scheduler get_zmq_socket is not the pinned symbol")
        scheduler_module.get_zmq_socket = _make_fail_closed_scheduler_zmq_bind(
            original, expected_scheduler_port
        )
    return {
        "installed": True,
        "scheduler_port": expected_scheduler_port,
        "max_bind_retries": 1,
        "same_port": True,
        "alternate_port_retry": False,
        "scheduler_source_sha256": scheduler_sha256,
        "common_source_sha256": common_sha256,
    }


def _strict_bool_env(name: str, default: str) -> bool:
    raw = os.environ.get(name, default)
    if raw not in {"0", "1"}:
        raise RuntimeError(f"{name} must be exactly 0 or 1, got {raw!r}")
    return raw == "1"


def _student_forwards_from_environment() -> tuple[int, bool]:
    raw = os.environ.get(
        "H3_FF_STUDENT_FORWARDS", str(PROCESS_LORA_PROFILE["distilled_nfe"])
    )
    try:
        forwards = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"H3_FF_STUDENT_FORWARDS must be an integer, got {raw!r}"
        ) from exc
    if forwards < 1:
        raise RuntimeError(f"H3_FF_STUDENT_FORWARDS must be positive, got {forwards}")
    allow_offgrid = _strict_bool_env("H3_FF_ALLOW_OFFGRID_STEPS", "0")
    recommended = tuple(int(value) for value in PROCESS_LORA_PROFILE["recommended_nfe"])
    if forwards not in recommended and not allow_offgrid:
        raise RuntimeError(
            f"{forwards}-forward student is outside the "
            f"{PROCESS_LORA_PROFILE_NAME} recommended NFE set {recommended}; set "
            "H3_FF_ALLOW_OFFGRID_STEPS=1 to acknowledge the experiment"
        )
    if forwards in recommended and allow_offgrid:
        raise RuntimeError(
            "H3_FF_ALLOW_OFFGRID_STEPS must remain 0 for a recommended NFE"
        )
    return forwards, allow_offgrid


PROCESS_STUDENT_FORWARDS, PROCESS_ALLOW_OFFGRID_STEPS = (
    _student_forwards_from_environment()
)


def _student_canvas_from_environment() -> tuple[int, int]:
    """Resolve the one-shape-per-process Stage-1 sweep canvas."""

    try:
        width = int(os.environ.get("H3_FF_STUDENT_WIDTH", "896"))
        height = int(os.environ.get("H3_FF_STUDENT_HEIGHT", "512"))
    except ValueError as exc:
        raise RuntimeError(
            "H3_FF_STUDENT_WIDTH/HEIGHT must be integer pixel dimensions"
        ) from exc
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise RuntimeError(
            "H3_FF_STUDENT_WIDTH/HEIGHT must be positive multiples of 32, got "
            f"{width}x{height}"
        )
    if width < height or not 1.0 <= width / height <= 4.0:
        raise RuntimeError(
            f"the controlled Stage-1 sweep requires a landscape H3 canvas, got {width}x{height}"
        )
    if width * height > 896 * 512:
        raise RuntimeError(
            f"the Stage-1 sweep may not exceed its 896x512 baseline, got {width}x{height}"
        )
    return width, height


PROCESS_STUDENT_WIDTH, PROCESS_STUDENT_HEIGHT = _student_canvas_from_environment()
ARM_CONFIGS = {
    "teacher": {
        "width": 1344,
        "height": 768,
        "short_edge": 768,
        "sigma_points": 50,
        "forward_evaluations": 49,
        "image_directory": "firstframes_1344x768",
        "manifest_image_key": "teacher_image",
        "lora": False,
    },
    "student": {
        "width": PROCESS_STUDENT_WIDTH,
        "height": PROCESS_STUDENT_HEIGHT,
        # 512 is the process-local overlay routing token.  The overlay resolves
        # it to the exact fixed width/height above before latent preparation.
        "short_edge": 512,
        "aspect_ratio": f"{PROCESS_STUDENT_WIDTH}:{PROCESS_STUDENT_HEIGHT}",
        "sigma_points": PROCESS_STUDENT_FORWARDS + 1,
        "forward_evaluations": PROCESS_STUDENT_FORWARDS,
        "image_directory": (
            f"firstframes_{PROCESS_STUDENT_WIDTH}x{PROCESS_STUDENT_HEIGHT}"
        ),
        "manifest_image_key": "student_image",
        "lora": True,
    },
}


# SGLang GPU workers use multiprocessing spawn and re-import this module.  The
# process-local patches therefore must be installed above the main guard.
PROCESS_ARM = os.environ.get("H3_FF_ARM")
PROCESS_COMPILE = _strict_bool_env("H3_FF_COMPILE", "0")
PROCESS_LORA_MERGE_MODE = os.environ.get("H3_FF_LORA_MERGE_MODE", "dynamic")
if PROCESS_LORA_MERGE_MODE not in {"dynamic", "merge"}:
    raise RuntimeError(
        "H3_FF_LORA_MERGE_MODE must be dynamic or merge, got "
        f"{PROCESS_LORA_MERGE_MODE!r}"
    )
PROCESS_ARM_OVERLAY: dict[str, Any] | None = None
PROCESS_VAE_COMPILE_OVERLAY: dict[str, Any] | None = None
PROCESS_ZMQ_BIND: dict[str, Any] | None = None
if PROCESS_ARM in ARM_CONFIGS:
    import torch

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
if PROCESS_ARM == "student":
    process_lora_path = os.environ.get("H3_LORA_PATH")
    if not process_lora_path:
        raise RuntimeError("H3_LORA_PATH is required for the student arm")
    from sglang_h3_firstframe_lora_overlay import (
        install_firstframe_student_lora_overlay,
    )

    PROCESS_ARM_OVERLAY = install_firstframe_student_lora_overlay(
        lora_path=process_lora_path,
        lora_nickname=str(PROCESS_LORA_PROFILE["nickname"]),
        lora_scale=float(PROCESS_LORA_PROFILE["scale"]),
        merge_mode=PROCESS_LORA_MERGE_MODE,
        compile_enabled=PROCESS_COMPILE,
        forward_evaluations=PROCESS_STUDENT_FORWARDS,
        distilled_nfe=int(PROCESS_LORA_PROFILE["distilled_nfe"]),
        recommended_nfe=tuple(PROCESS_LORA_PROFILE["recommended_nfe"]),
        video_shift=float(PROCESS_LORA_PROFILE["video_shift"]),
        audio_shift=float(PROCESS_LORA_PROFILE["audio_shift"]),
        allow_offgrid_steps=PROCESS_ALLOW_OFFGRID_STEPS,
    )
elif PROCESS_ARM == "teacher":
    from sglang_h3_512_overlay import install_512p_overlay

    PROCESS_ARM_OVERLAY = install_512p_overlay(768)
elif PROCESS_ARM is not None:
    raise RuntimeError(f"H3_FF_ARM must be teacher or student, got {PROCESS_ARM!r}")

if PROCESS_ARM in ARM_CONFIGS:
    from sglang_h3_single_gpu_compile_overlay import (
        install_single_gpu_vae_compile_overlay,
    )

    PROCESS_VAE_COMPILE_OVERLAY = install_single_gpu_vae_compile_overlay(
        arm=PROCESS_ARM,
        enabled=PROCESS_COMPILE,
        mode=os.environ.get(
            "SGLANG_VAE_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
        ),
    )
    PROCESS_ZMQ_BIND = _install_fail_closed_zmq_bind()

PROCESS_OVERLAY = {
    "arm": PROCESS_ARM_OVERLAY,
    "vae_compile": PROCESS_VAE_COMPILE_OVERLAY,
    "zmq_bind_policy": PROCESS_ZMQ_BIND,
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_item(
    manifest_path: Path,
    asset_root: Path,
    arm: str,
    index: int,
) -> tuple[dict[str, Any], Path]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("conditioning") != "first_frame_only":
        raise ValueError("manifest conditioning must be first_frame_only")
    if payload.get("reference_conditions") is not False:
        raise ValueError("manifest must explicitly disable reference conditions")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("manifest items must be a non-empty list")
    if index < 0 or index >= len(items):
        raise ValueError(f"prompt index {index} is outside [0, {len(items)})")
    item = dict(items[index])
    if int(item.get("index", -1)) != index:
        raise ValueError(f"manifest item {index} has a mismatched index")
    if item.get("first_frame_role") != "first_frame":
        raise ValueError(f"manifest item {index} is not marked first_frame")
    if item.get("reference_conditions") != []:
        raise ValueError(f"manifest item {index} contains forbidden references")
    prompt = item.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"manifest item {index} has no prompt")
    recorded = item.get(ARM_CONFIGS[arm]["manifest_image_key"])
    if not isinstance(recorded, str) or not Path(recorded).name:
        raise ValueError(f"manifest item {index} has no {arm} image basename")
    image_path = (
        asset_root / ARM_CONFIGS[arm]["image_directory"] / Path(recorded).name
    ).resolve()
    expected_parent = (asset_root / ARM_CONFIGS[arm]["image_directory"]).resolve()
    if image_path.parent != expected_parent or not image_path.is_file():
        raise FileNotFoundError(
            f"relocated {arm} first frame is unavailable: {image_path}"
        )

    from PIL import Image

    with Image.open(image_path) as image:
        actual_size = tuple(image.size)
    expected_size = (
        int(ARM_CONFIGS[arm]["width"]),
        int(ARM_CONFIGS[arm]["height"]),
    )
    if actual_size != expected_size:
        raise ValueError(f"first frame {image_path} is {actual_size}, expected {expected_size}")
    return item, image_path


def _validate_runtime(arm: str, lora_path: Path | None) -> dict[str, Any]:
    import torch
    import triton

    expected_torch = os.environ.get("H3_EXPECTED_TORCH", "2.11.0+cu130")
    expected_triton = os.environ.get("H3_EXPECTED_TRITON", "3.6.0")
    if torch.__version__ != expected_torch:
        raise RuntimeError(f"expected torch {expected_torch}, found {torch.__version__}")
    if triton.__version__ != expected_triton:
        raise RuntimeError(
            f"expected Triton {expected_triton}, found {triton.__version__}"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"single-card benchmark sees {torch.cuda.device_count()} GPUs, expected 1"
        )
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability != (10, 0):
        raise RuntimeError(f"GPU capability is {capability}, expected SM100")

    upstream_module = "sglang.multimodal_gen.runtime.models.dits.minimax_h3"
    spec = importlib.util.find_spec(upstream_module)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"pinned SGLang module unavailable: {upstream_module}")
    upstream_path = Path(spec.origin)
    upstream_sha256 = hashlib.sha256(upstream_path.read_bytes()).hexdigest()
    if upstream_sha256 != PINNED_SGLANG_MODEL_SHA256:
        raise RuntimeError(
            f"unexpected MiniMax-H3 source {upstream_sha256}; "
            f"expected {PINNED_SGLANG_MODEL_SHA256}"
        )
    if arm == "student":
        if lora_path is None or not lora_path.is_file() or lora_path.stat().st_size <= 0:
            raise FileNotFoundError(f"student LoRA unavailable: {lora_path}")
        if lora_path.name != PROCESS_LORA_PROFILE["filename"]:
            raise RuntimeError(
                f"student LoRA filename {lora_path.name!r} does not match "
                f"profile {PROCESS_LORA_PROFILE_NAME}"
            )
        if lora_path.stat().st_size != PROCESS_LORA_PROFILE["expected_bytes"]:
            raise RuntimeError(
                f"student LoRA has {lora_path.stat().st_size} bytes; expected "
                f"{PROCESS_LORA_PROFILE['expected_bytes']}"
            )
    return {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(capability),
        "visible_gpu_count": torch.cuda.device_count(),
        "upstream_model_source": str(upstream_path),
        "upstream_model_sha256": upstream_sha256,
        "lora_bytes": None if lora_path is None else lora_path.stat().st_size,
    }


def _sampling_params(
    *,
    prompt: str,
    image_path: Path,
    output_dir: Path,
    output_name: str,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    config = ARM_CONFIGS[arm]
    return {
        "prompt": prompt,
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
            "short_edge": int(config["short_edge"]),
            "aspect_ratio": str(config.get("aspect_ratio", "7:4")),
            "duration_seconds": 5.0,
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": int(config["sigma_points"]),
        "flow_shift": float(PROCESS_LORA_PROFILE["video_shift"]),
        "audio_flow_shift": float(PROCESS_LORA_PROFILE["audio_shift"]),
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
    peak = result.peak_memory_mb
    return {
        "inference_time_s": inference,
        "outer_generate_wall_s": outer,
        "generation_wall_s_reported": float(result.generation_time),
        "peak_memory_mb": _json_safe(peak),
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
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"ffprobe found {len(streams)} video streams in {path}")
    return dict(streams[0])


def _validate_service_ports(
    generator: Any,
    *,
    http_port: int,
    scheduler_port: int,
    master_port: int,
    nccl_port: int,
) -> dict[str, Any]:
    expected = {
        "http": http_port,
        "scheduler": scheduler_port,
        "master": master_port,
        "nccl": nccl_port,
    }
    server_args = generator.server_args
    actual = {
        "http": int(server_args.port),
        "scheduler": int(server_args.scheduler_port),
        "master": int(server_args.master_port),
        "nccl": int(server_args.nccl_port),
    }
    if actual != expected:
        raise RuntimeError(f"SGLang service ports {actual} != requested {expected}")
    if server_args.strict_ports is not True:
        raise RuntimeError("SGLang strict_ports must remain enabled")
    return {**actual, "strict": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(ARM_CONFIGS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-subfolder", default="FL2VA")
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--lora-path", type=Path)
    parser.add_argument(
        "--student-forwards", type=int, default=PROCESS_STUDENT_FORWARDS
    )
    parser.add_argument("--http-port", type=int, required=True)
    parser.add_argument("--scheduler-port", type=int, required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--nccl-port", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--measured-requests", type=int, default=1)
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    if PROCESS_ARM != args.arm:
        parser.error(f"H3_FF_ARM={PROCESS_ARM!r} disagrees with --arm={args.arm!r}")
    if args.student_forwards != PROCESS_STUDENT_FORWARDS:
        parser.error(
            f"--student-forwards={args.student_forwards} disagrees with "
            f"H3_FF_STUDENT_FORWARDS={PROCESS_STUDENT_FORWARDS}"
        )
    if args.model_revision != PINNED_MODEL_REVISION:
        parser.error(f"model revision is pinned to {PINNED_MODEL_REVISION}")
    if args.warmup_requests < 1 or args.measured_requests < 1:
        parser.error("warmup and measured request counts must both be positive")
    requested_ports = (
        args.http_port,
        args.scheduler_port,
        args.master_port,
        args.nccl_port,
    )
    if len(set(requested_ports)) != 4 or any(
        not 1024 <= port <= 65535 for port in requested_ports
    ):
        parser.error("service ports must be four distinct values in [1024, 65535]")
    args.manifest = args.manifest.resolve()
    args.asset_root = args.asset_root.resolve()
    args.out = args.out.resolve()
    args.lora_path = None if args.lora_path is None else args.lora_path.resolve()
    if args.arm == "student" and args.lora_path is None:
        parser.error("student arm requires --lora-path")
    if args.arm == "teacher" and args.lora_path is not None:
        parser.error("teacher arm must not receive a LoRA path")
    args.out.mkdir(parents=True, exist_ok=False)

    item, image_path = _load_item(
        args.manifest, args.asset_root, args.arm, args.prompt_index
    )
    runtime = _validate_runtime(args.arm, args.lora_path)
    config = ARM_CONFIGS[args.arm]

    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

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
        "port": args.http_port,
        "scheduler_port": args.scheduler_port,
        "master_port": args.master_port,
        "nccl_port": args.nccl_port,
        "strict_ports": True,
    }
    if args.arm == "student":
        generator_kwargs.update(
            {
                "lora_path": str(args.lora_path),
                "lora_nickname": str(PROCESS_LORA_PROFILE["nickname"]),
                "lora_scale": float(PROCESS_LORA_PROFILE["scale"]),
                "lora_merge_mode": PROCESS_LORA_MERGE_MODE,
                "lora_target_modules": [
                    "qkv_proj",
                    "out_proj",
                    "mlp.fc1",
                    "mlp.fc2",
                ],
            }
        )

    print(
        f"loading one-GPU SGLang {args.arm}: {config['width']}x{config['height']}, "
        f"{config['forward_evaluations']} forwards, compile={PROCESS_COMPILE}, "
        f"lora_profile={PROCESS_LORA_PROFILE_NAME if args.arm == 'student' else 'none'}",
        flush=True,
    )
    load_started = time.perf_counter()
    generator = DiffGenerator.from_pretrained(**generator_kwargs)
    load_s = time.perf_counter() - load_started
    service_ports = _validate_service_ports(
        generator,
        http_port=args.http_port,
        scheduler_port=args.scheduler_port,
        master_port=args.master_port,
        nccl_port=args.nccl_port,
    )
    warmups: list[dict[str, Any]] = []
    measured: list[dict[str, Any]] = []
    try:
        for ordinal in range(args.warmup_requests):
            output_name = f"warmup_{ordinal:02d}.mp4"
            record = _run_request(
                generator,
                _sampling_params(
                    prompt=item["prompt"],
                    image_path=image_path,
                    output_dir=args.out,
                    output_name=output_name,
                    arm=args.arm,
                    seed=int(item["seed"]) + 10_000 + ordinal,
                ),
            )
            record["excluded"] = True
            warmups.append(record)
            print(
                f"{args.arm} warmup {ordinal + 1}/{args.warmup_requests}: "
                f"{record['inference_time_s']:.3f}s excluded",
                flush=True,
            )
        for ordinal in range(args.measured_requests):
            output_name = f"measured_{ordinal:02d}.mp4"
            record = _run_request(
                generator,
                _sampling_params(
                    prompt=item["prompt"],
                    image_path=image_path,
                    output_dir=args.out,
                    output_name=output_name,
                    arm=args.arm,
                    seed=int(item["seed"]) + ordinal,
                ),
            )
            measured.append(record)
            print(
                f"{args.arm} measured {ordinal + 1}/{args.measured_requests}: "
                f"{record['inference_time_s']:.3f}s",
                flush=True,
            )
    finally:
        generator.shutdown()

    media = []
    for ordinal in range(args.measured_requests):
        output = args.out / f"measured_{ordinal:02d}.mp4"
        probe = _probe_media(output, args.ffprobe)
        actual = (int(probe["width"]), int(probe["height"]))
        expected = (int(config["width"]), int(config["height"]))
        if actual != expected:
            raise RuntimeError(f"measured output is {actual}, expected {expected}")
        if int(probe.get("nb_read_frames") or 0) != FRAME_COUNT:
            raise RuntimeError(
                f"measured output has {probe.get('nb_read_frames')} frames, expected {FRAME_COUNT}"
            )
        media.append({"path": str(output), "probe": probe})

    benchmark = {
        "schema_version": 1,
        "kind": (
            "sglang_minimax_h3_single_gb200_firstframe_compiled_hot"
            if PROCESS_COMPILE
            else "sglang_minimax_h3_single_gb200_firstframe_eager_smoke"
        ),
        "arm": args.arm,
        "framework": {
            "name": "SGLang multimodal_gen",
            "commit": PINNED_SGLANG_COMMIT,
            "container": os.environ.get("H3_CONTAINER_IMAGE", PINNED_IMAGE),
            "overlay": PROCESS_OVERLAY,
        },
        "runtime": runtime,
        "model": {
            "path": args.model_path,
            "subfolder": args.model_subfolder,
            "revision": args.model_revision,
            "variant": "fl2va",
            "lora_path": None if args.lora_path is None else str(args.lora_path),
            "lora_profile": PROCESS_LORA_PROFILE_NAME if args.arm == "student" else None,
            "lora_nickname": (
                PROCESS_LORA_PROFILE["nickname"] if args.arm == "student" else None
            ),
            "lora_scale": (
                PROCESS_LORA_PROFILE["scale"] if args.arm == "student" else None
            ),
            "lora_merge_mode": (
                PROCESS_LORA_MERGE_MODE if args.arm == "student" else None
            ),
            "lora_source_revision": (
                PROCESS_LORA_PROFILE["source_revision"]
                if args.arm == "student"
                else None
            ),
            "lora_release_rank": (
                PROCESS_LORA_PROFILE["rank"] if args.arm == "student" else None
            ),
            "lora_release_alpha": (
                PROCESS_LORA_PROFILE["alpha"] if args.arm == "student" else None
            ),
            "lora_distilled_nfe": (
                PROCESS_LORA_PROFILE["distilled_nfe"]
                if args.arm == "student"
                else None
            ),
            "lora_recommended_nfe": (
                list(PROCESS_LORA_PROFILE["recommended_nfe"])
                if args.arm == "student"
                else None
            ),
            "off_native_distillation_grid": (
                PROCESS_STUDENT_FORWARDS != PROCESS_LORA_PROFILE["distilled_nfe"]
                if args.arm == "student"
                else None
            ),
            "offgrid_acknowledged": (
                PROCESS_ALLOW_OFFGRID_STEPS
                if args.arm == "student"
                else None
            ),
            "officially_recommended_nfe": (
                PROCESS_STUDENT_FORWARDS
                in PROCESS_LORA_PROFILE["recommended_nfe"]
                if args.arm == "student"
                else None
            ),
        },
        "workload": {
            "task": "fl2va",
            "conditioning": "first_frame_only",
            "reference_conditions": False,
            "width": config["width"],
            "height": config["height"],
            "frames": FRAME_COUNT,
            "fps": FPS,
            "duration_seconds": 5.0,
            "sigma_points": config["sigma_points"],
            "transformer_forward_evaluations": config["forward_evaluations"],
            "flow_shift": float(PROCESS_LORA_PROFILE["video_shift"]),
            "audio_flow_shift": float(PROCESS_LORA_PROFILE["audio_shift"]),
            "prompt_index": args.prompt_index,
            "prompt_id": item.get("id"),
            "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
            "seed": int(item["seed"]),
            "first_frame": str(image_path),
        },
        "topology": {
            "num_gpus_per_service": 1,
            "tp_size": 1,
            "ulysses_degree": 1,
            "fsdp_inference": False,
            "offload": False,
            "service_ports": service_ports,
            "torch_compile": PROCESS_COMPILE,
            "compile_mode": (
                os.environ.get(
                    "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
                )
                if PROCESS_COMPILE
                else None
            ),
            "compile_scope": {
                "dit": "SGLang native" if PROCESS_COMPILE else "eager",
                "video_vae": (
                    "local encoder/decoder only; tiling/orchestration unchanged"
                    if PROCESS_COMPILE
                    else "eager"
                ),
                "lora": (
                    "merged_once_at_startup"
                    if args.arm == "student" and PROCESS_LORA_MERGE_MODE == "merge"
                    else (
                        "compiled_dynamic_fused_forward"
                        if args.arm == "student" and PROCESS_COMPILE
                        else "dynamic_eager" if args.arm == "student" else "none"
                    )
                ),
            },
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "timing": {
            "model_load_s_excluded": load_s,
            "warmups_excluded": warmups,
            "measured": measured,
        },
        "media": media,
    }
    _write_json(args.out / "benchmark.json", benchmark)
    print(f"wrote {args.out / 'benchmark.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
