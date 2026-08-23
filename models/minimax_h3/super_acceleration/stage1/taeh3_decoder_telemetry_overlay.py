"""Pinned MiniMax-H3 Stage-1 decoder and delivery telemetry overlay.

This module is installed in the process that constructs ``DiffGenerator``,
before SGLang creates ``MiniMaxH3Pipeline``.  It replaces only the pinned
MiniMax-H3 decoding stage and the ``gpu_worker.save_outputs`` symbol used by
that pinned worker.  The two supported video-decoder arms are:

* ``madebyollin_taeh3``: decode the normalized H3 diffusion latent directly;
* ``official_minimax_h3_video_vae``: preserve the stock H3 Video VAE path.

The normalized video and audio latents are cloned before either stock decoder
can reverse-normalize them in place.  This is important for the formal
same-latent comparison.  Video decode and official Audio VAE decode are timed
with CUDA synchronization and CUDA events.  The output writer is timed from a
synchronized boundary through H.264/AAC encoding, muxing, writer close, and a
non-empty/readable MP4 check.

Telemetry is append-only JSONL.  One ``decoder`` event and one ``encode_mux``
event share the same monotonically increasing ``request_sequence``.  The
``encode_mux`` event exposes both the raw writer time and the contract phase
time (official Audio VAE decode plus writer time).

The default delivery arm remains the original MP4 writer.  When
``H3_DIRECT_HANDOFF_ACTIVE=1`` the writer hook instead preprocesses the
decoded CUDA tensors for LTX, copies them into reusable pinned host buffers,
and synchronously stages the binary payload in the resident Stage-2 process.
The returned ``h3tensor://`` URI is a logical token, not a filesystem path.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any, TypeVar

import torch


PINNED_SGLANG_COMMIT = "12eadf86f12aec2e6f81a6e38b61b964a4c6b529"
PINNED_SGLANG_DECODING_SHA256 = (
    "5e2cf87da11e0c744d6c7703d8151abd7da7937e1ad67d576fdbf6c678380954"
)
PINNED_SGLANG_GPU_WORKER_SHA256 = (
    "f103c0861a11a36233e8ed71d6aeffa76635882bb5fb0a4fbce2d74fe1eb3f6b"
)
PINNED_TAEHV_COMMIT = "e589fddc076e77f5ba8cd6baabe4ba3260b261cd"
PINNED_TAEHV_SOURCE_SHA256 = (
    "28b452ad74f924d2d0922d6b3e805dfa7f504c523ced240435fefad5f6f650a7"
)
PINNED_TAEH3_CHECKPOINT_SHA256 = (
    "af92965c2d7986a89a757e7cccd26f9eeeff0c3f0d5495eb168aeb2d6d9be9ba"
)

DECODER_TAEH3 = "madebyollin_taeh3"
DECODER_OFFICIAL = "official_minimax_h3_video_vae"
DECODER_NAMES = (DECODER_TAEH3, DECODER_OFFICIAL)
TELEMETRY_SCHEMA = "minimax_h3_stage1_decoder_telemetry_v1"
DIRECT_VIDEO_INPUT_SHAPE = (1, 3, 124, 512, 896)
DIRECT_VIDEO_OUTPUT_SHAPE = (1, 3, 121, 384, 672)
DIRECT_AUDIO_INPUT_SHAPE = (1, 2, 165600)
DIRECT_AUDIO_OUTPUT_SHAPE = (1, 2, 161333)

_INSTALL_STATE: dict[str, Any] | None = None
_T = TypeVar("_T")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_absolute_file(
    value: str | os.PathLike[str],
    *,
    name: str,
    expected_sha256: str | None = None,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path, got {path}")
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{name} is not a non-empty file: {path}")
    if expected_sha256 is not None:
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{name} SHA-256 {actual_sha256} != {expected_sha256}"
            )
    return path


def _module_source_sha256(module: Any, *, name: str) -> tuple[Path, str]:
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        raise RuntimeError(f"{name} has no source path")
    path = Path(str(raw_path)).resolve(strict=True)
    if path.suffix in {".pyc", ".pyo"}:
        source_path = Path(importlib.util.source_from_cache(str(path)))
        path = source_path.resolve(strict=True)
    if path.suffix != ".py":
        raise RuntimeError(f"{name} source must be a .py file, got {path}")
    return path, _sha256_file(path)


def _prepare_telemetry_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"telemetry_path must be absolute, got {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_file():
        raise RuntimeError(f"telemetry_path is not a regular file: {path}")
    return path


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(dict(payload), sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(
                f"short JSONL write to {path}: wrote {written}/{len(encoded)} bytes"
            )
    finally:
        os.close(descriptor)


def _process_fields() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "cuda_device": (
            torch.cuda.current_device() if torch.cuda.is_initialized() else None
        ),
    }


def _tensor_description(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return {
        "shape": [int(dimension) for dimension in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "contiguous": bool(value.is_contiguous()),
    }


def _output_descriptions(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, torch.Tensor):
        description = _tensor_description(value)
        return [] if description is None else [description]
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        descriptions: list[dict[str, Any]] = []
        for item in value:
            description = _tensor_description(item)
            if description is not None:
                descriptions.append(description)
            else:
                descriptions.append({"python_type": type(item).__name__})
        return descriptions
    return [{"python_type": type(value).__name__}]


def _direct_handoff_config() -> dict[str, Any] | None:
    """Read spawn-safe direct-delivery configuration from the environment."""

    raw_active = os.environ.get("H3_DIRECT_HANDOFF_ACTIVE", "0")
    if raw_active not in {"0", "1"}:
        raise ValueError("H3_DIRECT_HANDOFF_ACTIVE must be 0 or 1")
    if raw_active == "0":
        return None
    endpoint = os.environ.get("H3_DIRECT_HANDOFF_ENDPOINT", "").strip()
    auth_token = os.environ.get("H3_DIRECT_HANDOFF_AUTH_TOKEN", "").strip()
    raw_pair_id = os.environ.get("H3_DIRECT_HANDOFF_PAIR_ID", "").strip()
    if not endpoint or not auth_token or not raw_pair_id:
        raise ValueError(
            "direct handoff requires H3_DIRECT_HANDOFF_ENDPOINT, "
            "H3_DIRECT_HANDOFF_AUTH_TOKEN, and H3_DIRECT_HANDOFF_PAIR_ID"
        )
    try:
        pair_id = int(raw_pair_id)
    except ValueError as exc:
        raise ValueError("H3_DIRECT_HANDOFF_PAIR_ID must be an integer") from exc
    if pair_id < 0:
        raise ValueError("H3_DIRECT_HANDOFF_PAIR_ID must be non-negative")
    return {"endpoint": endpoint, "auth_token": auth_token, "pair_id": pair_id}


def _direct_video_tensor(outputs: Any) -> torch.Tensor:
    if not isinstance(outputs, torch.Tensor):
        raise TypeError(
            "direct Stage-1 handoff requires save_outputs video to be a tensor, got "
            f"{type(outputs).__name__}"
        )
    if tuple(outputs.shape) != DIRECT_VIDEO_INPUT_SHAPE:
        raise RuntimeError(
            f"direct video shape {tuple(outputs.shape)} != {DIRECT_VIDEO_INPUT_SHAPE}"
        )
    if outputs.device.type != "cuda" or outputs.dtype != torch.float32:
        raise RuntimeError(
            "direct video must be FP32 CUDA, got "
            f"dtype={outputs.dtype} device={outputs.device}"
        )
    if not outputs.is_contiguous():
        raise RuntimeError("direct video input must be contiguous")
    return outputs


def _direct_audio_tensor(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            "direct Stage-1 handoff requires decoded audio to be a tensor, got "
            f"{type(value).__name__}"
        )
    if tuple(value.shape) != DIRECT_AUDIO_INPUT_SHAPE:
        raise RuntimeError(
            f"direct audio shape {tuple(value.shape)} != {DIRECT_AUDIO_INPUT_SHAPE}"
        )
    if value.device.type != "cuda" or value.dtype != torch.float32:
        raise RuntimeError(
            "direct audio must be FP32 CUDA, got "
            f"dtype={value.dtype} device={value.device}"
        )
    if not value.is_contiguous():
        raise RuntimeError("direct audio input must be contiguous")
    return value


def _prepare_direct_payload(
    video: torch.Tensor,
    audio: torch.Tensor,
    pinned_buffers: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Resize/crop on CUDA, then copy into reusable pinned CPU buffers."""

    video = _direct_video_tensor(video)
    audio = _direct_audio_tensor(audio)
    video_cpu = pinned_buffers.get("video")
    audio_cpu = pinned_buffers.get("audio")
    buffers_preexisting = video_cpu is not None and audio_cpu is not None
    if video_cpu is None:
        video_cpu = torch.empty(
            DIRECT_VIDEO_OUTPUT_SHAPE,
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=True,
        )
        pinned_buffers["video"] = video_cpu
    if audio_cpu is None:
        audio_cpu = torch.empty(
            DIRECT_AUDIO_OUTPUT_SHAPE,
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )
        pinned_buffers["audio"] = audio_cpu
    if (
        tuple(video_cpu.shape) != DIRECT_VIDEO_OUTPUT_SHAPE
        or video_cpu.dtype != torch.bfloat16
        or not video_cpu.is_pinned()
        or tuple(audio_cpu.shape) != DIRECT_AUDIO_OUTPUT_SHAPE
        or audio_cpu.dtype != torch.float32
        or not audio_cpu.is_pinned()
    ):
        raise RuntimeError("direct handoff reusable pinned-buffer contract changed")

    started_ns = time.perf_counter_ns()
    # Bilinear is spatial only. Flatten B/T so frames remain independent.
    frames = video[:, :, : DIRECT_VIDEO_OUTPUT_SHAPE[2]]
    frames_4d = frames.permute(0, 2, 1, 3, 4).reshape(
        -1, 3, DIRECT_VIDEO_INPUT_SHAPE[3], DIRECT_VIDEO_INPUT_SHAPE[4]
    )
    resized_4d = torch.nn.functional.interpolate(
        frames_4d,
        size=DIRECT_VIDEO_OUTPUT_SHAPE[-2:],
        mode="bilinear",
        align_corners=False,
    )
    resized = resized_4d.reshape(
        1,
        DIRECT_VIDEO_OUTPUT_SHAPE[2],
        3,
        DIRECT_VIDEO_OUTPUT_SHAPE[3],
        DIRECT_VIDEO_OUTPUT_SHAPE[4],
    ).permute(0, 2, 1, 3, 4)
    prepared_video = resized.clamp(0.0, 1.0).mul(2.0).sub(1.0)
    prepared_video = prepared_video.to(dtype=torch.bfloat16).contiguous()
    prepared_audio = audio[:, :, : DIRECT_AUDIO_OUTPUT_SHAPE[2]].clamp(
        -1.0, 1.0
    ).contiguous()
    if tuple(prepared_video.shape) != DIRECT_VIDEO_OUTPUT_SHAPE:
        raise RuntimeError("direct video preprocessing produced the wrong shape")
    if tuple(prepared_audio.shape) != DIRECT_AUDIO_OUTPUT_SHAPE:
        raise RuntimeError("direct audio preprocessing produced the wrong shape")

    video_cpu.copy_(prepared_video, non_blocking=True)
    audio_cpu.copy_(prepared_audio, non_blocking=True)
    torch.cuda.current_stream(video.device).synchronize()
    copied_ns = time.perf_counter_ns()
    return video_cpu, audio_cpu, {
        "preprocess_and_d2h_s": (copied_ns - started_ns) / 1_000_000_000.0,
        "video": _tensor_description(prepared_video),
        "audio": _tensor_description(prepared_audio),
        "video_cpu": _tensor_description(video_cpu),
        "audio_cpu": _tensor_description(audio_cpu),
        "pinned_buffers_reused": buffers_preexisting,
    }


def _require_single_visible_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-1 decoder telemetry requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Stage-1 decoder telemetry requires exactly one visible GPU, got "
            f"{torch.cuda.device_count()}"
        )
    return torch.device("cuda", torch.cuda.current_device())


def _timed_cuda_call(fn: Callable[[], _T]) -> tuple[_T, dict[str, Any]]:
    """Run one CUDA phase between synchronized wall/event boundaries."""

    _require_single_visible_cuda()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    started_ns = time.perf_counter_ns()
    start_event.record()
    try:
        result = fn()
    except BaseException:
        end_event.record()
        torch.cuda.synchronize()
        raise
    end_event.record()
    torch.cuda.synchronize()
    ended_ns = time.perf_counter_ns()
    return result, {
        "wall_s": (ended_ns - started_ns) / 1_000_000_000.0,
        "cuda_ms": float(start_event.elapsed_time(end_event)),
        "boundary": "cuda_synchronize_before_and_after",
    }


def _load_taeh3(
    *,
    source_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> Any:
    module_name = f"_stage1_pinned_taehv_{PINNED_TAEHV_COMMIT[:12]}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import pinned TAEHV source {source_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    tae_cls = getattr(module, "TAEHV", None)
    if not inspect.isclass(tae_cls):
        raise RuntimeError(f"pinned TAEHV source has no TAEHV class: {source_path}")
    model = tae_cls(checkpoint_path=str(checkpoint_path))
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("TAEHV(checkpoint_path=...) did not return torch.nn.Module")
    model = model.to(device=device, dtype=torch.float16).eval()
    model.requires_grad_(False)
    if not callable(getattr(model, "decode_video", None)):
        raise RuntimeError("pinned TAEHV model has no decode_video method")
    return model


def _normalize_saved_paths(value: Any) -> list[Path]:
    if isinstance(value, (str, os.PathLike)):
        raw_paths = [value]
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        raw_paths = list(value)
    else:
        raise RuntimeError(
            "pinned save_outputs returned unsupported path payload "
            f"{type(value).__name__}"
        )
    if not raw_paths:
        raise RuntimeError("pinned save_outputs returned no output paths")
    paths: list[Path] = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, (str, os.PathLike)):
            raise RuntimeError(
                "pinned save_outputs returned a non-path item "
                f"{type(raw_path).__name__}"
            )
        paths.append(Path(raw_path))
    return paths


def _validate_readable_mp4s(paths: Sequence[Path]) -> list[dict[str, Any]]:
    media: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix.lower() != ".mp4":
            raise RuntimeError(f"formal Stage-1 output is not MP4: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise RuntimeError(f"Stage-1 output is not a file: {resolved}")
        size = resolved.stat().st_size
        if size <= 0:
            raise RuntimeError(f"Stage-1 output is empty: {resolved}")
        with resolved.open("rb") as handle:
            first_byte = handle.read(1)
        if len(first_byte) != 1:
            raise RuntimeError(f"Stage-1 output is not readable: {resolved}")
        media.append(
            {
                "path": str(resolved),
                "bytes": int(size),
                "mp4_ready": True,
            }
        )
    return media


def install_taeh3_decoder_telemetry_overlay(
    *,
    decoder_name: str,
    decoder_telemetry_path: str | os.PathLike[str],
    encode_telemetry_path: str | os.PathLike[str],
    taehv_source_path: str | os.PathLike[str] | None = None,
    taeh3_checkpoint_path: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Install the pinned video decoder and output-writer telemetry hooks.

    This function must run before ``DiffGenerator.from_pretrained``.  Exactly
    one visible CUDA device is enforced when the stage is instantiated, not at
    overlay installation time, so importing/configuring the runner remains a
    lightweight CPU operation.
    """

    global _INSTALL_STATE
    if decoder_name not in DECODER_NAMES:
        raise ValueError(f"decoder_name must be one of {DECODER_NAMES}, got {decoder_name!r}")
    direct_config = _direct_handoff_config()
    if direct_config is not None and decoder_name != DECODER_TAEH3:
        raise ValueError("direct tensor handoff requires the TAEH3 decoder arm")
    stage_tensor_sender: Callable[..., dict[str, Any]] | None = None
    if direct_config is not None:
        # Imported only in the explicit direct arm.  The integration directory
        # is already on PYTHONPATH in both the client and spawned GPU worker.
        from handoff_protocol import stage_tensor as stage_tensor_sender

    telemetry = _prepare_telemetry_path(decoder_telemetry_path)
    encode_telemetry = _prepare_telemetry_path(encode_telemetry_path)
    if telemetry == encode_telemetry:
        raise ValueError("decoder and encode telemetry paths must be different")
    if run_id is None:
        run_id = os.environ.get("H3_STAGE1_BENCHMARK_ID", "").strip() or None
    if run_id is not None and not str(run_id).strip():
        raise ValueError("run_id must be non-empty when supplied")

    taehv_source: Path | None = None
    taeh3_checkpoint: Path | None = None
    if decoder_name == DECODER_TAEH3:
        if taehv_source_path is None or taeh3_checkpoint_path is None:
            raise ValueError("TAEH3 decoder requires source and checkpoint paths")
        taehv_source = _require_absolute_file(
            taehv_source_path,
            name="taehv_source_path",
            expected_sha256=PINNED_TAEHV_SOURCE_SHA256,
        )
        taeh3_checkpoint = _require_absolute_file(
            taeh3_checkpoint_path,
            name="taeh3_checkpoint_path",
            expected_sha256=PINNED_TAEH3_CHECKPOINT_SHA256,
        )
    elif taehv_source_path is not None or taeh3_checkpoint_path is not None:
        raise ValueError("official decoder arm must not declare TAEH3 files")

    requested = {
        "decoder_name": decoder_name,
        "decoder_telemetry_path": str(telemetry),
        "encode_telemetry_path": str(encode_telemetry),
        "taehv_source_path": None if taehv_source is None else str(taehv_source),
        "taeh3_checkpoint_path": (
            None if taeh3_checkpoint is None else str(taeh3_checkpoint)
        ),
        "direct_handoff": (
            None
            if direct_config is None
            else {
                "endpoint": direct_config["endpoint"],
                "pair_id": direct_config["pair_id"],
                "auth_token_sha256": hashlib.sha256(
                    direct_config["auth_token"].encode("utf-8")
                ).hexdigest(),
            }
        ),
        "run_id": run_id,
    }
    if _INSTALL_STATE is not None:
        if _INSTALL_STATE["config"] != requested:
            raise RuntimeError(
                f"a different Stage-1 decoder telemetry overlay is active: {_INSTALL_STATE}"
            )
        return dict(_INSTALL_STATE)

    from sglang.multimodal_gen.runtime.distributed import (
        get_world_group,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.runtime.entrypoints import utils as entrypoint_utils
    from sglang.multimodal_gen.runtime.managers import gpu_worker
    from sglang.multimodal_gen.runtime.pipelines import minimax_h3_pipeline
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
        minimax_h3 as minimax_h3_stages,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        video_adapter as minimax_h3_video_adapter,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
        decoding as decoding_module,
    )
    from sglang.multimodal_gen.runtime.utils.precision import (
        autocast_enabled,
        resolve_decode_precision,
    )

    decoding_source, decoding_sha256 = _module_source_sha256(
        decoding_module, name="MiniMax-H3 decoding module"
    )
    worker_source, worker_sha256 = _module_source_sha256(
        gpu_worker, name="SGLang GPU worker module"
    )
    if decoding_sha256 != PINNED_SGLANG_DECODING_SHA256:
        raise RuntimeError(
            f"unexpected pinned decoder source SHA-256 {decoding_sha256}"
        )
    if worker_sha256 != PINNED_SGLANG_GPU_WORKER_SHA256:
        raise RuntimeError(f"unexpected pinned GPU worker source SHA-256 {worker_sha256}")

    stock_stage = decoding_module.MiniMaxH3DecodingStage
    if minimax_h3_pipeline.MiniMaxH3DecodingStage is not stock_stage:
        raise RuntimeError("MiniMax-H3 pipeline decoder symbol was already replaced")
    if minimax_h3_stages.MiniMaxH3DecodingStage is not stock_stage:
        raise RuntimeError("exported MiniMax-H3 decoder symbol disagrees with pinned stage")
    expected_init_parameters = ["self", "video_vae", "audio_vae"]
    if list(inspect.signature(stock_stage.__init__).parameters) != expected_init_parameters:
        raise RuntimeError(
            f"unexpected pinned decoder __init__ signature {inspect.signature(stock_stage.__init__)}"
        )
    if gpu_worker.save_outputs is not entrypoint_utils.save_outputs:
        raise RuntimeError("gpu_worker.save_outputs was already replaced")

    # In a formal one-rank request, decoder forward and save_outputs are
    # strictly paired.  Keeping the audio timing in process lets the writer
    # event expose the full contract phase without reconstructing it later
    # from unrelated medians.
    pending_writer_events: deque[dict[str, Any]] = deque()
    sequence_state = {"decoder": 0, "writer": 0, "validator": 0}
    direct_pinned_buffers: dict[str, torch.Tensor] = {}

    # DiffGenerator performs one last, client-side media validation after the
    # GPU worker's save_outputs call returns.  In the stock path that validator
    # correctly ffprobes the returned MP4.  A direct handoff deliberately
    # returns a logical h3tensor:// token instead, so validate the token/sequence
    # contract here and never send it to ffprobe.  This patch is scoped strictly
    # to the explicit direct arm; the MP4 fallback keeps the pinned validator.
    stock_final_output_validator = None
    if direct_config is not None:
        adapter_cls = minimax_h3_video_adapter.MiniMaxH3VideoModelAdapter
        stock_final_output_validator = adapter_cls.validate_final_outputs_sync
        if getattr(
            stock_final_output_validator,
            "_h3_direct_tensor_validator_overlay",
            False,
        ):
            raise RuntimeError("direct tensor final-output validator was already installed")

        def validate_direct_tensor_final_outputs_sync(
            self: Any,
            output_paths: list[str],
            batch: Any,
        ) -> dict[str, str]:
            expected_outputs = int(getattr(batch, "num_outputs_per_prompt", 1))
            if len(output_paths) != expected_outputs or expected_outputs != 1:
                raise RuntimeError(
                    "direct MiniMax H3 handoff requires exactly one output token; "
                    f"got {len(output_paths)} outputs, expected {expected_outputs}"
                )
            sequence_state["validator"] += 1
            expected_sequence = sequence_state["validator"]
            if expected_sequence == 1:
                expected_token = (
                    "h3tensor://discard/"
                    f"pair-{direct_config['pair_id']}/seq-{expected_sequence}"
                )
                phase = "discard"
            else:
                expected_token = (
                    f"h3tensor://pair-{direct_config['pair_id']}/"
                    f"seq-{expected_sequence}"
                )
                phase = "warmup" if expected_sequence == 2 else "hot"
            actual_token = str(output_paths[0])
            if actual_token != expected_token:
                raise RuntimeError(
                    "direct MiniMax H3 output-token contract failed: "
                    f"got {actual_token!r}, expected {expected_token!r}"
                )
            return {
                "handoff_mode": "direct_tensor",
                "phase": phase,
                "tensor_token": actual_token,
            }

        validate_direct_tensor_final_outputs_sync._h3_direct_tensor_validator_overlay = (  # type: ignore[attr-defined]
            True
        )
        validate_direct_tensor_final_outputs_sync._h3_direct_tensor_original = (  # type: ignore[attr-defined]
            stock_final_output_validator
        )
        adapter_cls.validate_final_outputs_sync = (
            validate_direct_tensor_final_outputs_sync
        )

    class MiniMaxH3Stage1TelemetryDecodingStage(stock_stage):
        def __init__(self, video_vae: Any, audio_vae: Any) -> None:
            super().__init__(video_vae=video_vae, audio_vae=audio_vae)
            self._stage1_taeh3 = None
            if decoder_name == DECODER_TAEH3:
                assert taehv_source is not None and taeh3_checkpoint is not None
                self._stage1_taeh3 = _load_taeh3(
                    source_path=taehv_source,
                    checkpoint_path=taeh3_checkpoint,
                    device=_require_single_visible_cuda(),
                )

        @torch.no_grad()
        def forward(self, batch: Any, server_args: Any) -> OutputBatch:
            sequence_state["decoder"] += 1
            request_sequence = sequence_state["decoder"]
            event_base = {
                "schema": TELEMETRY_SCHEMA,
                "event": "decoder",
                "run_id": run_id,
                "request_sequence": request_sequence,
                "stage1_decoder_name": decoder_name,
                "time_unix_ns": time.time_ns(),
                **_process_fields(),
            }
            video_timing: dict[str, Any] | None = None
            audio_timing: dict[str, Any] | None = None
            finalize_timing: dict[str, Any] | None = None
            try:
                decoding_module._minimax_h3_decoder_task(batch)
                visual_latent = decoding_module._required_tensor(
                    batch.latents, "batch.latents"
                )
                audio_latent = decoding_module._required_tensor(
                    batch.audio_latents, "batch.audio_latents"
                )
                if visual_latent.ndim != 5:
                    raise ValueError("batch.latents must be [B, C, T, H, W]")
                if audio_latent.ndim != 3:
                    raise ValueError(
                        "batch.audio_latents must be [audio_channel, latent_dim, T]"
                    )
                if self.video_vae is None:
                    raise RuntimeError(
                        "MiniMax H3 tasks require the video_vae component"
                    )

                # Both official reverse-normalization helpers are in-place.
                # Preserve the final normalized latents for hashing and exact
                # cross-arm comparison.
                visual_decode_source = visual_latent.clone()
                audio_decode_source = audio_latent.clone()
                torch.cuda.synchronize()

                with self.use_declared_component(
                    component_name="video_vae", module=self.video_vae
                ) as selected_video_vae:
                    if selected_video_vae is None:
                        raise RuntimeError("video_vae became unavailable during decode")
                    self.video_vae = selected_video_vae
                    if selected_video_vae.training:
                        selected_video_vae.eval()

                    def decode_video() -> torch.Tensor:
                        if decoder_name == DECODER_TAEH3:
                            if self._stage1_taeh3 is None:
                                raise RuntimeError("TAEH3 decoder was not loaded")
                            tae_input = (
                                visual_decode_source.permute(0, 2, 1, 3, 4)
                                .to(dtype=torch.float16)
                                .contiguous()
                            )
                            tae_frames = decoding_module._required_tensor(
                                self._stage1_taeh3.decode_video(
                                    tae_input,
                                    parallel=True,
                                    show_progress_bar=False,
                                ),
                                "taeh3.decode_video",
                            )
                            if (
                                tae_frames.ndim != 5
                                or int(tae_frames.shape[0])
                                != int(visual_decode_source.shape[0])
                                or int(tae_frames.shape[2]) != 3
                            ):
                                raise RuntimeError(
                                    "TAEH3 output must be [B, T, 3, H, W], got "
                                    f"{tuple(tae_frames.shape)}"
                                )
                            visual_frames = tae_frames.permute(0, 2, 1, 3, 4)
                            visual_frames = decoding_module._crop_to_target_canvas(
                                batch, visual_frames
                            )
                        else:
                            visual_arch_config = (
                                server_args.pipeline_config.vae_config.arch_config
                            )
                            visual_decode_latent = (
                                decoding_module._reverse_normalize_latents_(
                                    visual_decode_source,
                                    mean_values=visual_arch_config.latents_mean,
                                    std_values=visual_arch_config.latents_std,
                                    name="video_vae",
                                )
                            )
                            video_vae_dtype = resolve_decode_precision(
                                server_args, "video_vae"
                            )
                            visual_autocast_enabled = (
                                visual_decode_source.device.type == "cuda"
                                and autocast_enabled(
                                    video_vae_dtype, server_args.disable_autocast
                                )
                            )
                            if visual_autocast_enabled:
                                selected_video_vae.prepare_decoder_autocast_weights(
                                    video_vae_dtype
                                )
                            with torch.autocast(
                                device_type=visual_decode_source.device.type,
                                dtype=video_vae_dtype,
                                enabled=visual_autocast_enabled,
                            ):
                                video_decode = self._get_vae_decode_fn(
                                    selected_video_vae,
                                    server_args,
                                    decode_fn=selected_video_vae.decode_base,
                                )
                                visual_frames = video_decode(visual_decode_latent)
                                visual_frames = (
                                    selected_video_vae.processor.revert_tensor(
                                        visual_frames
                                    )
                                )
                                visual_frames = decoding_module._required_tensor(
                                    visual_frames,
                                    "video_vae.processor.revert_tensor",
                                )
                                visual_frames = (
                                    decoding_module._canonical_visual_video_frames(
                                        visual_frames,
                                        batch_size=int(visual_decode_source.shape[0]),
                                    )
                                )
                                visual_frames = decoding_module._crop_to_target_canvas(
                                    batch, visual_frames
                                )
                        if (
                            visual_frames.dtype != torch.float32
                            or not visual_frames.is_contiguous()
                        ):
                            canonical_frames = torch.empty_like(
                                visual_frames,
                                dtype=torch.float32,
                                memory_format=torch.contiguous_format,
                            )
                            canonical_frames.copy_(visual_frames)
                            visual_frames = canonical_frames
                        return visual_frames

                    visual_frames, video_timing = _timed_cuda_call(decode_video)

                world_group = (
                    get_world_group() if model_parallel_is_initialized() else None
                )
                is_audio_owner = world_group is None or world_group.rank_in_group == 0
                owner_exception = None
                owner_error = None
                audio_payload = None
                if is_audio_owner:
                    try:
                        audio_payload, audio_timing = _timed_cuda_call(
                            lambda: self._decode_audio(
                                audio_decode_source, server_args
                            )
                        )
                    except Exception as exc:
                        owner_exception = exc
                        owner_error = f"{type(exc).__name__}: {exc}"
                if world_group is not None:
                    owner_error = world_group.broadcast_object(owner_error, src=0)
                if owner_error is not None:
                    if owner_exception is not None:
                        raise owner_exception
                    raise RuntimeError(
                        f"MiniMax H3 audio decode failed on rank 0: {owner_error}"
                    )
                if world_group is not None:
                    audio_payload = world_group.broadcast_tensor_dict(
                        audio_payload, src=0
                    )
                if not isinstance(audio_payload, dict):
                    raise RuntimeError(
                        "MiniMax H3 audio decode produced no output payload"
                    )
                audio_waveform = decoding_module._required_tensor(
                    audio_payload.get("waveform"), "audio_vae.decode"
                )
                audio_sample_rate = int(audio_payload["sample_rate"])

                def finalize_outputs() -> tuple[torch.Tensor, torch.Tensor]:
                    finalized_frames = server_args.pipeline_config.post_decoding(
                        visual_frames, server_args
                    )
                    finalized_audio = (
                        decoding_module._canonical_output_audio_waveform(
                            audio_waveform,
                            batch_size=int(finalized_frames.shape[0]),
                        )
                    )
                    return finalized_frames, finalized_audio

                (visual_frames, output_audio_waveform), finalize_timing = (
                    _timed_cuda_call(finalize_outputs)
                )
                decoder_event = {
                    **event_base,
                    "status": "ok",
                    "source_latent_cloned_before_decoder": True,
                    "video_latent": _tensor_description(visual_latent),
                    "audio_latent": _tensor_description(audio_latent),
                    "video_output": _tensor_description(visual_frames),
                    "audio_output": _tensor_description(output_audio_waveform),
                    # post_decoding produces the final canonical GPU RGB
                    # tensor and is therefore part of the video-decoder phase.
                    "stage1_decode_s": float(
                        video_timing["wall_s"] + finalize_timing["wall_s"]
                    ),
                    "stage1_decode_cuda_ms": float(
                        video_timing["cuda_ms"] + finalize_timing["cuda_ms"]
                    ),
                    "audio_decode_s": float(
                        0.0 if audio_timing is None else audio_timing["wall_s"]
                    ),
                    "audio_decode_cuda_ms": float(
                        0.0 if audio_timing is None else audio_timing["cuda_ms"]
                    ),
                    "post_decoding_s": float(finalize_timing["wall_s"]),
                    "post_decoding_cuda_ms": float(finalize_timing["cuda_ms"]),
                    "audio_sample_rate": audio_sample_rate,
                    "taehv_commit": (
                        PINNED_TAEHV_COMMIT
                        if decoder_name == DECODER_TAEH3
                        else None
                    ),
                    "taehv_source_sha256": (
                        PINNED_TAEHV_SOURCE_SHA256
                        if decoder_name == DECODER_TAEH3
                        else None
                    ),
                    "taeh3_checkpoint_sha256": (
                        PINNED_TAEH3_CHECKPOINT_SHA256
                        if decoder_name == DECODER_TAEH3
                        else None
                    ),
                }
                _append_jsonl(telemetry, decoder_event)
                if is_audio_owner:
                    pending_writer_events.append(
                        {
                            "request_sequence": request_sequence,
                            "stage1_decoder_name": decoder_name,
                            "audio_decode_s": decoder_event["audio_decode_s"],
                            "audio_decode_cuda_ms": decoder_event[
                                "audio_decode_cuda_ms"
                            ],
                            "post_decoding_s": decoder_event["post_decoding_s"],
                            "audio_sample_rate": audio_sample_rate,
                            # Retain the decoded CUDA PCM only until the
                            # immediately following writer hook.
                            "audio_output_tensor": output_audio_waveform,
                        }
                    )
                if not isinstance(getattr(batch, "extra", None), Mapping):
                    raise RuntimeError("MiniMax-H3 batch.extra is unavailable")
                batch.extra["minimax_h3_stage1_decoder_telemetry"] = {
                    "request_sequence": request_sequence,
                    "stage1_decoder_name": decoder_name,
                    "stage1_decode_s": decoder_event["stage1_decode_s"],
                    "audio_decode_s": decoder_event["audio_decode_s"],
                }
                return OutputBatch(
                    output=visual_frames,
                    audio=output_audio_waveform,
                    audio_sample_rate=audio_sample_rate,
                    trajectory_timesteps=batch.trajectory_timesteps,
                    trajectory_latents=batch.trajectory_latents,
                    rollout_trajectory_data=batch.rollout_trajectory_data,
                    trajectory_decoded=None,
                    metrics=batch.metrics,
                    noise_pred=None,
                )
            except BaseException as exc:
                _append_jsonl(
                    telemetry,
                    {
                        **event_base,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "stage1_decode_s": (
                            None if video_timing is None else video_timing["wall_s"]
                        ),
                        "audio_decode_s": (
                            None if audio_timing is None else audio_timing["wall_s"]
                        ),
                        "post_decoding_s": (
                            None
                            if finalize_timing is None
                            else finalize_timing["wall_s"]
                        ),
                    },
                )
                raise

    MiniMaxH3Stage1TelemetryDecodingStage.__name__ = (
        "MiniMaxH3Stage1TelemetryDecodingStage"
    )
    MiniMaxH3Stage1TelemetryDecodingStage.__qualname__ = (
        "MiniMaxH3Stage1TelemetryDecodingStage"
    )
    MiniMaxH3Stage1TelemetryDecodingStage._h3_stage1_decoder_telemetry_overlay = (
        True
    )
    MiniMaxH3Stage1TelemetryDecodingStage._h3_stage1_stock_stage = stock_stage

    stock_save_outputs = gpu_worker.save_outputs

    def timed_save_outputs(*args: Any, **kwargs: Any) -> Any:
        sequence_state["writer"] += 1
        writer_sequence = sequence_state["writer"]
        if not pending_writer_events:
            raise RuntimeError(
                "save_outputs has no matching successful MiniMax-H3 decoder event"
            )
        decoder_phase = pending_writer_events.popleft()
        request_sequence = int(decoder_phase["request_sequence"])
        if request_sequence != writer_sequence:
            raise RuntimeError(
                "decoder/save_outputs sequence mismatch: "
                f"decoder={request_sequence} writer={writer_sequence}"
            )
        outputs = args[0] if args else kwargs.get("outputs")
        fps = args[2] if len(args) > 2 else kwargs.get("fps")
        event_base = {
            "schema": TELEMETRY_SCHEMA,
            "event": (
                "tensor_handoff"
                if direct_config is not None
                else "encode_mux"
            ),
            "run_id": run_id,
            "request_sequence": request_sequence,
            "writer_sequence": writer_sequence,
            "stage1_decoder_name": decoder_phase["stage1_decoder_name"],
            "time_unix_ns": time.time_ns(),
            "fps": None if fps is None else float(fps),
            "outputs": _output_descriptions(outputs),
            **_process_fields(),
        }
        if direct_config is not None:
            if stage_tensor_sender is None:
                raise RuntimeError("direct tensor sender was not installed")
            audio_output = decoder_phase.pop("audio_output_tensor", None)
            phase = (
                "discard"
                if writer_sequence == 1
                else "warmup" if writer_sequence == 2 else "hot"
            )
            handoff_started_ns = time.perf_counter_ns()
            try:
                video = _direct_video_tensor(outputs)
                audio = _direct_audio_tensor(audio_output)
                if phase == "discard":
                    tensor_token = (
                        f"h3tensor://discard/pair-{direct_config['pair_id']}/seq-1"
                    )
                    preprocessing: dict[str, Any] = {
                        "preprocess_and_d2h_s": 0.0,
                        "discarded_before_preprocess": True,
                    }
                    ack: dict[str, Any] | None = None
                    tensor_staged = False
                else:
                    video_cpu, audio_cpu, preprocessing = _prepare_direct_payload(
                        video, audio, direct_pinned_buffers
                    )
                    ack = stage_tensor_sender(
                        direct_config["endpoint"],
                        {
                            "token": direct_config["auth_token"],
                            "pair_id": direct_config["pair_id"],
                            "seq": writer_sequence,
                            "op": "stage_tensor",
                            "metadata": {
                                "phase": phase,
                                "request_sequence": request_sequence,
                                "fps": None if fps is None else float(fps),
                                "audio_sample_rate": int(
                                    decoder_phase["audio_sample_rate"]
                                ),
                                "video_value_range": [-1.0, 1.0],
                                "audio_value_range": [-1.0, 1.0],
                            },
                        },
                        video_cpu,
                        audio_cpu,
                        connect_timeout_s=1800.0,
                        response_timeout_s=1800.0,
                    )
                    tensor_token = str(ack["tensor_token"])
                    if not tensor_token.startswith("h3tensor://"):
                        raise RuntimeError(
                            "Stage-2 returned a non-h3tensor token "
                            f"{tensor_token!r}"
                        )
                    tensor_staged = True
            except BaseException as exc:
                tensor_handoff_s = (
                    time.perf_counter_ns() - handoff_started_ns
                ) / 1_000_000_000.0
                _append_jsonl(
                    encode_telemetry,
                    {
                        **event_base,
                        "status": "error",
                        "handoff_mode": "direct_tensor",
                        "phase": phase,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "stage1_tensor_handoff_s": tensor_handoff_s,
                        "tensor_handoff_s": tensor_handoff_s,
                    },
                )
                raise
            tensor_handoff_s = (
                time.perf_counter_ns() - handoff_started_ns
            ) / 1_000_000_000.0
            audio_decode_s = float(decoder_phase["audio_decode_s"])
            _append_jsonl(
                encode_telemetry,
                {
                    **event_base,
                    "status": "ok",
                    "handoff_mode": "direct_tensor",
                    "phase": phase,
                    "audio_decode_s": audio_decode_s,
                    "audio_decode_cuda_ms": float(
                        decoder_phase["audio_decode_cuda_ms"]
                    ),
                    "post_decoding_s": float(decoder_phase["post_decoding_s"]),
                    "stage1_tensor_handoff_s": tensor_handoff_s,
                    "tensor_handoff_s": tensor_handoff_s,
                    "stage1_encode_mux_s": 0.0,
                    "stage1_encode_mux_definition": "not_materialized",
                    "preprocessing": preprocessing,
                    "tensor_token": tensor_token,
                    "tensor_staged": tensor_staged,
                    "destination_cuda_complete": bool(
                        False if ack is None else ack.get("copied_to_cuda")
                    ),
                    "stage_ack": ack,
                    "mp4_ready": False,
                },
            )
            # Match stock save_outputs' one-output list contract so the
            # DiffGenerator result exposes this URI as output_file_path.
            return [tensor_token]

        _require_single_visible_cuda()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        started_ns = time.perf_counter_ns()
        start_event.record()
        try:
            result = stock_save_outputs(*args, **kwargs)
        except BaseException as exc:
            end_event.record()
            torch.cuda.synchronize()
            ended_ns = time.perf_counter_ns()
            _append_jsonl(
                encode_telemetry,
                {
                    **event_base,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "output_write_s": (ended_ns - started_ns) / 1_000_000_000.0,
                    "output_write_cuda_ms": float(
                        start_event.elapsed_time(end_event)
                    ),
                },
            )
            raise
        end_event.record()
        torch.cuda.synchronize()
        writer_ended_ns = time.perf_counter_ns()
        output_write_s = (writer_ended_ns - started_ns) / 1_000_000_000.0
        output_write_cuda_ms = float(start_event.elapsed_time(end_event))
        readiness_started_ns = time.perf_counter_ns()
        try:
            saved_paths = _normalize_saved_paths(result)
            media = _validate_readable_mp4s(saved_paths)
        except BaseException as exc:
            _append_jsonl(
                encode_telemetry,
                {
                    **event_base,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "output_write_s": output_write_s,
                    "output_write_cuda_ms": output_write_cuda_ms,
                },
            )
            raise
        readiness_s = (
            time.perf_counter_ns() - readiness_started_ns
        ) / 1_000_000_000.0
        audio_decode_s = float(decoder_phase["audio_decode_s"])
        _append_jsonl(
            encode_telemetry,
            {
                **event_base,
                "status": "ok",
                "audio_decode_s": audio_decode_s,
                "audio_decode_cuda_ms": float(
                    decoder_phase["audio_decode_cuda_ms"]
                ),
                "post_decoding_s": float(decoder_phase["post_decoding_s"]),
                "output_write_s": output_write_s,
                "output_write_cuda_ms": output_write_cuda_ms,
                "readiness_check_s": readiness_s,
                "stage1_encode_mux_s": audio_decode_s + output_write_s,
                "stage1_encode_mux_definition": (
                    "official_audio_vae_decode_plus_gpu_to_cpu_h264_aac_mux_"
                    "and_writer_close"
                ),
                "media": media,
                "mp4_ready": all(bool(item["mp4_ready"]) for item in media),
            },
        )
        return result

    timed_save_outputs.__name__ = "stage1_timed_save_outputs"
    timed_save_outputs.__qualname__ = "stage1_timed_save_outputs"
    timed_save_outputs._h3_stage1_writer_telemetry_overlay = True  # type: ignore[attr-defined]
    timed_save_outputs._h3_stage1_stock_save_outputs = stock_save_outputs  # type: ignore[attr-defined]

    decoding_module.MiniMaxH3DecodingStage = (
        MiniMaxH3Stage1TelemetryDecodingStage
    )
    minimax_h3_stages.MiniMaxH3DecodingStage = (
        MiniMaxH3Stage1TelemetryDecodingStage
    )
    minimax_h3_pipeline.MiniMaxH3DecodingStage = (
        MiniMaxH3Stage1TelemetryDecodingStage
    )
    gpu_worker.save_outputs = timed_save_outputs

    _INSTALL_STATE = {
        "installed": True,
        "name": "minimax_h3_stage1_decoder_telemetry_v1",
        "pinned_sglang_commit": PINNED_SGLANG_COMMIT,
        "pinned_sglang_decoding_source": str(decoding_source),
        "pinned_sglang_decoding_sha256": decoding_sha256,
        "pinned_sglang_gpu_worker_source": str(worker_source),
        "pinned_sglang_gpu_worker_sha256": worker_sha256,
        "config": requested,
        "video_decode_boundary": "cuda_synchronize_before_and_after",
        "delivery_boundary": (
            "direct_preprocess_d2h_binary_stage_through_destination_h2d_ack"
            if direct_config is not None
            else "official_audio_vae_decode_plus_save_outputs_through_writer_close"
        ),
        "mp4_readability_checked_before_save_outputs_return": (
            direct_config is None
        ),
    }
    return dict(_INSTALL_STATE)


def install_delivery_overlay(
    *,
    decoder_name: str,
    decoder_telemetry_path: str | os.PathLike[str],
    encode_telemetry_path: str | os.PathLike[str],
    taehv_source_path: str | os.PathLike[str] | None = None,
    taeh3_checkpoint_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Runner-facing adapter using the concise arm names in Slurm."""

    mapped = {
        "taeh3": DECODER_TAEH3,
        "official_h3_vae": DECODER_OFFICIAL,
    }.get(decoder_name)
    if mapped is None:
        raise ValueError(f"unsupported delivery decoder {decoder_name!r}")
    if mapped == DECODER_OFFICIAL:
        taehv_source_path = None
        taeh3_checkpoint_path = None
    return install_taeh3_decoder_telemetry_overlay(
        decoder_name=mapped,
        decoder_telemetry_path=decoder_telemetry_path,
        encode_telemetry_path=encode_telemetry_path,
        taehv_source_path=taehv_source_path,
        taeh3_checkpoint_path=taeh3_checkpoint_path,
    )


__all__ = [
    "DECODER_NAMES",
    "DECODER_OFFICIAL",
    "DECODER_TAEH3",
    "PINNED_SGLANG_COMMIT",
    "PINNED_TAEH3_CHECKPOINT_SHA256",
    "PINNED_TAEHV_COMMIT",
    "PINNED_TAEHV_SOURCE_SHA256",
    "install_delivery_overlay",
    "install_taeh3_decoder_telemetry_overlay",
]
