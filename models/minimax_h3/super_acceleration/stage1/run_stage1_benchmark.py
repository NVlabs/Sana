#!/usr/bin/env python3
"""Hot single-GB200 Stage-1 delivery benchmark.

The measured wall boundary starts immediately before ``DiffGenerator.generate``
and ends only after the returned MP4 exists, is non-empty, and can be opened for
reading, or after direct tensor staging is acknowledged as destination-CUDA
complete.  Model load, startup LoRA merge, the excluded compile-prime request,
the one complete warmup request, ffprobe, and JSON serialization are outside
that boundary.

This file intentionally reuses the production Stage-1 grid overlay.  The
additional delivery overlay swaps only the video decoder (TAEH3 versus the
official H3 VAE) and records decoder plus encode/mux telemetry in the worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any


FPS = 24
WIDTH = 896
HEIGHT = 512
NFE = 4
SIGMA_POINTS = NFE + 1
PROFILE = "lx2v_4s_v01_544p"
FRAME_COUNTS = {5: 124, 10: 243}
TAEH3_COMMIT = "e589fddc076e77f5ba8cd6baabe4ba3260b261cd"


PROCESS_ACTIVE = os.environ.get("H3_DELIVERY_BENCH_ACTIVE", "0") == "1"
PROCESS_DECODER = os.environ.get("H3_DELIVERY_DECODER", "")
PROCESS_DECODER_TELEMETRY = os.environ.get("H3_DELIVERY_DECODER_TELEMETRY", "")
PROCESS_ENCODE_TELEMETRY = os.environ.get("H3_DELIVERY_ENCODE_TELEMETRY", "")
PROCESS_TAE_SOURCE = os.environ.get("H3_DELIVERY_TAE_SOURCE", "")
PROCESS_TAE_CHECKPOINT = os.environ.get("H3_DELIVERY_TAE_CHECKPOINT", "")
PROCESS_DIRECT_HANDOFF = os.environ.get("H3_DIRECT_HANDOFF_ACTIVE", "0") == "1"
PROCESS_DELIVERY_OVERLAY: dict[str, Any] | None = None

# Multiprocessing uses spawn.  These process overlays therefore have to be
# installed at module import time, not only under ``if __name__ == '__main__'``.
if PROCESS_ACTIVE:
    if PROCESS_DECODER not in {"taeh3", "official_h3_vae"}:
        raise RuntimeError(f"unsupported H3_DELIVERY_DECODER={PROCESS_DECODER!r}")
    if not PROCESS_DECODER_TELEMETRY or not PROCESS_ENCODE_TELEMETRY:
        raise RuntimeError("delivery decoder and encode telemetry paths are required")

    import sglang_stage1_grid_single_gpu as grid
    from taeh3_decoder_telemetry_overlay import install_delivery_overlay

    PROCESS_DELIVERY_OVERLAY = install_delivery_overlay(
        decoder_name=PROCESS_DECODER,
        decoder_telemetry_path=PROCESS_DECODER_TELEMETRY,
        encode_telemetry_path=PROCESS_ENCODE_TELEMETRY,
        taehv_source_path=PROCESS_TAE_SOURCE or None,
        taeh3_checkpoint_path=PROCESS_TAE_CHECKPOINT or None,
    )
else:
    grid = None


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
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def _one_new_jsonl(path: Path, before: int, *, label: str) -> tuple[int, dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    if len(lines) != before + 1:
        raise RuntimeError(
            f"{label} telemetry advanced from {before} to {len(lines)} rows; expected one"
        )
    return len(lines), dict(json.loads(lines[-1]))


def _sampling_params(
    *,
    item: dict[str, Any],
    image_path: Path,
    duration: int,
    output_dir: Path,
    output_name: str,
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
            "short_edge": HEIGHT,
            "aspect_ratio": "7:4",
            "duration_seconds": float(duration),
        },
        "num_outputs_per_prompt": 1,
        # SGLang consumes sigma points; five points mean four Transformer calls.
        "num_inference_steps": SIGMA_POINTS,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": int(item["seed"]),
        "output_path": str(output_dir),
        "output_file_name": output_name,
        "save_output": True,
        "return_file_paths_only": True,
    }


def _result_inference_s(result: Any) -> float:
    metrics = dict(result.metrics or {})
    if "total_duration_s" in metrics:
        return float(metrics["total_duration_s"])
    if "total_duration_ms" in metrics:
        return float(metrics["total_duration_ms"]) / 1000.0
    return float(result.generation_time)


def _probe_media(path: Path, ffprobe: str, expected_frames: int) -> dict[str, Any]:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_read_frames,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = list(json.loads(completed.stdout).get("streams") or [])
    videos = [row for row in streams if row.get("codec_type") == "video"]
    audios = [row for row in streams if row.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise RuntimeError(
            f"expected one video and one audio stream in {path}, got {streams}"
        )
    video = videos[0]
    if (
        (int(video.get("width", 0)), int(video.get("height", 0))) != (WIDTH, HEIGHT)
        or int(video.get("nb_read_frames") or 0) != expected_frames
        or video.get("r_frame_rate") != "24/1"
        or video.get("codec_name") != "h264"
        or audios[0].get("codec_name") != "aac"
        or int(audios[0].get("channels") or 0) != 2
        or int(audios[0].get("sample_rate") or 0) <= 0
    ):
        raise RuntimeError(f"media contract failed for {path}: {streams}")
    return {"streams": streams, "bytes": path.stat().st_size}


def _run_one(
    *,
    generator: Any,
    request: dict[str, Any],
    expected_output: Path,
    phase: str,
    repeat: int,
    duration: int,
    expected_frames: int,
    cache_telemetry_path: Path,
    decoder_telemetry_path: Path,
    encode_telemetry_path: Path,
    counters: dict[str, int],
    ffprobe: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = generator.generate(sampling_params_kwargs=request)
    # The worker-side save_outputs wrapper returns only after either H.264/AAC
    # materialization or destination-CUDA-complete tensor staging. Stop the
    # formal wall here; client-side telemetry validation is outside the timer.
    stage1_wall_s = time.perf_counter() - started
    if result is None or isinstance(result, list):
        raise RuntimeError(f"expected one generation result, got {result!r}")
    returned_reference = str(result.output_file_path)
    returned_path: Path | None = None
    if PROCESS_DIRECT_HANDOFF:
        if not returned_reference.startswith("h3tensor://"):
            raise RuntimeError(
                f"direct Stage-1 output is not an h3tensor token: {returned_reference!r}"
            )
    else:
        returned_path = Path(returned_reference)
        if returned_path.resolve() != expected_output.resolve():
            raise RuntimeError(
                f"returned output {returned_path} != expected {expected_output}"
            )
        # The worker has already performed this readability check before
        # returning. Repeat it client-side as an untimed fail-closed check.
        with returned_path.open("rb") as handle:
            if not handle.read(16):
                raise RuntimeError(f"Stage-1 MP4 is empty: {returned_path}")

    counters["cache"], cache_row = _one_new_jsonl(
        cache_telemetry_path, counters["cache"], label="denoise/cache"
    )
    counters["decoder"], decoder_row = _one_new_jsonl(
        decoder_telemetry_path, counters["decoder"], label="decoder"
    )
    counters["encode"], encode_row = _one_new_jsonl(
        encode_telemetry_path, counters["encode"], label="encode/mux"
    )
    expected_delivery_event = (
        "tensor_handoff" if PROCESS_DIRECT_HANDOFF else "encode_mux"
    )
    if (
        decoder_row.get("status") != "ok"
        or decoder_row.get("event") != "decoder"
        or encode_row.get("status") != "ok"
        or encode_row.get("event") != expected_delivery_event
        or int(decoder_row.get("request_sequence", -1))
        != int(encode_row.get("request_sequence", -2))
    ):
        raise RuntimeError(
            f"decoder/encode telemetry pairing failed: {decoder_row} / {encode_row}"
        )
    if PROCESS_DIRECT_HANDOFF:
        expected_direct_phase = (
            "discard"
            if int(encode_row["request_sequence"]) == 1
            else "warmup" if int(encode_row["request_sequence"]) == 2 else "hot"
        )
        if (
            encode_row.get("handoff_mode") != "direct_tensor"
            or encode_row.get("phase") != expected_direct_phase
            or encode_row.get("tensor_token") != returned_reference
            or encode_row.get("mp4_ready") is not False
            or (
                expected_direct_phase != "discard"
                and (
                    encode_row.get("tensor_staged") is not True
                    or encode_row.get("destination_cuda_complete") is not True
                )
            )
        ):
            raise RuntimeError(f"direct tensor delivery contract failed: {encode_row}")
    elif encode_row.get("mp4_ready") is not True:
        raise RuntimeError(f"MP4 delivery contract failed: {encode_row}")

    inference_s = _result_inference_s(result)
    stage1_decode_s = float(decoder_row["stage1_decode_s"])
    audio_decode_s = float(decoder_row["audio_decode_s"])
    stage1_encode_mux_s = float(encode_row["stage1_encode_mux_s"])
    stage1_tensor_handoff_s = float(
        encode_row.get("stage1_tensor_handoff_s", 0.0)
    )
    # GPUWorker.total_duration_s ends before output materialization.  Remove
    # the synchronized video and audio decoder phases to obtain the common H3
    # path through final normalized video/audio latents.
    h3_dit_and_shared_s = inference_s - stage1_decode_s - audio_decode_s
    if min(
        inference_s,
        stage1_decode_s,
        audio_decode_s,
        h3_dit_and_shared_s,
        stage1_wall_s,
    ) <= 0:
        raise RuntimeError(
            "non-positive Stage-1 timing: "
            f"inference={inference_s} decode={stage1_decode_s} "
            f"encode={stage1_encode_mux_s} shared={h3_dit_and_shared_s} "
            f"wall={stage1_wall_s}"
        )
    if PROCESS_DIRECT_HANDOFF:
        if stage1_encode_mux_s != 0.0 or stage1_tensor_handoff_s < 0.0:
            raise RuntimeError(f"invalid direct delivery timing: {encode_row}")
        if encode_row.get("phase") != "discard" and stage1_tensor_handoff_s <= 0.0:
            raise RuntimeError(f"non-positive tensor handoff timing: {encode_row}")
    elif stage1_encode_mux_s <= 0.0 or stage1_tensor_handoff_s != 0.0:
        raise RuntimeError(f"invalid MP4 delivery timing: {encode_row}")

    binding = grid._validate_model_binding(cache_row)
    coverage = binding.get("lora_coverage") or {}
    if (
        int(cache_row.get("scheduled_steps", -1)) != NFE
        or int(cache_row.get("computed_forwards", -1)) != NFE
        or int(cache_row.get("cached_steps", -1)) != 0
        or int(coverage.get("mapped_layers", -1)) != 208
        or int(coverage.get("merged_layers", -1)) != 208
        or int(coverage.get("active_dynamic_layers", -1)) != 0
        or coverage.get("merge_mode") != "merge"
    ):
        raise RuntimeError(f"LoRA/NFE/cache audit failed: {cache_row}")

    if PROCESS_DIRECT_HANDOFF:
        delivery: dict[str, Any] = {
            "tensor_token": returned_reference,
            "tensor_staged": bool(encode_row["tensor_staged"]),
            "destination_cuda_complete": bool(
                encode_row["destination_cuda_complete"]
            ),
        }
    else:
        assert returned_path is not None
        delivery = _probe_media(returned_path, ffprobe, expected_frames)
    metrics = dict(result.metrics or {})
    return {
        "phase": phase,
        "repeat": repeat,
        "duration": float(duration),
        "encoded_duration_s": expected_frames / FPS,
        "output_frames": expected_frames,
        "h3_dit_and_shared_s": h3_dit_and_shared_s,
        "h3_denoise_s": float(cache_row["denoise_total_s"]),
        "stage1_decoder_name": PROCESS_DECODER,
        "stage1_decode_s": stage1_decode_s,
        "stage1_audio_decode_s": audio_decode_s,
        "stage1_encode_mux_s": stage1_encode_mux_s,
        "stage1_tensor_handoff_s": stage1_tensor_handoff_s,
        "stage1_wall_s": stage1_wall_s,
        "worker_pre_save_s": inference_s,
        "client_and_transport_residual_s": (
            stage1_wall_s
            - inference_s
            - (
                stage1_tensor_handoff_s
                if PROCESS_DIRECT_HANDOFF
                else float(encode_row["output_write_s"])
            )
        ),
        "output_file": returned_reference,
        "delivery": delivery,
        "media": None if PROCESS_DIRECT_HANDOFF else delivery,
        "metrics": metrics,
        "denoise_telemetry": cache_row,
        "decoder_telemetry": decoder_row,
        "encode_mux_telemetry": encode_row,
        "model_binding": binding,
    }


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(row[key]) for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, choices=sorted(FRAME_COUNTS), required=True)
    parser.add_argument(
        "--decoder", choices=("taeh3", "official_h3_vae"), required=True
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-asset-root", type=Path, required=True)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-subfolder", default="FL2VA")
    parser.add_argument("--lora-path", type=Path, required=True)
    parser.add_argument("--http-port", type=int, required=True)
    parser.add_argument("--scheduler-port", type=int, required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--nccl-port", type=int, required=True)
    parser.add_argument("--compile-prime-requests", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--hot-repeats", type=int, default=10)
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    if not PROCESS_ACTIVE or grid is None:
        parser.error("H3_DELIVERY_BENCH_ACTIVE=1 is required")
    if args.decoder != PROCESS_DECODER:
        parser.error("--decoder disagrees with H3_DELIVERY_DECODER")
    if args.warmup_requests != 1:
        parser.error("the formal contract requires exactly one complete warmup")
    if args.compile_prime_requests != 1:
        parser.error("the compiled path requires exactly one excluded compile prime")
    if args.hot_repeats < 1:
        parser.error("--hot-repeats must be positive")
    if PROCESS_DIRECT_HANDOFF and (args.duration != 5 or args.decoder != "taeh3"):
        parser.error("direct tensor handoff supports only the fixed 5-second TAEH3 arm")
    if os.environ.get("H3_GRID_MODEL_PROFILE") != PROFILE:
        parser.error(f"H3_GRID_MODEL_PROFILE must be {PROFILE}")
    if os.environ.get("H3_GRID_CACHE_MODE") != "none":
        parser.error("Stage-1 delivery benchmark requires cache mode none")
    if os.environ.get("H3_GRID_COMPILE") != "1":
        parser.error("Stage-1 delivery benchmark requires compile enabled")

    for raw, label in (
        (PROCESS_DECODER_TELEMETRY, "decoder telemetry"),
        (PROCESS_ENCODE_TELEMETRY, "encode telemetry"),
        (os.environ.get("H3_GRID_TELEMETRY", ""), "denoise telemetry"),
    ):
        if not raw:
            parser.error(f"{label} path is missing")

    args.manifest = args.manifest.resolve()
    args.source_asset_root = args.source_asset_root.resolve()
    args.out = args.out.resolve()
    args.lora_path = args.lora_path.resolve()
    # Import-time overlays create telemetry parent directories in both the
    # client and spawned workers.  The Slurm worker already fail-closes if the
    # cell existed before launch, so accepting this empty directory is safe.
    args.out.mkdir(parents=True, exist_ok=True)
    expected_frames = FRAME_COUNTS[args.duration]
    items = grid._load_manifest(args.manifest, (args.prompt_index,))
    item = items[0]
    images = grid._prepare_images(
        items=items,
        source_asset_root=args.source_asset_root,
        output_dir=args.out / "_input",
        width=WIDTH,
        height=HEIGHT,
    )
    image_path = images[args.prompt_index]
    runtime = grid._validate_runtime(args.lora_path)

    cache_telemetry_path = Path(os.environ["H3_GRID_TELEMETRY"])
    decoder_telemetry_path = Path(PROCESS_DECODER_TELEMETRY)
    encode_telemetry_path = Path(PROCESS_ENCODE_TELEMETRY)
    for telemetry_path in (
        cache_telemetry_path,
        decoder_telemetry_path,
        encode_telemetry_path,
    ):
        if telemetry_path.exists():
            raise RuntimeError(f"refusing stale telemetry file {telemetry_path}")
    counters = {"cache": 0, "decoder": 0, "encode": 0}

    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

    ports = (args.http_port, args.scheduler_port, args.master_port, args.nccl_port)
    if len(set(ports)) != 4:
        parser.error("service ports must be distinct")
    generator_kwargs: dict[str, Any] = {
        "local_mode": True,
        "model_path": args.model_path,
        "model_subfolder": args.model_subfolder,
        "model_variant": "fl2va",
        "revision": grid.PINNED_MODEL_REVISION,
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
        "enable_torch_compile": True,
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
        "lora_path": str(args.lora_path),
        "lora_nickname": PROFILE,
        "lora_scale": 8.0 / 128.0,
        "lora_merge_mode": "merge",
        "lora_target_modules": ["qkv_proj", "out_proj", "mlp.fc1", "mlp.fc2"],
    }
    # The HSG production checkout predates the explicit decode-precision CLI
    # audit helper, while the newer mirrored checkout has it.  Preserve the
    # exact stock behavior on the former and validate the explicit value on
    # the latter.
    precision_helper = getattr(grid, "_video_vae_decode_precision_for_mode", None)
    if callable(precision_helper):
        generator_kwargs["vae_decode_precision"] = precision_helper(
            grid.PROCESS_DECODER_FULL_DTYPE
        )

    load_started = time.perf_counter()
    generator = DiffGenerator.from_pretrained(**generator_kwargs)
    model_load_s = time.perf_counter() - load_started
    try:
        service_ports = grid._validate_ports(generator, ports)
        precision_audit = getattr(grid, "_audit_video_vae_decode_precision", None)
        if callable(precision_audit):
            decode_precision = precision_audit(
                generator,
                decoder_full_dtype=grid.PROCESS_DECODER_FULL_DTYPE,
                expected_precision=generator_kwargs["vae_decode_precision"],
            )
        else:
            decode_precision = {
                "decoder_full_dtype": "stock",
                "video_vae_decode_precision": "pipeline_default",
                "binding_verified": "resolved inside pinned decoder stage",
            }
        videos_dir = args.out / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        # The production VAE compile overlay is installed at the denoise stage,
        # after the first-frame encoder already ran once.  A dedicated compile
        # prime lets it wrap the encoder/decoder; the following complete warmup
        # executes those wrapped graphs.  Both are excluded, leaving all hot
        # repeats free of lazy compile/autotune work.
        compile_prime_output = videos_dir / "compile_prime_excluded.mp4"
        compile_prime = _run_one(
            generator=generator,
            request=_sampling_params(
                item=item,
                image_path=image_path,
                duration=args.duration,
                output_dir=videos_dir,
                output_name=compile_prime_output.name,
            ),
            expected_output=compile_prime_output,
            phase="compile_prime_excluded",
            repeat=0,
            duration=args.duration,
            expected_frames=expected_frames,
            cache_telemetry_path=cache_telemetry_path,
            decoder_telemetry_path=decoder_telemetry_path,
            encode_telemetry_path=encode_telemetry_path,
            counters=counters,
            ffprobe=args.ffprobe,
        )

        warmup_output = videos_dir / "warmup_excluded.mp4"
        warmup = _run_one(
            generator=generator,
            request=_sampling_params(
                item=item,
                image_path=image_path,
                duration=args.duration,
                output_dir=videos_dir,
                output_name=warmup_output.name,
            ),
            expected_output=warmup_output,
            phase="warmup_excluded",
            repeat=0,
            duration=args.duration,
            expected_frames=expected_frames,
            cache_telemetry_path=cache_telemetry_path,
            decoder_telemetry_path=decoder_telemetry_path,
            encode_telemetry_path=encode_telemetry_path,
            counters=counters,
            ffprobe=args.ffprobe,
        )

        hot_repeats: list[dict[str, Any]] = []
        for repeat in range(args.hot_repeats):
            output = videos_dir / f"hot_{repeat:02d}.mp4"
            hot_repeats.append(
                _run_one(
                    generator=generator,
                    request=_sampling_params(
                        item=item,
                        image_path=image_path,
                        duration=args.duration,
                        output_dir=videos_dir,
                        output_name=output.name,
                    ),
                    expected_output=output,
                    phase="hot",
                    repeat=repeat,
                    duration=args.duration,
                    expected_frames=expected_frames,
                    cache_telemetry_path=cache_telemetry_path,
                    decoder_telemetry_path=decoder_telemetry_path,
                    encode_telemetry_path=encode_telemetry_path,
                    counters=counters,
                    ffprobe=args.ffprobe,
                )
            )
            _write_json(
                args.out / "progress.json",
                {
                    "status": "running",
                    "hot_completed": repeat + 1,
                    "hot_expected": args.hot_repeats,
                },
            )
    finally:
        generator.shutdown()

    decoder_display = (
        f"madebyollin/taeh3@{TAEH3_COMMIT}"
        if args.decoder == "taeh3"
        else "MiniMax-H3 official video VAE"
    )
    prompt_sha = hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest()
    result = {
        "schema_version": 1,
        "status": "complete",
        "kind": (
            "sglang_minimax_h3_stage1_direct_tensor_single_gb200_hot"
            if PROCESS_DIRECT_HANDOFF
            else "sglang_minimax_h3_stage1_delivery_single_gb200_hot"
        ),
        # Required page-facing fields: each is the median of ten hot repeats.
        "duration": float(args.duration),
        "output_frames": expected_frames,
        "h3_dit_and_shared_s": _median(hot_repeats, "h3_dit_and_shared_s"),
        "stage1_decoder_name": decoder_display,
        "stage1_decode_s": _median(hot_repeats, "stage1_decode_s"),
        "stage1_encode_mux_s": _median(hot_repeats, "stage1_encode_mux_s"),
        "stage1_tensor_handoff_s": _median(
            hot_repeats, "stage1_tensor_handoff_s"
        ),
        "stage1_wall_s": _median(hot_repeats, "stage1_wall_s"),
        "summary_method": f"median_of_{args.hot_repeats}_hot_repeats",
        "wall_boundary": (
            "start immediately before DiffGenerator.generate; stop after Stage-2 "
            "acknowledges the staged tensors as destination-CUDA-complete; excludes "
            "model load, startup LoRA merge, compile, warmup, and result serialization"
            if PROCESS_DIRECT_HANDOFF
            else "start immediately before DiffGenerator.generate; stop after the final "
            "MP4 writer/muxer has returned and the non-empty file can be opened for "
            "reading; excludes model load, startup LoRA merge, compile, warmup, "
            "ffprobe, and result serialization"
        ),
        "encoded_duration_s": expected_frames / FPS,
        "fps": FPS,
        "resolution": [WIDTH, HEIGHT],
        "api_sigma_points": SIGMA_POINTS,
        "actual_transformer_forwards": NFE,
        "video_sigmas": [1.0, 0.972972972972973, 0.9230769230769231, 0.8, 0.0],
        "audio_sigmas": [1.0, 0.9, 0.75, 0.5, 0.0],
        "prompt": {
            "index": args.prompt_index,
            "id": item.get("id"),
            "sha256": prompt_sha,
            "seed": int(item["seed"]),
            "first_frame": str(image_path),
        },
        "model": {
            "path": args.model_path,
            "profile": PROFILE,
            "lora_path": str(args.lora_path),
            "lora_scale": 8.0 / 128.0,
            "lora_merge_mode": "merge",
        },
        "execution": {
            "framework": "SGLang multimodal_gen",
            "sglang_commit": grid.PINNED_SGLANG_COMMIT,
            "visible_gpus_per_service": 1,
            "torch_compile": True,
            "cache_mode": "none",
            "attention": "dense",
            "handoff_mode": (
                "direct_tensor" if PROCESS_DIRECT_HANDOFF else "mp4"
            ),
            "service_ports": service_ports,
            "decode_precision": decode_precision,
            "delivery_overlay": PROCESS_DELIVERY_OVERLAY,
            "runtime": runtime,
        },
        "excluded": {
            "model_load_s": model_load_s,
            "compile_prime": compile_prime,
            "warmup": warmup,
        },
        "hot_repeats": hot_repeats,
    }
    _write_json(args.out / "benchmark.json", result)
    _write_json(
        args.out / "progress.json",
        {
            "status": "complete",
            "hot_completed": args.hot_repeats,
            "hot_expected": args.hot_repeats,
        },
    )
    print(json.dumps({key: result[key] for key in (
        "duration",
        "output_frames",
        "h3_dit_and_shared_s",
        "stage1_decoder_name",
        "stage1_decode_s",
        "stage1_encode_mux_s",
        "stage1_tensor_handoff_s",
        "stage1_wall_s",
    )}, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
