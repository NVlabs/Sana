#!/usr/bin/env python3
"""Run compiled H3 and synchronously hand each result to resident Stage 2.

The production arm stages decoded video/audio tensors from the SGLang GPU
worker before ``DiffGenerator.generate`` returns.  The older MP4 handoff is
kept as an explicit fallback for A/B and recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from handoff_protocol import request


# SGLang uses multiprocessing with the spawn start method.  The spawned GPU
# worker imports this entrypoint as ``__mp_main__`` without calling main().
# Import the benchmark (and therefore its process overlays) at module import
# time whenever the delivery environment is active, matching the original
# benchmark entrypoint's semantics.
if os.environ.get("H3_DELIVERY_BENCH_ACTIVE", "0") == "1":
    import run_stage1_benchmark as stage1
else:
    stage1 = None


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _map_path(path: Path, mapping: str) -> Path:
    source, separator, target = mapping.partition("=")
    if not separator or not source.startswith("/") or not target.startswith("/"):
        raise ValueError("--path-map must be ABS_CONTAINER=ABS_HOST")
    raw = str(path)
    if raw == source:
        return Path(target)
    if raw.startswith(source.rstrip("/") + "/"):
        return Path(target) / raw.removeprefix(source.rstrip("/") + "/")
    raise ValueError(f"Stage-1 output {path} is outside mapped prefix {source}")


def _map_host_path_to_container(path: Path, mapping: str) -> Path:
    """Reverse ``ABS_CONTAINER=ABS_HOST`` for untimed final-media checks."""

    container, separator, host = mapping.partition("=")
    if not separator or not container.startswith("/") or not host.startswith("/"):
        raise ValueError("--path-map must be ABS_CONTAINER=ABS_HOST")
    raw = str(path)
    if raw == host:
        return Path(container)
    if raw.startswith(host.rstrip("/") + "/"):
        return Path(container) / raw.removeprefix(host.rstrip("/") + "/")
    raise ValueError(f"Stage-2 output {path} is outside mapped host prefix {host}")


def _probe_final_av(path: Path) -> dict[str, Any]:
    """Verify the final H.264/AAC contract outside the E2E timing boundary."""

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels",
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
        raise RuntimeError(f"final output must contain one video and one audio: {streams}")
    video = videos[0]
    audio = audios[0]
    if (
        video.get("codec_name") != "h264"
        or (int(video.get("width", 0)), int(video.get("height", 0)))
        != (1344, 768)
        or video.get("avg_frame_rate") != "24/1"
        or audio.get("codec_name") != "aac"
        or int(audio.get("channels", 0)) != 2
        or int(audio.get("sample_rate", 0)) <= 0
    ):
        raise RuntimeError(f"final H.264/AAC media contract failed: {streams}")
    return {"path": str(path), "video": video, "audio": audio}


def _flag_value(arguments: list[str], flag: str) -> str:
    try:
        index = arguments.index(flag)
    except ValueError as exc:
        raise ValueError(f"underlying Stage-1 argument {flag} is required") from exc
    if index + 1 >= len(arguments):
        raise ValueError(f"missing value after {flag}")
    return arguments[index + 1]


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--auth-token", required=True)
    parser.add_argument("--pair-id", type=int, required=True)
    parser.add_argument("--path-map", required=True)
    parser.add_argument(
        "--handoff-mode", choices=("direct_tensor", "mp4"), required=True
    )
    parser.add_argument("--pair-metadata", type=Path, required=True)
    integration, stage1_argv = parser.parse_known_args()

    stage1_out = Path(_flag_value(stage1_argv, "--out"))
    manifest = Path(_flag_value(stage1_argv, "--manifest"))
    prompt_index = int(_flag_value(stage1_argv, "--prompt-index"))
    hot_expected = int(_flag_value(stage1_argv, "--hot-repeats"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError("Stage-1 source manifest must contain an items list")
    selected = next((row for row in items if int(row["index"]) == prompt_index), None)
    if selected is None:
        raise KeyError(f"prompt index {prompt_index} is absent from {manifest}")
    source_metadata = selected.get("source_metadata")
    input_image = (
        source_metadata.get("input_image")
        if isinstance(source_metadata, dict)
        else None
    )
    first_frame_sha256 = (
        input_image.get("image_sha256") if isinstance(input_image, dict) else None
    )
    if (
        not isinstance(first_frame_sha256, str)
        or len(first_frame_sha256) != 64
        or any(character not in "0123456789abcdef" for character in first_frame_sha256)
    ):
        raise ValueError("Stage-1 manifest lacks a lowercase first-frame SHA-256")
    identity = {
        "prompt_id": str(selected["id"]),
        "prompt": str(selected["prompt"]),
        "prompt_sha256": hashlib.sha256(str(selected["prompt"]).encode()).hexdigest(),
        "seed": int(selected["seed"]),
        "source_index": prompt_index,
        "first_frame_sha256": first_frame_sha256,
    }

    # The worker shell installed every overlay environment variable before
    # starting Python, so the import-time block above is authoritative.
    if stage1 is None:
        raise RuntimeError("H3_DELIVERY_BENCH_ACTIVE=1 is required")

    original_run_one = stage1._run_one
    paired_hot: list[dict[str, Any]] = []
    excluded_stage2_warmup: dict[str, Any] | None = None

    def integrated_run_one(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal excluded_stage2_warmup
        phase = str(kwargs["phase"])
        if phase not in {"warmup_excluded", "hot"}:
            return original_run_one(*args, **kwargs)
        operation = "warmup" if phase == "warmup_excluded" else "refine"
        repeat = int(kwargs["repeat"])

        # Insert Stage 2 directly at the generate-return boundary.  In tensor
        # mode the GPU worker has already copied the preprocessed BF16 frames
        # and PCM into the resident Stage-2 process and received a destination-
        # CUDA-complete ACK.  In fallback mode it has closed the Stage-1 MP4.
        # The original _run_one has not yet read telemetry or run ffprobe, so
        # this remains one real monotonic end-to-end interval.
        generator = kwargs["generator"]
        original_generate = generator.generate
        direct: dict[str, Any] = {}

        def generate_then_refine(*generate_args: Any, **generate_kwargs: Any) -> Any:
            e2e_started_ns = time.perf_counter_ns()
            result = original_generate(*generate_args, **generate_kwargs)
            stage1_ready_ns = time.perf_counter_ns()
            returned_reference = str(result.output_file_path)
            handoff_payload: dict[str, Any]
            if integration.handoff_mode == "direct_tensor":
                if not returned_reference.startswith("h3tensor://"):
                    raise RuntimeError(
                        "direct Stage-1 worker did not return an h3tensor token: "
                        f"{returned_reference!r}"
                    )
                handoff_payload = {"tensor_token": returned_reference}
            else:
                returned_path = Path(returned_reference)
                source_path = _map_path(returned_path, integration.path_map)
                handoff_payload = {"source_mp4": str(source_path)}
            handoff_started_ns = time.perf_counter_ns()
            response = request(
                integration.endpoint,
                {
                    "schema_version": 1,
                    "op": operation,
                    "token": integration.auth_token,
                    "pair_id": integration.pair_id,
                    "repeat": repeat,
                    "handoff_mode": integration.handoff_mode,
                    **handoff_payload,
                    **identity,
                },
                connect_timeout_s=1800.0,
                response_timeout_s=1800.0,
            )
            completed_ns = time.perf_counter_ns()
            if response.get("status") != "succeeded":
                raise RuntimeError(f"Stage-2 request failed: {response}")
            direct.update(
                {
                    "stage1_wall_s": (stage1_ready_ns - e2e_started_ns) / 1e9,
                    "handoff_roundtrip_s": (completed_ns - handoff_started_ns) / 1e9,
                    "direct_e2e_wall_s": (completed_ns - e2e_started_ns) / 1e9,
                    "response": response,
                }
            )
            return result

        generator.generate = generate_then_refine
        try:
            row = original_run_one(*args, **kwargs)
        finally:
            generator.generate = original_generate
        if not direct:
            raise RuntimeError("generate boundary wrapper did not execute")
        row["stage1_wall_s"] = direct["stage1_wall_s"]
        delivery_s = (
            float(row["stage1_tensor_handoff_s"])
            if integration.handoff_mode == "direct_tensor"
            else float(row["encode_mux_telemetry"]["output_write_s"])
        )
        row["client_and_transport_residual_s"] = (
            float(row["stage1_wall_s"])
            - float(row["worker_pre_save_s"])
            - delivery_s
        )
        response = direct["response"]
        stage2_wall_s = float(response["result"]["wall_s"])
        control_rpc_residual_s = float(direct["handoff_roundtrip_s"]) - stage2_wall_s
        if control_rpc_residual_s < 0:
            raise RuntimeError(
                "Stage-2 service wall exceeds its measured control roundtrip: "
                f"service={stage2_wall_s} "
                f"roundtrip={direct['handoff_roundtrip_s']}"
            )
        # The final mux is required output.  Probe its AV streams only after
        # the monotonic E2E endpoint above, so correctness checks do not inflate
        # the measured hot latency.  The excluded warmup file has been deleted.
        final_av_probe = None
        if operation == "refine":
            final_host_path = Path(str(response["result"]["output"]))
            final_container_path = _map_host_path_to_container(
                final_host_path, integration.path_map
            )
            final_av_probe = _probe_final_av(final_container_path)
        combined = {
            "repeat": repeat,
            "stage1": row,
            "handoff_roundtrip_s": direct["handoff_roundtrip_s"],
            "control_rpc_residual_s": control_rpc_residual_s,
            "e2e_wall_s": direct["direct_e2e_wall_s"],
            "stage2": response["result"],
            "stage2_output": response["result"]["output"],
            "final_av_probe_untimed": final_av_probe,
        }
        if operation == "warmup":
            excluded_stage2_warmup = combined
        else:
            paired_hot.append(combined)
        return row

    stage1._run_one = integrated_run_one
    old_argv = sys.argv
    try:
        sys.argv = ["run_stage1_benchmark.py", *stage1_argv]
        result = stage1.main()
    finally:
        sys.argv = old_argv
        stage1._run_one = original_run_one

    if result != 0 or len(paired_hot) != hot_expected or excluded_stage2_warmup is None:
        raise RuntimeError(
            f"incomplete pair: stage1_rc={result} hot={len(paired_hot)}/{hot_expected} "
            f"stage2_warmup={excluded_stage2_warmup is not None}"
        )
    benchmark_path = stage1_out / "benchmark.json"
    stage1_benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    output = {
        "schema_version": 1,
        "status": "complete",
        "kind": (
            "h3_taeh3_direct_tensor_to_ltx25_refiner_two_gpu_pair"
            if integration.handoff_mode == "direct_tensor"
            else "h3_taeh3_mp4_to_ltx25_refiner_two_gpu_pair"
        ),
        "handoff_mode": integration.handoff_mode,
        "pair_id": integration.pair_id,
        "hot_repeats": hot_expected,
        "boundary": (
            "one monotonic interval inside DiffGenerator.generate call: immediately before "
            "Stage-1 generation through destination-CUDA-complete tensor staging (or ready "
            "fallback MP4), localhost control handoff, and resident Stage-2 run_diagnostic "
            "return after the single final MP4 mux/verify"
        ),
        "stage1_benchmark": str(benchmark_path),
        "identity": identity,
        "excluded": {"stage2_warmup": excluded_stage2_warmup},
        "hot": paired_hot,
    }
    _atomic_json(integration.pair_metadata, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
