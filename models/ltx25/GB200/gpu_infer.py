#!/usr/bin/env python3
"""Persistent 4-GPU LTX-2.5 inference driver for the GB200 runtime.

The worker fleet and all model components are created once.  Warmup requests run
through the same path as measured requests, but model loading, compilation,
autotuning, and warmup are excluded from the reported steady-state latency.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from ltx_pipelines.ti2vid_two_stages_mgpu import (
    MGPUController,
    MultiModalGuiderParams,
    TI2VidTwoStagesRunner,
)
from ltx_pipelines.utils.args import (
    add_generated_keyframes_arg,
    default_2_stage_arg_parser,
    resolve_cli_params,
)
from ltx_pipelines.utils.constants import TDP_DISTILLED_SIGMAS


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _prompt_paths() -> list[Path]:
    raw = os.environ.get("LTX25_PROMPT_FILES", "")
    paths = [Path(value) for value in raw.split(os.pathsep) if value]
    if not paths:
        raise ValueError("LTX25_PROMPT_FILES must contain at least one prompt path")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing prompt files: {missing}")
    return paths


def _assert_hardware() -> None:
    visible = torch.cuda.device_count()
    if visible != 4:
        raise RuntimeError(
            f"the LTX-2.5 GB200 runtime requires exactly four visible GPUs; found {visible}"
        )
    capabilities = [torch.cuda.get_device_capability(index) for index in range(visible)]
    if any(capability != (10, 0) for capability in capabilities):
        if os.environ.get("LTX25_ALLOW_OTHER_HARDWARE", "0") != "1":
            raise RuntimeError(
                "the GB200 runtime requires compute capability 10.0; "
                f"found {capabilities}. Set LTX25_ALLOW_OTHER_HARDWARE=1 only for a diagnostic."
            )


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def _load_rank0_timing(base: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    pattern = f"{base.name}.*.requests.jsonl"
    for path in sorted(base.parent.glob(pattern)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("rank") == 0:
                rows[int(row["request"])] = row
    return rows


def _aggregate_stages(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for item in row.get("stages", []):
            values.setdefault(str(item["name"]), []).append(float(item["total"]))
        for item in row.get("tail", []):
            values.setdefault(str(item["name"]), []).append(float(item["total"]))
        values.setdefault("VideoIterator", []).append(float(row.get("video_iterator", 0.0)))
        values.setdefault("OutputEncode", []).append(float(row.get("output_encode_total", 0.0)))

    result = {f"{name}_s": _round(_mean(samples)) for name, samples in values.items()}
    video_vae = values.get("VideoDecoder", [])
    audio_vae = values.get("AudioDecoder", [])
    if video_vae and audio_vae and len(video_vae) == len(audio_vae):
        result["VAE_s"] = _round(_mean([v + a for v, a in zip(video_vae, audio_vae)]))
    return result


def _write_run_config(args: Any, prompt_paths: list[Path], warmup: int, measure: int) -> None:
    out_dir = Path(args.output_path).resolve().parent
    sigmas = [float(value) for value in TDP_DISTILLED_SIGMAS]
    config = {
        "model": "LTX-2.5 public dev BF16 + distilled LoRA 450",
        "hardware": "4x NVIDIA GB200",
        "profile": os.environ.get("LTX25_PROFILE", "default5s"),
        "variant": os.environ.get("LTX25_VARIANT", "unknown"),
        "final_width": args.width,
        "final_height": args.height,
        "stage1_width": args.width // 2,
        "stage1_height": args.height // 2,
        "num_frames": args.num_frames,
        "fps": args.frame_rate,
        "stage1_steps": args.num_inference_steps,
        "stage2_sigmas": sigmas,
        "stage2_updates": len(sigmas) - 1,
        "seed": args.seed,
        "warmup_requests": warmup,
        "measure_requests": measure,
        "prompt_files": [str(path) for path in prompt_paths],
        "stage1_parallelism": os.environ.get("LTX_S1_PARALLEL", "sp"),
        "stage2_parallelism": "2x2 tiled data parallel",
        "vae_parallelism": "2x2 distributed decode",
        "cache": {
            "policy": os.environ.get("LTX_STACK_CACHE", "off"),
            "threshold": float(os.environ.get("LTX_CACHE_THRESH", "0")),
            "warmup_steps": int(os.environ.get("LTX_CACHE_WARMUP", "1")),
            "max_consecutive": int(os.environ.get("LTX_CACHE_MAXCONSEC", "10")),
        },
        "compile": {
            "enabled": os.environ.get("LTX25_COMPILE", "0") == "1",
            "mode": "max-autotune-no-cudagraphs",
            "fullgraph": False,
            "capture": False,
        },
        "sol_attention": False,
        "timing_scope": "hot request wall including text encoder, both denoising stages, upsampler, VAE, and output encode",
        "excluded_from_timing": ["model loading", "torch.compile", "autotuning", "warmup requests"],
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    params = resolve_cli_params()
    parser = add_generated_keyframes_arg(
        default_2_stage_arg_parser(params=params, supports_auto_duration=True)
    )
    args = parser.parse_args()

    _assert_hardware()
    prompt_paths = _prompt_paths()
    warmup_requests = _env_int("LTX25_WARMUP_REQUESTS", 1)
    measure_requests = _env_int("LTX25_MEASURE_REQUESTS", 1)
    if warmup_requests < 1 or measure_requests < 1:
        raise ValueError("LTX25_WARMUP_REQUESTS and LTX25_MEASURE_REQUESTS must both be positive")

    output = Path(args.output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_run_config(args, prompt_paths, warmup_requests, measure_requests)

    vae_queue = torch.multiprocessing.get_context("spawn").SimpleQueue()
    controller = MGPUController(TI2VidTwoStagesRunner)
    load_started = time.perf_counter()
    controller.start(
        model_paths=args.model_paths,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        spatial_upsampler_path=args.spatial_upsampler_path,
        vae_queue=vae_queue,
        distilled_lora_path=args.distilled_lora[0].path,
        compilation_config=args.compile,
        diffvae_optimization=args.diffvae_optimization,
    )
    startup_s = time.perf_counter() - load_started
    print(f"[ltx25] startup={startup_s:.3f}s (excluded)", flush=True)

    request_rows: list[dict[str, Any]] = []
    total_requests = warmup_requests + measure_requests
    try:
        for index in range(total_requests):
            prompt_path = prompt_paths[index % len(prompt_paths)]
            prompt = prompt_path.read_text(encoding="utf-8").strip()
            phase = "warmup" if index < warmup_requests else "measured"
            measured_index = index - warmup_requests
            if phase == "measured" and measured_index == 0:
                request_output = output
            else:
                request_output = output.parent / f"{phase}-{index + 1:02d}-{prompt_path.stem}.mp4"

            print(
                f"[ltx25] request-start index={index + 1}/{total_requests} "
                f"phase={phase} prompt={prompt_path.name}",
                flush=True,
            )
            started = time.perf_counter()
            for _ in controller.stream(
                output_path=str(request_output),
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                frame_rate=args.frame_rate,
                num_inference_steps=args.num_inference_steps,
                video_guider_params=MultiModalGuiderParams(
                    cfg_scale=args.video_cfg_guidance_scale,
                    stg_scale=args.video_stg_guidance_scale,
                    rescale_scale=args.video_rescale_scale,
                    modality_scale=args.a2v_guidance_scale,
                    skip_step=args.video_skip_step,
                    stg_blocks=args.video_stg_blocks,
                ),
                audio_guider_params=MultiModalGuiderParams(
                    cfg_scale=args.audio_cfg_guidance_scale,
                    stg_scale=args.audio_stg_guidance_scale,
                    rescale_scale=args.audio_rescale_scale,
                    modality_scale=args.v2a_guidance_scale,
                    skip_step=args.audio_skip_step,
                    stg_blocks=args.audio_stg_blocks,
                ),
                images=args.images,
                enhance_prompt=args.enhance_prompt,
                enhance_static_cache=args.enhance_static_cache,
                hdr=args.hdr,
                generated_keyframes=args.num_generated_keyframes,
            ):
                pass
            elapsed = time.perf_counter() - started
            request_rows.append(
                {
                    "request": index + 1,
                    "phase": phase,
                    "prompt": prompt_path.name,
                    "request_s": elapsed,
                    "output": str(request_output),
                }
            )
            print(
                f"[ltx25] request-done index={index + 1}/{total_requests} "
                f"phase={phase} request_s={elapsed:.3f} output={request_output}",
                flush=True,
            )
            if phase == "warmup" and os.environ.get("LTX25_KEEP_WARMUP", "0") != "1":
                request_output.unlink(missing_ok=True)
    finally:
        controller.shutdown()

    measured = [row for row in request_rows if row["phase"] == "measured"]
    measured_times = [float(row["request_s"]) for row in measured]
    timing_base = Path(os.environ.get("LTX_TIME_FILE", str(output.parent / "timing")))
    rank0_by_request = _load_rank0_timing(timing_base)
    measured_worker_rows = [
        rank0_by_request[row["request"]]
        for row in measured
        if row["request"] in rank0_by_request
    ]

    cache_hits = [
        int(row["cache"]["skipped"])
        for row in measured_worker_rows
        if row.get("cache") is not None
    ]
    cache_calls = [
        int(row["cache"]["calls"])
        for row in measured_worker_rows
        if row.get("cache") is not None
    ]
    benchmark = {
        "schema_version": 1,
        "variant": os.environ.get("LTX25_VARIANT", "unknown"),
        "profile": os.environ.get("LTX25_PROFILE", "default5s"),
        "hardware": [torch.cuda.get_device_name(index) for index in range(4)],
        "host": platform.node(),
        "torch": torch.__version__,
        "startup_s_excluded": round(startup_s, 3),
        "warmup_requests_excluded": warmup_requests,
        "measured_requests": len(measured_times),
        "request_s": [round(value, 3) for value in measured_times],
        "request_s_mean": round(statistics.fmean(measured_times), 3),
        "request_s_median": round(statistics.median(measured_times), 3),
        "request_s_min": round(min(measured_times), 3),
        "request_s_max": round(max(measured_times), 3),
        "stage_s_mean_rank0": _aggregate_stages(measured_worker_rows),
        "cache_hits_per_request": cache_hits,
        "cache_calls_per_request": cache_calls,
        "timing_scope": "steady-state end-to-end request wall; load, compile, autotune, and warmup excluded",
        "requests": [
            {**row, "request_s": round(float(row["request_s"]), 3)} for row in measured
        ],
    }
    (output.parent / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(benchmark, indent=2, sort_keys=False), flush=True)
    print(f"[ltx25] wrote {output.parent / 'benchmark.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
