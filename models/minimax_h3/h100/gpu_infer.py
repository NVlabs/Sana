#!/usr/bin/env python3
"""Run one registered MiniMax-H3 profile through SGLang's offline API."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


RUNTIME_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUNTIME_ROOT.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.minimax_h3.h100.registration import (  # noqa: E402
    register_runtime,
)


# SGLang launches workers with multiprocessing spawn. Registration therefore
# stays at module scope so every worker receives the same local model class.
HARDWARE, PROFILE = register_runtime()

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (  # noqa: E402
    DiffGenerator,
)


DEFAULT_PROMPT_FILE = REPO_ROOT / "models/minimax_h3/demo_prompt.json"


def _load_prompt() -> tuple[str, str]:
    direct = os.environ.get("H3_PROMPT")
    if direct:
        return direct, "H3_PROMPT"
    prompt_file = Path(os.environ.get("H3_PROMPT_FILE", str(DEFAULT_PROMPT_FILE)))
    payload = json.loads(prompt_file.read_text(encoding="utf-8"))
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Prompt file has no non-empty prompt: {prompt_file}")
    return prompt, str(prompt_file)


def _metric_seconds(metrics: dict[str, Any], fallback: float) -> float:
    if "total_duration_s" in metrics:
        return float(metrics["total_duration_s"])
    if "total_duration_ms" in metrics:
        return float(metrics["total_duration_ms"]) / 1000.0
    return float(fallback)


def _result_record(result: Any) -> dict[str, Any]:
    metrics = dict(result.metrics or {})
    peak_memory = result.peak_memory_mb
    if isinstance(peak_memory, (list, tuple)):
        peak_memory = [float(value) for value in peak_memory]
    elif peak_memory is not None:
        peak_memory = float(peak_memory)
    return {
        "inference_time_s": _metric_seconds(metrics, result.generation_time),
        "generation_wall_s": float(result.generation_time),
        "peak_memory_mb_per_gpu": peak_memory,
        "output_file": result.output_file_path,
        "metrics": metrics,
    }


def _sampling_params(
    *,
    prompt: str,
    output_dir: Path,
    output_name: str,
    steps: int,
    duration_s: float,
    seed: int,
    save_output: bool,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "task": "t2va",
        "conditions": [],
        "target": {
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": duration_s,
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": steps,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": seed,
        "output_path": str(output_dir),
        "output_file_name": output_name,
        "save_output": save_output,
        "return_file_paths_only": True,
    }


def _run_request(
    generator: DiffGenerator,
    *,
    phase: str,
    epoch_file: Path,
    **sampling_params: Any,
) -> dict[str, Any]:
    epoch_file.write_text(
        f"{PROFILE.name}:{phase}:{time.time_ns()}\n", encoding="utf-8"
    )
    result = generator.generate(sampling_params_kwargs=sampling_params)
    if result is None or isinstance(result, list):
        raise RuntimeError(f"Expected one MiniMax-H3 result, got {result!r}")
    return _result_record(result)


def _event_files(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("sol_events_rank*.jsonl"))


def _collect_events(output_dir: Path) -> dict[str, Any] | None:
    if not PROFILE.sol_attention:
        return None
    events: list[dict[str, Any]] = []
    for path in _event_files(output_dir):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))

    expected_ranks = set(range(int(os.environ["H3_NUM_GPUS"])))
    measured_sparse_ranks = {
        int(event["rank"])
        for event in events
        if event.get("event") == "first_sparse_forward"
        and ":measured:" in str(event.get("request_epoch", ""))
    }
    if measured_sparse_ranks != expected_ranks:
        raise RuntimeError(
            "Sparse attention did not execute on every measured rank: "
            f"expected {sorted(expected_ranks)}, got {sorted(measured_sparse_ranks)}"
        )

    density_events = [
        event for event in events if event.get("event") == "route_density"
    ]
    if not density_events:
        raise RuntimeError("Sparse profile produced no warmup route-density event")
    weights = [
        int(event["blocks"]) * int(event["heads"]) for event in density_events
    ]
    total_weight = sum(weights)
    density = {
        "scope": "first sparse attention call of the warmup request",
        "threshold_density": sum(
            float(event["threshold_density"]) * weight
            for event, weight in zip(density_events, weights)
        )
        / total_weight,
        "effective_density": sum(
            float(event["effective_density"]) * weight
            for event, weight in zip(density_events, weights)
        )
        / total_weight,
        "sequence_tokens": int(density_events[0]["sequence_tokens"]),
        "sink_tokens": int(density_events[0]["sink_tokens"]),
    }

    cache_kind = None
    decision_event = None
    if PROFILE.cache == "easycache":
        cache_kind = "easycache"
        decision_event = "easycache_decision"
    elif PROFILE.cache == "firstblock":
        cache_kind = "firstblockcache"
        decision_event = "firstblockcache_decision"
    cache = None
    if decision_event is not None:
        decisions = [
            event
            for event in events
            if event.get("event") == decision_event
            and int(event.get("rank", -1)) == 0
            and ":measured:" in str(event.get("request_epoch", ""))
        ]
        reuse = sum(event.get("action") == "reuse" for event in decisions)
        cache = {
            "kind": cache_kind,
            "calls": len(decisions),
            "compute": len(decisions) - reuse,
            "reuse": reuse,
            "reuse_rate": reuse / len(decisions) if decisions else 0.0,
        }
    return {
        "measured_sparse_ranks": sorted(measured_sparse_ranks),
        "route_density": density,
        "cache": cache,
    }


def _validate_runtime() -> dict[str, Any]:
    import torch
    import triton

    expected_torch = os.environ.get("H3_EXPECTED_TORCH", "2.11.0+cu130")
    expected_triton = os.environ.get("H3_EXPECTED_TRITON", "3.6.0")
    if torch.__version__ != expected_torch:
        raise RuntimeError(
            f"Expected torch {expected_torch}, found {torch.__version__}"
        )
    if triton.__version__ != expected_triton:
        raise RuntimeError(
            f"Expected Triton {expected_triton}, found {triton.__version__}"
        )
    gpu_count = torch.cuda.device_count()
    if gpu_count != 4:
        raise RuntimeError(f"MiniMax-H3 release profiles require 4 visible GPUs, got {gpu_count}")
    devices = []
    for index in range(gpu_count):
        capability = tuple(torch.cuda.get_device_capability(index))
        if capability != HARDWARE.capability:
            raise RuntimeError(
                f"GPU {index} has capability {capability}; expected {HARDWARE.capability}"
            )
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(capability),
            }
        )
    backend = None
    if PROFILE.sol_attention:
        from sol_attn import get_sol_attn_backend

        backend = get_sol_attn_backend(0)
        if backend != HARDWARE.sol_backend:
            raise RuntimeError(
                f"Expected Sol-Attn backend {HARDWARE.sol_backend}, got {backend}"
            )
    return {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "devices": devices,
        "sol_attn_backend": backend,
    }


def main() -> int:
    output_dir = Path(os.environ.get("OUT_DIR", str(RUNTIME_ROOT / "outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in _event_files(output_dir):
        path.unlink()

    prompt, prompt_source = _load_prompt()
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    measured_steps = int(os.environ.get("H3_MEASURED_NUM_STEPS", "50"))
    warmup_steps = int(os.environ.get("H3_WARMUP_NUM_STEPS", str(measured_steps)))
    duration_s = float(os.environ.get("H3_DURATION_SECONDS", "5.166667"))
    seed = int(os.environ.get("H3_SEED", "0"))
    warmup_seed = int(os.environ.get("H3_WARMUP_SEED", str(seed + 10_000)))
    os.environ.setdefault("H3_EASYCACHE_NUM_FORWARDS", str(measured_steps - 1))
    epoch_file = output_dir / "request_epoch.txt"
    os.environ["H3_REQUEST_EPOCH_FILE"] = str(epoch_file)
    os.environ["H3_SOL_EVENT_LOG"] = str(output_dir / "sol_events_rank{rank}.jsonl")

    runtime = _validate_runtime()
    model_path = os.environ.get("H3_MODEL_PATH", "MiniMaxAI/MiniMax-H3")
    model_subfolder = os.environ.get("H3_MODEL_SUBFOLDER")
    if not model_subfolder:
        model_subfolder = "." if Path(model_path).name.lower() == "fl2va" else "FL2VA"

    warmup_path = output_dir / "warmup.mp4"
    generator = DiffGenerator.from_pretrained(
        local_mode=True,
        model_path=model_path,
        model_subfolder=model_subfolder,
        model_variant="fl2va",
        revision=os.environ["H3_MODEL_REVISION"],
        num_gpus=4,
        tp_size=1,
        ulysses_degree=4,
        enable_cfg_parallel=False,
        performance_mode="speed",
        use_fsdp_inference=True,
        layerwise_offload_components=[],
        enable_torch_compile=False,
        regional_compile=False,
        server_warmup=False,
        master_port=int(os.environ.get("H3_MASTER_PORT", "30005")),
    )
    try:
        warmup = _run_request(
            generator,
            phase="warmup",
            epoch_file=epoch_file,
            **_sampling_params(
                prompt=prompt,
                output_dir=output_dir,
                output_name="warmup.mp4",
                steps=warmup_steps,
                duration_s=duration_s,
                seed=warmup_seed,
                save_output=True,
            ),
        )
        warmup["output_file"] = None
        measured = _run_request(
            generator,
            phase="measured",
            epoch_file=epoch_file,
            **_sampling_params(
                prompt=prompt,
                output_dir=output_dir,
                output_name="out.mp4",
                steps=measured_steps,
                duration_s=duration_s,
                seed=seed,
                save_output=True,
            ),
        )
    finally:
        generator.shutdown()
        warmup_path.unlink(missing_ok=True)

    execution = _collect_events(output_dir)
    for path in _event_files(output_dir):
        path.unlink()
    epoch_file.unlink(missing_ok=True)

    benchmark = {
        "schema_version": 1,
        "profile": asdict(PROFILE),
        "hardware": asdict(HARDWARE),
        "runtime": runtime,
        "model": {
            "repo_or_path": model_path,
            "subfolder": model_subfolder,
            "revision": os.environ["H3_MODEL_REVISION"],
            "partition": "FL2VA",
            "dtype": "bfloat16",
        },
        "workload": {
            "task": "t2va",
            "width": 1344,
            "height": 768,
            "frames": 124,
            "duration_s": duration_s,
            "measured_steps": measured_steps,
            "seed": seed,
            "prompt_source": prompt_source,
            "prompt_sha256": prompt_sha256,
        },
        "topology": {
            "num_gpus": 4,
            "tensor_parallel": 1,
            "ulysses": 4,
            "fsdp_inference": True,
            "offload": False,
            "torch_compile": False,
            "reorder": False,
        },
        "timing_source": "SGLang GenerationResult.metrics.total_duration_s",
        "memory_source": "SGLang GenerationResult.peak_memory_mb",
        "warmup": warmup,
        "measured": measured,
        "execution": execution,
    }
    benchmark_path = output_dir / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(benchmark, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
