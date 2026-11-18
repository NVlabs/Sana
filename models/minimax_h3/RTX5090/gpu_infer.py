#!/usr/bin/env python3
"""Run the registered single-RTX-5090 MiniMax-H3 profile."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


RUNTIME_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUNTIME_ROOT.parents[2]
sys.path.insert(0, str(RUNTIME_ROOT))

from registration import register_runtime  # noqa: E402


# Multiprocessing uses spawn. Keeping registration at module scope makes the
# same local model/pipeline entries available in every SGLang worker process.
PROFILE = register_runtime()

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


def _result_record(result) -> dict[str, Any]:
    metrics = dict(result.metrics or {})
    peak_memory_mb = result.peak_memory_mb
    return {
        "inference_time_s": _metric_seconds(metrics, result.generation_time),
        "generation_wall_s": float(result.generation_time),
        "peak_memory_mb": (
            None if peak_memory_mb is None else float(peak_memory_mb)
        ),
        "output_file": result.output_file_path,
        "metrics": metrics,
    }


def _sampling_params(
    *,
    prompt: str,
    output_path: Path,
    output_file_name: str,
    steps: int,
    duration_s: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "task": "t2va",
        "conditions": [],
        "target": {
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": float(duration_s),
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": steps,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": seed,
        "output_path": str(output_path),
        "output_file_name": output_file_name,
        "save_output": True,
        "return_file_paths_only": True,
    }


def _run_request(
    generator: DiffGenerator,
    *,
    label: str,
    epoch_file: Path,
    **sampling_params: Any,
) -> dict[str, Any]:
    epoch_file.write_text(label + "\n", encoding="utf-8")
    result = generator.generate(sampling_params_kwargs=sampling_params)
    if result is None or isinstance(result, list):
        raise RuntimeError(f"Expected one MiniMax-H3 result, got {result!r}")
    return _result_record(result)


def main() -> int:
    model_path = os.environ.get("H3_MODEL_PATH", "MiniMaxAI/MiniMax-H3")
    # The FL2VA weights sit in a subfolder of the MiniMax-H3 repository, so the
    # subfolder depends on whether H3_MODEL_PATH already points inside it. Same
    # rule as the H100 and A100 cells; this one hardcoded ".", which meant a
    # repository id resolved to the repo root and the loader stopped on
    # "does not contain model_index.json" -- only a path that already ended in
    # FL2VA worked, and nothing said so outside the README.
    model_subfolder = os.environ.get("H3_MODEL_SUBFOLDER")
    if not model_subfolder:
        model_subfolder = "." if Path(model_path).name.lower() == "fl2va" else "FL2VA"

    output_dir = Path(os.environ.get("OUT_DIR", str(RUNTIME_ROOT / "outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)
    warmup_dir = output_dir / "warmup"
    warmup_dir.mkdir(exist_ok=True)

    prompt, prompt_source = _load_prompt()
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    measured_steps = int(os.environ.get("H3_MEASURED_NUM_STEPS", "50"))
    warmup_default = "5" if PROFILE == "dense" else str(measured_steps)
    warmup_steps = int(os.environ.get("H3_WARMUP_NUM_STEPS", warmup_default))
    duration_s = int(os.environ.get("H3_DURATION_SECONDS", "5"))
    seed = int(os.environ.get("H3_SEED", "0"))
    warmup_seed = int(os.environ.get("H3_WARMUP_SEED", str(seed + 1)))
    master_port = int(os.environ.get("H3_MASTER_PORT", "30005"))
    epoch_file = Path(
        os.environ.get("SOL_ATTN_REQUEST_EPOCH_FILE", output_dir / "request_epoch.txt")
    )

    compile_enabled = PROFILE == "fullopt"
    generator = DiffGenerator.from_pretrained(
        local_mode=True,
        model_path=model_path,
        model_subfolder=model_subfolder,
        num_gpus=1,
        tp_size=1,
        ulysses_degree=1,
        performance_mode="memory",
        layerwise_offload_components=["dit", "text_encoder", "vae"],
        dit_offload_prefetch_size=int(
            os.environ.get("H3_DIT_OFFLOAD_PREFETCH_SIZE", "1")
        ),
        dit_layerwise_resident_layers=int(
            os.environ.get("H3_DIT_RESIDENT_LAYERS", "0")
        ),
        pin_cpu_memory=False,
        enable_torch_compile=compile_enabled,
        regional_compile=compile_enabled,
        server_warmup=False,
        master_port=master_port,
    )
    try:
        warmup = _run_request(
            generator,
            label=f"warmup-{PROFILE}",
            epoch_file=epoch_file,
            **_sampling_params(
                prompt=prompt,
                output_path=warmup_dir,
                output_file_name="warmup.mp4",
                steps=warmup_steps,
                duration_s=duration_s,
                seed=warmup_seed,
            ),
        )
        measured = _run_request(
            generator,
            label=f"measured-{PROFILE}",
            epoch_file=epoch_file,
            **_sampling_params(
                prompt=prompt,
                output_path=output_dir,
                output_file_name="out.mp4",
                steps=measured_steps,
                duration_s=duration_s,
                seed=seed,
            ),
        )
    finally:
        generator.shutdown()

    benchmark = {
        "profile": PROFILE,
        "model": "MiniMaxAI/MiniMax-H3",
        "partition": "FL2VA",
        "dtype": "bfloat16",
        "hardware": "1x RTX 5090",
        "workload": {
            "task": "t2va",
            "width": 1344,
            "height": 768,
            "duration_s": duration_s,
            "measured_steps": measured_steps,
            "seed": seed,
            "prompt_source": prompt_source,
            "prompt_sha256": prompt_sha256,
        },
        "warmup": warmup,
        "measured": measured,
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
