#!/usr/bin/env python3
"""Run the registered LingBot-Video 4-GPU T2V workload.

The vendored LingBot source remains responsible for model execution. This
adapter supplies the fixed workload, launches torchrun, and translates the
result into the repository's canonical run artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable


RUNTIME_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = RUNTIME_DIR / "lingbot_src"
REPO_ROOT = Path(os.environ.get("AUTOVIDEO_REPO_ROOT", RUNTIME_DIR.parent.parent)).resolve()
PHASE_RE = re.compile(
    r"^PHASE\s+(?P<label>[A-Za-z0-9_]+)\s+"
    r"dt=(?P<dt>[0-9]+(?:\.[0-9]+)?)\s+"
    r"total=(?P<total>[0-9]+(?:\.[0-9]+)?)\s*$"
)
REQUIRED_HOT_PHASES = (
    "refiner_preloaded",
    "base_conditions_cached",
    "base_denoise_done",
    "base_vae_saved",
    "refiner_vae_encode_done",
    "refiner_conditions_cached",
    "refiner_denoise_done",
    "refiner_vae_saved",
)
PHASE_STAGE_NAMES = {
    "init_parallel_done": "init_parallel",
    "base_pipe_loaded": "base_model_setup",
    "refiner_preloaded": "refiner_model_setup",
    "base_conditions_cached": "base_condition_encode",
    "base_denoise_done": "host_interval_base_pipeline",
    "base_vae_saved": "host_interval_base_video_export",
    "refiner_vae_encode_done": "host_interval_refiner_input_prepare",
    "refiner_conditions_cached": "host_interval_refiner_condition_prepare",
    "refiner_denoise_done": "host_interval_refiner_pipeline",
    "refiner_vae_saved": "host_interval_refiner_video_export",
}


def env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise SystemExit(f"[lingbot_video] missing required environment variable: {name}")
    return "" if value is None else str(value)


def env_bool(name: str, default: bool) -> bool:
    raw = env(name, "1" if default else "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(
        f"[lingbot_video] {name} must be a boolean value, got {raw!r}"
    )


def attention_kernel() -> str:
    value = env("LINGBOT_ATTN_KERNEL", "fa2").strip().lower()
    if value not in {"fa2", "cudnn"}:
        raise SystemExit(
            f"[lingbot_video] LINGBOT_ATTN_KERNEL must be fa2 or cudnn, got {value!r}"
        )
    if RUNTIME_DIR.name == "lingbot_video_baseline" and value != "fa2":
        raise SystemExit(
            "[lingbot_video] the physically isolated baseline runtime supports only fa2; "
            "use runtime/lingbot_video_optimized for the cudnn candidate"
        )
    return value


def resolve_input(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for root in (REPO_ROOT, SOURCE_ROOT, RUNTIME_DIR):
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return (REPO_ROOT / path).resolve()


def load_source_snapshot() -> dict[str, Any]:
    snapshot_path = RUNTIME_DIR / "SOURCE_SNAPSHOT.json"
    snapshot = json.loads(snapshot_path.read_text())
    for relative, expected in snapshot.get("core_sha256", {}).items():
        path = SOURCE_ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        if actual != expected:
            raise SystemExit(
                f"[lingbot_video] vendored source integrity check failed for {relative}: "
                f"expected {expected}, got {actual}"
            )
    return snapshot


def parse_phase_lines(lines: Iterable[str]) -> dict[str, dict[str, float]]:
    phases: dict[str, dict[str, float]] = {}
    for line in lines:
        match = PHASE_RE.match(line.strip())
        if match:
            phases[match.group("label")] = {
                "dt_s": float(match.group("dt")),
                "total_s": float(match.group("total")),
            }
    return phases


def phase_dt(phases: dict[str, dict[str, float]], label: str) -> float | None:
    record = phases.get(label)
    return None if record is None else float(record["dt_s"])


def sum_if_complete(*values: float | None) -> float | None:
    if any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None))


class PeakMemorySampler:
    def __init__(self) -> None:
        self.peak_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        self._thread = threading.Thread(target=self._run, name="lingbot-gpu-memory", daemon=True)
        self._thread.start()

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.peak_mib

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                )
                values = [int(item.strip()) for item in proc.stdout.splitlines() if item.strip()]
                if values:
                    current = max(values)
                    self.peak_mib = current if self.peak_mib is None else max(self.peak_mib, current)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self._stop.wait(2.0)


def append_flag(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def validate_topology(nproc: int, cp_degree: int, fsdp_enabled: bool) -> None:
    if nproc != 4:
        raise SystemExit(f"[lingbot_video] registered workload requires exactly 4 GPUs, got {nproc}")
    if cp_degree not in {1, nproc}:
        raise SystemExit(
            "[lingbot_video] this registered runner uses cfg_parallel_degree=1, so "
            f"context parallel degree must be 1 or {nproc}, got {cp_degree}"
        )
    if cp_degree == 1 and not fsdp_enabled:
        raise SystemExit(
            "[lingbot_video] CP1 with four torchrun workers requires FSDP to initialize "
            "the distributed process group and place one worker on each GPU"
        )


def ensure_fresh_output_dir(out_dir: Path) -> None:
    expected = (
        "run.log",
        "t2v_base.mp4",
        "t2v_refined.mp4",
        "out.mp4",
        "benchmark.json",
        "run_config.json",
        "gpu_peak_mib.txt",
    )
    stale = [name for name in expected if (out_dir / name).exists()]
    if stale:
        raise SystemExit(
            "[lingbot_video] refusing to reuse stale output artifacts: " + ", ".join(stale)
        )


def build_command(
    out_dir: Path,
    prompt_json: Path | None = None,
    base_video: Path | None = None,
    refined_video: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    if not SOURCE_ROOT.is_dir():
        raise SystemExit(f"[lingbot_video] vendored source is missing: {SOURCE_ROOT}")

    python_bin = env("PYTHON_BIN", sys.executable)
    model_dir = Path(env("LINGBOT_MODEL_DIR", required=True)).expanduser().resolve()
    if prompt_json is None:
        prompt_json = resolve_input(
            env("LINGBOT_PROMPT_JSON", "models/lingbot_video/prompts/t2v_example_1.json")
        )
    if not model_dir.is_dir():
        raise SystemExit(f"[lingbot_video] model directory does not exist: {model_dir}")
    if not prompt_json.is_file():
        raise SystemExit(f"[lingbot_video] prompt JSON does not exist: {prompt_json}")

    nproc = int(env("LINGBOT_NPROC", "4"))
    cp_degree = int(env("LINGBOT_CONTEXT_PARALLEL_DEGREE", "4"))
    fsdp_enabled = env_bool("LINGBOT_ENABLE_FSDP", True)
    validate_topology(nproc, cp_degree, fsdp_enabled)
    attn_kernel = attention_kernel()
    ulysses_requested = env_bool("LINGBOT_CONTEXT_PARALLEL_ULYSSES", True)
    ulysses_enabled = cp_degree > 1 and ulysses_requested
    batch_cfg_enabled = env_bool("LINGBOT_BATCH_CFG", True)
    refiner_batch_cfg_enabled = env_bool("LINGBOT_REFINER_BATCH_CFG", True)
    reuse_conditions_enabled = env_bool("LINGBOT_REUSE_CONDITION_FEATURES", True)
    experimental_flags = {
        name: env_bool(name, False)
        for name in (
            "LINGBOT_MOE_EP",
            "LINGBOT_COMPILE_MOE",
            "LINGBOT_OVERLAP_REFINER_LOAD",
            "LINGBOT_BCAST_WEIGHTS",
            "LINGBOT_SHARDED_LOAD",
            "LINGBOT_OFFLOAD_BASE_BEFORE_REFINER",
        )
    }
    if not env_bool("LINGBOT_PHASE_TIMING", True):
        raise SystemExit("[lingbot_video] LINGBOT_PHASE_TIMING must remain enabled for benchmark integrity")
    active_experiments = [name for name, enabled in experimental_flags.items() if enabled]
    if active_experiments:
        raise SystemExit(
            "[lingbot_video] registered baseline/c5 candidates forbid unrelated experimental "
            "switches: " + ", ".join(active_experiments)
        )
    fixed_backends = {
        "DIFFUSERS_ATTN_BACKEND": "_native_flash",
        "LINGBOT_MOE_EXPERT_BACKEND": "grouped_mm",
        "LINGBOT_MOE_PAD_BACKEND": "vectorized",
        "LINGBOT_MOE_REORDER_BACKEND": "sort",
        "LINGBOT_MOE_RESTORE_BACKEND": "scatter",
        "LINGBOT_MOE_RESTORE_CHUNK_SIZE": "128",
        "LINGBOT_QWEN_ATTN_IMPLEMENTATION": "sdpa",
    }
    mismatched_backends = {
        name: env(name, expected)
        for name, expected in fixed_backends.items()
        if env(name, expected).strip().lower() != expected
    }
    # The kernel-fusion candidate is allowed to enable fused QKV (it only pays off
    # under torch.compile, which fuses the per-call weight-cat away).
    if env_bool("LINGBOT_FUSED_QKV_LINEAR", False) and env("AUTOVIDEO_CANDIDATE_ID", "") != "lingbot_video_cudnn_kernel_compile":
        mismatched_backends["LINGBOT_FUSED_QKV_LINEAR"] = "1"
    if mismatched_backends:
        raise SystemExit(
            "[lingbot_video] registered candidates require fixed non-c5 backends; "
            f"unexpected overrides: {mismatched_backends}"
        )
    if RUNTIME_DIR.name == "lingbot_video_optimized" and attn_kernel == "cudnn":
        if not (
            cp_degree == nproc
            and fsdp_enabled
            and ulysses_enabled
            and batch_cfg_enabled
            and refiner_batch_cfg_enabled
        ):
            raise SystemExit(
                "[lingbot_video] the registered c5 cudnn candidate requires CP4 Ulysses, "
                "FSDP, base batch_cfg, and refiner batch_cfg"
            )
    candidate_id = env("AUTOVIDEO_CANDIDATE_ID", required=True)
    expected_kernel = {
        "lingbot_video_cudnn_optimized": "cudnn",
        "lingbot_video_cudnn_off": "fa2",
        "lingbot_video_cudnn_pisa_full": "cudnn",
        "lingbot_video_cudnn_pisa_easycache": "cudnn",
        "lingbot_video_cudnn_pisa_easycache_refiner": "cudnn",
        "lingbot_video_cudnn_kernel_compile": "cudnn",
    }.get(candidate_id)
    if expected_kernel is None:
        raise SystemExit(f"[lingbot_video] unsupported optimized runtime candidate: {candidate_id}")
    if attn_kernel != expected_kernel:
        raise SystemExit(
            f"[lingbot_video] candidate {candidate_id} requires attention kernel {expected_kernel}"
        )
    _pisa_candidates = {
        "lingbot_video_cudnn_pisa_full",
        "lingbot_video_cudnn_pisa_easycache",
        "lingbot_video_cudnn_pisa_easycache_refiner",
        "lingbot_video_cudnn_kernel_compile",
    }
    pisa_enabled = env_bool("LINGBOT_PISA_ENABLED", candidate_id in _pisa_candidates)
    if candidate_id in _pisa_candidates and not pisa_enabled:
        raise SystemExit("[lingbot_video] a PISA candidate requires LINGBOT_PISA_ENABLED=1")
    if candidate_id not in _pisa_candidates and pisa_enabled:
        raise SystemExit(f"[lingbot_video] candidate {candidate_id} does not permit the PISA path")
    if not (
        cp_degree == nproc
        and fsdp_enabled
        and ulysses_requested
        and batch_cfg_enabled
        and refiner_batch_cfg_enabled
    ):
        raise SystemExit(
            "[lingbot_video] optimized runtime requires the registered CP4+FSDP+batched-CFG topology"
        )
    duration_s = float(env("LINGBOT_DURATION", "5"))
    fps = int(env("LINGBOT_FPS", "24"))
    frame_count = int(duration_s * fps)
    if frame_count < 1:
        raise SystemExit("[lingbot_video] duration*fps must produce at least one frame")
    expected_frames = ((frame_count - 1) // 4 + 1) * 4 + 1
    num_frames = int(env("LINGBOT_NUM_FRAMES", str(expected_frames)))
    if num_frames != expected_frames:
        raise SystemExit(
            "[lingbot_video] LINGBOT_NUM_FRAMES must match the runner's 4k+1 frame alignment because "
            f"runner derives frames from --duration: got {num_frames}, expected {expected_frames}"
        )

    if base_video is None:
        base_video = out_dir / "t2v_base.mp4"
    if refined_video is None:
        refined_video = out_dir / "t2v_refined.mp4"
    command = [
        python_bin,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        str(SOURCE_ROOT / "scripts/inference.py"),
        "--backend",
        "diffusers",
        "--model_dir",
        str(model_dir),
        "--run_refiner",
        "--mode",
        "t2v",
        "--prompt_json",
        str(prompt_json),
        "--duration",
        str(duration_s),
        "--num_frames",
        str(num_frames),
        "--output",
        str(base_video),
        "--refiner_output",
        str(refined_video),
        "--height",
        env("LINGBOT_HEIGHT", "480"),
        "--width",
        env("LINGBOT_WIDTH", "832"),
        "--steps",
        env("LINGBOT_STEPS", "40"),
        "--refiner_height",
        env("LINGBOT_REFINER_HEIGHT", "1088"),
        "--refiner_width",
        env("LINGBOT_REFINER_WIDTH", "1920"),
        "--refiner_steps",
        env("LINGBOT_REFINER_STEPS", "8"),
        "--guidance_scale",
        env("LINGBOT_GUIDANCE_SCALE", "3"),
        "--refiner_guidance_scale",
        env("LINGBOT_REFINER_GUIDANCE_SCALE", "3"),
        "--shift",
        env("LINGBOT_SHIFT", "3"),
        "--refiner_shift",
        env("LINGBOT_REFINER_SHIFT", "3"),
        "--refiner_t_thresh",
        env("LINGBOT_REFINER_T_THRESH", "0.85"),
        "--refiner_sigma_tail_steps",
        env("LINGBOT_REFINER_SIGMA_TAIL_STEPS", "2"),
        "--seed",
        env("LINGBOT_SEED", "42"),
        "--fps",
        str(fps),
        "--refiner_fps",
        str(fps),
        "--refiner_sample_fps",
        str(fps),
        "--transformer_dtype",
        "bf16",
        "--text_encoder_dtype",
        "bf16",
        "--vae_dtype",
        "fp32",
        "--refiner_vae_dtype",
        "fp32",
        "--context_parallel_degree",
        str(cp_degree),
    ]
    append_flag(
        command,
        ulysses_enabled,
        "--context_parallel_ulysses_anything",
    )
    append_flag(command, fsdp_enabled, "--enable_fsdp_inference")
    append_flag(command, batch_cfg_enabled, "--batch_cfg")
    append_flag(command, refiner_batch_cfg_enabled, "--refiner_batch_cfg")
    append_flag(
        command,
        reuse_conditions_enabled,
        "--reuse_condition_features",
    )

    config: dict[str, Any] = {
        "model": "lingbot_video_moe_30b_a3b",
        "model_dir": str(model_dir),
        "prompt_json": str(prompt_json),
        "task": "t2v_refiner",
        "height": int(env("LINGBOT_HEIGHT", "480")),
        "width": int(env("LINGBOT_WIDTH", "832")),
        "num_frames": num_frames,
        "fps": fps,
        "duration_s": duration_s,
        "steps": int(env("LINGBOT_STEPS", "40")),
        "refiner_height": int(env("LINGBOT_REFINER_HEIGHT", "1088")),
        "refiner_width": int(env("LINGBOT_REFINER_WIDTH", "1920")),
        "refiner_steps": int(env("LINGBOT_REFINER_STEPS", "8")),
        "guidance_scale": float(env("LINGBOT_GUIDANCE_SCALE", "3")),
        "refiner_guidance_scale": float(env("LINGBOT_REFINER_GUIDANCE_SCALE", "3")),
        "shift": float(env("LINGBOT_SHIFT", "3")),
        "refiner_shift": float(env("LINGBOT_REFINER_SHIFT", "3")),
        "refiner_t_thresh": float(env("LINGBOT_REFINER_T_THRESH", "0.85")),
        "refiner_sigma_tail_steps": int(env("LINGBOT_REFINER_SIGMA_TAIL_STEPS", "2")),
        "seed": int(env("LINGBOT_SEED", "42")),
        "num_gpus": nproc,
        "context_parallel_degree": cp_degree,
        "context_parallel_ulysses": ulysses_enabled,
        "fsdp": fsdp_enabled,
        "batch_cfg": batch_cfg_enabled,
        "refiner_batch_cfg": refiner_batch_cfg_enabled,
        "reuse_condition_features": reuse_conditions_enabled,
        "attention_kernel": attn_kernel,
        "pisa_enabled": pisa_enabled,
        "pisa_base_enabled": env_bool("LINGBOT_PISA_BASE_ENABLED", True),
        "pisa_refiner_enabled": env_bool("LINGBOT_PISA_REFINER_ENABLED", True),
        "experimental_flags": experimental_flags,
        "fixed_backends": fixed_backends,
        "source_variant": env("LINGBOT_SOURCE_VARIANT", "baseline"),
        "candidate_id": candidate_id,
    }
    return command, config


def normalized_environment(out_dir: Path) -> dict[str, str]:
    run_env = os.environ.copy()
    pythonpath = [str(SOURCE_ROOT), str(SOURCE_ROOT / "rewriter"), str(SOURCE_ROOT / "slurm/shims")]
    if run_env.get("PYTHONPATH"):
        pythonpath.append(run_env["PYTHONPATH"])
    run_env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    cache_root = Path(
        env("LINGBOT_CACHE_ROOT", str(REPO_ROOT.parent / ".cache/lingbot_video"))
    ).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_map = {
        "HF_HOME": cache_root / "huggingface",
        "TORCH_HOME": cache_root / "torch",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
        "TORCH_EXTENSIONS_DIR": cache_root / "torch_extensions",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "XDG_CACHE_HOME": cache_root / "xdg",
    }
    for key, value in cache_map.items():
        value.mkdir(parents=True, exist_ok=True)
        run_env.setdefault(key, str(value))
    tmp_dir = out_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TMP", "TEMP"):
        run_env[key] = str(tmp_dir)

    run_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run_env.setdefault("DIFFUSERS_ATTN_BACKEND", "_native_flash")
    run_env.setdefault("LINGBOT_MOE_PAD_BACKEND", "vectorized")
    run_env.setdefault("LINGBOT_MOE_EXPERT_BACKEND", "grouped_mm")
    run_env.setdefault("LINGBOT_FUSED_QKV_LINEAR", "0")
    run_env.setdefault("LINGBOT_MOE_REORDER_BACKEND", "sort")
    run_env.setdefault("LINGBOT_MOE_RESTORE_BACKEND", "scatter")
    run_env.setdefault("LINGBOT_MOE_RESTORE_CHUNK_SIZE", "128")
    run_env.setdefault("LINGBOT_QWEN_ATTN_IMPLEMENTATION", "sdpa")
    run_env.setdefault("LINGBOT_PISA_ENABLED", "0")
    run_env.setdefault("LINGBOT_PISA_BASE_ENABLED", "1")
    run_env.setdefault("LINGBOT_PISA_REFINER_ENABLED", "1")
    run_env.setdefault("LINGBOT_PISA_PHASE", "all")
    run_env.setdefault("LINGBOT_PHASE_TIMING", "1")
    return run_env


def compute_request_metrics(phases: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Validate required phase markers and derive the authoritative per-request
    timing metrics. Shared by the single-prompt and multi-prompt (validation
    set) benchmark writers so both apply identical validation."""
    missing = [label for label in REQUIRED_HOT_PHASES if label not in phases]
    if missing:
        raise SystemExit(
            "[lingbot_video] refusing to publish a cold-wall fallback benchmark; "
            f"missing required phase marker(s): {', '.join(missing)}"
        )
    phase_totals = [phases[label]["total_s"] for label in REQUIRED_HOT_PHASES]
    if any(current < previous for previous, current in zip(phase_totals, phase_totals[1:])):
        raise SystemExit("[lingbot_video] required phase totals are not monotonic")

    # Despite their historical names, the *_denoise_done markers wrap a full
    # pipeline call with output_type=np; that call includes VAE decode. Keep the
    # raw labels in run_config.json, but publish semantically accurate names.
    base_generate = phase_dt(phases, "base_denoise_done")
    refiner_input_prepare = phase_dt(phases, "refiner_vae_encode_done")
    refiner_generate = phase_dt(phases, "refiner_denoise_done")
    base_video_export = phase_dt(phases, "base_vae_saved")
    refiner_video_export = phase_dt(phases, "refiner_vae_saved")
    source_phase_subset_s = sum_if_complete(
        base_generate,
        refiner_input_prepare,
        refiner_generate,
        refiner_video_export,
    )
    request_start_s = phases["refiner_preloaded"]["total_s"]
    request_end_s = phases["refiner_vae_saved"]["total_s"]
    load_excluded_request_s = request_end_s - request_start_s
    if load_excluded_request_s <= 0:
        raise SystemExit(
            "[lingbot_video] invalid phase totals for load-excluded request timing: "
            f"start={request_start_s} end={request_end_s}"
        )
    return {
        "load_excluded_request_s": load_excluded_request_s,
        "source_phase_subset_s": source_phase_subset_s,
        "base_generation_pipeline_s": base_generate,
        "refiner_input_preparation_s": refiner_input_prepare,
        "refiner_generation_pipeline_s": refiner_generate,
        "base_video_export_s": base_video_export,
        "refiner_video_export_s": refiner_video_export,
    }


def write_benchmark(
    out_dir: Path,
    config: dict[str, Any],
    command: list[str],
    phases: dict[str, dict[str, float]],
    wall_s: float,
    peak_mib: int | None,
    run_env: dict[str, str],
) -> None:
    metrics = compute_request_metrics(phases)
    base_generate = metrics["base_generation_pipeline_s"]
    refiner_input_prepare = metrics["refiner_input_preparation_s"]
    refiner_generate = metrics["refiner_generation_pipeline_s"]
    base_video_export = metrics["base_video_export_s"]
    refiner_video_export = metrics["refiner_video_export_s"]
    source_phase_subset_s = metrics["source_phase_subset_s"]
    load_excluded_request_s = metrics["load_excluded_request_s"]

    stage_seconds = {
        PHASE_STAGE_NAMES.get(label, label): record["dt_s"]
        for label, record in phases.items()
    }
    benchmark = {
        "schema_version": 2,
        "total_s": load_excluded_request_s,
        "denoise_s": None,
        "decode_s": None,
        "load_excluded_request_s": load_excluded_request_s,
        "source_phase_subset_s": source_phase_subset_s,
        "wall_total_s": wall_s,
        "base_generation_pipeline_s": base_generate,
        "refiner_input_preparation_s": refiner_input_prepare,
        "refiner_generation_pipeline_s": refiner_generate,
        "base_video_export_s": base_video_export,
        "refiner_video_export_s": refiner_video_export,
        "timing_scope": "load_excluded_request_wall_from_refiner_preloaded_to_refiner_video_export",
        "timing_note": "Only the contiguous request total is authoritative. Per-phase values are host-observed intervals without CUDA synchronization; pipeline intervals include VAE decode and asynchronous work may spill across marker boundaries.",
        "warm_steady_state": False,
        "warmup_requests": 0,
        "includes_model_load": False,
        "max_device_memory_used_mib": peak_mib,
        "memory": {
            "max_device_memory_used_mib": peak_mib,
        },
        "stage_seconds": stage_seconds,
        "timings": {
            "wall_total_s": wall_s,
            "load_excluded_request_s": load_excluded_request_s,
            "source_phase_subset_s": source_phase_subset_s,
            "base_generation_pipeline_s": base_generate,
            "refiner_input_preparation_s": refiner_input_prepare,
            "refiner_generation_pipeline_s": refiner_generate,
            "refiner_video_export_s": refiner_video_export,
        },
        "aggregate": {
            "total_s": load_excluded_request_s,
            "load_excluded_request_s": load_excluded_request_s,
            "source_phase_subset_s": source_phase_subset_s,
            "wall_total_s": wall_s,
            "prompt_count": 1,
            "warmup_requests": 0,
        },
        "config": config,
    }
    source_snapshot = load_source_snapshot()
    runtime_snapshot_id = "snapshot:" + hashlib.sha256(
        (RUNTIME_DIR / "SOURCE_SNAPSHOT.json").read_bytes()
    ).hexdigest()
    effective_env_keys = (
        "AUTOVIDEO_CANDIDATE_ID",
        "DIFFUSERS_ATTN_BACKEND",
        "LINGBOT_ATTN_KERNEL",
        "LINGBOT_BATCH_CFG",
        "LINGBOT_BCAST_WEIGHTS",
        "LINGBOT_COMPILE_MOE",
        "LINGBOT_CONTEXT_PARALLEL_DEGREE",
        "LINGBOT_CONTEXT_PARALLEL_ULYSSES",
        "LINGBOT_ENABLE_FSDP",
        "LINGBOT_MOE_EP",
        "LINGBOT_MOE_EXPERT_BACKEND",
        "LINGBOT_FUSED_QKV_LINEAR",
        "LINGBOT_MOE_PAD_BACKEND",
        "LINGBOT_MOE_REORDER_BACKEND",
        "LINGBOT_MOE_RESTORE_BACKEND",
        "LINGBOT_MOE_RESTORE_CHUNK_SIZE",
        "LINGBOT_OFFLOAD_BASE_BEFORE_REFINER",
        "LINGBOT_OVERLAP_REFINER_LOAD",
        "LINGBOT_PHASE_TIMING",
        "LINGBOT_PISA_APPROX_REMAINDER",
        "LINGBOT_PISA_BLOCK_SIZE",
        "LINGBOT_PISA_DENSE_LAYERS",
        "LINGBOT_PISA_DENSE_HEAD_STEPS",
        "LINGBOT_PISA_DENSE_STEPS",
        "LINGBOT_PISA_DENSE_TAIL_STEPS",
        "LINGBOT_PISA_DENSITY",
        "LINGBOT_PISA_ENABLED",
        "LINGBOT_PISA_BASE_ENABLED",
        "LINGBOT_PISA_REFINER_ENABLED",
        "LINGBOT_PISA_KERNEL_NUM_STAGES",
        "LINGBOT_QWEN_ATTN_IMPLEMENTATION",
        "LINGBOT_REFINER_BATCH_CFG",
        "LINGBOT_SHARDED_LOAD",
        "LINGBOT_SOURCE_VARIANT",
        "NCCL_NET_PLUGIN",
        "PYTORCH_CUDA_ALLOC_CONF",
    )
    effective_environment = {
        key: run_env[key] for key in effective_env_keys if key in run_env
    }
    (out_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "config": config,
                "model_path": config["model_dir"],
                "num_frames": config["num_frames"],
                "command": command,
                "runtime_dir": str(RUNTIME_DIR),
                "source_root": str(SOURCE_ROOT),
                "base_commit": "a2bb04b78edd848500dc27a26e035a95442ae186",
                "runtime_snapshot_id": runtime_snapshot_id,
                "source_snapshot": source_snapshot,
                "effective_environment": effective_environment,
                "phase_records": phases,
            },
            indent=2,
        )
        + "\n"
    )


def execute_torchrun(
    command: list[str], run_env: dict[str, str], log
) -> tuple[int, list[str]]:
    """Run one torchrun invocation, streaming stdout to the console and the
    given open log handle, and return (return_code, captured_lines)."""
    lines: list[str] = []
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(SOURCE_ROOT),
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            lines.append(line)
        return_code = proc.wait()
    except BaseException:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise
    return return_code, lines


def resolve_validation_prompts() -> list[dict[str, Any]] | None:
    """When LINGBOT_VAL_PROMPTS points at a JSON list of prompt objects, return
    the first LINGBOT_NUM_PROMPTS of them for a median-over-prompts run.
    Otherwise return None to preserve the single-prompt legacy path."""
    raw = env("LINGBOT_VAL_PROMPTS", "").strip()
    if not raw:
        return None
    path = resolve_input(raw)
    if not path.is_file():
        raise SystemExit(f"[lingbot_video] validation prompt set does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit(
            f"[lingbot_video] LINGBOT_VAL_PROMPTS must be a non-empty JSON list of prompt objects: {path}"
        )
    for item in data:
        if not isinstance(item, dict):
            raise SystemExit(
                f"[lingbot_video] every entry in {path} must be a prompt object (dict)"
            )
    num = int(env("LINGBOT_NUM_PROMPTS", str(len(data))))
    return data[: max(1, num)]


def write_multi_benchmark(
    out_dir: Path,
    config: dict[str, Any],
    command: list[str],
    samples: list[dict[str, Any]],
    wall_s: float,
    peak_mib: int | None,
    run_env: dict[str, str],
) -> float:
    """Aggregate per-prompt request metrics into a median benchmark. Returns the
    median load-excluded request total that becomes the authoritative total_s."""
    def med(key: str) -> float | None:
        vals = [s[key] for s in samples if s.get(key) is not None]
        return statistics.median(vals) if vals else None

    median_total = med("total_s")
    if median_total is None:
        raise SystemExit("[lingbot_video] no valid per-prompt totals to aggregate")
    stage_seconds = {
        PHASE_STAGE_NAMES.get(label, label): record["dt_s"]
        for label, record in (samples[0].get("phases") or {}).items()
    }
    benchmark = {
        "schema_version": 2,
        "total_s": median_total,
        "denoise_s": None,
        "decode_s": None,
        "load_excluded_request_s": median_total,
        "source_phase_subset_s": med("source_phase_subset_s"),
        "wall_total_s": wall_s,
        "base_generation_pipeline_s": med("base_generation_pipeline_s"),
        "refiner_input_preparation_s": med("refiner_input_preparation_s"),
        "refiner_generation_pipeline_s": med("refiner_generation_pipeline_s"),
        "base_video_export_s": med("base_video_export_s"),
        "refiner_video_export_s": med("refiner_video_export_s"),
        "timing_scope": "load_excluded_request_wall_from_refiner_preloaded_to_refiner_video_export",
        "timing_note": (
            "Median of per-prompt load-excluded request totals over the official "
            "validation set; each prompt runs a fresh torchrun so model load is "
            "outside the measured scope. Per-phase values are host-observed intervals."
        ),
        "warm_steady_state": False,
        "warmup_requests": 0,
        "includes_model_load": False,
        "max_device_memory_used_mib": peak_mib,
        "memory": {"max_device_memory_used_mib": peak_mib},
        "stage_seconds": stage_seconds,
        "aggregate": {
            "total_s": median_total,
            "load_excluded_request_s": median_total,
            "source_phase_subset_s": med("source_phase_subset_s"),
            "wall_total_s": wall_s,
            "prompt_count": len(samples),
            "warmup_requests": 0,
            "reduction": "median",
        },
        "samples": [
            {k: v for k, v in s.items() if k != "phases"} for s in samples
        ],
        "config": config,
    }
    source_snapshot = load_source_snapshot()
    runtime_snapshot_id = "snapshot:" + hashlib.sha256(
        (RUNTIME_DIR / "SOURCE_SNAPSHOT.json").read_bytes()
    ).hexdigest()
    (out_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "config": config,
                "model_path": config["model_dir"],
                "num_frames": config["num_frames"],
                "command": command,
                "runtime_dir": str(RUNTIME_DIR),
                "source_root": str(SOURCE_ROOT),
                "base_commit": "a2bb04b78edd848500dc27a26e035a95442ae186",
                "runtime_snapshot_id": runtime_snapshot_id,
                "source_snapshot": source_snapshot,
                "prompt_count": len(samples),
                "samples": [
                    {k: v for k, v in s.items() if k != "phases"} for s in samples
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return median_total


def run_single_prompt(out_dir: Path, run_env: dict[str, str], sampler: PeakMemorySampler) -> int:
    """Legacy single-prompt path (LINGBOT_VAL_PROMPTS unset)."""
    command, config = build_command(out_dir)
    log_path = out_dir / "run.log"
    print(f"[lingbot_video] source_variant={config['source_variant']}", flush=True)
    print(f"[lingbot_video] attention_kernel={config['attention_kernel']}", flush=True)
    print(f"[lingbot_video] command={' '.join(command)}", flush=True)

    started = time.monotonic()
    try:
        with log_path.open("w") as log:
            return_code, lines = execute_torchrun(command, run_env, log)
    finally:
        wall_s = time.monotonic() - started
        peak_mib = sampler.stop()

    if peak_mib is not None:
        (out_dir / "gpu_peak_mib.txt").write_text(f"{peak_mib}\n")
    if return_code != 0:
        raise SystemExit(f"[lingbot_video] torchrun failed with exit code {return_code}")

    refined_video = out_dir / "t2v_refined.mp4"
    if not refined_video.is_file() or refined_video.stat().st_size == 0:
        raise SystemExit(f"[lingbot_video] expected refined video is missing: {refined_video}")
    shutil.copy2(refined_video, out_dir / "out.mp4")

    phases = parse_phase_lines(lines)
    write_benchmark(out_dir, config, command, phases, wall_s, peak_mib, run_env)
    print(
        f"[lingbot_video] done total_wall={wall_s:.2f}s "
        f"phases={len(phases)} peak_mib={peak_mib}",
        flush=True,
    )
    return 0


def run_validation_set(
    out_dir: Path,
    run_env: dict[str, str],
    sampler: PeakMemorySampler,
    prompts: list[dict[str, Any]],
) -> int:
    """Median-over-prompts path: run one fresh torchrun per official prompt and
    aggregate the load-excluded request totals."""
    log_path = out_dir / "run.log"
    samples: list[dict[str, Any]] = []
    first_command: list[str] | None = None
    first_config: dict[str, Any] | None = None
    started_all = time.monotonic()
    print(f"[lingbot_video] validation set: {len(prompts)} official prompt(s)", flush=True)

    with log_path.open("w") as log:
        for i, prompt_obj in enumerate(prompts):
            work = out_dir / f"prompt_{i:02d}"
            work.mkdir(parents=True, exist_ok=True)
            prompt_path = work / "prompt.json"
            # Strip harness-only metadata so the pristine runner receives an
            # object identical in shape to the official example prompt.json.
            runner_prompt = {k: v for k, v in prompt_obj.items() if not k.startswith("_")}
            prompt_path.write_text(json.dumps(runner_prompt, ensure_ascii=False, indent=2) + "\n")
            base_video = work / "t2v_base.mp4"
            refined_video = work / "t2v_refined.mp4"
            command, config = build_command(
                out_dir, prompt_json=prompt_path, base_video=base_video, refined_video=refined_video
            )
            if first_command is None:
                first_command, first_config = command, config
                print(f"[lingbot_video] source_variant={config['source_variant']}", flush=True)
                print(f"[lingbot_video] attention_kernel={config['attention_kernel']}", flush=True)
            header = f"\n===== [lingbot_video] prompt {i} ({prompt_obj.get('_source', 'prompt')}) =====\n"
            sys.stdout.write(header)
            log.write(header)
            log.flush()
            print(f"[lingbot_video] command={' '.join(command)}", flush=True)

            started = time.monotonic()
            return_code, lines = execute_torchrun(command, run_env, log)
            wall_i = time.monotonic() - started
            if return_code != 0:
                sampler.stop()
                raise SystemExit(
                    f"[lingbot_video] torchrun failed for prompt {i} with exit code {return_code}"
                )
            if not refined_video.is_file() or refined_video.stat().st_size == 0:
                sampler.stop()
                raise SystemExit(
                    f"[lingbot_video] expected refined video is missing for prompt {i}: {refined_video}"
                )
            phases = parse_phase_lines(lines)
            metrics = compute_request_metrics(phases)
            samples.append(
                {
                    "prompt_id": f"p{i}",
                    "source": prompt_obj.get("_source"),
                    "total_s": metrics["load_excluded_request_s"],
                    "source_phase_subset_s": metrics["source_phase_subset_s"],
                    "base_generation_pipeline_s": metrics["base_generation_pipeline_s"],
                    "refiner_input_preparation_s": metrics["refiner_input_preparation_s"],
                    "refiner_generation_pipeline_s": metrics["refiner_generation_pipeline_s"],
                    "base_video_export_s": metrics["base_video_export_s"],
                    "refiner_video_export_s": metrics["refiner_video_export_s"],
                    "wall_s": wall_i,
                    "refined_video": str(refined_video),
                    "base_video": str(base_video),
                    "phases": phases,
                }
            )
            print(
                f"[lingbot_video] prompt {i} total_s={metrics['load_excluded_request_s']:.2f}s "
                f"wall={wall_i:.2f}s",
                flush=True,
            )

    wall_all = time.monotonic() - started_all
    peak_mib = sampler.stop()
    if peak_mib is not None:
        (out_dir / "gpu_peak_mib.txt").write_text(f"{peak_mib}\n")

    assert first_command is not None and first_config is not None
    median_total = write_multi_benchmark(
        out_dir, first_config, first_command, samples, wall_all, peak_mib, run_env
    )

    # Canonical artifacts: the prompt whose total is nearest the median.
    canonical = min(samples, key=lambda s: abs(s["total_s"] - median_total))
    shutil.copy2(canonical["refined_video"], out_dir / "t2v_refined.mp4")
    shutil.copy2(canonical["base_video"], out_dir / "t2v_base.mp4")
    shutil.copy2(canonical["refined_video"], out_dir / "out.mp4")
    print(
        f"[lingbot_video] done median total_s={median_total:.2f}s over {len(samples)} prompt(s) "
        f"wall_total={wall_all:.2f}s peak_mib={peak_mib} canonical={canonical['prompt_id']}",
        flush=True,
    )
    return 0


def main() -> int:
    out_dir = Path(env("OUT_DIR", required=True)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_fresh_output_dir(out_dir)
    load_source_snapshot()
    run_env = normalized_environment(out_dir)

    sampler = PeakMemorySampler()
    sampler.start()
    prompts = resolve_validation_prompts()
    if prompts is None:
        return run_single_prompt(out_dir, run_env, sampler)
    return run_validation_set(out_dir, run_env, sampler, prompts)


if __name__ == "__main__":
    raise SystemExit(main())
