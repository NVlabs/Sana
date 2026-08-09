#!/usr/bin/env python3
"""Vanilla HunyuanVideo (diffusers) baseline runner — reconstructed.

This is the BASELINE runtime (no acceleration). It replaces the unobtainable
haozhel-local `Hunyuan-Diffusers` submodule for the baseline path: it runs the
stock diffusers ``HunyuanVideoPipeline`` once and emits exactly the artifacts the
auto-video harness consumes. The TeaCache acceleration seam
(``step_cache_runtime.py``) is added on top of this later — see README.md.

Interface (set by launch_config.py -> launch.sh, all via the environment):
  OUT_DIR                      destination for artifacts (required)
  MODEL_REPO                   HF repo id (default hunyuanvideo-community/HunyuanVideo)
  HUNYUAN_HEIGHT/WIDTH/NUM_FRAMES/FPS/NUM_INFERENCE_STEPS
  HUNYUAN_GUIDANCE_SCALE/TRUE_CFG_SCALE/MAX_SEQUENCE_LENGTH/DTYPE
  HUNYUAN_VAE_TILING/VAE_SLICING/DEVICE, SEED, PROMPT, NEGATIVE_PROMPT
  HF_HOME/HF_HUB_CACHE/HF_HUB_OFFLINE                 (consumed by diffusers/hf)

Artifacts written into $OUT_DIR (collect_run.py / plan_eval.py contract):
  out.mp4                      the generated clip (status gate needs it nonempty)
  frames/f_%05d.png            per-frame PNGs (aligned LPIPS/Gemini frame set)
  benchmark.json               NESTED timings + memory (see below)
  run_config.json              {model_path, num_frames, ...} (collector hint)

Timing convention (collect_run.py:189-216 + models/hunyuan_diffusers.toml
[baseline]): the diffusers ``pipe()`` call is a single fused generate
(denoise + VAE decode). We report it as ``timings.generate_s``; the collector
maps generate_s -> total_s/denoise_s as the GENERATION-time speedup metric, with
one-time load/placement EXCLUDED (recorded as evidence only).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Quiet, deterministic logging: keep run.log free of stray framework chatter so
# collect_run.determine_status() does not misread a benign warning as a failure.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity_error()
diffusers_logging.disable_progress_bar()

_DTYPES = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16,
    "fp32": torch.float32, "float32": torch.float32,
}


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(float(os.environ.get(name, default)))


def _b(name: str, default: str = "false") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in ("1", "true", "yes", "on")


def _save_frame(frame, path: Path) -> None:
    if hasattr(frame, "save"):  # PIL.Image
        frame.save(path)
        return
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype("uint8") if arr.max() <= 1.0 else arr.astype("uint8")
    Image.fromarray(arr).save(path)


def main() -> int:
    out_dir = Path(os.environ["OUT_DIR"])
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    model = os.environ.get("MODEL_REPO", "hunyuanvideo-community/HunyuanVideo")
    height, width = _i("HUNYUAN_HEIGHT", 720), _i("HUNYUAN_WIDTH", 1280)
    num_frames, fps = _i("HUNYUAN_NUM_FRAMES", 129), _i("HUNYUAN_FPS", 24)
    steps = _i("HUNYUAN_NUM_INFERENCE_STEPS", 50)
    guidance, true_cfg = _f("HUNYUAN_GUIDANCE_SCALE", 6.0), _f("HUNYUAN_TRUE_CFG_SCALE", 1.0)
    max_seq, seed = _i("HUNYUAN_MAX_SEQUENCE_LENGTH", 256), _i("SEED", 42)
    prompt = os.environ.get("PROMPT", "")
    negative = os.environ.get("NEGATIVE_PROMPT", "").strip() or None
    device = os.environ.get("HUNYUAN_DEVICE", "cuda")
    # HF_HUB_OFFLINE alone is NOT enough: diffusers' pipeline resolution still hits
    # the model_info API and crashes (OfflineModeIsEnabled). Pass local_files_only
    # so from_pretrained reads straight from the local cache.
    local_files_only = _b("HF_HUB_OFFLINE", "0")
    # Canonical diffusers HunyuanVideo recipe: transformer in bf16, the rest
    # (text encoders + VAE) in fp16. HUNYUAN_DTYPE selects the transformer dtype.
    tf_dtype = _DTYPES.get(os.environ.get("HUNYUAN_DTYPE", "bf16").lower(), torch.bfloat16)
    pipe_dtype = torch.float16

    print(f"[baseline] model={model} {width}x{height} frames={num_frames} steps={steps} "
          f"guidance={guidance} true_cfg={true_cfg} max_seq={max_seq} seed={seed} "
          f"tf_dtype={tf_dtype} pipe_dtype={pipe_dtype}", flush=True)

    if not torch.cuda.is_available():
        # Explicit, non-pattern message; raising marks the run failed for the gate.
        raise SystemExit("CUDA unavailable on this node; the baseline needs a GPU.")

    torch.cuda.reset_peak_memory_stats()
    wall0 = time.perf_counter()

    t = time.perf_counter()
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model, subfolder="transformer", torch_dtype=tf_dtype, local_files_only=local_files_only)
    pipe = HunyuanVideoPipeline.from_pretrained(
        model, transformer=transformer, torch_dtype=pipe_dtype, local_files_only=local_files_only)
    load_s = time.perf_counter() - t

    t = time.perf_counter()
    if _b("HUNYUAN_VAE_TILING", "true"):
        pipe.vae.enable_tiling()
    if _b("HUNYUAN_VAE_SLICING", "false"):
        pipe.vae.enable_slicing()
    pipe.to(device)
    placement_s = time.perf_counter() - t

    # Optional acceleration seam (TeaCache). No-op unless SGLANG_HQ_TEACACHE_* is
    # set, so the baseline config (which sets no such env) stays byte-identical.
    seam_diag = None
    try:
        import step_cache_runtime
        seam_diag = step_cache_runtime.maybe_enable(pipe)
        if seam_diag:
            print(f"[seam] TeaCache ON variant={seam_diag['variant']} "
                  f"threshold={seam_diag['threshold']} start_step={seam_diag['start_step']} "
                  f"max_hits={seam_diag['max_continuous_hits']}", flush=True)
    except Exception as exc:  # never let the seam break a run; fall back to baseline
        print(f"[seam] not enabled (type={type(exc).__name__})", flush=True)
        seam_diag = None

    # Warmup pass(es) for a HOT steady-state measurement (parity with the Wan
    # runners: timing_scope = text_to_video_hot_after_warmup_pass). Default 1;
    # set HUNYUAN_WARMUP_PASSES=0 for the legacy cold single-pass measurement.
    _warmup = int(os.environ.get("HUNYUAN_WARMUP_PASSES", "1") or "0")
    for _wi in range(_warmup):
        print(f"[baseline] warmup pass {_wi + 1}/{_warmup}", flush=True)
        _wg = torch.Generator(device=device).manual_seed(seed)
        pipe(
            prompt=prompt, negative_prompt=negative, height=height, width=width,
            num_frames=num_frames, num_inference_steps=steps, guidance_scale=guidance,
            true_cfg_scale=true_cfg, max_sequence_length=max_seq, generator=_wg,
        )
        torch.cuda.synchronize()
    # Cold-start the TeaCache controller (if the seam is enabled) so the timed
    # pass measures a fresh generation, not a warmup-warmed schedule.
    if _warmup and seam_diag is not None and callable(seam_diag.get("reset")):
        seam_diag["reset"]()

    generator = torch.Generator(device=device).manual_seed(seed)
    torch.cuda.synchronize()
    t = time.perf_counter()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        true_cfg_scale=true_cfg,
        max_sequence_length=max_seq,
        generator=generator,
    )
    torch.cuda.synchronize()
    generate_s = time.perf_counter() - t

    max_alloc_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    max_resv_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
    video = result.frames[0]

    t = time.perf_counter()
    export_to_video(video, str(out_dir / "out.mp4"), fps=fps)
    for idx, frame in enumerate(video, start=1):
        _save_frame(frame, frames_dir / f"f_{idx:05d}.png")
    export_s = time.perf_counter() - t
    wall_total_s = time.perf_counter() - wall0

    benchmark = {
        # Schema v2 flat keys (parity with the Wan runners). The diffusers
        # pipe() is a single fused generate (denoise + VAE decode), so decode
        # is not separately timed and total_s == denoise_s by convention.
        "schema_version": 2,
        "model_id": "hunyuan_video",
        "pipeline": "HunyuanVideoPipeline",
        "total_s": generate_s,
        "denoise_s": generate_s,
        "decode_s": 0.0,
        "timing_scope": ("text_to_video_hot_after_warmup_pass" if _warmup
                         else "text_to_video_single_pass"),
        "warm_steady_state": bool(_warmup),
        "baseline_class": "pristine_unoptimized",
        # Nested timings kept for existing collectors (collect_run reads both).
        "timings": {
            "generate_s": generate_s,   # collector maps -> total_s/denoise_s (speedup metric)
            "load_s": load_s,
            "placement_s": placement_s,
            "export_s": export_s,
            "wall_total_s": wall_total_s,
        },
        "memory": {
            "max_memory_allocated_gib": max_alloc_gib,
            "max_memory_reserved_gib": max_resv_gib,
        },
        "config": {
            "model": model, "width": width, "height": height, "num_frames": num_frames,
            "fps": fps, "steps": steps, "guidance_scale": guidance,
            "true_cfg_scale": true_cfg, "max_sequence_length": max_seq, "seed": seed,
            "transformer_dtype": str(tf_dtype), "pipe_dtype": str(pipe_dtype),
        },
    }
    if seam_diag:
        benchmark["seam"] = {
            k: seam_diag[k] for k in
            ("technique", "variant", "threshold", "start_step", "max_continuous_hits", "signal", "stats", "trace")
        }
        print(f"[seam] TeaCache stats: {seam_diag['stats']}", flush=True)
    (out_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    (out_dir / "run_config.json").write_text(json.dumps({
        "model_path": model, "num_frames": num_frames, "height": height, "width": width,
        "fps": fps, "num_inference_steps": steps, "seed": seed,
    }, indent=2) + "\n")

    print(f"[baseline] generate_s={generate_s:.2f} load_s={load_s:.2f} "
          f"placement_s={placement_s:.2f} export_s={export_s:.2f} "
          f"wall_total_s={wall_total_s:.2f} peak_alloc_gib={max_alloc_gib:.2f} "
          f"peak_reserved_gib={max_resv_gib:.2f} frames={len(video)}", flush=True)
    print("[baseline] DONE: out.mp4 + frames/ + benchmark.json + run_config.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
