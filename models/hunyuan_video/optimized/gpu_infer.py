#!/usr/bin/env python3
"""Vanilla HunyuanVideo (diffusers) baseline runner — reconstructed.

This is the BASELINE runtime (no acceleration). It replaces the unobtainable
haozhel-local `Hunyuan-Diffusers` submodule for the baseline path: it runs the
stock diffusers ``HunyuanVideoPipeline`` once and emits exactly the artifacts the
auto-video harness consumes. The TeaCache acceleration seam
(``step_cache_runtime.py``) is added on top of this later — see README.md.

Interface (set by launch_candidate.py -> launch.sh, all via the environment):
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

    # --- Optional SOL Attention (PISA2 CuTe DSL, SM100), gated on HUNYUAN_SOL_ATTN ---
    # HunyuanVideoAttnProcessor2_0 routes through diffusers' dispatch_attention_fn,
    # so the sol_attn dispatch hook installs the same way as for Wan. Installed
    # before torch.compile so the compiled graph captures it; dense fallback for
    # non-128 / causal / cross-attn / non-SM100.
    if os.environ.get("HUNYUAN_SOL_ATTN") == "1":
        import sys as _sys_sol
        _repo_sol = os.environ.get(
            "AUTOVIDEO_REPO_ROOT", str(Path(__file__).resolve().parents[3])
        )
        if _repo_sol not in _sys_sol.path:
            _sys_sol.path.insert(0, _repo_sol)
        from techniques.sparse_backends.sol_attn_backend import (
            make_sol_attn_dispatch, sol_attn_begin_forward,
        )
        from diffusers.models.transformers import transformer_hunyuan_video as _thv
        # HunyuanVideo self-attention is a JOINT [video, text] sequence with a
        # text-padding mask, so SOL runs the masked split-merge path: it needs
        # the video grid (F,H,W) and the video token count so the reorder +
        # sparse routing apply to the video sub-range only (text stays dense).
        _cfg = pipe.transformer.config
        _pt = int(getattr(_cfg, "patch_size", 2))
        _ptt = int(getattr(_cfg, "patch_size_t", 1))
        _vs = int(getattr(pipe, "vae_scale_factor_spatial", 8))
        _vt = int(getattr(pipe, "vae_scale_factor_temporal", 4))
        _Flat = (num_frames - 1) // _vt + 1
        _grid = (_Flat // _ptt, (height // _vs) // _pt, (width // _vs) // _pt)
        _video_len = _grid[0] * _grid[1] * _grid[2]
        _sol_kw = dict(
            target_density=float(os.environ.get("HUNYUAN_SOL_DENSITY", "0.05")),
            dense_steps=int(os.environ.get("HUNYUAN_SOL_DENSE_STEPS", "0")),
            dense_layers=os.environ.get("HUNYUAN_SOL_DENSE_LAYERS", ""),
            grid=_grid,
            video_len=_video_len,
        )
        _sol_tau = os.environ.get("HUNYUAN_SOL_TAU")
        if _sol_tau is not None:
            _sol_kw["tau"] = float(_sol_tau)
        _thv.dispatch_attention_fn = make_sol_attn_dispatch(
            _thv.dispatch_attention_fn, **_sol_kw
        )
        pipe.transformer.register_forward_pre_hook(lambda _m, _a: sol_attn_begin_forward())
        print(f"[opt] SOL Attention (Hunyuan joint video+text) installed "
              f"grid={_grid} video_len={_video_len} tau={_sol_tau} "
              f"density_target={_sol_kw['target_density']} "
              f"dense_steps={_sol_kw['dense_steps']} dense_layers='{_sol_kw['dense_layers']}'", flush=True)

    # --- Optional kernel: torch.compile the transformer, gated on HUNYUAN_COMPILE ---
    if os.environ.get("HUNYUAN_COMPILE") == "1":
        _mode = os.environ.get("HUNYUAN_COMPILE_MODE", "max-autotune-no-cudagraphs")
        pipe.transformer = torch.compile(pipe.transformer, mode=_mode, fullgraph=False)
        print(f"[opt] transformer torch.compile enabled mode={_mode}", flush=True)

    # Optional acceleration seam (TeaCache). No-op unless SGLANG_HQ_TEACACHE_* is
    # set, so the baseline candidate (which sets no such env) stays byte-identical.
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

    # Warmup pass(es) to amortize torch.compile autotune + CuteDSL kernel compile
    # BEFORE the timed pass, so generate_s reflects steady-state (not compile cost).
    _warmup = _i("HUNYUAN_WARMUP_PASSES", 0)
    for _wi in range(_warmup):
        print(f"[opt] warmup pass {_wi + 1}/{_warmup}", flush=True)
        _wg = torch.Generator(device=device).manual_seed(seed)
        pipe(
            prompt=prompt, negative_prompt=negative, height=height, width=width,
            num_frames=num_frames, num_inference_steps=steps, guidance_scale=guidance,
            true_cfg_scale=true_cfg, max_sequence_length=max_seq, generator=_wg,
        )
        torch.cuda.synchronize()

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
