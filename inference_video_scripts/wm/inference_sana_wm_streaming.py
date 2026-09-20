# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end streaming SANA-WM inference.

Three-stream chunk-pipelined recipe:

    * **Stage 1** — chunk-causal distilled student (``SanaMSVideoCamCtrlStreaming``)
      with self-forcing AR sampling (4 steps, ``cfg_scale=1``).
    * **Refiner** — chunk-causal LTX-2 with a sliding KV window (canonical
      3-step distilled schedule).
    * **VAE** — causal LTX-2 VAE that decodes one block at a time.

Stages run on dedicated CUDA streams; one decoded chunk per AR block is
appended to a progressive MP4 you can watch while generation continues.

The script applies the canonical fast configuration by default — no flags
needed:

    * ``torch.compile`` on the VAE decoder + refiner transformer
      (``max-autotune-no-cudagraphs``, numerically equivalent to eager).
    * Fast SDPA backend selection, Inductor ``coordinate_descent_tuning`` +
      ``epilogue_fusion``, cuDNN benchmark, expandable CUDA allocator.

Reaches ~0.93× of realtime in steady-state on a single H100 after the
one-time ``torch.compile`` warmup (~3 min cold, ~30 s warm cache).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

# Env knobs that must be set before any ``torch`` / ``diffusion.*`` import.
# DISABLE_XFORMERS keeps cross-attention on plain SDPA with Python-list
# seqlens (the layout the self-forcing scheduler expects). expandable_segments
# lets Inductor's max-autotune explore larger Triton workspaces without
# fragmentation OOMs on the refiner KV window.
os.environ.setdefault("DISABLE_XFORMERS", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import pyrallis  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from diffusion.utils.logger import get_root_logger  # noqa: E402
from inference_video_scripts.wm.inference_sana_wm import action_string_to_c2w  # noqa: F401  (re-export)
from inference_video_scripts.wm.inference_sana_wm import (  # noqa: E402
    GenerationParams,
    InferenceConfig,
    RefinerSettings,
    SanaWMPipeline,
    _resolve_trajectory,
    _snap_num_frames,
    estimate_intrinsics_with_pi3x,
    load_intrinsics,
    resize_and_center_crop,
    transform_intrinsics_for_crop,
)
from sana.tools import resolve_hf_path  # noqa: E402

# Canonical 4-step distilled-student schedule.
DEFAULT_DENOISING_STEP_LIST = "1000,960,889,727,0"

# Refiner KV sliding-window (used when --refiner_kv_max_frames is unset). Must be
# >= sink_size + refiner_block_size so each AR step still sees the full previous
# block; a smaller window (e.g. 2 < 1 sink + 3 block) drops cross-chunk context
# and makes the decoded video flicker at every chunk boundary.
DEFAULT_REFINER_KV_MAX_FRAMES = 11

# Streaming weights are auto-fetched from the Hub on first use, mirroring the
# bidirectional script. Each artefact resolves through ``resolve_hf_path`` so a
# local path / bundle still wins when present (it returns existing local paths
# unchanged and only snapshot-downloads bare ``hf://`` URIs).
HF_REPO = "Efficient-Large-Model/SANA-WM_streaming"
HF_STREAMING_DEFAULTS = {
    "model_path": f"hf://{HF_REPO}/sana_dit/model.pt",
    "causal_vae_path": f"hf://{HF_REPO}/ltx2_causal_vae",
    "refiner_root": f"hf://{HF_REPO}/refiner_diffusers",
    "gemma_root": f"hf://{HF_REPO}/gemma3_12b",
}

# The inference YAML ships in-repo (configs/sana_wm/), not in the weights repo.
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sana_wm" / "sana_wm_streaming_1600m_720p.yaml"

# Optional LOCAL bundle dir (``--streaming_root``). Unset by default so the
# hf:// defaults above drive a first-use download.
DEFAULT_STREAMING_ROOT = None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end streaming SANA-WM inference (stage-1 + refiner + causal VAE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--image", type=Path, required=True, help="First-frame RGB image.")
    p.add_argument("--prompt", type=Path, required=True, help="UTF-8 text file with the prompt.")
    p.add_argument("--output_dir", type=Path, required=True, help="Directory to write the progressive mp4.")
    p.add_argument("--name", default="output", help="Filename stem for outputs.")

    cam_group = p.add_mutually_exclusive_group(required=True)
    cam_group.add_argument("--camera", type=Path, help="(F,4,4) .npy camera-to-world poses.")
    cam_group.add_argument(
        "--action", type=str, help="Action DSL string, e.g. 'w-120,lw-80,...'. Rolled out internally."
    )

    p.add_argument(
        "--translation_speed", type=float, default=0.025, help="Per-frame translation magnitude when --action is used."
    )
    p.add_argument(
        "--rotation_speed_deg",
        type=float,
        default=0.6,
        help="Per-frame rotation magnitude (degrees) when --action is used.",
    )
    p.add_argument(
        "--intrinsics",
        type=Path,
        default=None,
        help="(3,3), (F,3,3), or (4,) intrinsics .npy. Pi3X-estimated if omitted.",
    )
    p.add_argument(
        "--num_frames",
        type=int,
        default=241,
        help="Total pixel frames (~15s @16fps; 241 = 24*10+1). Snapped to "
        "8*refiner_block_size*k+1 so the VAE and refiner chunking divide evenly.",
    )
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--flow_shift", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative_prompt", default="")

    # Streaming-only knobs. By default every weight is fetched from the Hub
    # (hf://Efficient-Large-Model/SANA-WM_streaming) on first use; pass a local
    # path / bundle to any of these to override.
    p.add_argument(
        "--streaming_root",
        type=Path,
        default=DEFAULT_STREAMING_ROOT,
        help="Optional LOCAL bundle dir holding sana_dit/, ltx2_causal_vae/, refiner_diffusers/, gemma3_12b/. "
        f"If unset, each artefact is fetched from hf://{HF_REPO}.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Streaming YAML (local path or hf:// URI). Default: in-repo configs/sana_wm/sana_wm_streaming_1600m_720p.yaml "
        "(or <streaming_root>/sana_wm_streaming_1600m_720p.yaml when --streaming_root is set).",
    )
    p.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=f"Override the streaming DiT checkpoint (local path or hf:// URI). Default: hf://{HF_REPO}/sana_dit/model.pt.",
    )
    p.add_argument(
        "--causal_vae_path",
        type=str,
        default=None,
        help=f"Override the causal LTX-2 VAE (local path or hf:// URI). Default: hf://{HF_REPO}/ltx2_causal_vae.",
    )
    p.add_argument(
        "--refiner_root",
        type=str,
        default=None,
        help=f"Override the chunk-causal refiner (local path or hf:// URI). Default: hf://{HF_REPO}/refiner_diffusers.",
    )
    p.add_argument(
        "--refiner_gemma_root",
        type=str,
        default=None,
        help=f"Override the Gemma diffusers root (local path or hf:// URI). Default: hf://{HF_REPO}/gemma3_12b.",
    )
    p.add_argument(
        "--denoising_step_list",
        default=DEFAULT_DENOISING_STEP_LIST,
        help="Comma-separated distilled-student timestep schedule (must end with 0).",
    )
    p.add_argument(
        "--num_frame_per_block",
        type=int,
        default=3,
        help="Latent frames per stage-1 AR chunk (must match the model's chunk_size).",
    )
    p.add_argument("--refiner_block_size", type=int, default=3, help="Refiner latent frames per AR block.")
    p.add_argument(
        "--refiner_kv_max_frames",
        type=int,
        default=DEFAULT_REFINER_KV_MAX_FRAMES,
        help="Refiner KV sliding-window size (sink + history + active).",
    )
    p.add_argument("--refiner_seed", type=int, default=42)
    p.add_argument("--sink_size", type=int, default=1)
    p.add_argument("--no_sink_token", action="store_true", help="Disable the stage-1 sink token (default: enabled).")
    p.add_argument(
        "--num_cached_blocks", type=int, default=2, help="Stage-1 KV sliding-window size (-1 keeps all past chunks)."
    )
    p.add_argument(
        "--streaming_crf", type=int, default=18, help="ffmpeg CRF for the progressive MP4 (lower = higher quality)."
    )
    p.add_argument("--streaming_preset", default="medium", help="ffmpeg libx264 preset for the progressive MP4 writer.")
    p.add_argument(
        "--streaming_encoder",
        default=os.environ.get("SANA_WM_STREAMING_MP4_ENCODER", "libx264"),
        help="MP4 encoder: libx264 or h264_nvenc.",
    )
    p.add_argument(
        "--output_mode",
        choices=["mp4", "cpu", "discard"],
        default="mp4",
        help="mp4 writes H.264, cpu copies decoded uint8 frames to host without writing, discard decodes on GPU and drops frames after synchronization.",
    )
    p.add_argument(
        "--no_mp4",
        action="store_true",
        help="Alias for --output_mode=discard; excludes uint8 CPU transfer and MP4 encoding from timings.",
    )
    p.add_argument(
        "--benchmark_json", type=Path, default=None, help="Optional JSON file with wall-clock throughput metrics."
    )
    p.add_argument(
        "--benchmark_repeats",
        type=int,
        default=1,
        help="Run generation this many times in one process; useful for warm/steady benchmark separation.",
    )
    p.add_argument(
        "--profile_cuda",
        action="store_true",
        help="Record per-stage CUDA event timings; useful for bottleneck breakdown but disabled by default for pure throughput.",
    )
    p.add_argument(
        "--sample_frames_npz", type=Path, default=None, help="Optional .npz path for sampled decoded uint8 frames."
    )
    p.add_argument(
        "--sample_frame_stride",
        type=int,
        default=16,
        help="Save every Nth output frame when --sample_frames_npz is set.",
    )
    p.add_argument("--no_compile", action="store_true", help="Disable torch.compile for smoke/debug runs.")

    p.add_argument("--offload_vae", action="store_true", help="Move the VAE to CPU between encode/decode steps.")
    p.add_argument(
        "--offload_refiner",
        action="store_true",
        help="Lazy-load the LTX-2 refiner only when needed; release afterwards.",
    )
    p.add_argument(
        "--offload_text_encoder",
        action="store_true",
        help="Move the stage-1 text encoder to CPU after prompt encoding to save GPU memory.",
    )

    # --- Quantized inference (per-component precision) ---
    p.add_argument(
        "--stage1_precision",
        choices=["bf16", "fp8", "fp4"],
        default="bf16",
        help="Stage-1 DiT compute precision. bf16 (default, any GPU); "
        "fp8 = FP8 W8A8 (Hopper+ / Blackwell); fp4 = NVFP4 W4A4 (Blackwell only). "
        "Quantizes self-attn + cross-attn + FFN, per-block scoped. Needs Transformer Engine.",
    )
    p.add_argument(
        "--refiner_precision",
        choices=["bf16", "fp8", "fp4"],
        default="bf16",
        help="LTX-2 refiner compute precision. bf16 (default, any GPU); "
        "fp8 = FP8 W8A8 (Hopper+ / Blackwell); fp4 = NVFP4 W4A4 (Blackwell only). "
        "Needs Transformer Engine.",
    )
    return p


def _resolve_streaming_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    """Materialise the five checkpoint paths, fetching from the Hub on first use.

    With ``--streaming_root`` set we read a LOCAL bundle (back-compat); otherwise
    each artefact falls back to its ``hf://`` default. Every path is then passed
    through :func:`resolve_hf_path` — local paths are returned unchanged, bare
    ``hf://`` URIs are snapshot-downloaded — and checked for existence so a bad
    local override still fails loudly.
    """
    root = args.streaming_root
    if root is not None:
        config_path = args.config or str(root / "sana_wm_streaming_1600m_720p.yaml")
        model_path = args.model_path or str(root / "sana_dit" / "model.pt")
        causal_vae_path = args.causal_vae_path or str(root / "ltx2_causal_vae")
        refiner_root = args.refiner_root or str(root / "refiner_diffusers")
        gemma_root = args.refiner_gemma_root or str(root / "gemma3_12b")
    else:
        config_path = args.config or str(DEFAULT_CONFIG)
        model_path = args.model_path or HF_STREAMING_DEFAULTS["model_path"]
        causal_vae_path = args.causal_vae_path or HF_STREAMING_DEFAULTS["causal_vae_path"]
        refiner_root = args.refiner_root or HF_STREAMING_DEFAULTS["refiner_root"]
        gemma_root = args.refiner_gemma_root or HF_STREAMING_DEFAULTS["gemma_root"]

    resolved: dict[str, Path] = {}
    for label, path in (
        ("--config", config_path),
        ("--model_path", model_path),
        ("--causal_vae_path", causal_vae_path),
        ("--refiner_root", refiner_root),
        ("--refiner_gemma_root", gemma_root),
    ):
        local = resolve_hf_path(str(path))
        if not Path(local).exists():
            raise SystemExit(f"{label} does not exist: {path}")
        resolved[label] = Path(local)
    return (
        resolved["--config"],
        resolved["--model_path"],
        resolved["--causal_vae_path"],
        resolved["--refiner_root"],
        resolved["--refiner_gemma_root"],
    )


def _apply_fast_defaults() -> None:
    """Set numerically-neutral perf knobs (cuDNN bench, SDPA, Inductor)."""
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    # Keep math SDP as a fallback for the Gemma text encoder shapes that are
    # not accepted by flash/cuDNN SDP on GB200. Main video attention still
    # selects flash/cuDNN where legal.
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(True)
    import torch._inductor.config as _ic

    _ic.coordinate_descent_tuning = True
    _ic.epilogue_fusion = True


def _apply_precision_args(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Translate the user-facing --stage1_precision/--refiner_precision flags into
    the internal TE quant env knobs, BEFORE the model is built. bf16 = no quant;
    fp8/fp4 quantize the same layers (self-attn + cross-attn + FFN, per-block
    scoped) and differ only in the TE recipe (FP8 block scaling vs NVFP4)."""
    if args.stage1_precision == "bf16":
        os.environ.pop("SANA_WM_STAGE1_NVFP4", None)
    else:
        os.environ["SANA_WM_STAGE1_NVFP4"] = "self_attn+cross+ffn"
        os.environ["SANA_WM_STAGE1_NVFP4_SCOPE"] = "block"
        os.environ["SANA_WM_STAGE1_LINEARIZE_FFN"] = "1"
        os.environ["SANA_WM_STAGE1_QUANT"] = "fp8block" if args.stage1_precision == "fp8" else "nvfp4"

    if args.refiner_precision == "bf16":
        os.environ.pop("SANA_WM_REFINER_NVFP4", None)
    else:
        os.environ["SANA_WM_REFINER_NVFP4"] = "1"
        os.environ["SANA_WM_REFINER_QUANT"] = "fp8block" if args.refiner_precision == "fp8" else "nvfp4"

    # fp8/fp4 need NVIDIA Transformer Engine (NOT in the default install). Fail fast
    # with an actionable message instead of a deep ImportError at model-build time.
    if args.stage1_precision != "bf16" or args.refiner_precision != "bf16":
        need = "NVFP4BlockScaling" if "fp4" in (args.stage1_precision, args.refiner_precision) else "Float8BlockScaling"
        try:
            import transformer_engine.common.recipe as _te_recipe
            import transformer_engine.pytorch  # noqa: F401

            if not hasattr(_te_recipe, need):
                raise ImportError(f"installed Transformer Engine lacks {need}")
        except Exception as exc:  # ImportError or version too old
            raise SystemExit(
                f"--stage1_precision/--refiner_precision fp8/fp4 require NVIDIA Transformer "
                f"Engine >= 2.x (for {need}), which is NOT part of the default SANA-WM install "
                f"({exc}). Install it with:\n"
                f"    pip install --no-build-isolation 'transformer_engine[pytorch]'\n"
                f"(the CUDA toolkit from environment_setup.sh is required to build it), or run "
                f"with bf16 (the default)."
            )

    if "fp4" in (args.stage1_precision, args.refiner_precision) and torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] < 10:  # NVFP4 needs Blackwell (sm_100/sm_120)
            logger.warning(
                "fp4 (NVFP4) requires a Blackwell GPU (sm_100/sm_120); detected sm_%d%d. " "Use fp8 on Hopper.",
                *torch.cuda.get_device_capability(),
            )
    logger.info("[precision] stage1=%s refiner=%s", args.stage1_precision, args.refiner_precision)


def main() -> None:
    args = _build_parser().parse_args()
    if args.benchmark_repeats < 1:
        raise SystemExit("--benchmark_repeats must be >= 1.")
    logger: logging.Logger = get_root_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _apply_fast_defaults()
    _apply_precision_args(args, logger)

    image = Image.open(args.image).convert("RGB")
    prompt = args.prompt.read_text(encoding="utf-8", errors="replace").strip()
    if not prompt:
        raise SystemExit(f"Prompt file is empty: {args.prompt}")

    c2w_full = _resolve_trajectory(args)
    num_frames = min(args.num_frames, c2w_full.shape[0])
    # Snap so both the LTX-2 VAE (8k+1) and the refiner chunking are satisfied:
    # the per-block check needs (F-1)/8 divisible by refiner_block_size, i.e.
    # F = 8*block*k + 1. Snapping to that stride covers both at once.
    snap_stride = 8 * args.refiner_block_size
    snapped = _snap_num_frames(num_frames, stride=snap_stride, upper_bound=c2w_full.shape[0])
    if snapped != args.num_frames:
        logger.warning(
            f"num_frames must be {snap_stride}k+1 (LTX-2 VAE 8k+1 + refiner block_size "
            f"{args.refiner_block_size}); --num_frames={args.num_frames} snapped to {snapped} "
            f"(trajectory has {c2w_full.shape[0]} frames)."
        )
    num_frames = snapped
    c2w = c2w_full[:num_frames]

    cropped, src_size, resized_size, crop_offset = resize_and_center_crop(image)
    if args.intrinsics is not None:
        intr_src = load_intrinsics(args.intrinsics, num_frames)
    else:
        intr_one = estimate_intrinsics_with_pi3x(image, device, logger)
        intr_src = np.broadcast_to(intr_one, (num_frames, 4)).copy()
    intrinsics_vec4 = transform_intrinsics_for_crop(intr_src, src_size, resized_size, crop_offset)

    config_path, model_path, causal_vae_path, refiner_root, gemma_root = _resolve_streaming_paths(args)
    config: InferenceConfig = pyrallis.parse(config_class=InferenceConfig, config_path=str(config_path), args=[])
    config.vae.vae_type = "LTX2VAE_diffusers_causal"
    config.vae.vae_pretrained = str(causal_vae_path)
    logger.info(f"[causal-vae] vae_pretrained -> {config.vae.vae_pretrained}")

    refiner = RefinerSettings(
        root=str(refiner_root),
        gemma_root=str(gemma_root),
        sink_size=args.sink_size,
        seed=args.refiner_seed,
        block_size=args.refiner_block_size,
        kv_max_frames=args.refiner_kv_max_frames,
    )

    pipeline = SanaWMPipeline(
        config=config,
        model_path=str(model_path),
        device=device,
        refiner=refiner,
        offload_vae=args.offload_vae,
        offload_refiner=args.offload_refiner,
        offload_text_encoder=args.offload_text_encoder,
        logger=logger,
    )

    if not args.no_compile:
        # Numerically-equivalent compile (default Inductor mode, no CUDA-graph
        # capture, no fp32->fp16 substitution). Cold compile takes ~3 min the
        # first time; subsequent runs reuse the Inductor cache.
        compile_mode = os.environ.get("SANA_WM_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs").strip()
        compile_dynamic_raw = os.environ.get("SANA_WM_TORCH_COMPILE_DYNAMIC", "1").strip().lower()
        compile_dynamic = compile_dynamic_raw not in {"0", "false", "no", "off"}
        compile_targets_raw = os.environ.get("SANA_WM_TORCH_COMPILE_TARGETS", "refiner")
        compile_targets = {
            item.strip().lower() for item in compile_targets_raw.replace("+", ",").split(",") if item.strip()
        }
        unsupported_targets = compile_targets - {"vae", "refiner"}
        if unsupported_targets:
            raise SystemExit(
                "SANA_WM_TORCH_COMPILE_TARGETS only supports 'vae' and 'refiner'; "
                f"got {sorted(unsupported_targets)}."
            )
        # The causal VAE streaming decoder must NOT be torch.compile'd: the
        # compiled graph corrupts its cross-chunk causal cache, so chunk 0
        # decodes correctly but every chunk >=1 decodes to zeros (uniform gray
        # frames from ~1.5s on). Refuse it unconditionally; compile the refiner
        # (the heavy module) instead.
        if "vae" in compile_targets:
            logger.warning(
                "[streaming] ignoring 'vae' in SANA_WM_TORCH_COMPILE_TARGETS: compiling the "
                "causal VAE decoder corrupts its streaming cache (chunk>=1 -> blank). Skipping it."
            )
            compile_targets.discard("vae")
        logger.info(
            "[streaming] torch.compile("
            f"targets={sorted(compile_targets)}, mode={compile_mode!r}, dynamic={compile_dynamic})"
        )
        if "refiner" in compile_targets:
            pipeline.refiner.transformer = torch.compile(
                pipeline.refiner.transformer, mode=compile_mode, dynamic=compile_dynamic
            )

    denoising_step_list = [int(t.strip()) for t in args.denoising_step_list.split(",") if t.strip()]
    if not denoising_step_list or denoising_step_list[-1] != 0:
        raise SystemExit("--denoising_step_list must be a comma-separated list ending with 0.")

    params = GenerationParams(
        num_frames=num_frames,
        fps=args.fps,
        cfg_scale=args.cfg_scale,
        flow_shift=args.flow_shift,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        sampling_algo="self_forcing",
        num_cached_blocks=args.num_cached_blocks,
        sink_token=not args.no_sink_token,
        num_frame_per_block=args.num_frame_per_block,
        denoising_step_list=denoising_step_list,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.no_mp4 and args.output_mode not in {"mp4", "discard"}:
        raise SystemExit("--no_mp4 cannot be combined with --output_mode=cpu.")
    output_mode = "discard" if args.no_mp4 else args.output_mode
    if args.sample_frames_npz is not None and args.sample_frame_stride < 1:
        raise SystemExit("--sample_frame_stride must be >= 1.")

    run_payloads = []
    result = None
    end_to_end_seconds = 0.0
    for run_idx in range(args.benchmark_repeats):
        suffix = "" if args.benchmark_repeats == 1 else f"_run{run_idx:02d}"
        streaming_path = out_dir / f"{args.name}{suffix}_streaming.mp4"
        sample_frames_path = None
        if args.sample_frames_npz is not None:
            sample_frames_path = args.sample_frames_npz
            if args.benchmark_repeats > 1:
                sample_frames_path = sample_frames_path.with_name(
                    f"{sample_frames_path.stem}{suffix}{sample_frames_path.suffix}"
                )
        logger.info(
            f"[streaming] starting interactive chunk-pipelined inference "
            f"run={run_idx}/{args.benchmark_repeats - 1} "
            f"output_mode={output_mode} -> {streaming_path}"
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        wall_start = time.perf_counter()
        result = pipeline.generate_streaming(
            cropped,
            prompt,
            c2w,
            intrinsics_vec4,
            params,
            output_path=streaming_path,
            streaming_crf=args.streaming_crf,
            streaming_preset=args.streaming_preset,
            streaming_encoder=args.streaming_encoder,
            output_mode=output_mode,
            profile_cuda=args.profile_cuda,
            sample_frames_path=sample_frames_path,
            sample_frame_stride=args.sample_frame_stride,
        )
        end_to_end_seconds = time.perf_counter() - wall_start
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        logger.info(
            f"[streaming] done: run={run_idx} mode={result['output_mode']} "
            f"frames={result['n_pixel_frames']} stream_wall={result['wall_seconds']:.3f}s "
            f"e2e={end_to_end_seconds:.3f}s fps={result['frames_per_second']:.3f} "
            f"realtime={result['realtime_factor']:.3f}x peak_mem={peak_mem_gb:.1f}GB "
            f"path={result['output_path']}"
        )
        run_payloads.append(
            {
                "run_index": int(run_idx),
                "peak_mem_gb": float(peak_mem_gb),
                "output_mode": result["output_mode"],
                "output_path": str(result["output_path"]) if result["output_path"] is not None else None,
                "n_pixel_frames": int(result["n_pixel_frames"]),
                "frames_per_second": float(result["frames_per_second"]),
                "realtime_factor": float(result["realtime_factor"]),
                "stream_wall_seconds": float(result["wall_seconds"]),
                "end_to_end_seconds": float(end_to_end_seconds),
                "first_chunk_seconds": result["first_chunk_seconds"],
                "first_chunk_frames": result["first_chunk_frames"],
                "steady_state_seconds": result["steady_state_seconds"],
                "steady_state_frames_per_second": result["steady_state_frames_per_second"],
                "steady_state_realtime_factor": result["steady_state_realtime_factor"],
                "stage1_cuda_seconds": result["stage1_cuda_seconds"],
                "refiner_cuda_seconds": result["refiner_cuda_seconds"],
                "decode_cuda_seconds": result["decode_cuda_seconds"],
                "sample_frames_path": (
                    str(result["sample_frames_path"]) if result["sample_frames_path"] is not None else None
                ),
                "sampled_frame_count": int(result["sampled_frame_count"]),
                "sampled_frame_indices": result["sampled_frame_indices"],
            }
        )

    assert result is not None
    if args.benchmark_json is not None:
        runtime_env = {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "DPM_TQDM",
                "FUSED_GDN_PRECISION",
                "PRECISION_OVERRIDE",
                "PYTORCH_CUDA_ALLOC_CONF",
                "SANA_WM_STREAMING_PROMPT_CACHE",
                "SANA_WM_STREAMING_MP4_ENCODER",
                "SANA_WM_STREAMING_PREDECODE_SINK",
                "SANA_WM_STREAMING_DIRECT_REFINED_BLOCKS",
                "SANA_WM_STREAMING_REFINER_FIRST",
                "SANA_WM_STREAMING_DECODE_CURRENT",
                "SANA_WM_STAGE1_NVFP4",
                "SANA_WM_STAGE1_NVFP4_MODE",
                "SANA_WM_STAGE1_NVFP4_INCLUDE_PATTERNS",
                "SANA_WM_STAGE1_NVFP4_SKIP_PATTERNS",
                "SANA_WM_STAGE1_NVFP4_TEXT_PAD_MULTIPLE",
                "SANA_WM_STAGE1_LINEARIZE_FFN",
                "SANA_WM_STAGE1_KV_SAVE_STRIDE",
                "SANA_WM_SDPA_D112_DIRECT",
                "SANA_WM_REFINER_NVFP4",
                "SANA_WM_REFINER_ATTN_BACKEND",
                "SANA_WM_REFINER_SELF_ATTN_KERNEL",
                "SANA_WM_REFINER_PRECONCAT_PREFIX",
                "SANA_WM_REFINER_KV_CACHE_DTYPE",
                "SANA_WM_REFINER_CAPTURE_KV_ONLY_LAST",
                "SANA_WM_REFINER_FUSE_SELF_QKV",
                "SANA_WM_REFINER_FAST_KV_CAPTURE",
                "SANA_WM_REFINER_PREGENERATE_NOISE",
                "SANA_WM_REFINER_TIMESTEP_CACHE",
                "SANA_WM_REFINER_FAST_KV_CLEAN_INTERVAL",
                "SANA_WM_REFINER_HISTORY_LAYERS",
                "SANA_WM_REFINER_HISTORY_LAYER_STRIDE",
                "SANA_WM_REFINER_HISTORY_LAYER_OFFSET",
                "SANA_WM_REFINER_HISTORY_KEEP_LAST",
                "SANA_WM_REFINER_CROSS_ATTN_KV_CACHE",
                "SANA_WM_REFINER_EMPTY_CACHE_BEFORE_PREFIX",
                "SANA_WM_REFINER_EMPTY_CACHE_BEFORE_CAPTURE",
                "SANA_WM_REFINER_PROFILE",
                "SANA_WM_REFINER_LAYER_PROFILE",
                "SANA_WM_REFINER_NVFP4_OFFLOAD_STAGE1",
                "SANA_WM_REFINER_NVFP4_OFFLOAD_VAE",
                "SANA_WM_TE_NVFP4_CPU_STAGING",
                "SANA_WM_STREAMING_LAZY_VAE_DECODER",
                "SANA_WM_TORCH_COMPILE_MODE",
                "SANA_WM_TORCH_COMPILE_DYNAMIC",
                "SANA_WM_TORCH_COMPILE_TARGETS",
                "SANA_WM_CUDAGRAPH_MARK_STEP",
            )
            if os.environ.get(name) is not None
        }
        payload = {
            "output_mode": result["output_mode"],
            "output_path": str(result["output_path"]) if result["output_path"] is not None else None,
            "num_frames": int(num_frames),
            "requested_num_frames": int(args.num_frames),
            "actual_num_frames": int(num_frames),
            "n_pixel_frames": int(result["n_pixel_frames"]),
            "fps_target": int(args.fps),
            "frames_per_second": float(result["frames_per_second"]),
            "realtime_factor": float(result["realtime_factor"]),
            "stream_wall_seconds": float(result["wall_seconds"]),
            "end_to_end_seconds": float(end_to_end_seconds),
            "first_chunk_seconds": result["first_chunk_seconds"],
            "first_chunk_frames": result["first_chunk_frames"],
            "steady_state_seconds": result["steady_state_seconds"],
            "steady_state_frames_per_second": result["steady_state_frames_per_second"],
            "steady_state_realtime_factor": result["steady_state_realtime_factor"],
            "stage1_cuda_seconds": result["stage1_cuda_seconds"],
            "refiner_cuda_seconds": result["refiner_cuda_seconds"],
            "decode_cuda_seconds": result["decode_cuda_seconds"],
            "sample_frames_path": (
                str(result["sample_frames_path"]) if result["sample_frames_path"] is not None else None
            ),
            "sampled_frame_count": int(result["sampled_frame_count"]),
            "sampled_frame_indices": result["sampled_frame_indices"],
            "n_refiner_blocks": int(result["n_refiner_blocks"]),
            "n_decode_chunks": int(result["n_decode_chunks"]),
            "denoising_step_list": denoising_step_list,
            "num_frame_per_block": int(args.num_frame_per_block),
            "refiner_block_size": int(args.refiner_block_size),
            "refiner_kv_max_frames": int(args.refiner_kv_max_frames),
            "streaming_encoder": str(args.streaming_encoder),
            "num_cached_blocks": int(args.num_cached_blocks),
            "torch_compile": not bool(args.no_compile),
            "profile_cuda": bool(args.profile_cuda),
            "streaming_root": str(args.streaming_root),
            "config": str(config_path),
            "model_path": str(model_path),
            "causal_vae_path": str(causal_vae_path),
            "refiner_root": str(refiner_root),
            "refiner_gemma_root": str(gemma_root),
            "benchmark_repeats": int(args.benchmark_repeats),
            "runtime_env": runtime_env,
            "runs": run_payloads,
            "best_stream_wall_seconds": min(run["stream_wall_seconds"] for run in run_payloads),
            "best_realtime_factor": max(run["realtime_factor"] for run in run_payloads),
            "best_steady_state_realtime_factor": max(
                (
                    run["steady_state_realtime_factor"]
                    for run in run_payloads
                    if run["steady_state_realtime_factor"] is not None
                ),
                default=None,
            ),
            "last_stream_wall_seconds": float(run_payloads[-1]["stream_wall_seconds"]),
            "last_realtime_factor": float(run_payloads[-1]["realtime_factor"]),
            "last_steady_state_realtime_factor": run_payloads[-1]["steady_state_realtime_factor"],
        }
        args.benchmark_json.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
