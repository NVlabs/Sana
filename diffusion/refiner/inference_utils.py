#!/usr/bin/env python3
"""Causal student refiner inference utilities.

Two inference modes are supported:
  - ``full``:  all 81 latent frames through the transformer at once
  - ``frame``: one latent frame at a time (autoregressive)

Both modes share VAE encode/decode and text encoding; only the denoising
loop differs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- vendor imports ---
from diffusion.refiner.ltx_core.causal_attention import apply_causal_mask, build_causal_video_mask
from diffusion.refiner.ltx_core.ltx_student_wrapper import LTXStudentWrapper
from diffusion.refiner.vendor.ltx_core.components.diffusion_steps import EulerDiffusionStep
from diffusion.refiner.vendor.ltx_core.components.noisers import GaussianNoiser
from diffusion.refiner.vendor.ltx_core.components.patchifiers import VideoLatentPatchifier
from diffusion.refiner.vendor.ltx_core.components.schedulers import LTX2Scheduler
from diffusion.refiner.vendor.ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from diffusion.refiner.vendor.ltx_core.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    TilingConfig,
    VideoDecoderConfigurator,
    VideoEncoderConfigurator,
)
from diffusion.refiner.vendor.ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    module_ops_from_gemma_root,
)
from diffusion.refiner.vendor.ltx_core.tools import VideoLatentTools
from diffusion.refiner.vendor.ltx_core.types import LatentState, VideoLatentShape
from diffusion.refiner.vendor.ltx_core.utils import find_matching_file
from diffusion.refiner.vendor.ltx_pipelines.utils.constants import (
    DEFAULT_NEGATIVE_PROMPT,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from diffusion.refiner.vendor.ltx_pipelines.utils.helpers import (
    modality_from_latent_state,
    post_process_latent,
)

try:
    import imageio.v2 as imageio_v2
except Exception:
    imageio_v2 = None

# ---- defaults ----
DEFAULT_FUSED_CHECKPOINT = str(
    (REPO_ROOT / "output" / "pretrained_models" / "LTX_2" / "ltx-2-19b-refiner-fused.safetensors").resolve()
)
DEFAULT_GEMMA_ROOT = str((REPO_ROOT / "output" / "pretrained_models" / "LTX_2" / "diffusers").resolve())
DEFAULT_LORA_CHECKPOINT = str(
    (REPO_ROOT / "output" / "distill_causal_refiner_cached_qkv_only" / "checkpoints" / "latest.pt").resolve()
)

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Causal student refiner inference.")
    p.add_argument(
        "--mode",
        choices=["full", "frame", "sequential", "streaming"],
        required=True,
        help="full = all frames at once, frame = 1 frame at a time, "
        "sequential = chunk-by-chunk with KV cache, "
        "streaming = long video chunk-by-chunk with incremental KV cache",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_LORA_CHECKPOINT,
        help="Trained LoRA checkpoint (.pt), or 'none' for zero-init LoRA",
    )
    p.add_argument("--fused-model", type=str, default=DEFAULT_FUSED_CHECKPOINT, help="Fused teacher safetensors")
    p.add_argument("--gemma-root", type=str, default=DEFAULT_GEMMA_ROOT, help="Gemma weights root for text encoder")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=81, help="Number of pixel frames to use from video (must be 8k+1)")
    p.add_argument(
        "--image-size", type=int, default=720, help="Resize video so shorter side = this (training used 720)"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-inference-steps", type=int, default=3, help="Denoising steps (sigma steps)")
    p.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Start from this sigma step (0=most noisy σ=0.909, 1=σ=0.725, 2=σ=0.422)",
    )
    p.add_argument("--fps", type=float, default=16.0)
    # Data source: use a zip dataset sample
    p.add_argument(
        "--data-zip",
        type=str,
        default=str(
            (REPO_ROOT / "data" / "sekai_game_train_961frames_16fps_ovl640" / "sekai_game_train_00000000.zip").resolve()
        ),
        help="Path to a SanaZip .zip file",
    )
    p.add_argument("--sample-index", type=int, default=0, help="Sample index within the zip")
    p.add_argument("--prompt", type=str, default=None, help="Text prompt (if None, reads from zip metadata)")
    p.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Chunk sizes for causal mask, e.g. 5 3 3. None = per-frame causal.",
    )
    p.add_argument(
        "--overlap-blend",
        action="store_true",
        help="Enable overlap-blend: run with shifted chunk configs and blend results (requires --chunk-sizes)",
    )
    p.add_argument(
        "--streaming-blend",
        action="store_true",
        help="Enable streaming blend: sliding window with overlap (true streaming)",
    )
    p.add_argument("--window-size", type=int, default=5, help="Window size for streaming blend (latent frames)")
    p.add_argument(
        "--window-overlap", type=int, default=2, help="Overlap between consecutive windows for streaming blend"
    )
    p.add_argument(
        "--drop-history",
        action="store_true",
        help="In streaming blend, drop context frames beyond the overlap (keep total ≤ window_size + overlap)",
    )
    p.add_argument(
        "--attention-mode",
        type=str,
        default="causal",
        choices=["causal", "sliding_window", "bidirectional"],
        help="Attention mode: causal (chunk), sliding_window, or bidirectional",
    )
    p.add_argument(
        "--window-radius", type=int, default=None, help="For sliding_window: each frame sees ±window_radius frames"
    )
    # --- FM refiner options ---
    p.add_argument(
        "--schedule",
        choices=["distilled", "full"],
        default="distilled",
        help="Sigma schedule: 'distilled' = 3-step STAGE_2 (default), 'full' = LTX2Scheduler multi-step",
    )
    p.add_argument(
        "--cfg-scale", type=float, default=1.0, help="CFG scale (1.0 = no guidance). FM refiner typically uses 3.0"
    )
    p.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt for CFG (default: vendor DEFAULT_NEGATIVE_PROMPT when cfg>1)",
    )
    p.add_argument(
        "--rescale-scale",
        type=float,
        default=0.0,
        help="Guidance rescale to prevent oversaturation (0=off, 0.7=vendor default)",
    )
    p.add_argument(
        "--flow-shift",
        type=float,
        default=None,
        help="Override flow shift for sigma schedule (default: auto from token count)",
    )
    p.add_argument(
        "--start-sigma",
        type=float,
        default=0.909375,
        help="Sigma at which refiner starts denoising (default: 0.909375)",
    )
    # --- Streaming long-video options ---
    p.add_argument(
        "--first-chunk-size", type=int, default=5, help="Frames in first chunk for streaming mode (default: 5)"
    )
    p.add_argument(
        "--stream-chunk-size", type=int, default=3, help="Frames in subsequent chunks for streaming mode (default: 3)"
    )
    p.add_argument(
        "--cache-mode",
        choices=["full", "window"],
        default="window",
        help="KV cache mode: full=keep all, window=keep last N chunks (default: window)",
    )
    p.add_argument(
        "--window-chunks", type=int, default=2, help="Number of past chunks to keep in window cache mode (default: 2)"
    )
    return p


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_video_from_zip(
    zip_path: str, sample_index: int, num_frames: int, image_size: int = 720
) -> tuple[torch.Tensor, str]:
    """Load a single video sample from a SanaZip file.

    Returns (pixel_video [B,C,F,H,W] in [-1,1] float32, caption string).
    """
    import zipfile

    import numpy as np

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = sorted([n for n in zf.namelist() if n.endswith(".mp4") or n.endswith(".webm")])
        if not namelist:
            raise ValueError(f"No video files found in {zip_path}")
        if sample_index >= len(namelist):
            raise IndexError(f"sample_index={sample_index} >= {len(namelist)} videos in zip")

        video_name = namelist[sample_index]
        video_bytes = zf.read(video_name)

        # Try to read caption from co-located json
        caption = "a video"
        base_name = video_name.rsplit(".", 1)[0]
        json_name = f"{base_name}.json"
        if json_name in zf.namelist():
            try:
                data = json.loads(zf.read(json_name))
                if isinstance(data, dict) and "prompt" in data:
                    caption = data["prompt"]
                elif isinstance(data, dict) and "caption" in data:
                    caption = data["caption"]
            except Exception:
                pass

    # Decode video using imageio
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        import imageio.v2 as iio

        reader = iio.get_reader(tmp.name, format="FFMPEG")
        frames = []
        for i, frame in enumerate(reader):
            if len(frames) >= num_frames:
                break
            frames.append(frame)
        reader.close()

    if len(frames) < num_frames:
        print(f"[WARN] Video has only {len(frames)} frames, requested {num_frames}. Using available frames.")
        num_frames = len(frames)
    # Ensure num_frames = 8k+1
    num_frames = ((num_frames - 1) // 8) * 8 + 1
    frames = frames[:num_frames]

    # Stack to tensor [F, H, W, C] uint8
    video_np = np.stack(frames, axis=0)
    video_tensor = torch.from_numpy(video_np).float()  # [F, H, W, C]

    # Resize so shorter side = image_size, preserving aspect ratio
    F, H, W, C = video_tensor.shape
    if min(H, W) != image_size:
        if H <= W:
            new_h = image_size
            new_w = int(round(W * image_size / H))
        else:
            new_w = image_size
            new_h = int(round(H * image_size / W))
        # Resize using torch interpolate: [F,H,W,C] -> [F,C,H,W] -> resize -> [F,H,W,C]
        video_chw = video_tensor.permute(0, 3, 1, 2)  # [F, C, H, W]
        video_chw = torch.nn.functional.interpolate(
            video_chw, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        video_tensor = video_chw.permute(0, 2, 3, 1)  # [F, H, W, C]
        F, H, W, C = video_tensor.shape

    # Align to 32-pixel grid
    target_h = (H // 32) * 32
    target_w = (W // 32) * 32
    video_tensor = video_tensor[:, :target_h, :target_w, :]

    # Convert to [B, C, F, H, W] in [-1, 1]
    video_bcfhw = video_tensor.permute(3, 0, 1, 2).unsqueeze(0) / 127.5 - 1.0  # [1, C, F, H, W]
    return video_bcfhw, caption


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def build_vae_encoder(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    builder = Builder(
        model_path=checkpoint_path,
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    )
    return builder.build(device=device, dtype=dtype).to(device).eval()


def build_vae_decoder(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    builder = Builder(
        model_path=checkpoint_path,
        model_class_configurator=VideoDecoderConfigurator,
        model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
    )
    return builder.build(device=device, dtype=dtype).to(device).eval()


@torch.no_grad()
def encode_video(encoder, pixel_video: torch.Tensor) -> torch.Tensor:
    """Encode [B, C, F, H, W] pixel video → [B, 128, F', H', W'] raw latent."""
    return encoder.tiled_encode(pixel_video, TilingConfig.default())


@torch.no_grad()
def decode_video(decoder, latent: torch.Tensor) -> torch.Tensor:
    """Decode [B, 128, F', H', W'] raw latent → [F, H, W, C] uint8."""
    chunks = list(decoder.decode_video(latent, TilingConfig.default()))
    return torch.cat(chunks, dim=0)  # [F, H, W, C] uint8


class TextEncoder:
    """Single-GPU Gemma text encoder with CPU offload (like CachedPromptEncoder in training)."""

    def __init__(self, checkpoint_path: str, gemma_root: str, dtype: torch.dtype, device: torch.device):
        self._dtype = dtype
        self._device = device

        module_ops = module_ops_from_gemma_root(gemma_root)
        model_folder = find_matching_file(gemma_root, "model*.safetensors").parent
        weight_paths = [str(p) for p in model_folder.rglob("*.safetensors")]
        self._text_enc_builder = Builder(
            model_path=tuple(weight_paths),
            model_class_configurator=GemmaTextEncoderConfigurator,
            model_sd_ops=GEMMA_LLM_KEY_OPS,
            module_ops=(GEMMA_MODEL_OPS, *module_ops),
        )
        self._emb_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
        )

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        """Encode a single prompt → v_context [1, seq_len, D]."""
        # Load Gemma to GPU, encode, offload
        text_encoder = self._text_enc_builder.build(device=torch.device("cpu"), dtype=self._dtype).eval()
        text_encoder.to(self._device)
        hs, mask = text_encoder.encode(prompt)
        text_encoder.to("cpu")
        del text_encoder
        torch.cuda.empty_cache()

        # Load embeddings processor, process, offload
        emb_proc = self._emb_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
        result = emb_proc.process_hidden_states(hs, mask)
        emb_proc.to("cpu")
        del emb_proc
        torch.cuda.empty_cache()

        # result.video_encoding may be [seq_len, D] or [1, seq_len, D]
        v = result.video_encoding.to(device=self._device, dtype=self._dtype)
        if v.dim() == 2:
            v = v.unsqueeze(0)  # [1, seq_len, D]
        return v


# ---------------------------------------------------------------------------
# Denoising loops
# ---------------------------------------------------------------------------


def create_noised_state(
    stage2_latent: torch.Tensor,
    *,
    seed: int,
    fps: float,
    device: torch.device,
    dtype: torch.dtype,
    start_step: int = 0,
    noise_scale: float | None = None,
    patchified_noise: torch.Tensor | None = None,
    temporal_offset: int = 0,
) -> tuple[LatentState, VideoLatentTools]:
    """Create patchified noised state from raw stage-2 latent [B, C, F, H, W].

    Parameters
    ----------
    start_step : int
        Which sigma to use for noise injection (0 = most noisy, 1 = less noisy, etc.)
        Only used when noise_scale is None.
    noise_scale : float or None
        Explicit noise scale. Overrides start_step when provided.
    patchified_noise : optional [B, T, C] tensor
        Pre-generated noise in patchified space.  When provided the ``seed``
        parameter is ignored and this noise is used directly.
    temporal_offset : int
        Global latent-frame offset for this window.
    """
    B, C, Fr, H, W = stage2_latent.shape
    tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=VideoLatentShape(batch=B, channels=C, frames=Fr, height=H, width=W),
        fps=fps,
    )
    state = tools.create_initial_state(
        device=device,
        dtype=dtype,
        initial_latent=stage2_latent.to(device=device, dtype=dtype),
        temporal_offset=temporal_offset,
    )
    scale = float(noise_scale) if noise_scale is not None else float(STAGE_2_DISTILLED_SIGMA_VALUES[start_step])
    if patchified_noise is not None:
        scaled_mask = state.denoise_mask * scale
        noised = patchified_noise * scaled_mask + state.latent * (1 - scaled_mask)
        state = replace(state, latent=noised.to(dtype))
    else:
        noiser = GaussianNoiser(torch.Generator(device=device).manual_seed(seed))
        state = noiser(state, scale)
    return state, tools


def build_sigma_schedule(
    schedule: str,
    num_steps: int,
    start_sigma: float,
    device: torch.device,
    *,
    flow_shift: float | None = None,
    start_step: int = 0,
) -> torch.Tensor:
    """Build sigma schedule for denoising.

    Returns a 1-D tensor of sigma values from start_sigma down to 0.
    """
    if schedule == "distilled":
        sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device)
        sigmas = sigmas[start_step:]
        if num_steps < len(sigmas) - 1:
            sigmas = sigmas[: num_steps + 1]
        return sigmas

    # "full" schedule via LTX2Scheduler, truncated to start_sigma
    if flow_shift is not None:
        import math

        raw = torch.linspace(1.0, 0.0, num_steps + 1)
        raw = torch.where(raw != 0, math.exp(flow_shift) / (math.exp(flow_shift) + (1 / raw - 1)), 0.0)
        non_zero = raw != 0
        one_minus_z = 1.0 - raw[non_zero]
        scale_factor = one_minus_z[-1] / (1.0 - 0.1)
        raw[non_zero] = 1.0 - (one_minus_z / scale_factor)
        full_sigmas = raw.to(dtype=torch.float32, device=device)
    else:
        full_sigmas = LTX2Scheduler().execute(steps=num_steps).to(dtype=torch.float32, device=device)

    # Truncate to start_sigma
    for i in range(len(full_sigmas)):
        if full_sigmas[i].item() <= start_sigma + 1e-6:
            sigmas = full_sigmas[i:].clone()
            sigmas[0] = start_sigma
            return sigmas

    raise ValueError(f"No sigma <= {start_sigma} in schedule")


@torch.inference_mode()
def denoise_full_cfg(
    student: LTXStudentWrapper,
    state: LatentState,
    tools: VideoLatentTools,
    v_context: torch.Tensor,
    v_context_neg: torch.Tensor | None,
    sigmas: torch.Tensor,
    *,
    cfg_scale: float = 1.0,
    rescale_scale: float = 0.0,
    chunk_sizes: list[int] | None = None,
    attention_mode: str = "causal",
    window_radius: int | None = None,
) -> torch.Tensor:
    """Multi-step Euler denoise with optional CFG.

    When cfg_scale > 1.0, runs two forward passes per step (cond + uncond).
    Returns raw latent [B, C, F, H, W].
    """
    stepper = EulerDiffusionStep()
    use_cfg = cfg_scale > 1.0 and v_context_neg is not None
    apply_causal = attention_mode != "bidirectional"

    for step_idx in range(len(sigmas) - 1):
        sigma = sigmas[step_idx]

        # Conditioned prediction
        denoised_cond = student.forward_denoise(
            video_state=state,
            v_context=v_context,
            sigma=sigma,
            apply_causal=apply_causal,
            chunk_sizes=chunk_sizes,
            attention_mode=attention_mode if attention_mode != "bidirectional" else "causal",
            window_radius=window_radius,
        )

        if use_cfg:
            # Unconditioned prediction
            denoised_uncond = student.forward_denoise(
                video_state=state,
                v_context=v_context_neg,
                sigma=sigma,
                apply_causal=apply_causal,
                chunk_sizes=chunk_sizes,
                attention_mode=attention_mode if attention_mode != "bidirectional" else "causal",
                window_radius=window_radius,
            )
            # CFG combine: pred = uncond + cfg_scale * (cond - uncond)
            denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)

            # Rescale to prevent oversaturation
            if rescale_scale > 0:
                factor = denoised_cond.float().std() / denoised.float().std().clamp(min=1e-8)
                factor = rescale_scale * factor + (1 - rescale_scale)
                denoised = denoised * factor
        else:
            denoised = denoised_cond

        denoised_pp = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
        state = replace(state, latent=stepper.step(state.latent, denoised_pp, sigmas, step_idx))

    final = tools.unpatchify(tools.clear_conditioning(state)).latent
    return final


@torch.inference_mode()
def denoise_full(
    student: LTXStudentWrapper,
    state: LatentState,
    tools: VideoLatentTools,
    v_context: torch.Tensor,
    num_steps: int,
    chunk_sizes: list[int] | None = None,
    start_step: int = 0,
    attention_mode: str = "causal",
    window_radius: int | None = None,
) -> torch.Tensor:
    """Run full denoising (all frames at once) with causal/sliding-window mask.

    Parameters
    ----------
    start_step : int
        Skip the first ``start_step`` denoising steps.
    attention_mode : str
        ``"causal"`` or ``"sliding_window"``.
    window_radius : int or None
        For sliding_window: each frame sees ±window_radius frames.

    Returns raw latent [B, C, F, H, W].
    """
    sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=student.device)
    sigmas = sigmas[start_step:]  # trim to start from the right sigma
    if num_steps < len(sigmas) - 1:
        sigmas = sigmas[: num_steps + 1]
    stepper = EulerDiffusionStep()

    for step_idx in range(len(sigmas) - 1):
        denoised = student.forward_denoise_step(
            video_state=state,
            v_context=v_context,
            sigmas=sigmas,
            step_index=step_idx,
            apply_causal=True,
            chunk_sizes=chunk_sizes,
            attention_mode=attention_mode,
            window_radius=window_radius,
        )
        denoised_pp = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
        state = replace(state, latent=stepper.step(state.latent, denoised_pp, sigmas, step_idx))

    # Unpatchify to get [B, C, F, H, W]
    final = tools.unpatchify(tools.clear_conditioning(state)).latent
    return final


def _generate_shifted_chunk_configs(base_chunks: list[int], num_frames: int) -> list[list[int]]:
    """Generate shifted chunk configurations for overlap-blend.

    Given base [5,3,3] (11 frames, 3 chunks), produce configs with different
    boundary positions so every frame is "interior" in at least one config.

    Strategy: shift boundaries by ±1 where possible, keeping the same number
    of chunks and ensuring each chunk has size >= 1.
    """
    n_chunks = len(base_chunks)
    configs = [base_chunks]

    # Generate shifted versions by moving 1 frame between adjacent chunks
    for i in range(n_chunks - 1):
        # Shift boundary right: chunk[i] gets +1, chunk[i+1] gets -1
        if base_chunks[i + 1] > 1:
            shifted = list(base_chunks)
            shifted[i] += 1
            shifted[i + 1] -= 1
            if shifted not in configs:
                configs.append(shifted)

        # Shift boundary left: chunk[i] gets -1, chunk[i+1] gets +1
        if base_chunks[i] > 1:
            shifted = list(base_chunks)
            shifted[i] -= 1
            shifted[i + 1] += 1
            if shifted not in configs:
                configs.append(shifted)

    # Also try a more aggressive shift: move 2 frames if possible
    for i in range(n_chunks - 1):
        if base_chunks[i + 1] > 2:
            shifted = list(base_chunks)
            shifted[i] += 2
            shifted[i + 1] -= 2
            if shifted not in configs:
                configs.append(shifted)
        if base_chunks[i] > 2:
            shifted = list(base_chunks)
            shifted[i] -= 2
            shifted[i + 1] += 2
            if shifted not in configs:
                configs.append(shifted)

    return configs


def _compute_blend_weights(chunk_configs: list[list[int]], num_frames: int) -> torch.Tensor:
    """Compute per-frame blend weights for each chunk config.

    Each frame gets weight proportional to its distance from the nearest
    chunk boundary in that config. Interior frames get high weight,
    boundary frames get low weight.

    Returns: (num_configs, num_frames) float tensor, normalized per frame.
    """
    weights = torch.zeros(len(chunk_configs), num_frames)

    for ci, chunks in enumerate(chunk_configs):
        offset = 0
        for chunk_size in chunks:
            for local_pos in range(chunk_size):
                frame_idx = offset + local_pos
                # Distance from nearest chunk boundary (0 = at boundary)
                dist = min(local_pos, chunk_size - 1 - local_pos)
                # +0.5 so boundary frames still get some weight
                weights[ci, frame_idx] = dist + 0.5
            offset += chunk_size

    # Normalize per frame across configs
    weights = weights / weights.sum(dim=0, keepdim=True).clamp(min=1e-8)
    return weights


@torch.inference_mode()
def denoise_overlap_blend(
    student: LTXStudentWrapper,
    stage2_latent: torch.Tensor,
    v_context: torch.Tensor,
    *,
    seed: int,
    fps: float,
    num_steps: int,
    base_chunk_sizes: list[int],
    start_step: int = 0,
) -> torch.Tensor:
    """Denoise with multiple shifted chunk configs and blend results.

    For each chunk configuration (the base + shifted variants):
      1. Create noised state from the SAME noise (same seed)
      2. Run full denoising with that chunk config
      3. Collect the denoised latent

    Then blend all results per-frame, weighting by distance from chunk
    boundaries — frames near a boundary in one config get low weight
    from that config but high weight from a config where they're interior.

    Returns raw latent [B, C, F, H, W].
    """
    device = student.device
    dtype = student.dtype
    num_frames = stage2_latent.shape[2]  # latent frames

    configs = _generate_shifted_chunk_configs(base_chunk_sizes, num_frames)
    print(f"  Overlap-blend: {len(configs)} configs: {configs}")

    weights = _compute_blend_weights(configs, num_frames).to(device=device, dtype=dtype)
    print(f"  Blend weights per config:")
    for ci, cfg in enumerate(configs):
        w = weights[ci].cpu().tolist()
        print(f"    {cfg}: [{', '.join(f'{x:.2f}' for x in w)}]")

    all_denoised = []
    for ci, chunk_cfg in enumerate(configs):
        print(f"  Config {ci + 1}/{len(configs)}: {chunk_cfg} ...", end="", flush=True)
        t0 = time.perf_counter()

        # Same noise for all configs (same seed)
        state, tools = create_noised_state(
            stage2_latent, seed=seed, fps=fps, device=device, dtype=dtype, start_step=start_step
        )
        denoised = denoise_full(
            student, state, tools, v_context, num_steps, chunk_sizes=chunk_cfg, start_step=start_step
        )
        all_denoised.append(denoised)
        print(f" {time.perf_counter() - t0:.1f}s")

    # Blend: weighted average over configs per latent frame
    # all_denoised[i] shape: (B, C, F, H, W)
    # weights shape: (num_configs, F)
    stacked = torch.stack(all_denoised, dim=0)  # (N, B, C, F, H, W)
    # Reshape weights for broadcasting: (N, 1, 1, F, 1, 1)
    w = weights[:, None, None, :, None, None]
    blended = (stacked * w).sum(dim=0)  # (B, C, F, H, W)

    return blended


@torch.inference_mode()
def denoise_streaming_blend(
    student: LTXStudentWrapper,
    stage2_latent: torch.Tensor,
    v_context: torch.Tensor,
    *,
    seed: int,
    fps: float,
    num_steps: int,
    window_size: int = 5,
    overlap: int = 2,
    start_step: int = 0,
    drop_history: bool = False,
    attention_mode: str = "causal",
    window_radius: int | None = None,
) -> torch.Tensor:
    """Streaming denoising with overlapping windows and linear blending.

    Processes the video chunk-by-chunk in a sliding window fashion:
      1. Each window of ``window_size`` latent frames is denoised.
         When attention_mode="causal": chunk causal (context=past, window=bidirectional).
         When attention_mode="sliding_window": sliding window ±window_radius on all frames.
      2. Adjacent windows overlap by ``overlap`` frames.
      3. In the overlap region, outputs are linearly blended.

    Parameters
    ----------
    window_size : int
        Number of latent frames per processing window.
    overlap : int
        Number of frames shared between consecutive windows.
    attention_mode : str
        "causal" or "sliding_window".
    window_radius : int or None
        For sliding_window: each frame sees ±window_radius frames.

    Returns raw latent [B, C, F, H, W].
    """
    device = student.device
    dtype = student.dtype
    B, C, Fr, H, W = stage2_latent.shape

    assert 0 <= overlap < window_size, f"overlap={overlap} must be < window_size={window_size}"
    stride = window_size - overlap

    # Compute window start positions
    windows = []
    start = 0
    while start < Fr:
        end = min(start + window_size, Fr)
        windows.append((start, end))
        if end == Fr:
            break
        start += stride
    print(f"  Streaming blend: window_size={window_size}, overlap={overlap}, stride={stride}")
    print(f"  Windows: {windows} ({len(windows)} total)")

    sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device)
    sigmas = sigmas[start_step:]  # trim to start from the right sigma
    if num_steps < len(sigmas) - 1:
        sigmas = sigmas[: num_steps + 1]

    # Output buffer — accumulate denoised frames
    denoised_latent = torch.zeros_like(stage2_latent)
    committed_up_to = 0  # frames [0, committed_up_to) are finalized

    for wi, (win_start, win_end) in enumerate(windows):
        win_len = win_end - win_start
        print(f"  Window {wi + 1}/{len(windows)}: frames [{win_start}:{win_end}] ...", end="", flush=True)
        t0 = time.perf_counter()

        # Build input: context + noised current window
        if win_start > 0:
            if drop_history:
                # Only keep overlap frames as context (drop older history)
                ctx_start = max(0, win_start - overlap)
            else:
                ctx_start = 0
            context_latent = denoised_latent[:, :, ctx_start:win_start, :, :]
            window_latent = stage2_latent[:, :, win_start:win_end, :, :]
            full_latent = torch.cat([context_latent, window_latent], dim=2)
        else:
            ctx_start = 0
            full_latent = stage2_latent[:, :, :win_end, :, :]

        # Create noised state for the sequence
        state, tools = create_noised_state(
            full_latent, seed=seed, fps=fps, device=device, dtype=dtype, start_step=start_step
        )

        ctx_len = win_start - ctx_start

        # Build attention config based on mode
        if attention_mode == "sliding_window":
            # Sliding window: all frames (context + window) use ±radius attention
            chunk_sizes = None
            fwd_attention_mode = "sliding_window"
            fwd_window_radius = window_radius
        else:
            # Chunk causal: context as one chunk, window as another
            if ctx_len > 0:
                chunk_sizes = [ctx_len, win_len]
            else:
                chunk_sizes = [win_len]
            fwd_attention_mode = "causal"
            fwd_window_radius = None

        stepper = EulerDiffusionStep()
        for step_idx in range(len(sigmas) - 1):
            denoised = student.forward_denoise_step(
                video_state=state,
                v_context=v_context,
                sigmas=sigmas,
                step_index=step_idx,
                apply_causal=True,
                chunk_sizes=chunk_sizes,
                attention_mode=fwd_attention_mode,
                window_radius=fwd_window_radius,
            )
            denoised_pp = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
            state = replace(state, latent=stepper.step(state.latent, denoised_pp, sigmas, step_idx))

        # Extract denoised result for the current window
        final = tools.unpatchify(tools.clear_conditioning(state)).latent
        # final has shape (B, C, ctx_len + win_len, H, W); window is the last win_len frames
        window_output = final[:, :, ctx_len : ctx_len + win_len, :, :]  # (B, C, win_len, H, W)

        # Blend overlap region, then commit new frames
        if win_start > 0 and overlap > 0:
            ovl_start = win_start
            ovl_end = min(win_start + overlap, win_end)
            ovl_len = ovl_end - ovl_start

            # Linear blend weights: 0 at start of overlap → 1 at end
            alpha = torch.linspace(0.0, 1.0, ovl_len, device=device, dtype=dtype)
            alpha = alpha[None, None, :, None, None]  # (1, 1, ovl_len, 1, 1)

            prev_ovl = denoised_latent[:, :, ovl_start:ovl_end, :, :]
            curr_ovl = window_output[:, :, :ovl_len, :, :]
            denoised_latent[:, :, ovl_start:ovl_end, :, :] = (1 - alpha) * prev_ovl + alpha * curr_ovl

            # Non-overlap part of window
            if ovl_end < win_end:
                denoised_latent[:, :, ovl_end:win_end, :, :] = window_output[:, :, ovl_len:, :, :]
        else:
            denoised_latent[:, :, win_start:win_end, :, :] = window_output

        committed_up_to = win_end
        dt = time.perf_counter() - t0
        print(f" {dt:.1f}s (committed frames 0-{committed_up_to - 1})")

    return denoised_latent


# ---------------------------------------------------------------------------
# Sequential chunk-by-chunk inference (teacher-forcing style)
# ---------------------------------------------------------------------------


def _cache_context_kv(
    student: LTXStudentWrapper,
    context_latent: torch.Tensor,
    v_context: torch.Tensor,
    *,
    fps: float,
    chunk_sizes: list[int],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Forward clean context at σ=0 through all 48 blocks, cache post-RoPE KV.

    Returns a list of (K, V) tuples, one per transformer block.
    Context attn1 KV is independent of text (text enters via attn2 cross-attention)
    and σ=0 is constant, so this cache is valid across all denoising steps AND
    both CFG passes (conditioned + unconditioned).
    """
    B, C, Fr, H, W = context_latent.shape
    ctx_tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=VideoLatentShape(batch=B, channels=C, frames=Fr, height=H, width=W),
        fps=fps,
    )
    context_state = ctx_tools.create_initial_state(
        device=context_latent.device,
        dtype=context_latent.dtype,
        initial_latent=context_latent,
    )
    if len(chunk_sizes) > 1:
        context_state = apply_causal_mask(context_state, chunk_sizes=chunk_sizes)

    ctx_mod = modality_from_latent_state(
        context_state,
        v_context,
        torch.zeros(B, device=context_latent.device, dtype=torch.float32),
    )

    vm = student.transformer.velocity_model
    preprocessor = vm.video_args_preprocessor
    if hasattr(preprocessor, "simple_preprocessor"):
        preprocessor = preprocessor.simple_preprocessor

    ctx_args = preprocessor.prepare(ctx_mod)

    from diffusion.refiner.vendor.ltx_core.guidance.perturbations import BatchedPerturbationConfig

    perturbations = BatchedPerturbationConfig.empty(B)

    all_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
    for block in vm.transformer_blocks:
        block.attn1._tf_capture_kv = True
        ctx_args, _ = block(video=ctx_args, audio=None, perturbations=perturbations)
        all_kv.append(block.attn1._tf_cached_kv)
        block.attn1._tf_capture_kv = False

    return all_kv


def _inject_kv(student: LTXStudentWrapper, all_kv: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    for block, kv in zip(student.transformer.velocity_model.transformer_blocks, all_kv):
        block.attn1._tf_kv_prefix = kv


def _clear_kv(student: LTXStudentWrapper) -> None:
    for block in student.transformer.velocity_model.transformer_blocks:
        block.attn1._tf_kv_prefix = None


@torch.inference_mode()
def denoise_sequential_chunks(
    student: LTXStudentWrapper,
    stage2_latent: torch.Tensor,
    v_context: torch.Tensor,
    v_context_neg: torch.Tensor | None,
    sigmas: torch.Tensor,
    *,
    chunk_sizes: list[int],
    cfg_scale: float = 1.0,
    rescale_scale: float = 0.0,
    seed: int = 42,
    fps: float = 16.0,
) -> torch.Tensor:
    """Sequential chunk-by-chunk denoising matching teacher-forcing training.

    For each chunk i:
      1. If i > 0: forward already-denoised chunks 0..i-1 at σ=0 through all
         48 blocks, cache post-RoPE attn1 K,V (one-time per chunk).
      2. Denoise chunk i through N Euler steps, injecting context KV at every
         step. Each step only forwards chunk i's tokens.
      3. Store denoised chunk i; move to chunk i+1.

    The context KV is cached once per chunk and reused across all denoising
    steps + both CFG passes (context KV is text-independent and σ=0-constant).

    Returns raw latent [B, C, F, H, W].
    """
    device = student.device
    dtype = student.dtype
    B, C, Fr, H, W = stage2_latent.shape

    denoised_latent = torch.zeros_like(stage2_latent)
    stepper = EulerDiffusionStep()
    use_cfg = cfg_scale > 1.0 and v_context_neg is not None

    offset = 0
    for chunk_idx, chunk_size in enumerate(chunk_sizes):
        chunk_start = offset
        chunk_end = offset + chunk_size
        offset = chunk_end

        print(
            f"  Chunk {chunk_idx + 1}/{len(chunk_sizes)}: " f"frames [{chunk_start}:{chunk_end}] ...",
            end="",
            flush=True,
        )
        t0 = time.perf_counter()

        # Create noised state for this chunk (correct temporal_offset for RoPE)
        chunk_input = stage2_latent[:, :, chunk_start:chunk_end, :, :]
        state, tools = create_noised_state(
            chunk_input,
            seed=seed,
            fps=fps,
            device=device,
            dtype=dtype,
            noise_scale=float(sigmas[0]),
            temporal_offset=chunk_start,
        )

        # Cache context KV from already-denoised chunks
        context_kv = None
        if chunk_idx > 0:
            context_latent = denoised_latent[:, :, :chunk_start, :, :].clone()
            context_kv = _cache_context_kv(
                student,
                context_latent.to(device=device, dtype=dtype),
                v_context,
                fps=fps,
                chunk_sizes=chunk_sizes[:chunk_idx],
            )
            _inject_kv(student, context_kv)

        # Multi-step Euler denoising
        for step_idx in range(len(sigmas) - 1):
            sigma = sigmas[step_idx]

            denoised_cond = student.forward_denoise(
                video_state=state,
                v_context=v_context,
                sigma=sigma,
                apply_causal=False,
            )

            if use_cfg:
                denoised_uncond = student.forward_denoise(
                    video_state=state,
                    v_context=v_context_neg,
                    sigma=sigma,
                    apply_causal=False,
                )
                denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
                if rescale_scale > 0:
                    factor = denoised_cond.float().std() / denoised.float().std().clamp(min=1e-8)
                    factor = rescale_scale * factor + (1 - rescale_scale)
                    denoised = denoised * factor
            else:
                denoised = denoised_cond

            denoised_pp = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
            state = replace(state, latent=stepper.step(state.latent, denoised_pp, sigmas, step_idx))

        # Clear context KV
        if context_kv is not None:
            _clear_kv(student)
            del context_kv

        # Store denoised chunk
        final = tools.unpatchify(tools.clear_conditioning(state)).latent
        denoised_latent[:, :, chunk_start:chunk_end, :, :] = final

        dt = time.perf_counter() - t0
        print(f" {dt:.1f}s")

    return denoised_latent


# ---------------------------------------------------------------------------
# Streaming long-video inference with incremental KV cache
# ---------------------------------------------------------------------------


def _cache_chunk_kv_incremental(
    student: LTXStudentWrapper,
    chunk_latent: torch.Tensor,
    v_context: torch.Tensor,
    *,
    fps: float,
    temporal_offset: int,
    existing_cache: list[tuple[torch.Tensor, torch.Tensor]] | None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Forward ONE clean chunk at σ=0 with existing KV cache injected.

    Captures this chunk's post-RoPE K,V at each layer.  The existing cache
    provides context from previously denoised chunks — the hidden states at
    each layer correctly incorporate that context through the injected KV.

    Returns a list of (K, V) tuples for this chunk only (not accumulated).
    """
    B, C, Fr, H, W = chunk_latent.shape
    tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=VideoLatentShape(batch=B, channels=C, frames=Fr, height=H, width=W),
        fps=fps,
    )
    state = tools.create_initial_state(
        device=chunk_latent.device,
        dtype=chunk_latent.dtype,
        initial_latent=chunk_latent,
        temporal_offset=temporal_offset,
    )
    mod = modality_from_latent_state(
        state,
        v_context,
        torch.zeros(B, device=chunk_latent.device, dtype=torch.float32),
    )

    vm = student.transformer.velocity_model
    preprocessor = vm.video_args_preprocessor
    if hasattr(preprocessor, "simple_preprocessor"):
        preprocessor = preprocessor.simple_preprocessor

    args = preprocessor.prepare(mod)

    from diffusion.refiner.vendor.ltx_core.guidance.perturbations import BatchedPerturbationConfig

    perturbations = BatchedPerturbationConfig.empty(B)

    chunk_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx, block in enumerate(vm.transformer_blocks):
        if existing_cache is not None:
            block.attn1._tf_kv_prefix = existing_cache[layer_idx]
        block.attn1._tf_capture_kv = True
        args, _ = block(video=args, audio=None, perturbations=perturbations)
        chunk_kv.append(block.attn1._tf_cached_kv)
        block.attn1._tf_capture_kv = False
        block.attn1._tf_kv_prefix = None

    return chunk_kv


def _accumulate_cache(
    accumulated: list[tuple[torch.Tensor, torch.Tensor]] | None,
    new_chunk_kv: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Append new chunk KV to accumulated cache (full mode)."""
    if accumulated is None:
        return [(k.clone(), v.clone()) for k, v in new_chunk_kv]
    result = []
    for (old_k, old_v), (new_k, new_v) in zip(accumulated, new_chunk_kv):
        result.append(
            (
                torch.cat([old_k, new_k], dim=1),
                torch.cat([old_v, new_v], dim=1),
            )
        )
    return result


def _window_cache(
    window_kvs: list[list[tuple[torch.Tensor, torch.Tensor]]],
    window_chunks: int,
) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
    """Build accumulated cache from the last ``window_chunks`` chunk KVs."""
    if not window_kvs:
        return None
    # Take last window_chunks entries
    kvs = window_kvs[-window_chunks:]
    n_layers = len(kvs[0])
    result = []
    for layer_idx in range(n_layers):
        ks = [kv[layer_idx][0] for kv in kvs]
        vs = [kv[layer_idx][1] for kv in kvs]
        result.append((torch.cat(ks, dim=1), torch.cat(vs, dim=1)))
    return result


def _cache_memory_gb(cache: list[tuple[torch.Tensor, torch.Tensor]] | None) -> float:
    if cache is None:
        return 0.0
    total = sum(k.nelement() * k.element_size() + v.nelement() * v.element_size() for k, v in cache)
    return total / (1024**3)


@torch.inference_mode()
def denoise_streaming_long(
    student: LTXStudentWrapper,
    stage2_latent: torch.Tensor,
    v_context: torch.Tensor,
    v_context_neg: torch.Tensor | None,
    sigmas: torch.Tensor,
    *,
    first_chunk_size: int = 5,
    chunk_size: int = 3,
    cache_mode: str = "window",
    window_chunks: int = 2,
    cfg_scale: float = 1.0,
    rescale_scale: float = 0.0,
    seed: int = 42,
    fps: float = 16.0,
) -> torch.Tensor:
    """Streaming long-video inference with incremental KV cache.

    Processes video chunk-by-chunk.  After denoising each chunk, only that
    chunk is forwarded at σ=0 to incrementally build the KV cache for the
    next chunk — no need to re-forward the entire context each time.

    Parameters
    ----------
    first_chunk_size : int
        Frames in the first chunk (default 5).
    chunk_size : int
        Frames in all subsequent chunks (default 3).
    cache_mode : "full" or "window"
        "full" — keep all previous chunks' KV (memory grows linearly).
        "window" — keep only the last ``window_chunks`` chunks' KV.
    window_chunks : int
        Number of past chunks to keep in window mode (default 2).

    Returns raw latent [B, C, F, H, W].
    """
    device = student.device
    dtype = student.dtype
    B, C, Fr, H, W = stage2_latent.shape

    # Build chunk schedule: [5, 3, 3, 3, ..., 3]
    chunks: list[int] = [first_chunk_size]
    remaining = Fr - first_chunk_size
    while remaining > 0:
        cs = min(chunk_size, remaining)
        chunks.append(cs)
        remaining -= cs
    n_chunks = len(chunks)

    print(f"  Streaming long video: {Fr} latent frames, {n_chunks} chunks")
    print(f"  Chunk schedule: [{chunks[0]}] + [{chunk_size}] × {n_chunks - 1}")
    print(f"  Cache mode: {cache_mode}" + (f" (window={window_chunks})" if cache_mode == "window" else ""))
    print(f"  Denoise steps: {len(sigmas) - 1}, CFG: {cfg_scale}")

    denoised_latent = torch.zeros_like(stage2_latent)
    stepper = EulerDiffusionStep()
    use_cfg = cfg_scale > 1.0 and v_context_neg is not None

    # KV cache state
    accumulated_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    window_kvs: list[list[tuple[torch.Tensor, torch.Tensor]]] = []

    offset = 0
    for chunk_idx, cs in enumerate(chunks):
        chunk_start = offset
        chunk_end = offset + cs
        offset = chunk_end

        print(f"  [{chunk_idx + 1}/{n_chunks}] frames [{chunk_start}:{chunk_end}]", end="", flush=True)
        t0 = time.perf_counter()

        # --- Denoise this chunk ---
        chunk_input = stage2_latent[:, :, chunk_start:chunk_end, :, :]
        state, tools = create_noised_state(
            chunk_input,
            seed=seed,
            fps=fps,
            device=device,
            dtype=dtype,
            noise_scale=float(sigmas[0]),
            temporal_offset=chunk_start,
        )

        # Inject context KV for denoising
        current_cache = accumulated_cache if cache_mode == "full" else _window_cache(window_kvs, window_chunks)
        if current_cache is not None:
            _inject_kv(student, current_cache)

        for step_idx in range(len(sigmas) - 1):
            denoised_cond = student.forward_denoise(
                video_state=state,
                v_context=v_context,
                sigma=sigmas[step_idx],
                apply_causal=False,
            )
            if use_cfg:
                denoised_uncond = student.forward_denoise(
                    video_state=state,
                    v_context=v_context_neg,
                    sigma=sigmas[step_idx],
                    apply_causal=False,
                )
                denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
                if rescale_scale > 0:
                    factor = denoised_cond.float().std() / denoised.float().std().clamp(min=1e-8)
                    factor = rescale_scale * factor + (1 - rescale_scale)
                    denoised = denoised * factor
            else:
                denoised = denoised_cond
            denoised_pp = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
            state = replace(state, latent=stepper.step(state.latent, denoised_pp, sigmas, step_idx))

        if current_cache is not None:
            _clear_kv(student)

        # Store denoised chunk
        final = tools.unpatchify(tools.clear_conditioning(state)).latent
        denoised_latent[:, :, chunk_start:chunk_end, :, :] = final
        del state

        # --- Incrementally cache this chunk's KV ---
        chunk_kv = _cache_chunk_kv_incremental(
            student,
            final.to(device=device, dtype=dtype),
            v_context,
            fps=fps,
            temporal_offset=chunk_start,
            existing_cache=accumulated_cache if cache_mode == "full" else _window_cache(window_kvs, window_chunks),
        )

        if cache_mode == "full":
            accumulated_cache = _accumulate_cache(accumulated_cache, chunk_kv)
            cache_gb = _cache_memory_gb(accumulated_cache)
        else:
            window_kvs.append(chunk_kv)
            if len(window_kvs) > window_chunks:
                window_kvs.pop(0)
            cache_gb = _cache_memory_gb(_window_cache(window_kvs, window_chunks))

        dt = time.perf_counter() - t0
        print(f" {dt:.1f}s (cache: {cache_gb:.1f}GB)")

    # Cleanup
    del accumulated_cache, window_kvs
    torch.cuda.empty_cache()

    return denoised_latent


@torch.inference_mode()
def denoise_frame_by_frame(
    student: LTXStudentWrapper,
    stage2_latent: torch.Tensor,
    v_context: torch.Tensor,
    *,
    seed: int,
    fps: float,
    num_steps: int,
) -> torch.Tensor:
    """Run denoising one latent frame at a time (autoregressive).

    For each latent frame t:
      - Build the noised state for frames [0..t]
      - Apply causal mask (frame t can attend to [0..t])
      - Run denoising loop
      - Keep the denoised result for frame t
      - Previous frames use their already-denoised values

    Returns raw latent [B, C, F, H, W].
    """
    device = student.device
    dtype = student.dtype
    B, C, Fr, H, W = stage2_latent.shape

    sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device)
    if num_steps < len(sigmas) - 1:
        sigmas = sigmas[: num_steps + 1]

    # We'll build up the denoised latent frame by frame
    denoised_latent = torch.zeros_like(stage2_latent)

    for frame_idx in range(Fr):
        print(f"  [frame {frame_idx + 1}/{Fr}]", end="", flush=True)
        t0 = time.perf_counter()

        # Use frames [0..frame_idx] — already denoised frames + current noised frame
        # For the current frame, use the original latent; for previous frames, use denoised values
        partial_latent = (
            torch.cat(
                [
                    denoised_latent[:, :, :frame_idx, :, :],
                    stage2_latent[:, :, frame_idx : frame_idx + 1, :, :],
                ],
                dim=2,
            )
            if frame_idx > 0
            else stage2_latent[:, :, :1, :, :]
        )

        # Create noised state for partial video
        state, tools = create_noised_state(partial_latent, seed=seed, fps=fps, device=device, dtype=dtype)
        stepper = EulerDiffusionStep()

        for step_idx in range(len(sigmas) - 1):
            denoised = student.forward_denoise_step(
                video_state=state,
                v_context=v_context,
                sigmas=sigmas,
                step_index=step_idx,
                apply_causal=True,
            )
            denoised_pp = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
            state = replace(state, latent=stepper.step(state.latent, denoised_pp, sigmas, step_idx))

        # Extract the last frame (frame_idx) from denoised result
        final = tools.unpatchify(tools.clear_conditioning(state)).latent
        denoised_latent[:, :, frame_idx : frame_idx + 1, :, :] = final[:, :, -1:, :, :]

        dt = time.perf_counter() - t0
        print(f" {dt:.1f}s", flush=True)

    print()
    return denoised_latent


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------


def save_video(video_thwc: torch.Tensor, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_uint8 = video_thwc.detach().cpu().to(torch.uint8).numpy()
    writer = imageio_v2.get_writer(
        str(output_path),
        fps=max(1, int(round(fps))),
        format="FFMPEG",
        codec="libx264",
        quality=8,
    )
    try:
        for frame in video_uint8:
            writer.append_data(frame)
    finally:
        writer.close()
    print(f"Saved video to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] mode={args.mode}, checkpoint={args.checkpoint}")
    print(f"[INFO] fused_model={args.fused_model}")
    print(f"[INFO] device={device}, dtype={dtype}")

    # ---- 1. Load video from zip ----
    print("[1/5] Loading video from zip...")
    t0 = time.perf_counter()
    pixel_video, caption = load_video_from_zip(args.data_zip, args.sample_index, args.num_frames, args.image_size)
    if args.prompt is not None:
        caption = args.prompt
    print(f"  video shape: {tuple(pixel_video.shape)}, caption: {caption[:80]}...")
    print(f"  took {time.perf_counter() - t0:.1f}s")

    # Save input video for comparison
    input_thwc = ((pixel_video[0].permute(1, 2, 3, 0) + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    save_video(input_thwc, args.output_dir / "input.mp4", args.fps)

    # ---- 2. VAE Encode ----
    print("[2/5] VAE encoding...")
    t0 = time.perf_counter()
    vae_encoder = build_vae_encoder(args.fused_model, device=device, dtype=dtype)
    stage2_latent = encode_video(vae_encoder, pixel_video.to(device=device, dtype=dtype))
    print(f"  latent shape: {tuple(stage2_latent.shape)}")
    vae_encoder.to("cpu")
    del vae_encoder
    torch.cuda.empty_cache()
    print(f"  took {time.perf_counter() - t0:.1f}s")

    # ---- 3. Text Encode (positive + negative for CFG) ----
    print("[3/5] Text encoding...")
    t0 = time.perf_counter()
    text_enc = TextEncoder(args.fused_model, args.gemma_root, dtype=dtype, device=device)
    v_context = text_enc.encode(caption)
    print(f"  v_context shape: {tuple(v_context.shape)}")

    # Encode negative prompt for CFG
    v_context_neg = None
    if args.cfg_scale > 1.0:
        neg_prompt = args.negative_prompt if args.negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT
        print(f"  CFG={args.cfg_scale}, encoding negative prompt ({len(neg_prompt)} chars)...")
        v_context_neg = text_enc.encode(neg_prompt)
        print(f"  v_context_neg shape: {tuple(v_context_neg.shape)}")

    del text_enc
    torch.cuda.empty_cache()
    print(f"  took {time.perf_counter() - t0:.1f}s")

    # ---- 4. Load student + LoRA ----
    print("[4/5] Loading causal student...")
    t0 = time.perf_counter()

    lora_rank = 384
    lora_alpha = 384.0
    lora_targets = ["attn1.to_q", "attn1.to_k", "attn1.to_v"]
    load_ckpt = args.checkpoint.lower() not in ("none", "zero", "")

    if load_ckpt:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        lora_rank = ckpt.get("lora_rank", lora_rank)
        lora_alpha = ckpt.get("lora_alpha", lora_alpha)
        lora_targets = ckpt.get("lora_target_modules", lora_targets)

    student = LTXStudentWrapper(
        checkpoint_path=args.fused_model,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target_modules=lora_targets,
        device=device,
        dtype=dtype,
    )

    if load_ckpt:
        # Convert DTensor (from FSDP2 training) to regular tensors.
        lora_sd = {}
        for k, v in ckpt["lora_state_dict"].items():
            if hasattr(v, "_local_tensor"):
                t = v._local_tensor
            elif torch.is_tensor(v):
                t = v
            else:
                lora_sd[k] = v
                continue
            t = t.cpu()
            # Check if this is a partial shard from FSDP2
            target_param = None
            for name, lora in student._lora_modules.items():
                if k == f"{name}.lora_A":
                    target_param = lora.lora_A
                    break
                elif k == f"{name}.lora_B":
                    target_param = lora.lora_B
                    break
            if target_param is not None and t.shape != target_param.shape:
                print(
                    f"  [WARN] {k}: checkpoint shape {tuple(t.shape)} != expected {tuple(target_param.shape)}, "
                    f"likely FSDP2 shard. Skipping load (using zero-init)."
                )
                continue
            lora_sd[k] = t
        if lora_sd:
            student.load_lora_state_dict(lora_sd)
            print(f"  loaded LoRA weights from checkpoint")
        else:
            print(f"  [WARN] No compatible LoRA weights found in checkpoint. Using zero-init.")
    else:
        print(f"  using zero-init LoRA (no checkpoint)")

    student._transformer.eval()
    print(f"  LoRA: rank={lora_rank}, alpha={lora_alpha}, targets={lora_targets}")
    print(f"  base frozen: {student.base_is_frozen()}")
    print(f"  took {time.perf_counter() - t0:.1f}s")

    # ---- 5. Denoise ----
    print(f"[5/5] Denoising (mode={args.mode}, schedule={args.schedule}, steps={args.num_inference_steps})...")
    t0 = time.perf_counter()

    if args.chunk_sizes is not None:
        print(f"  chunk_sizes={args.chunk_sizes}")
    if args.cfg_scale > 1.0:
        print(f"  CFG scale={args.cfg_scale}, rescale={args.rescale_scale}")
    if args.overlap_blend:
        if args.chunk_sizes is None:
            raise ValueError("--overlap-blend requires --chunk-sizes")
        print(f"  overlap-blend: enabled")
    if args.streaming_blend:
        print(f"  streaming-blend: window_size={args.window_size}, overlap={args.window_overlap}")

    # Build sigma schedule
    sigmas = build_sigma_schedule(
        args.schedule,
        args.num_inference_steps,
        args.start_sigma,
        device,
        flow_shift=args.flow_shift,
        start_step=args.start_step,
    )
    print(f"  sigma schedule ({len(sigmas)-1} steps): [{sigmas[0]:.4f} -> {sigmas[-1]:.4f}]")

    if args.mode == "streaming":
        # Streaming long-video with incremental KV cache
        denoised_latent = denoise_streaming_long(
            student,
            stage2_latent,
            v_context,
            v_context_neg,
            sigmas,
            first_chunk_size=args.first_chunk_size,
            chunk_size=args.stream_chunk_size,
            cache_mode=args.cache_mode,
            window_chunks=args.window_chunks,
            cfg_scale=args.cfg_scale,
            rescale_scale=args.rescale_scale,
            seed=args.seed,
            fps=args.fps,
        )
    elif args.mode == "sequential":
        # Sequential chunk-by-chunk with KV cache (teacher-forcing style)
        if args.chunk_sizes is None:
            raise ValueError("--mode sequential requires --chunk-sizes")
        print(f"  sequential: chunk-by-chunk with KV cache")
        denoised_latent = denoise_sequential_chunks(
            student,
            stage2_latent,
            v_context,
            v_context_neg,
            sigmas,
            chunk_sizes=args.chunk_sizes,
            cfg_scale=args.cfg_scale,
            rescale_scale=args.rescale_scale,
            seed=args.seed,
            fps=args.fps,
        )
    elif args.schedule == "full" or args.cfg_scale > 1.0:
        # FM refiner path: multi-step + optional CFG via denoise_full_cfg
        state, tools = create_noised_state(
            stage2_latent,
            seed=args.seed,
            fps=args.fps,
            device=device,
            dtype=dtype,
            noise_scale=args.start_sigma,
        )
        denoised_latent = denoise_full_cfg(
            student,
            state,
            tools,
            v_context,
            v_context_neg,
            sigmas,
            cfg_scale=args.cfg_scale,
            rescale_scale=args.rescale_scale,
            chunk_sizes=args.chunk_sizes,
            attention_mode=args.attention_mode,
            window_radius=args.window_radius,
        )
    elif args.streaming_blend:
        denoised_latent = denoise_streaming_blend(
            student,
            stage2_latent,
            v_context,
            seed=args.seed,
            fps=args.fps,
            num_steps=args.num_inference_steps,
            window_size=args.window_size,
            overlap=args.window_overlap,
            start_step=args.start_step,
            drop_history=args.drop_history,
            attention_mode=args.attention_mode,
            window_radius=args.window_radius,
        )
    elif args.overlap_blend:
        denoised_latent = denoise_overlap_blend(
            student,
            stage2_latent,
            v_context,
            seed=args.seed,
            fps=args.fps,
            num_steps=args.num_inference_steps,
            base_chunk_sizes=args.chunk_sizes,
            start_step=args.start_step,
        )
    elif args.mode == "full":
        state, tools = create_noised_state(
            stage2_latent, seed=args.seed, fps=args.fps, device=device, dtype=dtype, start_step=args.start_step
        )
        denoised_latent = denoise_full(
            student,
            state,
            tools,
            v_context,
            args.num_inference_steps,
            chunk_sizes=args.chunk_sizes,
            start_step=args.start_step,
            attention_mode=args.attention_mode,
            window_radius=args.window_radius,
        )
    else:
        denoised_latent = denoise_frame_by_frame(
            student,
            stage2_latent,
            v_context,
            seed=args.seed,
            fps=args.fps,
            num_steps=args.num_inference_steps,
        )

    denoise_time = time.perf_counter() - t0
    print(f"  denoised latent shape: {tuple(denoised_latent.shape)}")
    print(f"  took {denoise_time:.1f}s")

    # Free student to make room for VAE decoder
    student.close()
    del student
    torch.cuda.empty_cache()

    # ---- 6. VAE Decode ----
    print("[6/5] VAE decoding...")
    t0 = time.perf_counter()
    vae_decoder = build_vae_decoder(args.fused_model, device=device, dtype=dtype)
    decoded_thwc = decode_video(vae_decoder, denoised_latent.to(device=device, dtype=dtype))
    print(f"  decoded shape: {tuple(decoded_thwc.shape)}")
    vae_decoder.to("cpu")
    del vae_decoder
    torch.cuda.empty_cache()
    print(f"  took {time.perf_counter() - t0:.1f}s")

    # ---- 7. Save output ----
    if args.streaming_blend:
        mode_tag = f"streaming_w{args.window_size}_o{args.window_overlap}"
    elif args.overlap_blend:
        mode_tag = f"{args.mode}_blend"
    else:
        mode_tag = args.mode
    if args.start_step > 0:
        mode_tag += f"_ss{args.start_step}"
    output_path = args.output_dir / f"output_{mode_tag}.mp4"
    save_video(decoded_thwc, output_path, args.fps)

    # Save metadata
    meta = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "fused_model": str(args.fused_model),
        "data_zip": str(args.data_zip),
        "sample_index": args.sample_index,
        "prompt": caption,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "fps": args.fps,
        "latent_shape": list(stage2_latent.shape),
        "output_shape": list(decoded_thwc.shape),
        "denoise_time_s": round(denoise_time, 2),
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metadata to: {meta_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
