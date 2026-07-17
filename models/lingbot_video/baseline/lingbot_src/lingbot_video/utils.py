from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

LOW_NOISE_TAIL_V1_NAME = "low_noise_tail_v1"
LOW_NOISE_TAIL_V1_DEFAULT_STEPS = 2


def num_frames_from_duration(duration: float, fps: int) -> int:
    frame_count = int(float(duration) * int(fps))
    return ((frame_count - 1) // 4 + 1) * 4 + 1


def validate_refiner_sigmas(
    sigmas: Sequence[float] | np.ndarray,
    t_thresh: float | None = None,
) -> np.ndarray:
    arr = np.asarray(list(sigmas), dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("refiner sigma schedule must be a non-empty 1D list")
    if not np.all(np.isfinite(arr)):
        raise ValueError("refiner sigma schedule contains non-finite values")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"refiner sigma schedule values must be in [0, 1], got {arr.tolist()}")
    if arr.size > 1 and not np.all(np.diff(arr) < 0.0):
        raise ValueError(f"refiner sigma schedule must be strictly descending, got {arr.tolist()}")
    if t_thresh is not None and abs(float(arr[0]) - float(t_thresh)) > 1e-6:
        raise ValueError(f"refiner sigma schedule must start at t_thresh={float(t_thresh)}, got {float(arr[0])}")
    return arr


def compute_refiner_sigmas(
    *,
    sigma_max: float,
    sigma_min: float,
    num_inference_steps: int,
    shift: float,
    t_thresh: float | None,
    tail_steps: int = 0,
) -> np.ndarray | None:
    if t_thresh is None:
        return None
    t_value = float(t_thresh)
    if not (0.0 < t_value <= 1.0):
        raise ValueError(f"refiner t_thresh must lie in (0, 1], got {t_value}")
    steps = int(num_inference_steps)
    if steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {steps}")
    tail = int(tail_steps or 0)
    if tail < 0:
        raise ValueError(f"refiner_sigma_tail_steps must be >= 0, got {tail}")

    base = np.linspace(float(sigma_max), float(sigma_min), steps + 1).copy()[:-1]
    shift_value = float(shift)
    shifted = shift_value * base / (1.0 + (shift_value - 1.0) * base)
    eps = 1e-6
    sigmas = shifted[shifted <= t_value + eps]
    if sigmas.size == 0 or abs(float(sigmas[0]) - t_value) > eps:
        sigmas = np.concatenate([[t_value], sigmas])
    if tail > 0:
        start = float(sigmas[-1])
        stop = min(float(sigma_min), start)
        extra = np.linspace(start, stop, tail + 2, dtype=np.float64)[1:-1]
        sigmas = np.concatenate([sigmas, extra])
    return validate_refiner_sigmas(sigmas, t_value).astype(np.float32)


def prepare_refiner_latent(
    x_up: torch.Tensor,
    noise: torch.Tensor,
    t_thresh: float | torch.Tensor,
) -> torch.Tensor:
    if not torch.is_tensor(t_thresh):
        t_thresh = torch.tensor(float(t_thresh), device=x_up.device, dtype=x_up.dtype)
    while t_thresh.ndim < x_up.ndim:
        t_thresh = t_thresh.view(*t_thresh.shape, *([1] * (x_up.ndim - t_thresh.ndim)))
    return (1.0 - t_thresh) * x_up + t_thresh * noise


def _pad_prompt_embeds(
    embeds: torch.Tensor,
    mask: torch.Tensor,
    target_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if embeds.ndim != 3:
        raise ValueError(f"prompt embeds must be rank-3, got {tuple(embeds.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"prompt mask must be rank-2, got {tuple(mask.shape)}")
    if embeds.shape[:2] != mask.shape:
        raise ValueError(
            f"prompt embeds/mask shape mismatch: {tuple(embeds.shape)} vs {tuple(mask.shape)}"
        )
    if embeds.shape[0] != 1:
        raise ValueError(f"batched CFG helper expects batch=1 inputs, got {embeds.shape[0]}")
    if embeds.shape[1] > target_length:
        raise ValueError(f"cannot pad length {embeds.shape[1]} down to {target_length}")
    pad_len = target_length - embeds.shape[1]
    if pad_len == 0:
        return embeds, mask
    embed_pad = torch.zeros(
        embeds.shape[0],
        pad_len,
        embeds.shape[2],
        dtype=embeds.dtype,
        device=embeds.device,
    )
    mask_pad = torch.zeros(mask.shape[0], pad_len, dtype=mask.dtype, device=mask.device)
    return torch.cat([embeds, embed_pad], dim=1), torch.cat([mask, mask_pad], dim=1)


def batch_cfg_prompt_inputs(
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    negative_embeds: torch.Tensor,
    negative_mask: torch.Tensor,
    *,
    null_cond_clone_zero: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if null_cond_clone_zero:
        zero_negative = torch.zeros_like(prompt_embeds)
        return (
            torch.cat([prompt_embeds, zero_negative], dim=0),
            torch.cat([prompt_mask, prompt_mask.clone()], dim=0),
        )

    target_length = max(int(prompt_embeds.shape[1]), int(negative_embeds.shape[1]))
    prompt_padded, prompt_mask_padded = _pad_prompt_embeds(
        prompt_embeds, prompt_mask, target_length
    )
    negative_padded, negative_mask_padded = _pad_prompt_embeds(
        negative_embeds, negative_mask, target_length
    )
    return (
        torch.cat([prompt_padded, negative_padded], dim=0),
        torch.cat([prompt_mask_padded, negative_mask_padded], dim=0),
    )


def caption_from_sample(sample: dict[str, Any]) -> str:
    if "caption" in sample:
        caption = sample["caption"]
    else:
        runtime_keys = {
            "duration",
            "fps",
            "height",
            "width",
            "num_frames",
            "resolution",
            "ratio",
        }
        caption = {key: value for key, value in sample.items() if key not in runtime_keys}
    if isinstance(caption, (dict, list)):
        return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
    return str(caption)


def compute_training_frame_budget(
    num_source_frames: int,
    source_fps: float,
    sample_fps: int = 24,
    vae_tc: int = 4,
) -> tuple[int, float, int]:
    if num_source_frames <= 0:
        return 1, 0.0, 1
    if source_fps > sample_fps:
        raw_val = int(num_source_frames / source_fps * sample_fps)
    else:
        raw_val = int(num_source_frames)
    sample_frame = ((raw_val - 1) // vae_tc) * vae_tc + 1
    sample_frame = max(sample_frame, 1)
    vae_fps = sample_frame / num_source_frames * float(source_fps)
    t_vae = (sample_frame - 1) // vae_tc + 1
    return int(sample_frame), float(vae_fps), int(t_vae)


def compute_training_aligned_indices(
    num_source_frames: int,
    sample_frame: int,
) -> np.ndarray:
    if sample_frame <= 0:
        return np.zeros(0, dtype=int)
    if num_source_frames <= 0:
        return np.zeros(sample_frame, dtype=int)
    if num_source_frames >= sample_frame:
        return np.linspace(0, num_source_frames - 1, sample_frame, dtype=int)
    head = np.arange(num_source_frames, dtype=int)
    pad = np.full(sample_frame - num_source_frames, num_source_frames - 1, dtype=int)
    return np.concatenate([head, pad])


def resize_video_tensor(video: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"video tensor must have shape [B,C,T,H,W], got {tuple(video.shape)}")
    bsz, channels, frames, _height, _width = video.shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channels, _height, _width)
    resized = F.interpolate(flat, size=(height, width), mode="bicubic", align_corners=False)
    resized = resized.clamp(0.0, 1.0)
    return resized.reshape(bsz, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()


def load_refiner_video_tensor(
    path: str | Path,
    height: int,
    width: int,
    *,
    sample_fps: int = 24,
    vae_tc: int = 4,
    max_frames: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    from decord import VideoReader, cpu

    vr = VideoReader(str(path), ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Video has no frames: {path}")
    src_fps = float(vr.get_avg_fps())
    sample_frame, vae_fps, t_vae = compute_training_frame_budget(
        total,
        src_fps,
        sample_fps=sample_fps,
        vae_tc=vae_tc,
    )
    sample_frame_uncapped = int(sample_frame)
    truncated = False
    if max_frames is not None and sample_frame > int(max_frames):
        sample_frame = int(max_frames)
        truncated = True
        vae_fps = float(sample_frame) / max(total, 1) * src_fps
        t_vae = (sample_frame - 1) // vae_tc + 1
    indices = compute_training_aligned_indices(total, sample_frame)
    frames = torch.from_numpy(vr.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).float()
    frames = frames / 255.0
    video = frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    video = resize_video_tensor(video, height=height, width=width)
    meta = {
        "src_fps": float(src_fps),
        "sample_frame": int(sample_frame),
        "sample_frame_uncapped": int(sample_frame_uncapped),
        "max_frames": None if max_frames is None else int(max_frames),
        "truncated_by_max_frames": bool(truncated),
        "vae_fps": float(vae_fps),
        "t_vae": int(t_vae),
        "num_source_frames": int(total),
        "align_to_training": True,
    }
    return video, meta


def load_first_frame_condition_tensor(
    path: str | Path,
    target_height: int,
    target_width: int,
    geometry_height: int,
    geometry_width: int,
) -> torch.Tensor:
    """Load the clean first frame center-cropped to the lowres video's aspect.

    The refiner latent grid is encoded from the lowres video, so the injected
    frame-0 condition must share that video's geometry, not the raw image's.
    """
    image = Image.open(path).convert("RGB")
    image_width, image_height = image.size
    geometry_aspect = float(geometry_width) / float(geometry_height)
    image_aspect = float(image_width) / float(image_height)
    if image_aspect > geometry_aspect:
        crop_height = image_height
        crop_width = max(1, int(round(crop_height * geometry_aspect)))
        left = int(round((image_width - crop_width) / 2.0))
        top = 0
    else:
        crop_width = image_width
        crop_height = max(1, int(round(crop_width / geometry_aspect)))
        left = 0
        top = int(round((image_height - crop_height) / 2.0))
    crop = image.crop((left, top, left + crop_width, top + crop_height))
    crop = crop.resize((target_width, target_height), resample=Image.BICUBIC)
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    frame = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return frame.permute(1, 0, 2, 3).unsqueeze(0)
