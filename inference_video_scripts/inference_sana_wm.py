"""Sana-WM camera-controlled image-to-video inference.

Given a starting image, a text prompt, and a camera trajectory (either a
``(F, 4, 4)`` ``.npy`` of camera-to-world poses or a WASD/IJKL action
string that we roll out for you), samples a latent video with the Sana
DiT and decodes it to pixels with either the LTX-2 sink-bidirectional
Euler refiner (default, high quality) or the Sana VAE (fast).

All weights default to the public Hugging Face release
``Efficient-Large-Model/SANA-WM_bidirectional`` and are downloaded on
first use.

The output frame size is fixed at ``704 x 1280``. Input images are
aspect-preserving resized + center-cropped to that resolution. Intrinsics
may be omitted — we estimate them with Pi3X from the input image, but you
should pass them when available because intrinsics estimation error will
propagate into the generated geometry.
"""

import argparse
import gc
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# xformers' memory_efficient_attention interacts badly with our cross-attention
# mask path on torch 2.9 + xformers 0.0.33; fall back to PyTorch SDPA, which is
# numerically equivalent here. Must be set before any sana imports.
os.environ.setdefault("DISABLE_XFORMERS", "1")

import imageio.v3 as iio
import numpy as np
import pyrallis
import torch
from PIL import Image
from torchvision import transforms as T

# Importing diffusion.model.nets registers all Sana / Sana-WM blocks.
import diffusion.model.nets  # noqa: F401
from diffusion import DPMS, FlowEuler, LTXFlowEuler
from diffusion.model.builder import (
    build_model,
    get_tokenizer_and_text_encoder,
    get_vae,
    vae_decode,
    vae_encode,
)
from diffusion.model.utils import get_weight_dtype
from diffusion.refiner.diffusers_ltx2_refiner import DiffusersLTX2Refiner, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusion.utils.action_overlay import apply_overlay
from diffusion.utils.cam_utils import compute_raymap, get_pose_inverse
from diffusion.utils.camctrl_config import ModelVideoCamCtrlConfig, model_video_camctrl_init_config
from diffusion.utils.chunk_utils import get_chunk_index_from_config
from diffusion.utils.config import AEConfig, SchedulerConfig, TextEncoderConfig
from diffusion.utils.logger import get_root_logger
from sana.tools import resolve_hf_path
from tools.download import find_model

SamplingAlgo = Literal["flow_euler_ltx", "flow_euler", "flow_dpm-solver"]

# Sana-WM is trained at this single resolution.
TARGET_HEIGHT = 704
TARGET_WIDTH = 1280

# Pi3X intrinsics sanity check. Outside this range we refuse to proceed.
MIN_FOV_DEG = 25.0
MAX_FOV_DEG = 120.0

# Public release on Hugging Face. Override on the CLI for local files.
HF_REPO = "Efficient-Large-Model/SANA-WM_bidirectional"
HF_DEFAULTS = {
    "model_path": f"hf://{HF_REPO}/dit/sana_wm_1600m_720p.safetensors",
    "config": f"hf://{HF_REPO}/config.yaml",
    "refiner_checkpoint": f"hf://{HF_REPO}/refiner/refiner.safetensors",
    "refiner_diffusers_root": None,
    "refiner_gemma_root": f"hf://{HF_REPO}/refiner/text_encoder",
}

# Action-string rollout defaults. Rotation is intentionally slower than
# translation so casual WASD+IJKL strings produce natural trajectories.
DEFAULT_TRANSLATION_SPEED = 0.05
DEFAULT_ROTATION_SPEED_DEG = 1.2
DEFAULT_PITCH_LIMIT_DEG = 85.0
ALLOWED_ACTION_KEYS: frozenset[str] = frozenset("wasdijkl")

# ============================================================================
# Config
# ============================================================================


@dataclass
class InferenceConfig:
    """Slim YAML config: model + VAE + text encoder + scheduler only."""

    model: ModelVideoCamCtrlConfig
    vae: AEConfig
    text_encoder: TextEncoderConfig
    scheduler: SchedulerConfig
    # The base Sana class checks ``config.work_dir`` to decide where to tee
    # initialization logs; an empty string means "log to stdout".
    work_dir: str = ""


@dataclass
class GenerationParams:
    """Per-call generation knobs."""

    num_frames: int = 161
    fps: int = 16
    step: int = 60
    cfg_scale: float = 5.0
    flow_shift: float | None = None
    seed: int = 42
    negative_prompt: str = ""
    sampling_algo: SamplingAlgo = "flow_euler_ltx"


@dataclass
class RefinerSettings:
    """LTX-2 sink-bidirectional Euler refiner configuration."""

    checkpoint: Path | str
    gemma_root: Path | str
    diffusers_root: Path | str | None = None
    sink_size: int = 1
    seed: int = 42


# ============================================================================
# Action-string → camera-to-world trajectory
# ============================================================================


def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _parse_action_string(action: str) -> list[list[str]]:
    """``"w-10,iw-5,none-3"`` → list of per-frame held-key lists."""
    cleaned = "".join(action.replace("，", ",").split())
    if not cleaned:
        raise ValueError("action string is empty")
    per_frame: list[list[str]] = []
    for segment in cleaned.split(","):
        if not segment or "-" not in segment:
            raise ValueError(f"Invalid action segment {segment!r}: expected '<keys>-<duration>'.")
        keys_part, dur_str = segment.rsplit("-", 1)
        if not dur_str.isdigit() or int(dur_str) <= 0:
            raise ValueError(f"Action segment {segment!r} has a non-positive duration {dur_str!r}.")
        n = int(dur_str)
        keys_lower = keys_part.lower()
        if keys_lower == "none":
            keys: list[str] = []
        else:
            bad = sorted({c for c in keys_lower if c not in ALLOWED_ACTION_KEYS})
            if bad:
                raise ValueError(
                    f"Action segment {segment!r} contains unknown keys {bad}; "
                    f"allowed: {''.join(sorted(ALLOWED_ACTION_KEYS))}."
                )
            keys = sorted(set(keys_lower))
        per_frame.extend([list(keys) for _ in range(n)])
    return per_frame


def action_string_to_c2w(
    action: str,
    *,
    translation_speed: float = DEFAULT_TRANSLATION_SPEED,
    rotation_speed_deg: float = DEFAULT_ROTATION_SPEED_DEG,
    pitch_limit_deg: float = DEFAULT_PITCH_LIMIT_DEG,
) -> np.ndarray:
    """Roll out a ``(N+1, 4, 4)`` camera-to-world trajectory from an action string.

    The DSL groups segments as ``<keys>-<frames>`` joined by commas. ``"none"``
    means no keys held. Movement keys (``wasd``) translate on the world XZ
    plane; rotation keys (``ijkl``) apply pitch / yaw. Coordinate convention:
    OpenCV (``+X right, +Y down, +Z forward``).
    """
    per_frame = _parse_action_string(action)
    rotate_rad = math.radians(rotation_speed_deg)
    pitch_limit_rad = math.radians(pitch_limit_deg)
    current = np.eye(4, dtype=np.float64)
    poses = [current.copy()]
    current_pitch = 0.0

    for keys in per_frame:
        held = set(keys)
        R = current[:3, :3]
        T_ = current[:3, 3]

        # Pitch (camera-local X).
        pitch_delta = (rotate_rad if "i" in held else 0.0) - (rotate_rad if "k" in held else 0.0)
        new_pitch = current_pitch + pitch_delta
        if not (-pitch_limit_rad <= new_pitch <= pitch_limit_rad):
            pitch_delta = 0.0
        else:
            current_pitch = new_pitch

        # Yaw (world Y).
        yaw_delta = (rotate_rad if "l" in held else 0.0) - (rotate_rad if "j" in held else 0.0)
        R_new = _rot_y(yaw_delta) @ R @ _rot_x(pitch_delta)

        # Horizontal-plane WASD translation.
        forward = R_new[:, 2].copy()
        forward[1] = 0.0
        right = R_new[:, 0].copy()
        right[1] = 0.0
        fn = float(np.linalg.norm(forward))
        rn = float(np.linalg.norm(right))
        if fn > 0:
            forward /= fn + 1e-6
        if rn > 0:
            right /= rn + 1e-6
        move = np.zeros(3, dtype=np.float64)
        if "w" in held:
            move += forward * translation_speed
        if "s" in held:
            move -= forward * translation_speed
        if "d" in held:
            move += right * translation_speed
        if "a" in held:
            move -= right * translation_speed

        current = np.eye(4, dtype=np.float64)
        current[:3, :3] = R_new
        current[:3, 3] = T_ + move
        poses.append(current.copy())

    return np.stack(poses, axis=0).astype(np.float32)


# ============================================================================
# Intrinsics: load from .npy or estimate with Pi3X
# ============================================================================


def load_intrinsics(path: Path, num_frames: int) -> np.ndarray:
    """Return ``(num_frames, 4)`` intrinsics as ``[fx, fy, cx, cy]``.

    Accepts ``.npy`` arrays shaped ``(3, 3)``, ``(F, 3, 3)``, or ``(4,)``.
    """
    arr = np.load(path).astype(np.float32)
    if arr.shape == (4,):
        return np.broadcast_to(arr, (num_frames, 4)).copy()
    if arr.shape == (3, 3):
        v = np.array([arr[0, 0], arr[1, 1], arr[0, 2], arr[1, 2]], dtype=np.float32)
        return np.broadcast_to(v, (num_frames, 4)).copy()
    if arr.ndim == 3 and arr.shape[1:] == (3, 3) and arr.shape[0] >= num_frames:
        K = arr[:num_frames]
        return np.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], axis=1)
    raise ValueError(
        f"Unsupported intrinsics shape {arr.shape} for num_frames={num_frames}; " f"expected (3,3), (F,3,3), or (4,)."
    )


def estimate_intrinsics_with_pi3x(image: Image.Image, device: torch.device, logger: logging.Logger) -> np.ndarray:
    """Estimate ``(fx, fy, cx, cy)`` for ``image`` using Pi3X.

    The image is internally resized to a Pi3X-friendly shape; the returned
    intrinsics are scaled back to ``image.size``. We assert
    ``MIN_FOV_DEG < horizontal_fov < MAX_FOV_DEG`` and abort otherwise so
    the user knows to provide intrinsics manually.
    """
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import recover_intrinsic_from_rays_d

    logger.warning(
        "Intrinsics not provided — estimating with Pi3X from the input image. "
        "Estimation errors propagate into the generated camera geometry; please "
        "supply --intrinsics when accurate values are available."
    )

    W_orig, H_orig = image.size
    pixel_limit = 255_000
    scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1.0
    W_t, H_t = W_orig * scale, H_orig * scale
    k, m = max(1, round(W_t / 14)), max(1, round(H_t / 14))
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_t / H_t:
            k -= 1
        else:
            m -= 1
    W_model, H_model = max(1, k) * 14, max(1, m) * 14
    resized = image.resize((W_model, H_model), Image.Resampling.LANCZOS)
    tensor = T.ToTensor()(resized).unsqueeze(0).unsqueeze(0).to(device)

    dtype = (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
    model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
    model.disable_multimodal()
    model.requires_grad_(False)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        out = model(imgs=tensor)
    rays_d = torch.nn.functional.normalize(out["local_points"], dim=-1)
    K_model = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)
    K_model_np = K_model[0, 0].detach().cpu().float().numpy()

    sx, sy = W_orig / W_model, H_orig / H_model
    fx, fy = float(K_model_np[0, 0] * sx), float(K_model_np[1, 1] * sy)
    cx, cy = float(K_model_np[0, 2] * sx), float(K_model_np[1, 2] * sy)

    fov_x = math.degrees(2.0 * math.atan(W_orig / (2.0 * fx)))
    fov_y = math.degrees(2.0 * math.atan(H_orig / (2.0 * fy)))
    logger.info(
        f"Pi3X intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} " f"(FOV: H={fov_x:.1f}° V={fov_y:.1f}°)"
    )
    if not (MIN_FOV_DEG < fov_x < MAX_FOV_DEG and MIN_FOV_DEG < fov_y < MAX_FOV_DEG):
        raise SystemExit(
            f"Pi3X-estimated FOV (H={fov_x:.1f}°, V={fov_y:.1f}°) falls outside "
            f"[{MIN_FOV_DEG}°, {MAX_FOV_DEG}°]. Intrinsics estimation likely failed; "
            f"pass --intrinsics with a trusted .npy."
        )

    # Free Pi3X memory before the heavy models load.
    del model, out, K_model, rays_d, tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return np.array([fx, fy, cx, cy], dtype=np.float32)


def transform_intrinsics_for_crop(
    intrinsics_vec4: np.ndarray,
    src_size: tuple[int, int],
    resized_size: tuple[int, int],
    crop_offset: tuple[int, int],
) -> np.ndarray:
    """Adjust ``[fx, fy, cx, cy]`` to match a resize-then-center-crop image."""
    src_w, src_h = src_size
    rw, rh = resized_size
    cl, ct = crop_offset
    sx, sy = rw / src_w, rh / src_h
    out = intrinsics_vec4.copy()
    out[..., 0] *= sx
    out[..., 2] = out[..., 2] * sx - cl
    out[..., 1] *= sy
    out[..., 3] = out[..., 3] * sy - ct
    return out


# ============================================================================
# Image preprocessing → 704 x 1280
# ============================================================================


def resize_and_center_crop(
    image: Image.Image, target_h: int = TARGET_HEIGHT, target_w: int = TARGET_WIDTH
) -> tuple[Image.Image, tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Aspect-preserving resize then center-crop to ``(target_h, target_w)``.

    Returns ``(cropped_image, src_size, resized_size, crop_offset)`` where
    ``crop_offset = (left, top)``. The source size is what we'd use to map
    user-supplied intrinsics into the cropped image's pixel grid.
    """
    src_w, src_h = image.size
    scale = max(target_h / src_h, target_w / src_w)
    rw = max(target_w, int(round(src_w * scale)))
    rh = max(target_h, int(round(src_h * scale)))
    resized = image.resize((rw, rh), Image.LANCZOS)
    left = (rw - target_w) // 2
    top = (rh - target_h) // 2
    cropped = resized.crop((left, top, left + target_w, top + target_h))
    return cropped, (src_w, src_h), (rw, rh), (left, top)


# ============================================================================
# Camera conditioning tensors
# ============================================================================


def _pack_camera_conditions(
    poses: torch.Tensor,
    intrinsics_latent: torch.Tensor,
    num_frames: int,
    latent_frames: int,
    latent_h: int,
    latent_w: int,
    vae_time_stride: int,
) -> dict[str, torch.Tensor]:
    """Build raymap + chunk_plucker tensors the model consumes."""
    time_indices = torch.arange(0, num_frames, vae_time_stride)
    if len(time_indices) > latent_frames:
        time_indices = time_indices[:latent_frames]

    raymap = torch.cat(
        [poses[time_indices].reshape(len(time_indices), -1), intrinsics_latent[time_indices]],
        dim=-1,
    )

    chunk_starts = time_indices - (vae_time_stride - 1)
    chunks = []
    for start in chunk_starts:
        s = max(0, int(start))
        e = s + vae_time_stride
        chunk_poses, chunk_intrs = poses[s:e], intrinsics_latent[s:e]
        if chunk_poses.shape[0] < vae_time_stride:
            pad = vae_time_stride - chunk_poses.shape[0]
            chunk_poses = torch.cat([chunk_poses, chunk_poses[-1:].repeat(pad, 1, 1)], dim=0)
            chunk_intrs = torch.cat([chunk_intrs, chunk_intrs[-1:].repeat(pad, 1)], dim=0)
        plucker = compute_raymap(chunk_intrs, chunk_poses, latent_h, latent_w, use_plucker=True)
        chunks.append(plucker.permute(0, 3, 1, 2).reshape(-1, latent_h, latent_w))
    chunk_plucker = torch.stack(chunks).permute(1, 0, 2, 3)
    return {"raymap": raymap, "chunk_plucker": chunk_plucker}


def prepare_camera(
    poses_c2w: np.ndarray,
    intrinsics_vec4: np.ndarray,
    *,
    target_size: tuple[int, int],
    vae_stride: tuple[int, int, int] | list[int],
) -> dict[str, torch.Tensor]:
    """Relativise poses to frame 0 and build the model-input tensors."""
    num_frames = poses_c2w.shape[0]
    vae_time_stride, vae_spatial_stride = vae_stride[0], vae_stride[-1]
    H_pixel, W_pixel = target_size
    latent_h = H_pixel // vae_spatial_stride
    latent_w = W_pixel // vae_spatial_stride
    latent_frames = (num_frames - 1) // vae_time_stride + 1

    poses = torch.from_numpy(poses_c2w).float()
    first_inv = get_pose_inverse(poses[0:1]).squeeze(0)
    poses_rel = torch.matmul(first_inv, poses[1:])
    poses = torch.cat([torch.eye(4).unsqueeze(0), poses_rel], dim=0)

    intrinsics = torch.from_numpy(intrinsics_vec4).float()
    intrinsics_latent = intrinsics.clone()
    intrinsics_latent[:, [0, 2]] *= latent_w / float(W_pixel)
    intrinsics_latent[:, [1, 3]] *= latent_h / float(H_pixel)

    return _pack_camera_conditions(
        poses,
        intrinsics_latent,
        num_frames,
        latent_frames,
        latent_h,
        latent_w,
        vae_time_stride,
    )


# ============================================================================
# Output
# ============================================================================


def write_video(output_dir: Path, name: str, video_hwc: np.ndarray, fps: int, logger: logging.Logger) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{name}_generated.mp4"
    iio.imwrite(video_path, video_hwc, fps=fps)
    logger.info(f"Saved {video_path}")
    return video_path


# ============================================================================
# Pipeline
# ============================================================================


class SanaWMPipeline:
    """End-to-end Sana-WM inference pipeline.

    Builds the Sana DiT, VAE, text encoder, and (optionally) the LTX-2
    refiner once and exposes :meth:`generate` for repeated sampling.

    By default every component is loaded eagerly and stays resident on
    ``device``. Pass ``offload_vae=True`` or ``offload_refiner=True`` to
    instead instantiate lazily and return to CPU after each call.
    """

    def __init__(
        self,
        config: InferenceConfig,
        model_path: str | Path,
        *,
        device: torch.device | str = "cuda",
        refiner: RefinerSettings | None = None,
        offload_vae: bool = False,
        offload_refiner: bool = False,
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.device = torch.device(device)
        self.refiner_settings = refiner
        self.offload_vae = offload_vae
        self.offload_refiner = offload_refiner
        self.logger = logger or get_root_logger()
        self.weight_dtype = get_weight_dtype(config.model.mixed_precision)
        self.vae_dtype = get_weight_dtype(config.vae.weight_dtype)
        self._refiner_built = False

        self._build_vae()
        self._build_text_encoder()
        self._build_model(model_path)
        if refiner is not None and not offload_refiner:
            self._build_refiner()

    # ------- construction -------

    def _build_vae(self) -> None:
        self.config.vae.vae_pretrained = resolve_hf_path(self.config.vae.vae_pretrained)
        vae = get_vae(
            self.config.vae.vae_type,
            self.config.vae.vae_pretrained,
            device=self.device,
            dtype=self.vae_dtype,
            config=self.config.vae,
        )
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        if hasattr(vae, "use_framewise_encoding"):
            vae.use_framewise_encoding = True
            vae.use_framewise_decoding = True
            vae.tile_sample_stride_num_frames = getattr(self.config.vae, "tile_sample_stride_num_frames", 64)
            vae.tile_sample_min_num_frames = getattr(self.config.vae, "tile_sample_min_num_frames", 96)
        self.vae = vae

    def _build_text_encoder(self) -> None:
        self.tokenizer, self.text_encoder = get_tokenizer_and_text_encoder(
            name=self.config.text_encoder.text_encoder_name, device=self.device
        )

    def _build_model(self, model_path: str | Path) -> None:
        latent_size = self.config.model.image_size // self.config.vae.vae_stride[-1]
        kwargs = model_video_camctrl_init_config(self.config, latent_size=latent_size)
        model = build_model(
            self.config.model.model,
            use_fp32_attention=self.config.model.get("fp32_attention", False),
            **kwargs,
        ).to(self.device)
        self.logger.info(f"Loaded {self.config.model.model} ({sum(p.numel() for p in model.parameters()):,} params)")

        state = find_model(str(model_path))
        if "generator" in state:
            state = state["generator"]
        if "state_dict" not in state:
            stripped = {(k[len("model.") :] if k.startswith("model.") else k): v for k, v in state.items()}
            state = {"state_dict": stripped}
        state["state_dict"].pop("pos_embed", None)
        missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
        if missing:
            self.logger.warning(f"Missing keys: {missing}")
        if unexpected:
            self.logger.warning(f"Unexpected keys: {unexpected}")
        self.model = model.eval().to(self.weight_dtype)

    def _build_refiner(self) -> None:
        if self.refiner_settings is None:
            self._refiner_built = False
            return
        if "LTX2VAE_diffusers" not in self.config.vae.vae_type:
            raise ValueError(f"The refiner requires LTX2VAE_diffusers, got {self.config.vae.vae_type!r}.")
        refiner_root = self._resolve_refiner_root(self.refiner_settings)
        gemma = resolve_hf_path(str(self.refiner_settings.gemma_root))
        self.refiner = DiffusersLTX2Refiner(
            refiner_root=refiner_root,
            gemma_root=gemma,
            dtype=self.weight_dtype,
            device=self.device,
        )
        self._refiner_built = True

    def _resolve_refiner_root(self, refiner: RefinerSettings) -> str:
        candidates: list[Path] = []
        repo_root = Path(__file__).resolve().parents[1]

        if refiner.diffusers_root is not None:
            root = Path(resolve_hf_path(str(refiner.diffusers_root)))
            candidates.append(root)

        checkpoint = str(refiner.checkpoint)
        checkpoint_path = Path(checkpoint).expanduser()
        if checkpoint_path.exists():
            if checkpoint_path.is_dir():
                candidates.append(checkpoint_path)
            else:
                candidates.extend(
                    [
                        checkpoint_path.parent / "refiner_diffusers",
                        checkpoint_path.parent.parent / "refiner_diffusers",
                    ]
                )

        candidates.append(repo_root / "output" / "pretrained_models" / "SANA-WM_bidirectional" / "refiner_diffusers")

        for candidate in candidates:
            if (candidate / "transformer" / "config.json").is_file() and (
                candidate / "connectors" / "config.json"
            ).is_file():
                return str(candidate)

        raise FileNotFoundError(
            "Could not find a diffusers-format LTX-2 refiner. Pass --refiner_diffusers_root "
            "or convert the single-file checkpoint with tools/convert_sana_wm_refiner_to_diffusers.py."
        )

    def _release_refiner(self) -> None:
        if not self._refiner_built:
            return
        for attr in ("refiner",):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            try:
                obj.to("meta")
            except Exception:
                obj.to("cpu")
            setattr(self, attr, None)
        self._refiner_built = False
        torch.cuda.empty_cache()
        gc.collect()

    def _offload_stage1(self) -> None:
        for attr in ("model", "text_encoder", "vae"):
            module = getattr(self, attr, None)
            if module is None:
                continue
            try:
                module.to("meta")
            except Exception:
                module.to("cpu")
            setattr(self, attr, None)
        torch.cuda.empty_cache()
        gc.collect()

    # ------- generation -------

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        c2w: np.ndarray,
        intrinsics_vec4: np.ndarray,
        params: GenerationParams = GenerationParams(),
    ) -> dict[str, object]:
        """Generate a video.

        Args:
            image: First-frame RGB image, already cropped to ``(704, 1280)``.
            prompt: Text prompt.
            c2w: ``(F, 4, 4)`` camera-to-world matrices for ``params.num_frames`` frames.
            intrinsics_vec4: ``(F, 4)`` ``[fx, fy, cx, cy]`` matching ``image``.
            params: Per-call generation knobs.

        Returns:
            Dict with ``video`` ``(T, H, W, 3)`` uint8, ``c2w``, and ``latent``.
        """
        vae_stride = self.config.vae.vae_stride
        latent_T = (params.num_frames - 1) // vae_stride[0] + 1
        latent_h, latent_w = TARGET_HEIGHT // vae_stride[-1], TARGET_WIDTH // vae_stride[-1]

        camera = prepare_camera(
            c2w[: params.num_frames],
            intrinsics_vec4[: params.num_frames],
            target_size=(TARGET_HEIGHT, TARGET_WIDTH),
            vae_stride=vae_stride,
        )

        sana_latent = self._sample_stage1(image, prompt, camera, params, latent_T, latent_h, latent_w)

        if self.refiner_settings is not None:
            video = self._refine(sana_latent, prompt, params, self.refiner_settings)
            # _refine drops the sink anchor frame; realign the trajectory.
            video_c2w = c2w[1 : params.num_frames]
        else:
            video = self._decode_with_sana_vae(sana_latent)
            video_c2w = c2w[: params.num_frames]

        return {"video": video, "c2w": video_c2w, "latent": sana_latent.cpu()}

    # ------- stage 1: Sana DiT -------

    def _encode_prompts(
        self, prompt: str, negative_prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_length = self.config.text_encoder.model_max_length
        chi_prompt = "\n".join(self.config.text_encoder.chi_prompt or [])
        if chi_prompt:
            prompt = chi_prompt + prompt
            max_length_all = len(self.tokenizer.encode(chi_prompt)) + max_length - 2
        else:
            max_length_all = max_length

        def encode(text: str, length: int) -> tuple[torch.Tensor, torch.Tensor]:
            tok = self.tokenizer(
                [text], max_length=length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(self.device)
            return self.text_encoder(tok.input_ids, tok.attention_mask)[0], tok.attention_mask

        cond, cond_mask = encode(prompt, max_length_all)
        select = [0] + list(range(-max_length + 1, 0))
        cond = cond[:, None][:, :, select]
        cond_mask = cond_mask[:, select]
        neg, neg_mask = encode(negative_prompt, max_length)
        return cond, cond_mask, neg[:, None], neg_mask

    def _sample_stage1(
        self,
        image: Image.Image,
        prompt: str,
        camera: dict[str, torch.Tensor],
        params: GenerationParams,
        latent_T: int,
        latent_h: int,
        latent_w: int,
    ) -> torch.Tensor:
        if self.offload_vae:
            self.vae.to(self.device)
        img = (T.ToTensor()(image) * 2.0 - 1.0).unsqueeze(0).unsqueeze(2)
        first_latent = vae_encode(
            self.config.vae.vae_type,
            self.vae,
            img.to(self.device, dtype=self.vae_dtype),
            device=self.device,
        ).to(self.weight_dtype)
        if self.offload_vae:
            self.vae.to("cpu")
            torch.cuda.empty_cache()

        cond, cond_mask, neg, neg_mask = self._encode_prompts(prompt, params.negative_prompt)
        raymap = camera["raymap"].unsqueeze(0).to(self.device, dtype=self.weight_dtype)
        chunk_plucker = camera["chunk_plucker"].unsqueeze(0).to(self.device, dtype=self.weight_dtype)
        if params.cfg_scale > 1.0:
            mask_cfg = torch.cat([neg_mask, cond_mask], dim=0)
            raymap_cfg = torch.cat([raymap, raymap], dim=0)
            chunk_plucker_cfg = torch.cat([chunk_plucker, chunk_plucker], dim=0)
        else:
            mask_cfg, raymap_cfg, chunk_plucker_cfg = cond_mask, raymap, chunk_plucker

        latent_channels = first_latent.shape[1]
        generator = torch.Generator(device=self.device).manual_seed(params.seed)
        z = torch.randn(
            1,
            latent_channels,
            latent_T,
            latent_h,
            latent_w,
            dtype=self.weight_dtype,
            device=self.device,
            generator=generator,
        )
        z[:, :, :1] = first_latent

        chunk_index = get_chunk_index_from_config(self.config, num_frames=latent_T)
        model_kwargs: dict[str, object] = dict(
            data_info={
                "img_hw": torch.tensor([[TARGET_HEIGHT, TARGET_WIDTH]], dtype=torch.float, device=self.device),
                "condition_frame_info": {0: 0.0},
            },
            mask=mask_cfg,
            camera_conditions=raymap_cfg,
            chunk_plucker=chunk_plucker_cfg,
        )
        if chunk_index is not None:
            model_kwargs["chunk_index"] = chunk_index

        flow_shift = self._resolve_flow_shift(params.flow_shift)
        samples = self._dispatch_solver(
            params.sampling_algo,
            z,
            cond,
            neg,
            params.cfg_scale,
            flow_shift,
            params.step,
            model_kwargs,
            chunk_index,
            generator,
        )
        torch.cuda.empty_cache()
        return samples.detach()

    def _resolve_flow_shift(self, override: float | None) -> float:
        if override is not None:
            return override
        return (
            self.config.scheduler.inference_flow_shift
            if self.config.scheduler.inference_flow_shift is not None
            else self.config.scheduler.flow_shift
        )

    def _dispatch_solver(
        self,
        algo: SamplingAlgo,
        z: torch.Tensor,
        cond: torch.Tensor,
        neg: torch.Tensor,
        cfg_scale: float,
        flow_shift: float,
        steps: int,
        model_kwargs: dict,
        chunk_index: object,
        generator: torch.Generator,
    ) -> torch.Tensor:
        base = dict(
            condition=cond, uncondition=neg, cfg_scale=cfg_scale, flow_shift=flow_shift, model_kwargs=model_kwargs
        )
        if algo == "flow_euler_ltx":
            return LTXFlowEuler(self.model, **base).sample(z, steps=steps, generator=generator)
        if algo == "flow_euler":
            return FlowEuler(self.model, **base).sample(z, steps=steps)
        if algo == "flow_dpm-solver":
            return DPMS(
                self.model,
                condition=cond,
                uncondition=neg,
                cfg_scale=cfg_scale,
                model_type="flow",
                guidance_type="classifier-free",
                model_kwargs=model_kwargs,
                schedule="FLOW",
            ).sample(z, steps=steps, order=2, skip_type="time_uniform_flow", method="multistep", flow_shift=flow_shift)
        raise ValueError(f"Unknown sampling_algo: {algo}")

    # ------- stage 2: decode -------

    def _decode_with_sana_vae(self, sana_latent: torch.Tensor) -> np.ndarray:
        self.logger.info(f"[sana-vae] decoding {sana_latent.shape[2]} latent frames")
        if getattr(self, "vae", None) is None:
            self._build_vae()
        if self.offload_vae:
            self.vae.to(self.device)
        samples = sana_latent.to(device=self.device, dtype=self.vae_dtype)
        decoded = vae_decode(self.config.vae.vae_type, self.vae, samples)
        if isinstance(decoded, list):
            decoded = torch.stack(decoded, dim=0)
        video = (
            torch.clamp(127.5 * decoded + 127.5, 0, 255).permute(0, 2, 3, 4, 1).to("cpu", dtype=torch.uint8).numpy()[0]
        )
        if self.offload_vae:
            self.vae.to("cpu")
        del samples, decoded
        torch.cuda.empty_cache()
        return video

    def _refine(
        self,
        sana_latent: torch.Tensor,
        prompt: str,
        params: GenerationParams,
        refiner: RefinerSettings,
    ) -> np.ndarray:
        if self.offload_refiner:
            self._offload_stage1()
            self._build_refiner()

        sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device)
        start_sigma = float(sigmas[0])
        self.logger.info(f"[refiner] {len(sigmas) - 1}-step Euler, start_sigma={start_sigma:.4f}")

        refined = self.refiner.refine_latents(
            sana_latent,
            prompt,
            fps=float(params.fps),
            sink_size=int(refiner.sink_size),
            seed=int(refiner.seed),
            progress=True,
        )
        if self.offload_refiner:
            self._release_refiner()

        self.logger.info(f"[refiner] decoding {refined.shape[2]} latent frames with diffusers LTX2 VAE")
        video = self._decode_with_sana_vae(refined)
        # The refiner's first decoded frame is the clean sink anchor; drop it so
        # the output starts from the first refined frame.
        video = video[1:]
        del refined
        torch.cuda.empty_cache()
        gc.collect()
        return video


# ============================================================================
# CLI
# ============================================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sana_wm",
        description="Sana-WM camera-controlled image-to-video inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", type=Path, required=True, help="First-frame RGB image.")
    p.add_argument("--prompt", type=Path, required=True, help="UTF-8 text file with the prompt.")
    p.add_argument("--output_dir", type=Path, required=True, help="Directory to write the mp4.")
    p.add_argument("--name", default="output", help="Filename stem for outputs.")

    # Camera trajectory (one of --camera or --action).
    cam_group = p.add_mutually_exclusive_group(required=True)
    cam_group.add_argument("--camera", type=Path, help="(F,4,4) .npy camera-to-world poses.")
    cam_group.add_argument(
        "--action", type=str, help="Action DSL string, e.g. 'w-80,jw-40,w-40'. Rolled out internally."
    )
    p.add_argument(
        "--translation_speed",
        type=float,
        default=DEFAULT_TRANSLATION_SPEED,
        help="Per-frame translation magnitude when a WASD key is held.",
    )
    p.add_argument(
        "--rotation_speed_deg",
        type=float,
        default=DEFAULT_ROTATION_SPEED_DEG,
        help="Per-frame rotation magnitude in degrees when an IJKL key is held.",
    )

    # Intrinsics: optional — Pi3X-estimated from the image if omitted.
    p.add_argument(
        "--intrinsics",
        type=Path,
        default=None,
        help=".npy intrinsics, shape (3,3), (F,3,3), or (4,) = (fx,fy,cx,cy). "
        "If omitted, we estimate intrinsics from --image with Pi3X.",
    )

    # Generation knobs.
    p.add_argument(
        "--num_frames",
        type=int,
        default=161,
        help="Total frames (10 s @ 16 fps default). With --action, "
        "this is the upper bound on the rolled-out trajectory.",
    )
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--step", type=int, default=60, help="DiT sampling steps.")
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--flow_shift", type=float, default=None, help="Override the scheduler's inference flow_shift.")
    p.add_argument(
        "--sampling_algo", default="flow_euler_ltx", choices=["flow_euler_ltx", "flow_euler", "flow_dpm-solver"]
    )
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no_action_overlay",
        action="store_true",
        help="Skip rendering the WASD + joystick overlay on the output video.",
    )

    # Weights and config.
    p.add_argument(
        "--config", default=HF_DEFAULTS["config"], help="Slim inference YAML config (local path or hf:// URI)."
    )
    p.add_argument(
        "--model_path", default=HF_DEFAULTS["model_path"], help="Stage-1 Sana DiT checkpoint (local path or hf:// URI)."
    )

    # Refiner: ON by default; pass --no_refiner to use Sana VAE decode.
    p.add_argument(
        "--no_refiner", action="store_true", help="Skip the LTX-2 refiner; decode stage-1 latents with the Sana VAE."
    )
    p.add_argument(
        "--refiner_checkpoint",
        default=HF_DEFAULTS["refiner_checkpoint"],
        help="Legacy LTX-2 refiner safetensors path. Used only to locate a sibling diffusers conversion.",
    )
    p.add_argument(
        "--refiner_diffusers_root",
        default=HF_DEFAULTS["refiner_diffusers_root"],
        help="Diffusers-format LTX-2 refiner root containing transformer/ and connectors/.",
    )
    p.add_argument(
        "--refiner_gemma_root",
        default=HF_DEFAULTS["refiner_gemma_root"],
        help="Gemma diffusers root for the refiner text encoder.",
    )
    p.add_argument("--refiner_seed", type=int, default=42)
    p.add_argument("--sink_size", type=int, default=1)

    # Memory.
    p.add_argument("--offload_vae", action="store_true", help="Move the VAE to CPU between encode/decode steps.")
    p.add_argument(
        "--offload_refiner",
        action="store_true",
        help="Lazy-load the LTX-2 refiner only when needed; release afterwards.",
    )
    return p


def _resolve_trajectory(args: argparse.Namespace) -> np.ndarray:
    """Materialise the camera-to-world trajectory from --camera or --action."""
    if args.action is not None:
        return action_string_to_c2w(
            args.action,
            translation_speed=args.translation_speed,
            rotation_speed_deg=args.rotation_speed_deg,
        )
    c2w_raw = np.load(args.camera).astype(np.float32)
    if c2w_raw.ndim != 3 or c2w_raw.shape[1:] != (4, 4):
        raise SystemExit(f"--camera must be a (F, 4, 4) .npy; got {c2w_raw.shape}.")
    return c2w_raw


def main() -> None:
    args = _build_parser().parse_args()

    logger = get_root_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(args.image).convert("RGB")
    prompt = args.prompt.read_text(encoding="utf-8", errors="replace").strip()
    if not prompt:
        raise SystemExit(f"Prompt file is empty: {args.prompt}")

    c2w_full = _resolve_trajectory(args)
    num_frames = min(args.num_frames, c2w_full.shape[0])
    c2w = c2w_full[:num_frames]

    cropped, src_size, resized_size, crop_offset = resize_and_center_crop(image)
    if args.intrinsics is not None:
        intr_src = load_intrinsics(args.intrinsics, num_frames)
    else:
        intr_one = estimate_intrinsics_with_pi3x(image, device, logger)
        intr_src = np.broadcast_to(intr_one, (num_frames, 4)).copy()
    intrinsics_vec4 = transform_intrinsics_for_crop(intr_src, src_size, resized_size, crop_offset)

    config: InferenceConfig = pyrallis.parse(
        config_class=InferenceConfig, config_path=resolve_hf_path(args.config), args=[]
    )

    refiner = (
        None
        if args.no_refiner
        else RefinerSettings(
            checkpoint=args.refiner_checkpoint,
            gemma_root=args.refiner_gemma_root,
            diffusers_root=args.refiner_diffusers_root,
            sink_size=args.sink_size,
            seed=args.refiner_seed,
        )
    )

    pipeline = SanaWMPipeline(
        config=config,
        model_path=resolve_hf_path(args.model_path),
        device=device,
        refiner=refiner,
        offload_vae=args.offload_vae,
        offload_refiner=args.offload_refiner,
        logger=logger,
    )

    params = GenerationParams(
        num_frames=num_frames,
        fps=args.fps,
        step=args.step,
        cfg_scale=args.cfg_scale,
        flow_shift=args.flow_shift,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        sampling_algo=args.sampling_algo,
    )

    out = pipeline.generate(cropped, prompt, c2w, intrinsics_vec4, params)
    video_hwc = out["video"]

    if not args.no_action_overlay:
        logger.info("Compositing action overlay onto the output video.")
        video_hwc = apply_overlay(video_hwc, out["c2w"])

    write_video(args.output_dir, args.name, video_hwc, params.fps, logger)


if __name__ == "__main__":
    main()
