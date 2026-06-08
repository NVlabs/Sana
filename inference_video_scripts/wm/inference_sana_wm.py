# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

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
import hashlib
import json
import logging
import math
import os
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

# xformers' memory_efficient_attention interacts badly with our cross-attention
# mask path on torch 2.9 + xformers 0.0.33; fall back to PyTorch SDPA, which is
# numerically equivalent here. Must be set before any sana imports.
os.environ.setdefault("DISABLE_XFORMERS", "1")

import imageio.v3 as iio
import numpy as np
import pyrallis
import torch
import torch.nn as nn
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
from diffusion.refiner.diffusers_ltx2_refiner import (
    STAGE_2_DISTILLED_SIGMA_VALUES,
    DiffusersLTX2Refiner,
)
from diffusion.refiner.diffusers_ltx2_refiner import _env_flag as _te_env_flag
from diffusion.refiner.diffusers_ltx2_refiner import _env_flag_default_true as _te_env_flag_default_true
from diffusion.refiner.diffusers_ltx2_refiner import _env_tuple as _te_env_tuple
from diffusion.refiner.diffusers_ltx2_refiner import (
    _replace_linear_with_te_nvfp4,
)
from diffusion.scheduler.self_forcing_flow_euler_sampler import SelfForcingFlowEulerCamCtrl
from diffusion.utils.action_overlay import apply_overlay
from diffusion.utils.cam_utils import compute_raymap, get_pose_inverse
from diffusion.utils.camctrl_config import ModelVideoCamCtrlConfig, model_video_camctrl_init_config
from diffusion.utils.chunk_utils import get_chunk_index_from_config
from diffusion.utils.config import AEConfig, SchedulerConfig, TextEncoderConfig
from diffusion.utils.logger import get_root_logger
from inference_video_scripts.wm.camera_control import (
    DEFAULT_PITCH_LIMIT_DEG,
    DEFAULT_ROTATION_SPEED_DEG,
    DEFAULT_TRANSLATION_SPEED,
    DSL_KEY_TO_CONTROL,
)
from inference_video_scripts.wm.camera_control import FPS as _CAM_FPS  # shared camera-control core (demo + inference)
from inference_video_scripts.wm.camera_control import (
    CameraPoseIntegrator,
    VelocityState,
    controls_to_target_velocity,
)
from sana.tools import resolve_hf_path
from tools.download import find_model

SamplingAlgo = Literal["flow_euler_ltx", "flow_euler", "flow_dpm-solver", "self_forcing"]

# Sana-WM is trained at this single resolution.
TARGET_HEIGHT = 704
TARGET_WIDTH = 1280

# Pi3X intrinsics sanity check. Outside this range we refuse to proceed.
MIN_FOV_DEG = 25.0
MAX_FOV_DEG = 120.0

# Public release on Hugging Face. Override on the CLI for local files.
# NOTE: The default HF checkpoint is bidirectional and is incompatible with
# ``--sampling_algo=self_forcing``. Self-forcing requires a chunk-causal trained
# checkpoint (HF release pending) — pass ``--model_path`` and a matching
# ``--config`` pointing to a chunk-causal model when using that algorithm.
HF_REPO = "Efficient-Large-Model/SANA-WM_bidirectional"
HF_DEFAULTS = {
    "model_path": f"hf://{HF_REPO}/dit/sana_wm_1600m_720p.safetensors",
    "config": f"hf://{HF_REPO}/config.yaml",
    "refiner_root": f"hf://{HF_REPO}/refiner",
    "refiner_gemma_root": f"hf://{HF_REPO}/refiner/text_encoder",
}

_ENV_TRUE = {"1", "true", "yes", "on"}
_ENV_FALSE = {"", "0", "false", "no", "off"}
_STAGE1_NVFP4_SKIP_DEFAULTS = (
    "^x_embedder",
    "^raymap_embedder",
    "^plucker_embedder",
    "^t_embedder",
    "^t_block",
    "^y_embedder",
    "^final_layer",
)
_STAGE1_NVFP4_INCLUDE_BY_MODE = {
    "self_proj": (r"^blocks\.\d+\.attn\.proj$",),
    "self_qkv": (r"^blocks\.\d+\.attn\.qkv$",),
    "self_attn": (
        r"^blocks\.\d+\.attn\.qkv$",
        r"^blocks\.\d+\.attn\.proj$",
        r"^blocks\.\d+\.attn\.beta_proj$",
        r"^blocks\.\d+\.attn\.gate_proj$",
        r"^blocks\.\d+\.attn\.output_gate$",
    ),
    "cam": (
        r"^blocks\.\d+\.attn\.q_proj_cam$",
        r"^blocks\.\d+\.attn\.k_proj_cam$",
        r"^blocks\.\d+\.attn\.v_proj_cam$",
        r"^blocks\.\d+\.attn\.out_proj_cam$",
    ),
    "cross": (r"^blocks\.\d+\.cross_attn\.",),
    "ffn": (
        r"^blocks\.\d+\.mlp\.inverted_conv\.linear$",
        r"^blocks\.\d+\.mlp\.point_conv\.linear$",
    ),
}


def _prepared_module_cache_root() -> Path | None:
    if os.environ.get("SANA_WM_PREPARED_MODULE_CACHE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    root = os.environ.get("SANA_WM_PREPARED_MODULE_CACHE_DIR", "").strip()
    return Path(root).expanduser() if root else Path.home() / ".cache" / "sana_wm_prepared_modules"


def _prepared_module_cache_hash(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:20]


def _path_fingerprint(path: str | Path) -> dict[str, object]:
    raw = str(path)
    try:
        resolved = Path(raw).expanduser().resolve()
    except Exception:
        return {"path": raw}
    try:
        stat = resolved.stat()
    except OSError:
        return {"path": str(resolved)}
    return {"path": str(resolved), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _is_local_callable_for_pickle(value: object) -> bool:
    if isinstance(value, types.MethodType):
        value = value.__func__
    if not isinstance(value, types.FunctionType):
        return False
    qualname = getattr(value, "__qualname__", "")
    return "<locals>" in qualname or getattr(value, "__name__", "") == "<lambda>"


def _strip_local_callables_for_pickle(root: object) -> list[tuple[object, object, object, str]]:
    restore: list[tuple[object, object, object, str]] = []
    seen: set[int] = set()
    leaf_types = (str, bytes, int, float, bool, type(None), Path, torch.device, torch.dtype)

    def set_value(owner: object, key: object, old_value: object, new_value: object, kind: str) -> None:
        if kind == "dict":
            owner[key] = new_value
        elif kind == "list":
            owner[key] = new_value
        else:
            setattr(owner, str(key), new_value)
        restore.append((owner, key, old_value, kind))

    def scrub_value(value: object) -> tuple[object, bool]:
        if _is_local_callable_for_pickle(value):
            return None, True
        if hasattr(value, "_replace") and hasattr(value, "init_fn"):
            updates = {}
            if _is_local_callable_for_pickle(getattr(value, "init_fn", None)):
                updates["init_fn"] = None
            if _is_local_callable_for_pickle(getattr(value, "get_rng_state_tracker", None)):
                updates["get_rng_state_tracker"] = None
            if updates:
                return value._replace(**updates), True
        return value, False

    def walk(obj: object) -> None:
        if isinstance(obj, leaf_types) or isinstance(obj, (torch.Tensor, nn.Parameter)):
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                new_value, changed = scrub_value(value)
                if changed:
                    set_value(obj, key, value, new_value, "dict")
                else:
                    walk(value)
            return
        if isinstance(obj, list):
            for index, value in enumerate(list(obj)):
                new_value, changed = scrub_value(value)
                if changed:
                    set_value(obj, index, value, new_value, "list")
                else:
                    walk(value)
            return
        if isinstance(obj, tuple):
            return

        try:
            items = list(vars(obj).items())
        except TypeError:
            return
        for key, value in items:
            if key.startswith("__"):
                continue
            new_value, changed = scrub_value(value)
            if changed:
                set_value(obj, key, value, new_value, "attr")
            elif key not in {"_parameters", "_buffers"}:
                walk(value)

    walk(root)
    return restore


def _restore_stripped_pickle_values(restore: list[tuple[object, object, object, str]]) -> None:
    for owner, key, value, kind in reversed(restore):
        if kind == "dict":
            owner[key] = value
        elif kind == "list":
            owner[key] = value
        else:
            setattr(owner, str(key), value)


def _stage1_nvfp4_mode() -> str:
    raw = os.environ.get("SANA_WM_STAGE1_NVFP4", "").strip().lower()
    if raw in _ENV_FALSE:
        return ""
    mode = os.environ.get("SANA_WM_STAGE1_NVFP4_MODE", "").strip().lower()
    if not mode and raw not in _ENV_TRUE:
        mode = raw
    return mode or "self_qkv"


def _stage1_nvfp4_include_patterns(mode: str) -> tuple[str, ...] | None:
    if mode in {"all", "full"}:
        if not _te_env_flag("SANA_WM_STAGE1_NVFP4_ALLOW_FULL"):
            raise ValueError(
                "Stage-1 full NVFP4 is experimental and showed visible quality loss in validation; "
                "set SANA_WM_STAGE1_NVFP4_ALLOW_FULL=1 to force it."
            )
        return None
    patterns: list[str] = []
    for item in mode.replace("+", ",").split(","):
        key = item.strip()
        if not key:
            continue
        if key not in _STAGE1_NVFP4_INCLUDE_BY_MODE:
            raise ValueError(
                f"Unsupported SANA_WM_STAGE1_NVFP4_MODE={mode!r}; "
                f"supported={sorted([*list(_STAGE1_NVFP4_INCLUDE_BY_MODE), 'all', 'full'])}"
            )
        patterns.extend(_STAGE1_NVFP4_INCLUDE_BY_MODE[key])
    patterns.extend(_te_env_tuple("SANA_WM_STAGE1_NVFP4_INCLUDE_PATTERNS"))
    return tuple(dict.fromkeys(patterns))


def _stage1_nvfp4_uses_cross_attention() -> bool:
    mode = _stage1_nvfp4_mode()
    if not mode:
        return False
    if mode in {"all", "full"}:
        return True
    if any(item.strip() == "cross" for item in mode.replace("+", ",").split(",")):
        return True
    return any("cross_attn" in pattern for pattern in _te_env_tuple("SANA_WM_STAGE1_NVFP4_INCLUDE_PATTERNS"))


def _stage1_forward_long_nvfp4(_self, *args, **kwargs):
    import transformer_engine.pytorch as te

    with te.fp8_autocast(enabled=True, fp8_recipe=_self._sana_wm_stage1_nvfp4_recipe):
        forward_long_impl = getattr(_self, "_sana_wm_stage1_forward_long_impl", None)
        if forward_long_impl is not None:
            return forward_long_impl(*args, **kwargs)
        return type(_self).forward_long(_self, *args, **kwargs)


def _make_stage1_nvfp4_recipe():
    import transformer_engine.common.recipe as te_recipe

    # RHT (random Hadamard transform) spreads outliers so a small signal in a
    # block isn't rounded to zero by a scale set by the large entries; with it
    # off, the low-magnitude camera-control residual is quantized away and the
    # video stops following the pose. Default both ON (env can disable to recover
    # the older, faster-but-pose-losing recipe).
    return te_recipe.NVFP4BlockScaling(
        disable_rht=not _te_env_flag_default_true("SANA_WM_STAGE1_NVFP4_RHT"),
        disable_stochastic_rounding=not _te_env_flag_default_true("SANA_WM_STAGE1_NVFP4_STOCHASTIC"),
    )


def _restore_stage1_nvfp4_runtime(model: nn.Module) -> None:
    model._sana_wm_stage1_nvfp4_recipe = _make_stage1_nvfp4_recipe()
    model.forward_long = types.MethodType(_stage1_forward_long_nvfp4, model)


class _LinearizedPointwiseConv(nn.Module):
    def __init__(self, conv_layer: nn.Module) -> None:
        super().__init__()
        conv = getattr(conv_layer, "conv", None)
        if not isinstance(conv, nn.Conv2d):
            raise ValueError("expected ConvLayer.conv to be nn.Conv2d")
        if conv.kernel_size != (1, 1) or conv.stride != (1, 1) or conv.padding != (0, 0):
            raise ValueError("only exact 1x1 pointwise Conv2d can be linearized")
        if conv.dilation != (1, 1) or conv.groups != 1:
            raise ValueError("grouped or dilated pointwise Conv2d cannot be linearized")
        if getattr(conv_layer, "dropout", None) is not None or getattr(conv_layer, "norm", None) is not None:
            raise ValueError("pointwise ConvLayer with dropout/norm is not supported")

        self.linear = nn.Linear(
            conv.in_channels,
            conv.out_channels,
            bias=conv.bias is not None,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            self.linear.weight.copy_(conv.weight.flatten(1))
            if conv.bias is not None:
                self.linear.bias.copy_(conv.bias)
        self.act = getattr(conv_layer, "act", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.act is not None:
            x = self.act(x)
        return x


class _CachedGLUMBConvTempLinearized(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        depth_conv = source.depth_conv
        depthwise = getattr(depth_conv, "conv", None)
        if not isinstance(depthwise, nn.Conv2d):
            raise ValueError("expected depth_conv.conv to be nn.Conv2d")

        self.inverted_conv = _LinearizedPointwiseConv(source.inverted_conv)
        self.depth_conv = depth_conv
        self.point_conv = _LinearizedPointwiseConv(source.point_conv)
        self.t_conv = source.t_conv
        self.glu_act = source.glu_act
        self.hidden_features = depthwise.out_channels // 2
        self.out_feature = self.t_conv.in_channels

    def forward(
        self,
        x: torch.Tensor,
        HW=None,
        save_kv_cache: bool = False,
        kv_cache=None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        B, N, _ = x.shape
        assert len(HW) == 3, "HW must be a tuple of (T, H, W)"
        T, H, W = HW

        x = self.inverted_conv(x)
        x = x.reshape(B * T, H, W, -1).permute(0, 3, 1, 2)
        x = self.depth_conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = x * self.glu_act(gate)
        x = x.reshape(B * T, self.hidden_features, H * W).permute(0, 2, 1)
        x = x.reshape(B, N, self.hidden_features)
        x = self.point_conv(x)

        x_reshaped = x.reshape(B, T, H * W, self.out_feature).permute(0, 3, 1, 2)
        padding_size = self.t_conv.kernel_size[0] // 2
        x_t_conv_in = x_reshaped
        padded_size = 0
        if kv_cache is not None:
            if kv_cache[-1] is not None:
                x_t_conv_in = torch.cat([kv_cache[-1][:, :, -padding_size:], x_reshaped], dim=2)
                padded_size = x_t_conv_in.shape[2] - x_reshaped.shape[2]
            if save_kv_cache:
                kv_cache[-1] = x_reshaped[:, :, -padding_size:, :].detach().clone()

        t_conv_out = self.t_conv(x_t_conv_in)[:, :, padded_size:]
        x_out = x_reshaped + t_conv_out
        x_out = x_out.permute(0, 2, 3, 1).reshape(B, N, self.out_feature)

        if kv_cache is not None:
            return x_out, kv_cache
        return x_out


def _linearize_stage1_ffn_for_nvfp4(module: nn.Module, *, prefix: str = "") -> tuple[int, int]:
    from diffusion.model.nets.basic_modules import CachedGLUMBConvTemp

    converted = 0
    skipped = 0
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, CachedGLUMBConvTemp):
            try:
                replacement = _CachedGLUMBConvTempLinearized(child)
            except ValueError:
                skipped += 1
                continue
            replacement.train(child.training)
            setattr(module, name, replacement)
            converted += 1
            continue
        child_converted, child_skipped = _linearize_stage1_ffn_for_nvfp4(child, prefix=child_prefix)
        converted += child_converted
        skipped += child_skipped
    return converted, skipped


# DEFAULT_TRANSLATION_SPEED / DEFAULT_ROTATION_SPEED_DEG / DEFAULT_PITCH_LIMIT_DEG
# come from camera_control (shared with the interactive demo so the two never drift).
ALLOWED_ACTION_KEYS: frozenset[str] = frozenset("wasdijkl")

# One-time notice (the action DSL key mapping changed vs the first public release).
_ACTION_MAPPING_NOTICE_SHOWN = False

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
    # Self-forcing autoregressive sampler knobs (only used when
    # ``sampling_algo == "self_forcing"``).
    num_cached_blocks: int = 2
    sink_token: bool = False
    num_frame_per_block: int = 3
    denoising_step_list: list[int] | None = None
    # When the refiner is enabled, additionally Sana-VAE-decode the unrefined
    # stage-1 latent so callers can compare stage-1 against refined output.
    save_stage1: bool = False


@dataclass
class RefinerSettings:
    """LTX-2 refiner configuration.

    ``block_size`` controls the inference mode: ``None`` (default) keeps the
    legacy single-shot sink-bidirectional Euler path; setting ``block_size``
    (canonical: 3) enables chunk-causal AR with a sliding KV window of
    ``kv_max_frames`` context frames, matching tian's
    ``run_reforcing_inference`` ``distilled-3step + source-sink-1`` recipe.
    """

    root: Path | str
    gemma_root: Path | str
    sink_size: int = 1
    seed: int = 42
    block_size: int | None = None
    kv_max_frames: int = 11


# ============================================================================
# Action-string → camera-to-world trajectory
# ============================================================================


# Rotation matrices + the pose integrator live in camera_control (shared core).


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
    smooth: bool = True,
) -> np.ndarray:
    """Roll out a ``(N+1, 4, 4)`` camera-to-world trajectory from an action string.

    The DSL groups segments as ``<keys>-<frames>`` joined by commas; ``"none"``
    means no keys held. Letters map to the **unified** control scheme (shared
    with the interactive demo via :mod:`camera_control`):

        w / s   forward / back translation        a / d   yaw left / right
        i / k   pitch up / down                    j / l   strafe left / right

    With ``smooth=True`` the same exponential velocity model as the live demo is
    applied (instant on a new key press, gentle coast on release), so a held key
    eases to rest instead of snapping. Coordinate convention: OpenCV
    (``+X right, +Y down, +Z forward``).
    """
    global _ACTION_MAPPING_NOTICE_SHOWN
    if not _ACTION_MAPPING_NOTICE_SHOWN:
        _ACTION_MAPPING_NOTICE_SHOWN = True
        logging.getLogger("sana_wm").warning(
            "[sana-wm] The --action control mapping was updated: a/d = YAW, j/l = STRAFE "
            "(previously a/d = strafe, j/l = yaw); i/k = pitch is unchanged. Motion is now "
            "smoothed with gentler default speeds (0.025 / 0.6 deg). If you have action strings "
            "from the earlier release, swap a/d <-> j/l to keep the same motion. See docs/sana_wm.md."
        )
    per_frame = _parse_action_string(action)
    rotation_speed_rad = math.radians(rotation_speed_deg)
    integrator = CameraPoseIntegrator(math.radians(pitch_limit_deg))
    velocity = VelocityState()
    poses = [integrator.pose.copy()]
    last_controls: set[str] = set()
    dt = 1.0 / _CAM_FPS

    for keys in per_frame:
        controls = {DSL_KEY_TO_CONTROL[c] for c in keys if c in DSL_KEY_TO_CONTROL}
        target = controls_to_target_velocity(
            controls, translation_speed=translation_speed, rotation_speed_rad=rotation_speed_rad
        )
        if smooth:
            # Snap on a fresh press (so a new key takes effect immediately);
            # otherwise ease toward the target (gentle coast on release).
            if controls - last_controls:
                velocity.snap_to(target)
            else:
                velocity.step_toward(target, dt)
            last_controls = controls
        else:
            velocity = target
        poses.append(integrator.step(velocity))

    return np.stack(poses, axis=0).astype(np.float32)


# ============================================================================
# Intrinsics: load from .npy or estimate with Pi3X
# ============================================================================


def _fit_intrinsics_sequence(arr: np.ndarray, num_frames: int) -> np.ndarray:
    """Return ``arr`` fitted to ``num_frames`` along axis 0."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] == num_frames:
        return arr.copy()
    if arr.shape[0] > num_frames:
        return arr[:num_frames].copy()
    if arr.shape[0] == 1:
        return np.broadcast_to(arr[:1], (num_frames, *arr.shape[1:])).copy()

    old_t = np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    flat = arr.reshape(arr.shape[0], -1)
    fitted = np.empty((num_frames, flat.shape[1]), dtype=np.float32)
    for idx in range(flat.shape[1]):
        fitted[:, idx] = np.interp(new_t, old_t, flat[:, idx]).astype(np.float32)
    return fitted.reshape((num_frames, *arr.shape[1:]))


def load_intrinsics(path: Path, num_frames: int) -> np.ndarray:
    """Return ``(num_frames, 4)`` intrinsics as ``[fx, fy, cx, cy]``.

    Accepts ``.npy`` arrays shaped ``(3, 3)``, ``(F, 3, 3)``, ``(4,)``, or
    ``(F, 4)``. Per-frame intrinsics are truncated or resampled in time to
    match ``num_frames``.
    """
    arr = np.load(path).astype(np.float32)
    if arr.shape == (4,):
        return np.broadcast_to(arr, (num_frames, 4)).copy()
    if arr.shape == (3, 3):
        v = np.array([arr[0, 0], arr[1, 1], arr[0, 2], arr[1, 2]], dtype=np.float32)
        return np.broadcast_to(v, (num_frames, 4)).copy()
    if arr.ndim == 2 and arr.shape[1] == 4:
        return _fit_intrinsics_sequence(arr, num_frames)
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        K = _fit_intrinsics_sequence(arr, num_frames)
        return np.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], axis=1)
    raise ValueError(
        f"Unsupported intrinsics shape {arr.shape} for num_frames={num_frames}; "
        f"expected (3,3), (F,3,3), (4,), or (F,4)."
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
        offload_text_encoder: bool = False,
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.device = torch.device(device)
        self.refiner_settings = refiner
        self.offload_vae = offload_vae
        self.offload_refiner = offload_refiner
        self.offload_text_encoder = offload_text_encoder
        self.logger = logger or get_root_logger()
        self._model_path = model_path
        self.weight_dtype = get_weight_dtype(config.model.mixed_precision)
        self.vae_dtype = get_weight_dtype(config.vae.weight_dtype)
        self._refiner_built = False
        self._streaming_stage1_prompt_cache: dict[
            tuple[object, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._streaming_refiner_prompt_cache: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}

        self._build_vae()
        self._build_text_encoder()
        if self.offload_text_encoder:
            self._offload_text_encoder()
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
        cache_path = self._stage1_prepared_cache_path(model_path)
        if cache_path is not None and cache_path.is_file():
            t0 = time.perf_counter()
            self.logger.info("[stage1-cache] loading prepared Stage-1 model from %s", cache_path)
            try:
                model = torch.load(cache_path, map_location=self.device, weights_only=False)
                self.model = model.eval()
                if getattr(self.model, "_sana_wm_stage1_nvfp4_converted", False):
                    _restore_stage1_nvfp4_runtime(self.model)
                else:
                    self.model = self.model.to(device=self.device, dtype=self.weight_dtype)
                self.logger.info("[stage1-cache] loaded prepared Stage-1 model in %.1fs", time.perf_counter() - t0)
                return
            except Exception as exc:
                self.logger.warning("[stage1-cache] failed to load %s: %s; rebuilding", cache_path, exc)

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

    def _stage1_prepared_cache_path(self, model_path: str | Path) -> Path | None:
        root = _prepared_module_cache_root()
        mode = _stage1_nvfp4_mode()
        if root is None or not mode:
            return None
        payload = {
            "kind": "stage1_prepared_model_v2",
            "model_path": _path_fingerprint(model_path),
            "model_name": self.config.model.model,
            "dtype": str(self.weight_dtype),
            "torch": torch.__version__,
            "stage1_nvfp4_mode": mode,
            "stage1_linearize_ffn": os.environ.get("SANA_WM_STAGE1_LINEARIZE_FFN", ""),
            "stage1_text_pad_multiple": os.environ.get("SANA_WM_STAGE1_NVFP4_TEXT_PAD_MULTIPLE", ""),
            "stage1_skip_patterns": os.environ.get("SANA_WM_STAGE1_NVFP4_SKIP_PATTERNS", ""),
            "stage1_include_patterns": os.environ.get("SANA_WM_STAGE1_NVFP4_INCLUDE_PATTERNS", ""),
            "stage1_no_default_skips": os.environ.get("SANA_WM_STAGE1_NVFP4_NO_DEFAULT_SKIPS", ""),
            "te_cpu_staging": os.environ.get("SANA_WM_TE_NVFP4_CPU_STAGING", ""),
        }
        try:
            import transformer_engine

            payload["transformer_engine"] = getattr(transformer_engine, "__version__", "unknown")
        except Exception:
            payload["transformer_engine"] = "unavailable"
        return root / "stage1" / f"{_prepared_module_cache_hash(payload)}.pt"

    def _save_stage1_prepared_cache(self, model_path: str | Path) -> None:
        if os.environ.get("SANA_WM_PREPARED_MODULE_CACHE_SAVE", "1").strip().lower() in {
            "",
            "0",
            "false",
            "no",
            "off",
        }:
            return
        cache_path = self._stage1_prepared_cache_path(model_path)
        model = getattr(self, "model", None)
        if cache_path is None or model is None or cache_path.is_file():
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
        t0 = time.perf_counter()
        self.logger.info("[stage1-cache] saving prepared Stage-1 model to %s", cache_path)
        forward_long = model.__dict__.pop("forward_long", None)
        restore = _strip_local_callables_for_pickle(model)
        if restore:
            self.logger.info("[stage1-cache] stripped %d init-only callables before save", len(restore))
        try:
            torch.save(model, tmp_path)
            os.replace(tmp_path, cache_path)
        except Exception as exc:
            self.logger.warning("[stage1-cache] failed to save %s: %s", cache_path, exc)
        finally:
            _restore_stripped_pickle_values(restore)
            if forward_long is not None:
                model.forward_long = forward_long
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
        if cache_path.is_file():
            self.logger.info("[stage1-cache] saved prepared Stage-1 model in %.1fs", time.perf_counter() - t0)

    def _prepare_stage1_nvfp4(self) -> None:
        mode = _stage1_nvfp4_mode()
        if not mode:
            return
        model = getattr(self, "model", None)
        if model is None or getattr(model, "_sana_wm_stage1_nvfp4_converted", False):
            return

        if _te_env_flag("SANA_WM_STAGE1_LINEARIZE_FFN") and not getattr(model, "_sana_wm_stage1_ffn_linearized", False):
            converted_ffn, skipped_ffn = _linearize_stage1_ffn_for_nvfp4(model)
            if converted_ffn <= 0:
                raise RuntimeError(
                    "SANA_WM_STAGE1_LINEARIZE_FFN=1 converted no CachedGLUMBConvTemp blocks; " f"skipped={skipped_ffn}."
                )
            model._sana_wm_stage1_ffn_linearized = True
            self.logger.info(
                "[Stage-1 linearized FFN] converted %d CachedGLUMBConvTemp blocks, skipped %d",
                converted_ffn,
                skipped_ffn,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        recipe = _make_stage1_nvfp4_recipe()
        skip_patterns: tuple[str, ...] = ()
        if not _te_env_flag("SANA_WM_STAGE1_NVFP4_NO_DEFAULT_SKIPS"):
            skip_patterns = _STAGE1_NVFP4_SKIP_DEFAULTS
        skip_patterns = tuple(dict.fromkeys((*skip_patterns, *_te_env_tuple("SANA_WM_STAGE1_NVFP4_SKIP_PATTERNS"))))
        include_patterns = _stage1_nvfp4_include_patterns(mode)
        converted, skipped = _replace_linear_with_te_nvfp4(
            model,
            recipe=recipe,
            params_dtype=self.weight_dtype,
            skip_patterns=skip_patterns,
            include_patterns=include_patterns,
        )
        if converted <= 0:
            raise RuntimeError(
                f"SANA_WM_STAGE1_NVFP4={mode!r} converted no Linear layers; "
                f"skipped={skipped}, include_patterns={include_patterns}."
            )

        model._sana_wm_stage1_nvfp4_converted = True
        model._sana_wm_stage1_nvfp4_recipe = recipe
        forward_long = getattr(model, "forward_long", None)
        forward_long_func = getattr(forward_long, "__func__", forward_long)
        if forward_long_func is not getattr(type(model), "forward_long", None):
            model._sana_wm_stage1_forward_long_impl = forward_long
        model.forward_long = types.MethodType(_stage1_forward_long_nvfp4, model)
        self.logger.info(
            "[stage1-nvfp4] mode=%s converted %d Linear layers (skipped %d)",
            mode,
            converted,
            skipped,
        )
        torch.cuda.empty_cache()
        if hasattr(self, "_model_path"):
            self._save_stage1_prepared_cache(self._model_path)

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
        root = Path(resolve_hf_path(str(refiner.root)))
        if not (root / "transformer" / "config.json").is_file() or not (root / "connectors" / "config.json").is_file():
            raise FileNotFoundError(
                f"LTX-2 refiner not found at {root}. Expected " "transformer/config.json and connectors/config.json."
            )
        return str(root)

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

    def _offload_text_encoder(self) -> None:
        if not self.offload_text_encoder:
            return
        text_encoder = getattr(self, "text_encoder", None)
        if text_encoder is None:
            return
        text_encoder.to("cpu")
        torch.cuda.empty_cache()

    def _offload_vae_encoder_for_streaming(self) -> None:
        vae = getattr(self, "vae", None)
        encoder = getattr(vae, "encoder", None)
        if encoder is not None:
            encoder.to("cpu")
            torch.cuda.empty_cache()

    def _move_vae_decoder_for_streaming(self, device: torch.device | str) -> None:
        vae = getattr(self, "vae", None)
        decoder = getattr(vae, "decoder", None)
        if decoder is not None:
            decoder.to(device)
        elif vae is not None:
            vae.to(device)

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

        # When the refiner is enabled, optionally decode the unrefined stage-1
        # latent with the Sana VAE first so the caller can compare both paths.
        # We must do this BEFORE ``_refine``, which offloads stage-1 components
        # when ``offload_refiner`` is set.
        stage1_video = None
        if self.refiner_settings is not None and params.save_stage1:
            stage1_video = self._decode_with_sana_vae(sana_latent)

        if self.refiner_settings is not None:
            video = self._refine(sana_latent, prompt, params, self.refiner_settings)
            # _refine drops the sink anchor frame; realign the trajectory.
            video_c2w = c2w[1 : params.num_frames]
        else:
            video = self._decode_with_sana_vae(sana_latent)
            video_c2w = c2w[: params.num_frames]

        result: dict[str, object] = {"video": video, "c2w": video_c2w, "latent": sana_latent.cpu()}
        if stage1_video is not None:
            result["stage1_video"] = stage1_video
            # stage-1 covers all ``num_frames`` frames; refined drops frame 0.
            result["stage1_c2w"] = c2w[: params.num_frames]
        return result

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

        text_encoder_on_target = True
        try:
            text_encoder_device = next(self.text_encoder.parameters()).device
            target_device = torch.device(self.device)
            text_encoder_on_target = text_encoder_device.type == target_device.type and (
                target_device.type != "cuda"
                or target_device.index is None
                or text_encoder_device.index == target_device.index
            )
        except StopIteration:
            pass

        move_text_encoder_for_encode = self.offload_text_encoder or not text_encoder_on_target
        if move_text_encoder_for_encode:
            self.text_encoder.to(self.device)

        def encode(text: str, length: int) -> tuple[torch.Tensor, torch.Tensor]:
            tok = self.tokenizer(
                [text], max_length=length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(self.device)
            return self.text_encoder(tok.input_ids, tok.attention_mask)[0], tok.attention_mask

        try:
            cond, cond_mask = encode(prompt, max_length_all)
            select = [0] + list(range(-max_length + 1, 0))
            cond = cond[:, None][:, :, select]
            cond_mask = cond_mask[:, select]
            neg, neg_mask = encode(negative_prompt, max_length)
            return cond, cond_mask, neg[:, None], neg_mask
        finally:
            if move_text_encoder_for_encode:
                self.text_encoder.to("cpu")
                torch.cuda.empty_cache()

    def _pad_stage1_text_for_nvfp4(
        self,
        cond: torch.Tensor,
        cond_mask: torch.Tensor,
        neg: torch.Tensor,
        neg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not _stage1_nvfp4_uses_cross_attention():
            return cond, cond_mask, neg, neg_mask
        multiple = int(os.environ.get("SANA_WM_STAGE1_NVFP4_TEXT_PAD_MULTIPLE", "8"))
        if multiple <= 1:
            return cond, cond_mask, neg, neg_mask

        def pad_pair(text: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            pad = (-text.shape[-2]) % multiple
            if pad == 0:
                return text, mask
            text_pad_shape = list(text.shape)
            text_pad_shape[-2] = pad
            mask_pad_shape = list(mask.shape)
            mask_pad_shape[-1] = pad
            text = torch.cat([text, text.new_zeros(text_pad_shape)], dim=-2)
            mask = torch.cat([mask, mask.new_zeros(mask_pad_shape)], dim=-1)
            return text, mask

        cond, cond_mask = pad_pair(cond, cond_mask)
        neg, neg_mask = pad_pair(neg, neg_mask)
        return cond, cond_mask, neg, neg_mask

    def _streaming_prompt_cache_enabled(self) -> bool:
        return os.environ.get("SANA_WM_STREAMING_PROMPT_CACHE", "1").lower() in _ENV_TRUE

    def _get_streaming_stage1_prompt(
        self, prompt: str, negative_prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (
            prompt,
            negative_prompt,
            str(self.device),
            str(self.weight_dtype),
            _stage1_nvfp4_mode(),
            os.environ.get("SANA_WM_STAGE1_NVFP4_TEXT_PAD_MULTIPLE", "8"),
        )
        cache = self._streaming_stage1_prompt_cache
        if self._streaming_prompt_cache_enabled() and key in cache:
            return cache[key]

        cond, cond_mask, neg, neg_mask = self._encode_prompts(prompt, negative_prompt)
        cond, cond_mask, neg, neg_mask = self._pad_stage1_text_for_nvfp4(cond, cond_mask, neg, neg_mask)
        if self._streaming_prompt_cache_enabled():
            cache.clear()
            cache[key] = (
                cond.detach(),
                cond_mask.detach(),
                neg.detach(),
                neg_mask.detach(),
            )
        return cond, cond_mask, neg, neg_mask

    def _get_streaming_refiner_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.refiner is None:
            raise RuntimeError("Streaming refiner prompt requested before the refiner is built.")
        key = (
            prompt.strip(),
            str(self.device),
            str(self.refiner.dtype),
            str(self.refiner.refiner_root),
            str(self.refiner.gemma_root),
            int(self.refiner.text_max_sequence_length),
        )
        cache = self._streaming_refiner_prompt_cache
        if self._streaming_prompt_cache_enabled() and key in cache:
            return cache[key]

        # A 32GB 5090 cannot hold Gemma plus the resident DiT/VAE/refiner stack,
        # so temporarily spill the generation models while the prompt is encoded.
        # TE NVFP4 tensors cannot be moved to CPU without dequantizing; for those
        # prepared-cache models, release them and reload from cache after encoding.
        dropped_stage1 = False
        dropped_refiner = False
        stage1_text_encoder = getattr(self, "text_encoder", None)
        if stage1_text_encoder is not None:
            stage1_text_encoder.to("cpu")
        if getattr(self.model, "_sana_wm_stage1_nvfp4_converted", False):
            self.model = None
            dropped_stage1 = True
        else:
            self.model.to("cpu")
        self._move_vae_decoder_for_streaming("cpu")
        if getattr(self.refiner, "_te_nvfp4_converted", False):
            self.refiner.transformer = None
            dropped_refiner = True
        else:
            self.refiner.move_video_modules("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        try:
            embeds, attention_mask = self.refiner._encode_prompt(prompt)
        finally:
            if dropped_refiner:
                self._build_refiner()
            if dropped_stage1:
                self._build_model(self._model_path)
        if self._streaming_prompt_cache_enabled():
            cache.clear()
            cache[key] = (embeds.detach(), attention_mask.detach())
        return embeds, attention_mask

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
        else:
            self._offload_vae_encoder_for_streaming()

        cond, cond_mask, neg, neg_mask = self._encode_prompts(prompt, params.negative_prompt)
        cond, cond_mask, neg, neg_mask = self._pad_stage1_text_for_nvfp4(cond, cond_mask, neg, neg_mask)
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
        self._prepare_stage1_nvfp4()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
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
            params,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.logger.info(
            f"[timing] stage1 sample: {time.perf_counter() - t0:.3f}s " f"(latent shape {tuple(samples.shape)})"
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
        params: GenerationParams,
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
        if algo == "self_forcing":
            if chunk_index is None:
                raise ValueError(
                    "--sampling_algo=self_forcing requires the config to expose chunk_index "
                    "(chunk-causal model). The default bidirectional Sana-WM checkpoint is "
                    "incompatible; supply a chunk-causal --config and --model_path."
                )
            # ``use_softmax_attention=True`` selects the 10-slot accumulator
            # (``_accumulate_softmax_kv_cache``) which transparently handles both
            # the old concat-layout and the new state-or-concat dual-mode flag in
            # slot 6 — required for the hybrid GDN+Softmax chunk-causal Sana-WM.
            solver = SelfForcingFlowEulerCamCtrl(
                self.model,
                condition=cond,
                uncondition=neg,
                cfg_scale=cfg_scale,
                flow_shift=flow_shift,
                model_kwargs=model_kwargs,
                base_chunk_frames=params.num_frame_per_block,
                num_cached_blocks=params.num_cached_blocks,
                sink_token=params.sink_token,
                use_softmax_attention=True,
            )
            return solver.sample(
                z,
                steps=steps,
                generator=generator,
                denoising_step_list=params.denoising_step_list,
            )
        raise ValueError(f"Unknown sampling_algo: {algo}")

    # ------- stage 2: decode -------

    def _decode_with_sana_vae(self, sana_latent: torch.Tensor) -> np.ndarray:
        self.logger.info(f"[sana-vae] decoding {sana_latent.shape[2]} latent frames")
        if getattr(self, "vae", None) is None:
            self._build_vae()
        if self.offload_vae:
            self.vae.to(self.device)
        samples = sana_latent.to(device=self.device, dtype=self.vae_dtype)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        decoded = vae_decode(self.config.vae.vae_type, self.vae, samples)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.logger.info(
            f"[timing] vae decode: {time.perf_counter() - t0:.3f}s "
            f"(latent T={sana_latent.shape[2]} -> pixels {tuple(decoded.shape)})"
        )
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
            block_size=refiner.block_size,
            kv_max_frames=int(refiner.kv_max_frames),
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

    @torch.inference_mode()
    def generate_streaming(
        self,
        image: "Image.Image",
        prompt: str,
        c2w: torch.Tensor,
        intrinsics_vec4: torch.Tensor,
        params: GenerationParams,
        *,
        output_path: str | Path,
        streaming_crf: int = 18,
        streaming_preset: str = "medium",
        streaming_encoder: str = "libx264",
        output_mode: str = "mp4",
        profile_cuda: bool = False,
        sample_frames_path: str | Path | None = None,
        sample_frame_stride: int = 0,
        decoded_chunk_callback=None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, object]:
        """Chunk-pipelined interactive generation.

        Runs stage-1 sampling, refiner AR blocks, and causal-VAE decode on
        three CUDA streams, emitting one decoded chunk per AR block to a
        progressive MP4. Called from
        :mod:`inference_video_scripts.wm.inference_sana_wm_streaming`, which
        is the canonical entrypoint. Requires the pipeline to be built
        with the causal LTX-2 VAE and a refiner with ``block_size`` set,
        and ``params.sampling_algo == 'self_forcing'``.

        Returns a dict with ``output_path``, ``n_pixel_frames``, and ``c2w``
        aligned with the emitted frames (first frame dropped).
        """
        from diffusion.model.ltx2 import CausalVaeStreamingDecoder
        from diffusion.refiner.diffusers_ltx2_refiner import (
            STAGE_2_DISTILLED_SIGMA_VALUES,
            RefinerChunkRunner,
        )
        from inference_video_scripts.wm.streaming_pipeline import (
            StreamingPipelineConfig,
            run_streaming_inference,
        )

        if "LTX2VAE_diffusers_causal" not in self.config.vae.vae_type:
            raise ValueError(
                "generate_streaming requires the causal LTX-2 VAE "
                f"(config.vae.vae_type must include 'LTX2VAE_diffusers_causal'); "
                f"got {self.config.vae.vae_type!r}. Use inference_sana_wm_streaming.py."
            )
        if self.refiner_settings is None or self.refiner_settings.block_size is None:
            raise ValueError(
                "generate_streaming requires a refiner with block_size set "
                "(canonical: 3). Use inference_sana_wm_streaming.py."
            )
        if params.sampling_algo != "self_forcing":
            raise ValueError(
                "generate_streaming requires sampling_algo='self_forcing'; "
                f"got {params.sampling_algo!r}. Use inference_sana_wm_streaming.py."
            )
        if self.offload_refiner and not self._refiner_built:
            # Streaming keeps all three models resident; swapping the refiner
            # in/out per chunk would dominate runtime.
            self._build_refiner()

        def _progress(message: str, phase: str = "prepare", **extra: object) -> None:
            if progress_callback is not None:
                progress_callback({"phase": phase, "message": message, **extra})

        vae_stride = self.config.vae.vae_stride
        latent_T = (params.num_frames - 1) // vae_stride[0] + 1
        latent_h = TARGET_HEIGHT // vae_stride[-1]
        latent_w = TARGET_WIDTH // vae_stride[-1]

        _progress("preparing camera conditioning")
        camera = prepare_camera(
            c2w[: params.num_frames],
            intrinsics_vec4[: params.num_frames],
            target_size=(TARGET_HEIGHT, TARGET_WIDTH),
            vae_stride=vae_stride,
        )

        # Encode the refiner prompt before allocating resident Stage-1 tensors.
        # On 32GB cards Gemma's temporary activations can otherwise collide with
        # latents/raymaps even after the large model modules are offloaded.
        _progress("encoding refiner prompt")
        refiner_prompt_embeds, refiner_prompt_attention_mask = self._get_streaming_refiner_prompt(prompt)
        if _te_env_flag("SANA_WM_REFINER_NVFP4"):
            _progress("preparing refiner NVFP4")
            cpu_stage_nvfp4 = _te_env_flag("SANA_WM_TE_NVFP4_CPU_STAGING")
            offload_stage1_for_refiner_nvfp4 = (not cpu_stage_nvfp4) and _te_env_flag(
                "SANA_WM_REFINER_NVFP4_OFFLOAD_STAGE1"
            )
            offload_vae_for_refiner_nvfp4 = (not cpu_stage_nvfp4) and _te_env_flag("SANA_WM_REFINER_NVFP4_OFFLOAD_VAE")
            if offload_stage1_for_refiner_nvfp4:
                self.model.to("cpu")
                torch.cuda.empty_cache()
            if offload_vae_for_refiner_nvfp4:
                self.vae.to("cpu")
                torch.cuda.empty_cache()
            if not cpu_stage_nvfp4:
                self.refiner.move_video_modules(self.device)
            self.refiner.offload_video_unused_audio_modules("cpu")
            self.refiner.prepare_transformer_nvfp4()
            if cpu_stage_nvfp4:
                self.refiner.move_video_modules(self.device)
                self.refiner.offload_video_unused_audio_modules("cpu")
            torch.cuda.empty_cache()

        # First-frame encode (uses self.vae, which is the causal LTX-2 VAE in
        # streaming mode — same encoder as the bidirectional sibling).
        _progress("encoding first frame")
        if self.offload_vae or _te_env_flag("SANA_WM_REFINER_NVFP4_OFFLOAD_VAE"):
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
        else:
            # The encoder is no longer needed after the conditioning frame is
            # latent-encoded. Move it out before refiner NVFP4 conversion, where
            # 32GB cards are sensitive to temporary TE allocation peaks.
            self._offload_vae_encoder_for_streaming()

        _progress("encoding stage-1 prompt")
        cond, cond_mask, neg, neg_mask = self._get_streaming_stage1_prompt(prompt, params.negative_prompt)
        _progress("allocating latent buffers")
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

        solver = SelfForcingFlowEulerCamCtrl(
            self.model,
            condition=cond,
            uncondition=neg,
            cfg_scale=params.cfg_scale,
            flow_shift=flow_shift,
            model_kwargs=model_kwargs,
            base_chunk_frames=params.num_frame_per_block,
            num_cached_blocks=params.num_cached_blocks,
            sink_token=params.sink_token,
            use_softmax_attention=True,
        )
        stage1_chunk_indices = tuple(int(v) for v in solver.create_autoregressive_segments(latent_T))
        n_stage1_chunks = len(stage1_chunk_indices) - 1
        _progress(
            "initializing chunk pipeline",
            stage1_total=n_stage1_chunks,
            decode_total=max(latent_T - int(self.refiner_settings.sink_size), 0)
            // int(self.refiner_settings.block_size),
        )
        stage1_iter = solver.sample_chunks(
            z,
            steps=params.step,
            generator=generator,
            denoising_step_list=params.denoising_step_list,
        )

        _progress("moving refiner and stage-1 models to GPU")
        self.refiner.move_video_modules(self.device)
        self.refiner.offload_video_unused_audio_modules("cpu")
        self.refiner.prepare_transformer_nvfp4()
        torch.cuda.empty_cache()
        self.model.to(self.device)
        self._prepare_stage1_nvfp4()
        torch.cuda.empty_cache()
        lazy_vae_decoder = os.environ.get("SANA_WM_STREAMING_LAZY_VAE_DECODER", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        sequential_offload = os.environ.get("SANA_WM_STREAMING_SEQUENTIAL_OFFLOAD", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if lazy_vae_decoder:
            self._move_vae_decoder_for_streaming("cpu")
        else:
            self._move_vae_decoder_for_streaming(self.device)

        def _offload_stage1_after_streaming_sample() -> None:
            self.model.to("cpu")
            self._move_vae_decoder_for_streaming("cpu")
            torch.cuda.empty_cache()

        sigmas_t = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device)
        refiner_runner = RefinerChunkRunner(
            self.refiner,
            prompt_embeds=refiner_prompt_embeds,
            prompt_attention_mask=refiner_prompt_attention_mask,
            fps=float(params.fps),
            sigmas=sigmas_t,
            source_sink_frames=int(self.refiner_settings.sink_size),
            block_size=int(self.refiner_settings.block_size),
            kv_max_frames=int(self.refiner_settings.kv_max_frames),
            seed=int(self.refiner_settings.seed),
            spatial_shape=(int(z.shape[3]), int(z.shape[4])),
            n_active_frames=max(int(z.shape[2]) - int(self.refiner_settings.sink_size), 0),
            latent_channels=int(z.shape[1]),
            batch_size=int(z.shape[0]),
        )

        # Causal VAE streaming decoder.
        vae_streaming_decoder = CausalVaeStreamingDecoder(self.vae)

        cfg = StreamingPipelineConfig(
            sink_size=int(self.refiner_settings.sink_size),
            block_size=int(self.refiner_settings.block_size),
            fps=int(params.fps),
            output_path=output_path,
            mp4_crf=int(streaming_crf),
            mp4_preset=str(streaming_preset),
            mp4_encoder=str(streaming_encoder),
            drop_first_pixel=True,
            output_mode=output_mode,
            profile_cuda=bool(profile_cuda),
            sample_frames_path=sample_frames_path,
            sample_frame_stride=int(sample_frame_stride),
            lazy_vae_decoder=lazy_vae_decoder,
            sequential_offload=sequential_offload,
            stage1_done_callback=_offload_stage1_after_streaming_sample if sequential_offload else None,
            stage1_chunk_ends=stage1_chunk_indices[1:],
            decoded_chunk_callback=decoded_chunk_callback,
            progress_callback=progress_callback,
        )
        _progress("running streaming pipeline", phase="stream")
        result = run_streaming_inference(
            stage1_chunk_iter=stage1_iter,
            n_stage1_chunks=n_stage1_chunks,
            z_init=z,
            refiner_runner=refiner_runner,
            vae_streaming_decoder=vae_streaming_decoder,
            pixel_h=TARGET_HEIGHT,
            pixel_w=TARGET_WIDTH,
            config=cfg,
            logger=self.logger,
        )

        return {
            "output_path": result.output_path,
            "n_pixel_frames": result.n_pixel_frames,
            "n_refiner_blocks": result.n_refiner_blocks,
            "n_decode_chunks": result.n_decode_chunks,
            "output_mode": result.output_mode,
            "wall_seconds": result.wall_seconds,
            "first_chunk_seconds": result.first_chunk_seconds,
            "first_chunk_frames": result.first_chunk_frames,
            "steady_state_seconds": result.steady_state_seconds,
            "steady_state_frames_per_second": result.steady_state_frames_per_second,
            "steady_state_realtime_factor": result.steady_state_realtime_factor,
            "frames_per_second": result.frames_per_second,
            "realtime_factor": result.realtime_factor,
            "stage1_cuda_seconds": result.stage1_cuda_seconds,
            "refiner_cuda_seconds": result.refiner_cuda_seconds,
            "decode_cuda_seconds": result.decode_cuda_seconds,
            "sample_frames_path": result.sample_frames_path,
            "sampled_frame_count": result.sampled_frame_count,
            "sampled_frame_indices": result.sampled_frame_indices,
            "c2w": c2w[1 : 1 + result.n_pixel_frames],
        }


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
        "--sampling_algo",
        default="flow_euler_ltx",
        choices=["flow_euler_ltx", "flow_euler", "flow_dpm-solver", "self_forcing"],
    )
    p.add_argument(
        "--num_cached_blocks",
        type=int,
        default=2,
        help="Self-forcing: number of past chunks to retain in KV cache "
        "(set to a sliding-window size; -1 keeps all). Only used when "
        "--sampling_algo=self_forcing.",
    )
    p.add_argument(
        "--sink_token",
        action="store_true",
        help="Self-forcing: anchor chunk 0 in the KV cache permanently as a "
        "sink token. Only used when --sampling_algo=self_forcing.",
    )
    p.add_argument(
        "--num_frame_per_block",
        type=int,
        default=3,
        help="Self-forcing: number of latent frames per autoregressive chunk. " "Must match the model's chunk_size.",
    )
    p.add_argument(
        "--denoising_step_list",
        type=str,
        default="",
        help="Self-forcing distilled student schedule, comma-separated integer "
        "timesteps that must end with 0 (e.g., '1000,750,500,250,0'). When "
        "provided, --step is ignored and these exact timesteps are used.",
    )
    p.add_argument(
        "--save_stage1",
        action="store_true",
        help="Also Sana-VAE-decode the unrefined stage-1 latent and write it as "
        "a separate '<name>_stage1.mp4'. No-op when --no_refiner is set.",
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
        "--refiner_root",
        default=HF_DEFAULTS["refiner_root"],
        help="LTX-2 refiner root containing transformer/ and connectors/.",
    )
    p.add_argument(
        "--refiner_gemma_root",
        default=HF_DEFAULTS["refiner_gemma_root"],
        help="Gemma diffusers root for the refiner text encoder.",
    )
    p.add_argument("--refiner_seed", type=int, default=42)
    p.add_argument("--sink_size", type=int, default=1)
    p.add_argument(
        "--refiner_block_size",
        type=int,
        default=None,
        help="LTX-2 refiner: latent frames per AR block. When set (canonical: 3) "
        "the refiner runs chunk-causal AR with a sliding KV window; when unset "
        "it falls back to the legacy single-shot sink-bidirectional Euler path.",
    )
    p.add_argument(
        "--refiner_kv_max_frames",
        type=int,
        default=11,
        help="LTX-2 refiner: maximum (sink + history + active) latent frames "
        "retained in the AR sliding window. Canonical: 11 = 1 sink + 10 recent.",
    )

    # Causal VAE + interactive chunk-pipelined streaming.
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


def _snap_num_frames(n: int, stride: int = 8, *, upper_bound: int | None = None) -> int:
    """Snap ``n`` to the nearest ``stride*k + 1`` (LTX-2 VAE constraint).

    Ties round up to keep the user's requested length when possible. If the
    rounded value would exceed ``upper_bound`` (e.g., trajectory length), the
    floor candidate is returned instead.
    """
    if n < 1:
        return 1
    if (n - 1) % stride == 0:
        return n
    floor_cand = n - ((n - 1) % stride)
    ceil_cand = floor_cand + stride
    snapped = floor_cand if (n - floor_cand) < (ceil_cand - n) else ceil_cand
    if upper_bound is not None and snapped > upper_bound:
        snapped = floor_cand
    return max(snapped, 1)


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
    snapped = _snap_num_frames(num_frames, stride=8, upper_bound=c2w_full.shape[0])
    if snapped != args.num_frames:
        logger.warning(
            f"LTX-2 VAE requires num_frames = 8k+1; "
            f"--num_frames={args.num_frames} snapped to {snapped} "
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

    config: InferenceConfig = pyrallis.parse(
        config_class=InferenceConfig, config_path=resolve_hf_path(args.config), args=[]
    )

    refiner = (
        None
        if args.no_refiner
        else RefinerSettings(
            root=args.refiner_root,
            gemma_root=args.refiner_gemma_root,
            sink_size=args.sink_size,
            seed=args.refiner_seed,
            block_size=args.refiner_block_size,
            kv_max_frames=args.refiner_kv_max_frames,
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

    denoising_step_list: list[int] | None = None
    if args.denoising_step_list:
        denoising_step_list = [int(t.strip()) for t in args.denoising_step_list.split(",") if t.strip()]
        if not denoising_step_list or denoising_step_list[-1] != 0:
            raise SystemExit("--denoising_step_list must be a comma-separated list ending with 0.")

    params = GenerationParams(
        num_frames=num_frames,
        fps=args.fps,
        step=args.step,
        cfg_scale=args.cfg_scale,
        flow_shift=args.flow_shift,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        sampling_algo=args.sampling_algo,
        num_cached_blocks=args.num_cached_blocks,
        sink_token=args.sink_token,
        num_frame_per_block=args.num_frame_per_block,
        denoising_step_list=denoising_step_list,
        save_stage1=args.save_stage1,
    )

    out = pipeline.generate(cropped, prompt, c2w, intrinsics_vec4, params)
    video_hwc = out["video"]

    if not args.no_action_overlay:
        logger.info("Compositing action overlay onto the output video.")
        video_hwc = apply_overlay(video_hwc, out["c2w"])

    write_video(args.output_dir, args.name, video_hwc, params.fps, logger)

    stage1_video = out.get("stage1_video")
    if stage1_video is not None:
        stage1_hwc = stage1_video
        if not args.no_action_overlay:
            stage1_hwc = apply_overlay(stage1_hwc, out["stage1_c2w"])
        write_video(args.output_dir, f"{args.name}_stage1", stage1_hwc, params.fps, logger)


if __name__ == "__main__":
    main()
