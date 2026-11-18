"""Stable library API for converting normalized or raw H3 latents."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .constants import H3_LATENTS_MEAN, H3_LATENTS_STD
from .geometry import align_h3_to_ltx, h3_temporal_positions, ltx_temporal_positions, padded_ltx_pixel_frames
from .model import build_model


def expected_shapes(pixel_frames: int, pixel_height: int, pixel_width: int) -> dict[str, list[int]]:
    if pixel_height % 32 or pixel_width % 32:
        raise ValueError("pixel height and width must be divisible by 32")
    padded_frames = padded_ltx_pixel_frames(pixel_frames)
    return {
        "h3": [24, len(h3_temporal_positions(pixel_frames)), pixel_height // 16, pixel_width // 16],
        "aligned_h3": [384, len(ltx_temporal_positions(padded_frames)), pixel_height // 32, pixel_width // 32],
        "ltx": [128, len(ltx_temporal_positions(padded_frames)), pixel_height // 32, pixel_width // 32],
    }


class H3ToLTXAdapter:
    """Load the frozen checkpoint and translate H3 latents to normalized LTX latents."""

    def __init__(self, model, config: dict, device: torch.device, dtype: torch.dtype):
        self.model = model
        self.config = config
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "H3ToLTXAdapter":
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        if config.get("format") != "h3_ltx_adapter_safetensors_v1":
            raise ValueError(f"unsupported adapter format: {config.get('format')!r}")
        geometry = config.get("geometry", {})
        expected_geometry = {
            "spatial_mode": "pixel_unshuffle",
            "temporal_mode": "linear_plus_nearest_pack",
            "temporal_pack_slots": 3,
        }
        if geometry != expected_geometry:
            raise ValueError(f"checkpoint geometry {geometry} != {expected_geometry}")
        resolved_device = torch.device(device)
        model = build_model(config["model_config"])
        state = load_file(str(model_dir / "model.safetensors"), device="cpu")
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False).to(device=resolved_device, dtype=dtype)
        return cls(model, config, resolved_device, dtype)

    @staticmethod
    def normalize_raw_h3(raw_h3: torch.Tensor) -> torch.Tensor:
        mean = raw_h3.new_tensor(H3_LATENTS_MEAN).view(1, -1, 1, 1, 1)
        std = raw_h3.new_tensor(H3_LATENTS_STD).view(1, -1, 1, 1, 1)
        return (raw_h3 - mean) / std

    @torch.inference_mode()
    def convert(
        self,
        h3_latent: torch.Tensor,
        *,
        pixel_frames: int,
        pixel_height: int,
        pixel_width: int,
        input_normalization: str = "normalized",
    ) -> torch.Tensor:
        if h3_latent.ndim == 4:
            h3_latent = h3_latent.unsqueeze(0)
        expected = expected_shapes(pixel_frames, pixel_height, pixel_width)
        if list(h3_latent.shape[1:]) != expected["h3"]:
            raise ValueError(f"H3 latent shape {list(h3_latent.shape[1:])} != expected {expected['h3']}")
        if input_normalization == "raw":
            h3_latent = self.normalize_raw_h3(h3_latent.float())
        elif input_normalization != "normalized":
            raise ValueError("input_normalization must be 'normalized' or 'raw'")
        aligned = align_h3_to_ltx(
            h3_latent.to(device=self.device, dtype=self.dtype),
            pixel_frames=pixel_frames,
            target_height=pixel_height // 32,
            target_width=pixel_width // 32,
            slots=3,
        )
        if list(aligned.shape[1:]) != expected["aligned_h3"]:
            raise RuntimeError(f"aligned H3 shape {list(aligned.shape[1:])} != expected {expected['aligned_h3']}")
        output = self.model(aligned)
        if list(output.shape[1:]) != expected["ltx"]:
            raise RuntimeError(f"LTX output shape {list(output.shape[1:])} != expected {expected['ltx']}")
        if not torch.isfinite(output).all():
            raise FloatingPointError("adapter produced NaN or Inf")
        return output
