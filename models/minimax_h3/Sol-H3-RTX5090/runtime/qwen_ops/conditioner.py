#!/usr/bin/env python3
"""Resident MiniMax-H3 conditioner backed by ComfyUI's NVFP4 loader.

This module deliberately delegates checkpoint interpretation and quantized
execution to ComfyUI.  It does not translate the packed NVFP4-AWQ weights or
reimplement the MiniMax Qwen3-VL presentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch


DEVICE = torch.device("cuda:0")
HIDDEN_SIZE = 5120
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@dataclass(frozen=True)
class MiniMaxConditioning:
    """Validated CPU handoff returned by the resident Comfy CLIP."""

    cond: torch.Tensor
    minimax_token_tags: torch.Tensor


def _device_key(value: object) -> tuple[str, int | None]:
    device = torch.device(value)
    index = device.index
    if device.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return device.type, index


def _import_comfy_sd(comfy_root: Path) -> Any:
    root = comfy_root.expanduser().resolve()
    expected_sd = root / "comfy" / "sd.py"
    if not expected_sd.is_file():
        raise FileNotFoundError(f"ComfyUI source is missing {expected_sd}")

    loaded = sys.modules.get("comfy.sd")
    if loaded is not None:
        loaded_path = Path(loaded.__file__).resolve()
        if loaded_path != expected_sd:
            raise RuntimeError(
                f"comfy.sd was already imported from {loaded_path}, expected {expected_sd}"
            )
        return loaded

    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    import comfy.sd as comfy_sd

    loaded_path = Path(comfy_sd.__file__).resolve()
    if loaded_path != expected_sd:
        raise RuntimeError(f"imported comfy.sd from {loaded_path}, expected {expected_sd}")
    return comfy_sd


def normalize_and_validate_conditioning(encoded: object) -> MiniMaxConditioning:
    """Validate the exact MiniMax-H3 conditioning ABI and normalize tags to [L]."""

    if not isinstance(encoded, dict):
        raise TypeError(f"encode_from_tokens(return_dict=True) returned {type(encoded)!r}")
    cond = encoded.get("cond")
    tags = encoded.get("minimax_token_tags")
    if not isinstance(cond, torch.Tensor):
        raise TypeError("encoded['cond'] is not a torch.Tensor")
    if not isinstance(tags, torch.Tensor):
        raise TypeError("encoded['minimax_token_tags'] is not a torch.Tensor")
    if not cond.is_floating_point():
        raise TypeError(f"cond must be floating point, got {cond.dtype}")
    if cond.ndim != 3 or cond.shape[0] != 1 or cond.shape[2] != HIDDEN_SIZE:
        raise ValueError(
            f"cond must have shape [1,L,{HIDDEN_SIZE}], got {list(cond.shape)}"
        )
    if not bool(torch.isfinite(cond).all().item()):
        raise ValueError("cond contains non-finite values")
    # Comfy intentionally returns conditioning on intermediate_device(), which
    # is CPU unless --gpu-only is active.  This process boundary is also the
    # correct IPC form for the separate resident Stage-1 process.
    cond = cond.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()

    if tags.ndim == 2 and tags.shape[0] == 1:
        tags = tags[0]
    if tags.ndim != 1 or tags.shape[0] != cond.shape[1]:
        raise ValueError(
            f"minimax_token_tags must have shape [{cond.shape[1]}], got {list(tags.shape)}"
        )
    if tags.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"minimax_token_tags must use an integer dtype, got {tags.dtype}")
    tag_values = set(int(value) for value in tags.detach().to("cpu").unique().tolist())
    if not tag_values.issubset({0, 1}):
        raise ValueError(f"minimax_token_tags contains values outside {{0,1}}: {tag_values}")
    tags = tags.detach().to(device="cpu", dtype=torch.long).contiguous()
    return MiniMaxConditioning(cond=cond, minimax_token_tags=tags)


class MiniMaxNVFP4Conditioner:
    """One resident ComfyUI MiniMax CLIP fixed to logical device cuda:0."""

    def __init__(self, checkpoint: Path, comfy_root: Path) -> None:
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if checkpoint.suffix != ".safetensors":
            raise ValueError(f"expected a .safetensors checkpoint, got {checkpoint}")
        if not torch.cuda.is_available():
            raise RuntimeError("cuda:0 is required but CUDA is unavailable")

        torch.cuda.set_device(DEVICE)
        comfy_sd = _import_comfy_sd(comfy_root)
        self.checkpoint = checkpoint
        self.comfy_root = comfy_root.expanduser().resolve()
        self.clip = comfy_sd.load_clip(
            ckpt_paths=[str(checkpoint)],
            embedding_directory=None,
            clip_type=comfy_sd.CLIPType.MINIMAX,
            model_options={
                "load_device": DEVICE,
                "offload_device": DEVICE,
                "initial_device": DEVICE,
            },
            disable_dynamic=True,
        )

        cond_stage_model = getattr(self.clip, "cond_stage_model", None)
        qwen_container = getattr(cond_stage_model, "qwen3vl_32b", None)
        transformer = getattr(qwen_container, "transformer", None)
        transformer_type = type(transformer)
        if (
            transformer_type.__module__ != "comfy.text_encoders.minimax"
            or transformer_type.__name__ != "MiniMaxQwen3VL"
        ):
            raise RuntimeError(
                "unexpected MiniMax conditioner implementation: "
                f"{transformer_type.__module__}.{transformer_type.__name__}"
            )
        self.transformer = transformer

        patcher = self.clip.patcher
        expected = _device_key(DEVICE)
        load_device = _device_key(patcher.load_device)
        offload_device = _device_key(patcher.offload_device)
        if load_device != expected or offload_device != expected:
            raise RuntimeError(
                "Comfy CLIP is not fully resident on cuda:0: "
                f"load={patcher.load_device}, offload={patcher.offload_device}"
            )

    @property
    def device_contract(self) -> dict[str, str]:
        patcher = self.clip.patcher
        return {
            "logical_device": str(DEVICE),
            "load_device": str(patcher.load_device),
            "offload_device": str(patcher.offload_device),
            "initial_device": str(DEVICE),
            "disable_dynamic": "true",
        }

    @property
    def loader_identity(self) -> dict[str, str]:
        transformer_type = type(self.transformer)
        return {
            "clip": f"{type(self.clip).__module__}.{type(self.clip).__name__}",
            "transformer": (
                f"{transformer_type.__module__}.{transformer_type.__name__}"
            ),
        }

    def residency_summary(self) -> dict[str, object]:
        parameter_devices: dict[str, int] = {}
        parameter_count = 0
        meta_parameters: list[str] = []
        non_cuda_parameters: list[str] = []
        for name, parameter in self.transformer.named_parameters():
            parameter_count += 1
            device = parameter.device
            device_name = str(device)
            parameter_devices[device_name] = parameter_devices.get(device_name, 0) + 1
            if device.type == "meta":
                meta_parameters.append(name)
            elif device.type != "cuda" or device.index not in (None, 0):
                non_cuda_parameters.append(name)
        if parameter_count == 0:
            raise RuntimeError("MiniMaxQwen3VL exposes no parameters")
        if meta_parameters or non_cuda_parameters:
            raise RuntimeError(
                "MiniMaxQwen3VL is not fully resident on cuda:0: "
                f"meta={meta_parameters[:8]}, non_cuda={non_cuda_parameters[:8]}"
            )
        return {
            "parameter_count": parameter_count,
            "parameter_devices": parameter_devices,
            "all_parameters_cuda0_non_meta": True,
        }
