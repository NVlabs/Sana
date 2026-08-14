"""Prequant checkpoint discovery + load-time SDOps for NVFP4.
Loads on-disk packed NVFP4 weights and scales into ``NVFP4Linear`` without
rewriting nibble order.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import torch
from torch import nn

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from ltx_core.model.transformer import LTXModel
from ltx_core.quantization.nvfp4.types import ActScale

SafetensorsHeader = dict[str, dict[str, Any]]


def _read_safetensors_header(path: str) -> SafetensorsHeader:
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))
    return {k: v for k, v in header.items() if k != "__metadata__"}


def _discover_nvfp4_layers(header: SafetensorsHeader) -> dict[str, int]:
    """Return ``{layer_path: weight_scale_rows}`` for paired U8/F8/F32 NVFP4 tensors."""
    weight_keys = frozenset(k.removesuffix(".weight") for k in header if k.endswith(".weight"))
    layers: dict[str, int] = {}
    for layer in weight_keys:
        has_scale = f"{layer}.weight_scale" in header
        has_scale_2 = f"{layer}.weight_scale_2" in header
        if has_scale != has_scale_2:
            raise ValueError(
                f"{layer!r} has exactly one of .weight_scale/.weight_scale_2 — "
                "expected both or neither (NVFP4 checkpoints pair them 1:1:1)."
            )
        if not (has_scale and has_scale_2):
            continue
        weight_info = header[f"{layer}.weight"]
        scale_info = header[f"{layer}.weight_scale"]
        scale_2_info = header[f"{layer}.weight_scale_2"]
        if not (weight_info["dtype"] == "U8" and scale_info["dtype"] == "F8_E4M3" and scale_2_info["dtype"] == "F32"):
            continue
        layers[layer] = scale_info["shape"][0]
    return layers


def _layers_with_input_scale(header: SafetensorsHeader, layers: dict[str, int]) -> frozenset[str]:
    """Subset of ``layers`` that also have a scalar FP32 ``.input_scale``."""
    out: set[str] = set()
    for layer in layers:
        info = header.get(f"{layer}.input_scale")
        if info is not None and info["dtype"] == "F32":
            out.add(layer)
    return frozenset(out)


def _require_static_input_scales(
    header: SafetensorsHeader,
    layers: dict[str, int],
) -> None:
    with_input = _layers_with_input_scale(header, layers)
    missing = sorted(set(layers) - with_input)
    if missing:
        raise ValueError(
            f"act_scale=ActScale.STATIC requires .input_scale on every NVFP4 layer; "
            f"missing for {len(missing)} layers (e.g. {missing[0]!r})."
        )


def _discover_prequant(
    checkpoint_path: str,
    *,
    act_scale: ActScale,
) -> dict[str, int]:
    """Read the checkpoint header once; return ``{layer_path: weight_scale_rows}``.
    Every discovered NVFP4 Linear is hosted as ``NVFP4Linear``, including layers whose
    ``weight_scale`` rows are cuBLAS-padded above ``out_features`` (``NVFP4Linear``
    accepts ``weight_scale_rows`` independently of ``out_features``). There is no BF16
    dequant fallback for those; revisit only if a real checkpoint fails to load or run.
    """
    header = _read_safetensors_header(checkpoint_path)
    layers = _discover_nvfp4_layers(header)
    if not layers:
        raise ValueError(
            f"nvfp4 prequant requires a pre-quantized checkpoint with paired "
            f".weight_scale/.weight_scale_2 tensors, but {checkpoint_path!r} has none."
        )
    match act_scale:
        case ActScale.STATIC:
            _require_static_input_scales(header, layers)
        case ActScale.FIXED_1 | ActScale.DYNAMIC:
            pass
    return layers


def _matching_layer(name: str, layer_set: dict) -> str | None:
    suffix = "." + name
    return next((p for p in layer_set if p == name or p.endswith(suffix)), None)


def _layer_from_param_key(key: str, suffix: str, layer_set: dict) -> str | None:
    path = key.removesuffix(suffix)
    if path in layer_set:
        return path
    return _matching_layer(path, layer_set)


def build_prequant_sd_ops(checkpoint_path: str, *, act_scale: ActScale = ActScale.STATIC) -> SDOps:
    """Pass through packed NVFP4 weights/scales into ``NVFP4Linear`` parameters.
    When ``act_scale=ActScale.STATIC``, also loads calibrated ``.input_scale`` scalars.
    """
    nvfp4_layers = _discover_prequant(checkpoint_path, act_scale=act_scale)

    def _on_weight(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        if _layer_from_param_key(key, ".weight", nvfp4_layers) is None:
            return [KeyValueOperationResult(key, value)]
        if value.dtype != torch.uint8:
            raise ValueError(f"expected U8 NVFP4 weight for {key!r}, got {value.dtype}")
        return [KeyValueOperationResult(key, value)]

    def _on_weight_scale(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        if _layer_from_param_key(key, ".weight_scale", nvfp4_layers) is None:
            return [KeyValueOperationResult(key, value)]
        return [KeyValueOperationResult(key, value.view(torch.uint8).contiguous())]

    def _on_weight_scale_2(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        if _layer_from_param_key(key, ".weight_scale_2", nvfp4_layers) is None:
            return [KeyValueOperationResult(key, value)]
        return [KeyValueOperationResult(key, value.float().reshape(()))]

    def _on_input_scale(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        if _layer_from_param_key(key, ".input_scale", nvfp4_layers) is None:
            return [KeyValueOperationResult(key, value)]
        match act_scale:
            case ActScale.STATIC:
                return [KeyValueOperationResult(key, value.float().reshape(()))]
            case ActScale.FIXED_1 | ActScale.DYNAMIC:
                # Not consumed by the module; drop so unused-key checks stay quiet.
                return []

    return (
        SDOps("NVFP4_PREQUANT")
        .with_kv_operation(key_suffix=".input_scale", operation=_on_input_scale)
        .with_kv_operation(key_suffix=".weight_scale_2", operation=_on_weight_scale_2)
        .with_kv_operation(key_suffix=".weight_scale", operation=_on_weight_scale)
        .with_kv_operation(key_suffix=".weight", operation=_on_weight)
    )


def get_prequant_swap_module_ops(
    checkpoint_path: str,
    *,
    act_scale: ActScale = ActScale.STATIC,
) -> tuple[ModuleOps, ...]:
    """Swap discovered NVFP4 layers to ``NVFP4Linear``."""
    # Imported here so header discovery stays kernels-free.
    from ltx_core.quantization.nvfp4.linear import NVFP4Linear  # noqa: PLC0415

    nvfp4_layers = _discover_prequant(checkpoint_path, act_scale=act_scale)

    def _matching(name: str) -> str | None:
        return _matching_layer(name, nvfp4_layers)

    def _swap(model: nn.Module) -> nn.Module:
        replacements: list[tuple[nn.Module, str, nn.Linear, int]] = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear) or isinstance(module, NVFP4Linear):
                continue
            matched = _matching(name)
            if matched is None:
                continue
            if "." in name:
                parent_name, attr_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            replacements.append((parent, attr_name, module, nvfp4_layers[matched]))

        for parent, attr_name, linear, weight_scale_rows in replacements:
            setattr(
                parent,
                attr_name,
                NVFP4Linear(
                    linear.in_features,
                    linear.out_features,
                    bias=linear.bias is not None,
                    act_scale=act_scale,
                    weight_scale_rows=weight_scale_rows,
                    device=linear.weight.device,
                ),
            )
        return model

    return (
        ModuleOps(
            name="nvfp4_prequant_swap",
            matcher=lambda model: isinstance(model, LTXModel),
            mutator=_swap,
        ),
    )
