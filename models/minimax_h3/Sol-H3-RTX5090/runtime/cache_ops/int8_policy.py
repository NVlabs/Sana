#!/usr/bin/env python3
"""Native ConvRot INT8 policies for the offline Gemma and AV connector only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .int8_header import (
    Int8ConvRotLayerSpec,
    partition_dev_int8_convrot_specs,
    read_dev_int8_convrot_specs,
    read_gemma_int8_convrot_specs,
    uncompiled_module_name,
)
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from ltx_core.quantization import QuantizationPolicy
from ltx_core.text_encoders.gemma.encoders.base_encoder import LTXGemmaTextEncoder
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor


def _int8_output_dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    raise TypeError(f"unsupported INT8 Linear output dtype {dtype}")


def _comfy_kitchen() -> Any:
    try:
        import comfy_kitchen as ck
        from comfy_kitchen.tensor import TensorWiseINT8Layout
    except ImportError as exc:
        raise RuntimeError(
            "Full Dev INT8 ConvRot requires comfy-kitchen in the Stage2 Python "
            "environment; install the pinned backend before building this cache"
        ) from exc
    for attribute in ("int8_linear",):
        if not hasattr(ck, attribute):
            raise RuntimeError(f"comfy-kitchen lacks required API {attribute}")
    for attribute in ("quantize", "dequantize"):
        if not hasattr(TensorWiseINT8Layout, attribute):
            raise RuntimeError(
                f"TensorWiseINT8Layout lacks required API {attribute}"
            )
    return ck, TensorWiseINT8Layout


class DevInt8ConvRotLinear(nn.Module):
    """Stock-LTX-compatible Linear backed by Comfy row-scaled ConvRot INT8."""

    def __init__(
        self,
        spec: Int8ConvRotLayerSpec,
        *,
        bias: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.in_features = spec.weight_shape[1]
        self.out_features = spec.weight_shape[0]
        self.convrot_groupsize = spec.convrot_groupsize
        self.weight = nn.Parameter(
            torch.empty(spec.weight_shape, dtype=torch.int8, device=device),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(spec.scale_shape, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.in_features:
            raise ValueError(
                f"INT8 Linear expected K={self.in_features}, got {value.shape[-1]}"
            )
        # Use comfy-kitchen's registered opaque custom op. Calling its public
        # Python registry wrapper here makes torch.compile trace into nanobind's
        # DLPack bridge and fail on FakeTensor before the kernel can run.
        return torch.ops.comfy_kitchen.int8_linear(
            value.contiguous(),
            self.weight.contiguous(),
            self.weight_scale,
            self.bias,
            _int8_output_dtype_code(value.dtype),
            True,
            self.convrot_groupsize,
            None,
        )


def _replace_linears(
    model: nn.Module,
    specs: tuple[Int8ConvRotLayerSpec, ...],
) -> nn.Module:
    for spec in specs:
        try:
            current = model.get_submodule(spec.module_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"Full Dev INT8 checkpoint names unknown LTX module {spec.module_name!r}"
            ) from exc
        if not isinstance(current, nn.Linear):
            raise TypeError(
                f"quantized target {spec.module_name!r} is {type(current).__name__}, "
                "not torch.nn.Linear"
            )
        actual_shape = tuple(current.weight.shape)
        if actual_shape != spec.weight_shape:
            raise ValueError(
                f"{spec.module_name} checkpoint/model shape mismatch: "
                f"{spec.weight_shape} != {actual_shape}"
            )
        parent_name, _, child_name = spec.module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        dtype = current.bias.dtype if current.bias is not None else torch.bfloat16
        replacement = DevInt8ConvRotLinear(
            spec,
            bias=current.bias is not None,
            device=current.weight.device,
            dtype=dtype,
        )
        replacement.train(current.training)
        setattr(parent, child_name, replacement)
    return model


def _metadata_drop(
    _key: str, _value: torch.Tensor
) -> list[KeyValueOperationResult]:
    return []


def _scale_to_fp32(
    key: str,
    value: torch.Tensor,
    quantized_modules: frozenset[str],
) -> list[KeyValueOperationResult]:
    module_name = uncompiled_module_name(key.removesuffix(".weight_scale"))
    if module_name not in quantized_modules:
        return [KeyValueOperationResult(key, value)]
    return [KeyValueOperationResult(key, value.to(dtype=torch.float32))]


def _policy_sd_ops(
    specs: tuple[Int8ConvRotLayerSpec, ...],
    *,
    name: str,
    drop_prefixes: tuple[str, ...] = (),
) -> SDOps:
    quantized_modules = frozenset(spec.module_name for spec in specs)
    ops = SDOps(name)
    # These keys pass the stock transformer's broad model.diffusion_model
    # matcher. Drop them before the suffix handlers can retain their scales or
    # markers; the PromptEncoder loads the same tensors through its own builder.
    for prefix in drop_prefixes:
        ops = ops.with_kv_operation(key_prefix=prefix, operation=_metadata_drop)
    return (
        ops
        .with_kv_operation(key_suffix=".comfy_quant", operation=_metadata_drop)
        .with_kv_operation(
            key_suffix=".weight_scale",
            operation=lambda key, value: _scale_to_fp32(
                key, value, quantized_modules
            ),
        )
    )


def _map_specs_through_sd_ops(
    specs: tuple[Int8ConvRotLayerSpec, ...],
    stock_sd_ops: SDOps,
) -> tuple[Int8ConvRotLayerSpec, ...]:
    """Map raw checkpoint module names through the pinned Gemma key policy."""

    mapped: list[Int8ConvRotLayerSpec] = []
    seen: set[str] = set()
    for spec in specs:
        mapped_weight = stock_sd_ops.apply_to_key(f"{spec.module_name}.weight")
        if mapped_weight is None or not mapped_weight.endswith(".weight"):
            raise RuntimeError(
                f"stock Gemma SDOps rejected INT8 module {spec.module_name!r}"
            )
        module_name = mapped_weight.removesuffix(".weight")
        if module_name in seen:
            raise RuntimeError(f"stock Gemma SDOps collapsed duplicate {module_name!r}")
        seen.add(module_name)
        mapped.append(
            Int8ConvRotLayerSpec(
                module_name=module_name,
                weight_shape=spec.weight_shape,
                scale_shape=spec.scale_shape,
                convrot_groupsize=spec.convrot_groupsize,
            )
        )
    return tuple(mapped)


def _spec_fingerprint(specs: tuple[Int8ConvRotLayerSpec, ...]) -> str:
    payload = [
        {
            "module": spec.module_name,
            "weight": spec.weight_shape,
            "scale": spec.scale_shape,
            "group": spec.convrot_groupsize,
        }
        for spec in specs
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_dev_int8_convrot_embeddings_policy(
    checkpoint_path: str | Path,
) -> QuantizationPolicy:
    """Build the INT8 policy for the separate prompt embeddings processor."""

    all_specs = read_dev_int8_convrot_specs(checkpoint_path)
    _, connector_specs = partition_dev_int8_convrot_specs(all_specs)
    _comfy_kitchen()
    module_op = ModuleOps(
        name=(
            "full_dev_int8_convrot_embeddings_"
            f"{_spec_fingerprint(connector_specs)}"
        ),
        matcher=lambda model: isinstance(model, EmbeddingsProcessor),
        mutator=lambda model: _replace_linears(model, connector_specs),
    )
    return QuantizationPolicy(
        sd_ops=_policy_sd_ops(
            connector_specs,
            name="FULL_DEV_COMFY_INT8_CONVROT_EMBEDDINGS",
        ),
        module_ops=(module_op,),
    )


def build_gemma_int8_convrot_policy(
    checkpoint_path: str | Path,
    stock_sd_ops: SDOps,
) -> QuantizationPolicy:
    """Build the exact 328-Linear Gemma 4 INT8 policy for d151147."""

    raw_specs = read_gemma_int8_convrot_specs(checkpoint_path)
    specs = _map_specs_through_sd_ops(raw_specs, stock_sd_ops)
    if len(specs) != 328:
        raise RuntimeError(f"expected 328 mapped Gemma INT8 Linears, got {len(specs)}")
    _comfy_kitchen()
    module_op = ModuleOps(
        name=f"gemma4_int8_convrot_{_spec_fingerprint(specs)}",
        matcher=lambda model: isinstance(model, LTXGemmaTextEncoder),
        mutator=lambda model: _replace_linears(model, specs),
    )
    return QuantizationPolicy(
        sd_ops=_policy_sd_ops(
            specs,
            name="GEMMA4_COMFY_INT8_CONVROT",
        ),
        module_ops=(module_op,),
    )


__all__ = [
    "DevInt8ConvRotLinear",
    "build_dev_int8_convrot_embeddings_policy",
    "build_gemma_int8_convrot_policy",
]
