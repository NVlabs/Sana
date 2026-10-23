# SPDX-License-Identifier: Apache-2.0
"""Exact external PEFT LoRA loading for meta-initialized MiniMax-H3 models.

The normal PEFT state-dict loader copies into existing parameters. That is a
no-op for a meta tensor, even though every checkpoint key can match. This
module builds the PEFT topology without passing the external weight to that
loader, then assigns the validated LoRA leaves when the topology is meta or
copies them when it is already materialized.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dev.yanzuolu.common.config import CfgNode
from dev.yanzuolu.common.logging import get_logger
from dev.yanzuolu.common.model.adapters import PeftLoraAdapter
from dev.yanzuolu.common.model.weights import load_state_dict_file

logger = get_logger()

_EXPORTED_LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")
_LIVE_LORA_PATTERN = re.compile(r"\.lora_[AB]\.default\.weight$")


def live_lora_leaves(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return every live default-adapter A/B leaf by exact state-dict FQN."""
    leaves = {
        key: tensor
        for key, tensor in model.state_dict().items()
        if _LIVE_LORA_PATTERN.search(key)
    }
    assert leaves, "MiniMax-H3 adapter topology exposes no live LoRA A/B leaves"
    return leaves


def load_external_lora_exact(
    model: nn.Module,
    path: Path,
    *,
    assign: bool,
) -> dict[str, torch.Tensor]:
    """Validate and load one complete default-adapter LoRA in FP32."""
    assert path.suffix in {
        ".safetensor",
        ".safetensors",
    }, f"external LoRA must be safetensors, got {path}"
    exported = load_state_dict_file(path)
    assert exported, f"external LoRA is empty: {path}"

    invalid = sorted(
        key for key in exported if not key.endswith(_EXPORTED_LORA_SUFFIXES)
    )
    assert not invalid, (
        "external LoRA checkpoint must contain only LoRA A/B tensors; "
        f"first invalid keys: {invalid[:3]}"
    )
    mapped: dict[str, torch.Tensor] = {}
    for key, tensor in exported.items():
        live_key = f"{key.removesuffix('.weight')}.default.weight"
        assert live_key not in mapped, f"external LoRA key collision at {live_key}"
        assert (
            isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
        ), f"external LoRA tensor {key} must be floating point"
        mapped[live_key] = tensor.to(device="cpu", dtype=torch.float32)

    live = live_lora_leaves(model)
    assert set(mapped) == set(live), (
        "external and live LoRA keys differ; "
        f"missing external: {sorted(set(live) - set(mapped))[:3]}, "
        f"unknown external: {sorted(set(mapped) - set(live))[:3]}"
    )
    shape_mismatches = [
        (key, tuple(mapped[key].shape), tuple(live[key].shape))
        for key in sorted(live)
        if mapped[key].shape != live[key].shape
    ]
    assert not shape_mismatches, (
        "external and live LoRA shapes differ; first "
        f"(key, external, live): {shape_mismatches[:3]}"
    )

    result = model.load_state_dict(mapped, strict=False, assign=assign)
    assert (
        not result.unexpected_keys
    ), f"external LoRA produced unexpected live keys: {result.unexpected_keys[:3]}"

    loaded = live_lora_leaves(model)
    invalid_loaded = sorted(
        key
        for key, tensor in loaded.items()
        if tensor.is_meta or tensor.dtype != torch.float32
    )
    assert not invalid_loaded, (
        "external LoRA did not materialize as FP32; " f"first: {invalid_loaded[:3]}"
    )
    return mapped


class MiniMaxH3ExternalPeftLoraAdapter:
    """Build PEFT and exactly load an external LoRA on real or meta storage."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        adapter_config = CfgNode(copy.deepcopy(dict(self.config)))
        assert adapter_config.get(
            "weight", None
        ), "MiniMaxH3ExternalPeftLoraAdapter requires adapter.weight"
        weight_path = Path(adapter_config.pop("weight"))
        state = PeftLoraAdapter(adapter_config)(state)

        model = state["model"]
        assert isinstance(model, nn.Module)
        leaves = live_lora_leaves(model)
        meta = [key for key, tensor in leaves.items() if tensor.is_meta]
        assert len(meta) in {0, len(leaves)}, (
            "live LoRA A/B leaves are partially meta; refusing mixed loading: "
            f"{len(meta)}/{len(leaves)} meta"
        )
        assign = bool(meta) or any(
            tensor.dtype != torch.float32 for tensor in leaves.values()
        )
        load_external_lora_exact(model, weight_path, assign=assign)
        logger.info(
            "[%s] %s %d external LoRA tensors from %s",
            state["name"],
            "assigned" if assign else "loaded",
            len(leaves),
            weight_path,
        )
        return state


__all__ = [
    "MiniMaxH3ExternalPeftLoraAdapter",
    "live_lora_leaves",
    "load_external_lora_exact",
]
