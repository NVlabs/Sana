"""Fuse MiniMax-H3 inference adapters into the base transformer.

The loader applies the adapter directly to the resident transformer, avoiding
a PEFT runtime dependency after startup.

FastVideo's ``fastvideo-lora-v2`` checkpoints are hybrid adapters: low-rank
updates use ``W += B @ A`` and selected parameters additionally carry tiny
``.diff``/``.diff_b`` corrections. Those corrections must be accumulated in
float32 or BF16 rounding can erase them. Replacement-weight checkpoints are
rejected; this delivery accepts the dense-datafree adapter only.

Checkpoint conventions and defaults follow:
https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/inference_minimax_h3.py
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


LORA_SUFFIXES = {
    ".lora_A.default.weight": "A",
    ".lora_B.default.weight": "B",
    ".lora_A.weight": "A",
    ".lora_B.weight": "B",
}
LORA_TARGET_MODULES = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)


@dataclass(frozen=True)
class FuseReport:
    path: str
    format: str
    pairs: int
    diffs: int
    replacements: int
    rank: int
    alpha: int
    scale: float
    effective_scale: float
    device: str
    fuse_s: float


def _payload(
    keys: list[str], path: Path, *, hybrid: bool
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    a_keys, b_keys, diffs, replacements, unsupported = {}, {}, {}, [], []
    for key in keys:
        matched = False
        for suffix, side in LORA_SUFFIXES.items():
            if key.endswith(suffix):
                module_name = key[: -len(suffix)].removeprefix("diffusion_model.")
                (a_keys if side == "A" else b_keys)[module_name] = key
                matched = True
                break
        if matched:
            continue
        if hybrid and key.endswith(".diff_b"):
            param_name = key[:-len(".diff_b")].removeprefix("diffusion_model.") + ".bias"
            diffs[param_name] = (key, "bias")
        elif hybrid and key.endswith(".diff"):
            param_name = key[:-len(".diff")].removeprefix("diffusion_model.") + ".weight"
            diffs[param_name] = (key, "weight")
        elif key.endswith(".set_weight"):
            replacements.append(key)
        else:
            unsupported.append(key)
    if replacements:
        raise ValueError(
            f"{path} contains {len(replacements)} replacement tensors; "
            "this runtime requires the dense-datafree adapter"
        )
    if unsupported:
        raise ValueError(
            f"{path} contains unsupported adapter keys: "
            f"{unsupported[:3]}"
        )
    if not a_keys:
        raise ValueError(f"No LoRA A tensors found in {path}")
    missing_a = sorted(b_keys.keys() - a_keys.keys())
    missing_b = sorted(a_keys.keys() - b_keys.keys())
    if missing_a or missing_b:
        raise ValueError(
            f"Unpaired LoRA tensors: missing A={missing_a[:3]}, "
            f"missing B={missing_b[:3]}"
        )
    pairs = {name: (a_keys[name], b_keys[name]) for name in sorted(a_keys)}
    return pairs, {name: diffs[name] for name in sorted(diffs)}


@torch.no_grad()
def fuse_lora(
    transformer: torch.nn.Module,
    path: str | Path,
    *,
    alpha: int = 8,
    scale: float = 1.0,
) -> FuseReport:
    """Validate and fuse a PEFT LoRA or FastVideo v2 hybrid adapter in-place."""
    started = time.perf_counter()
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint does not exist: {path}")
    if alpha < 1:
        raise ValueError("LoRA alpha must be at least 1")
    if not torch.isfinite(torch.tensor(scale)) or scale < 0:
        raise ValueError("LoRA scale must be finite and non-negative")

    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        adapter_format = metadata.get("format", "peft-lora")
        hybrid = adapter_format == "fastvideo-lora-v2"
        pairs, diffs = _payload(list(checkpoint.keys()), path, hybrid=hybrid)
        if hybrid:
            expected_low_rank = int(metadata.get("low_rank_tensors", len(pairs) * 2))
            expected_diffs = int(metadata.get("diff_tensors", len(diffs)))
            expected_replacements = int(metadata.get("set_weight_tensors", 0))
            if expected_low_rank != len(pairs) * 2 or expected_diffs != len(diffs):
                raise ValueError(
                    "FastVideo adapter metadata/payload mismatch: "
                    f"low_rank={expected_low_rank}/{len(pairs) * 2}, "
                    f"diffs={expected_diffs}/{len(diffs)}"
                )
            if expected_replacements:
                raise ValueError(
                    f"FastVideo metadata declares {expected_replacements} replacement tensors"
                )
        validated = []
        validated_diffs = []
        ranks = set()
        # Validate the complete checkpoint before modifying any base weight.
        for module_name, (a_key, b_key) in pairs.items():
            if not hybrid and not module_name.endswith(LORA_TARGET_MODULES):
                raise ValueError(f"Unsupported LoRA target module: {module_name}")
            try:
                module = transformer.get_submodule(module_name)
            except AttributeError as error:
                raise ValueError(f"LoRA target is absent from transformer: {module_name}") from error
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(
                    f"LoRA target {module_name} is {type(module).__name__}, expected Linear"
                )
            a_shape = checkpoint.get_slice(a_key).get_shape()
            b_shape = checkpoint.get_slice(b_key).get_shape()
            if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != b_shape[1]:
                raise ValueError(
                    f"Invalid LoRA pair for {module_name}: A{a_shape}, B{b_shape}"
                )
            if tuple(module.weight.shape) != (b_shape[0], a_shape[1]):
                raise ValueError(
                    f"LoRA/base mismatch for {module_name}: base{tuple(module.weight.shape)}, "
                    f"A{a_shape}, B{b_shape}"
                )
            ranks.add(a_shape[0])
            validated.append((module, a_key, b_key))
        for param_name, (diff_key, _) in diffs.items():
            try:
                parameter = transformer.get_parameter(param_name)
            except AttributeError as error:
                raise ValueError(f"Adapter diff target is absent: {param_name}") from error
            diff_shape = checkpoint.get_slice(diff_key).get_shape()
            if tuple(parameter.shape) != tuple(diff_shape):
                raise ValueError(
                    f"Adapter diff/base mismatch for {param_name}: "
                    f"base{tuple(parameter.shape)}, diff{diff_shape}"
                )
            validated_diffs.append((parameter, diff_key))
        if len(ranks) != 1:
            raise ValueError(f"Mixed LoRA ranks are unsupported: {sorted(ranks)}")

        devices = {module.weight.device for module, _, _ in validated}
        devices.update(parameter.device for parameter, _ in validated_diffs)
        if len(devices) != 1:
            raise RuntimeError(f"LoRA targets span multiple devices: {sorted(map(str, devices))}")
        device = devices.pop()
        if device.type != "cuda":
            raise RuntimeError(
                "Refusing to fuse the adapter on CPU: the low-rank matrix products "
                "would make service startup impractically slow. Move the "
                "transformer to its rank-local CUDA device before calling fuse_lora()."
            )

        rank = ranks.pop()
        if hybrid:
            metadata_rank = int(metadata.get("rank", rank))
            if metadata_rank != rank:
                raise ValueError(
                    f"FastVideo metadata rank {metadata_rank} does not match tensor rank {rank}"
                )
            applied_alpha = rank
        else:
            applied_alpha = int(alpha)
        multiplier = float(scale) * applied_alpha / rank
        for module, a_key, b_key in validated:
            device, dtype = module.weight.device, module.weight.dtype
            a = checkpoint.get_tensor(a_key).to(device=device, dtype=dtype)
            b = checkpoint.get_tensor(b_key).to(device=device, dtype=dtype)
            module.weight.addmm_(b, a, beta=1.0, alpha=multiplier)
            del a, b
        for parameter, diff_key in validated_diffs:
            delta = checkpoint.get_tensor(diff_key).to(device=parameter.device)
            merged = parameter.float().add_(delta.float(), alpha=float(scale))
            parameter.copy_(merged.to(dtype=parameter.dtype))
            del delta, merged

    gc.collect()
    torch.cuda.empty_cache()
    transformer.requires_grad_(False)
    transformer.eval()
    return FuseReport(
        path=str(path),
        format=adapter_format,
        pairs=len(pairs),
        diffs=len(diffs),
        replacements=0,
        rank=rank,
        alpha=applied_alpha,
        scale=float(scale),
        effective_scale=multiplier,
        device=str(device),
        fuse_s=time.perf_counter() - started,
    )
