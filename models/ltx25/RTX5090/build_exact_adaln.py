#!/usr/bin/env python3
"""Build the compact Exact AdaLN table used by the NVFP4 profile."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .exact_adaln import MODULE_PREFIXES, TABLE_FORMAT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--sigmas", default="0.909375,0.725,0.421875")
    return parser.parse_args()


def load_prefixed_state(handle, prefix: str) -> dict[str, torch.Tensor]:
    state_prefix = f"{prefix}."
    return {
        key.removeprefix(state_prefix): handle.get_tensor(key)
        for key in handle.keys()
        if key.startswith(state_prefix)
    }


def load_module(prefix: str, checkpoint: Path, device: torch.device):
    from ltx_core.model.transformer.adaln import AdaLayerNormSingle

    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(f"{prefix}.linear.weight")
        embedding_dim = int(weight.shape[1])
        coefficient = int(weight.shape[0] // embedding_dim)
        state = load_prefixed_state(handle, prefix)
    module = AdaLayerNormSingle(embedding_dim, embedding_coefficient=coefficient)
    module.load_state_dict(state, strict=True)
    return module.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16), list(weight.shape)


def compact_rows(tensor: torch.Tensor, name: str) -> torch.Tensor:
    reference = tensor[:1]
    for chunk in tensor.split(1024, dim=0):
        if not torch.equal(chunk, reference.expand_as(chunk)):
            raise RuntimeError(f"{name} is not row-exact")
    return reference.cpu().clone()


def build_module(name, prefix, checkpoint, timesteps, tokens, device):
    module, weight_shape = load_module(prefix, checkpoint, device)
    projected_rows = []
    embedded_rows = []
    for timestep in timesteps:
        values = torch.full((tokens,), float(timestep.item()), device=device)
        with torch.inference_mode():
            projected, embedded = module(values, hidden_dtype=torch.bfloat16)
        projected_rows.append(compact_rows(projected, f"{name}.projected"))
        embedded_rows.append(compact_rows(embedded, f"{name}.embedded"))
        del values, projected, embedded
    del module
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "linear_weight_shape": weight_shape,
        "projected": torch.stack(projected_rows),
        "embedded": torch.stack(embedded_rows),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    sigmas = [float(value) for value in args.sigmas.split(",")]
    scaled_timesteps = torch.tensor(sigmas, dtype=torch.float32) * 1000.0
    modules = {
        name: build_module(
            name,
            prefix,
            args.checkpoint,
            scaled_timesteps,
            args.tokens,
            device,
        )
        for name, prefix in MODULE_PREFIXES.items()
    }
    payload = {
        "format": TABLE_FORMAT,
        "source_size": args.checkpoint.stat().st_size,
        "token_count": args.tokens,
        "sigmas": sigmas,
        "scaled_timesteps": scaled_timesteps,
        "modules": modules,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"table": str(args.output), "tokens": args.tokens, "sigmas": sigmas}))


if __name__ == "__main__":
    main()
