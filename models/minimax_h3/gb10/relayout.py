"""ComfyUI's `pruned_fp8_scaled` key layout, and the reader that resolves it.

Split out of the single-file port to match `optimized/relayout.py` next door. Three things
stand between the released file and `MiniMaxH3Transformer3DModel`, and all three live here:
the rename table, the per-head QKV question, and the SwiGLU half-swap. The rename and the
transforms are lifted from `scripts/convert_minimax_h3_to_diffusers.py`, which is the authority
on the mapping.

One correction against that script is load-bearing. It de-interleaves per-head QKV because the
*raw MiniMax shards* need it; the ComfyUI file has already been converted, so applying it again
scrambles attention. `checks/validate_layout.py` settles it against the released block 0:
contiguous thirds score 0.999649, de-interleaved 0.034. `reorder_interleaved_qkv` is kept
because it is the authority for the raw layout, and is deliberately not called on this file.
The SwiGLU swap *is* needed — 0.9996 against 0.0002.
"""

from __future__ import annotations

import json
import re

import torch
from safetensors.torch import safe_open

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0

_RENAMES = (
    ("time_embedder.proj_in.", "time_embedder.linear_1."),
    ("time_embedder.proj_out.", "time_embedder.linear_2."),
    ("video_patch_proj.", "proj_in."),
    ("audio_patch_proj.", "audio_proj_in."),
    ("condition_proj.", "context_embedder."),
    ("final_layer.norm.", "norm_out.norm."),
    ("final_layer.adaln_proj.linear.", "norm_out.linear."),
    ("final_layer.video_out.", "proj_out."),
    ("final_layer.audio_out.", "audio_proj_out."),
    (".attn.q_norm.", ".attn.norm_q."),
    (".attn.k_norm.", ".attn.norm_k."),
    (".attn.out_proj.", ".attn.to_out.0."),
)

# `rope.inv_freq` is recomputed by the port from `rope_theta` / `rope_freq_dim`, bit-for-bit.
_DROPPED = ("rope.inv_freq",)


def rename_key(key: str) -> str:
    """Original -> diffusers, for everything that is a pure rename."""
    if key.startswith("token_refiner.blocks."):
        key = key.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif key.startswith("blocks."):
        key = key.replace("blocks.", "transformer_blocks.", 1)
    for src, dst in _RENAMES:
        key = key.replace(src, dst)
    return key


def reorder_interleaved_qkv(weight: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """`[head0: q k v, head1: q k v, ...]` -> `[q_all; k_all; v_all]`. No transpose."""
    expected = num_heads * 3 * head_dim
    if weight.shape[0] != expected:
        raise ValueError(f"fused qkv has {weight.shape[0]} rows, expected {expected}")
    grouped = weight.reshape(num_heads, 3 * head_dim, *weight.shape[1:])
    q, k, v = grouped.split(head_dim, dim=1)
    return torch.cat(
        [t.reshape(num_heads * head_dim, *weight.shape[1:]) for t in (q, k, v)], dim=0
    )


def _fp8_safe(fn, tensor: torch.Tensor):
    """Run a pure-movement op on an FP8 tensor by borrowing uint8's kernels where needed.

    `reshape` / `split` / `cat` are all defined on `float8_e4m3fn` only patchily, and the
    ones that are missing are exactly the ones this module needs. Since every op here moves
    bytes without reading them, running on a `uint8` view is equivalent.
    """
    if tensor.dtype is not FP8_DTYPE:
        return fn(tensor)
    result = fn(tensor.view(torch.uint8))
    if isinstance(result, (tuple, list)):
        return type(result)(part.view(FP8_DTYPE) for part in result)
    return result.view(FP8_DTYPE)


def _swap_swiglu_halves(weight: torch.Tensor) -> torch.Tensor:
    """`[gate; value]` (reference) -> `[value; gate]` (diffusers' `SwiGLU`).

    Unlike the QKV rows, this one *is* needed: the checkpoint keeps the reference's fused
    order here, and swapping scores 0.9996 against the released block 0 where leaving it
    alone scores 0.0002.
    """
    gate, value = weight.chunk(2, dim=0)
    return torch.cat([value, gate], dim=0)



_QUANT_SUFFIXES = (".weight_scale", ".input_scale", ".comfy_quant")


def read_pruned_fp8_checkpoint(
    path: str, num_heads: int, head_dim: int, fuse_qkv: bool = False,
    keep_split_qkv: bool = False,
) -> dict:
    """Read the ComfyUI file into diffusers-keyed plain tensors plus a per-layer quant spec.

    Returns `{"tensors": {diffusers_key: tensor}, "quant": {module_path: spec}, "table": ...}`,
    where `spec` carries the FP8 weight, its scales and the decoded `comfy_quant` JSON.
    """
    tensors: dict[str, torch.Tensor] = {}
    quant: dict[str, dict] = {}
    table: torch.Tensor | None = None

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

        # 1. Collect the quantisation sidecars first, keyed by their *original* module path.
        sidecar: dict[str, dict] = {}
        for key in keys:
            for suffix in _QUANT_SUFFIXES:
                if key.endswith(suffix):
                    sidecar.setdefault(key[: -len(suffix)], {})[suffix[1:]] = f.get_tensor(key)
                    break

        for module, entry in sidecar.items():
            blob = entry.get("comfy_quant")
            entry["spec"] = json.loads(bytes(blob.tolist()).decode()) if blob is not None else {}

        # 2. Walk the real tensors.
        for key in keys:
            if key.endswith(_QUANT_SUFFIXES) or key in _DROPPED:
                continue
            if key == "adaln_t_table":
                table = f.get_tensor(key)
                continue

            tensor = f.get_tensor(key)
            module = key.rsplit(".", 1)[0]
            info = sidecar.get(module)

            if key.endswith(".attn.qkv_proj.weight"):
                # Contiguous thirds, *not* a per-head de-interleave. ComfyUI quantised the
                # reference model's in-memory state dict, which is already `[q_all; k_all;
                # v_all]` — only the raw MiniMax shards are per-head interleaved, and
                # normalising this file as if it were one scrambles attention into a flat
                # grey field. Measured against the released block 0: contiguous thirds give
                # cosine 0.9996 (the FP8 E4M3 noise floor), de-interleaving gives 0.03.
                prefix = rename_key(key).removesuffix("qkv_proj.weight")
                inner = num_heads * head_dim
                if fuse_qkv:
                    # `[q_all; k_all; v_all]` is what `MiniMaxH3AttnProcessor` expects behind
                    # `attn.fused_projections`, where it chunks the output in three. One GEMM
                    # instead of three, and the shared input is quantised once instead of
                    # three times.
                    tensors[f"{prefix}to_qkv.weight"] = tensor
                    if info is not None:
                        quant[f"{prefix}to_qkv"] = info
                # `keep_split_qkv` carries both forms so `--sweep` can toggle
                # `fused_projections` in one process — the only way to measure the baseline
                # against everything else without crossing a process boundary. It is not free:
                # the row-slices are contiguous views of the same storage, but `model.to(device)`
                # copies each buffer separately, so the DiT goes from 23.3 to 29.1 GiB. On a
                # shared 119 GB machine with no swap that margin is the difference between
                # running and being OOM-killed, so anything that is not the sweep leaves it off.
                if fuse_qkv and not keep_split_qkv:
                    continue
                parts = _fp8_safe(lambda t: t.split(inner, dim=0), tensor)
                for name, part in zip(("to_q", "to_k", "to_v"), parts):
                    target = f"{prefix}{name}"
                    tensors[f"{target}.weight"] = part
                    if info is not None:
                        quant[target] = info
                continue

            if key.endswith(".mlp.fc1.weight"):
                swapped = _fp8_safe(_swap_swiglu_halves, tensor)
                target = rename_key(key).removesuffix(".weight").replace(
                    ".mlp.fc1", ".ff.net.0.proj"
                )
                tensors[f"{target}.weight"] = swapped.contiguous()
                if info is not None:
                    quant[target] = info
                continue

            target_key = rename_key(key).replace(".mlp.fc2.", ".ff.net.2.")
            tensors[target_key] = tensor
            if info is not None and target_key.endswith(".weight"):
                quant[target_key.removesuffix(".weight")] = info

    if table is None:
        raise ValueError(f"{path} has no `adaln_t_table`; this is not a pruned checkpoint")
    return {"tensors": tensors, "quant": quant, "table": table}


