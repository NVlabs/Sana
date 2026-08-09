"""Assemble `MiniMaxH3Transformer3DModel` from the pruned FP8 checkpoint.

Sibling of the model construction inside `optimized/gpu_infer.py`. The config is the released
one; what changes is that quantised linears arrive as `Fp8Linear` and the AdaLN projections are
rebuilt at their pruned width before any weight is read, so nothing is ever materialised at
BF16 and then thrown away.
"""

from __future__ import annotations

import torch
from torch import nn

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0

from adaln import patch_pruned_adaln
from fp8_linear import Fp8Linear
from fusion_install import (
    patch_fused_adaln,
    patch_fused_rope,
    patch_fused_swiglu,
    patch_sol_attn,
)
from relayout import read_pruned_fp8_checkpoint

def _set_module(root: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if name.isdigit():
        parent[int(name)] = module
    else:
        setattr(parent, name, module)


def build_pruned_fp8_transformer(
    checkpoint: str,
    config: dict,
    device: str = "cuda",
    compute_dtype: torch.dtype = torch.bfloat16,
    fuse_qkv: bool = False,
    keep_split_qkv: bool = False,
    quantizer: str = "eager",
    fuse_adaln: bool = False,
    fuse_rope: bool = False,
    fuse_swiglu: bool = False,
    sol_attn_tau: float | None = None,
):
    """Materialise `MiniMaxH3Transformer3DModel` around the pruned FP8 weights.

    The model is built on the meta device and every parameter is then *assigned* rather than
    copied into, so the 33B of BF16 the config describes is never allocated: the quantised
    linears arrive as `Fp8Linear` and the AdaLN projections are rebuilt at their pruned
    `8 -> N` shape before any load happens.
    """
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3Transformer3DModel

    num_heads = config["num_attention_heads"]
    head_dim = config["attention_head_dim"]

    read = read_pruned_fp8_checkpoint(
        checkpoint, num_heads, head_dim, fuse_qkv=fuse_qkv, keep_split_qkv=keep_split_qkv
    )
    tensors, quant, table = read["tensors"], read["quant"], read["table"]

    with init_empty_weights():
        model = MiniMaxH3Transformer3DModel.from_config(config)

    rebuilt: set[str] = set()

    # 1. The pruned AdaLN projections take 8 inputs, not `time_embed_dim`. Rebuild them at the
    #    checkpoint's shape and keep them float32: they are 39M parameters in total, so the
    #    precision is free, and their output is cast back to the block stack's dtype.
    for name, module in list(model.named_modules()):
        if not (name.endswith("adaln_proj.linear") or name == "norm_out.linear"):
            continue
        weight = tensors.get(f"{name}.weight")
        if weight is None or weight.shape[1] != table.shape[1]:
            continue
        replacement = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
        replacement.weight = nn.Parameter(weight.float(), requires_grad=False)
        replacement.bias = nn.Parameter(tensors[f"{name}.bias"].float(), requires_grad=False)
        replacement._out_dtype = compute_dtype
        _set_module(model, name, replacement)
        tensors.pop(f"{name}.weight"), tensors.pop(f"{name}.bias")
        rebuilt.add(name)

    # 2. Swap every quantised linear for its FP8 counterpart.
    for name, info in quant.items():
        spec = info.get("spec", {})
        if spec.get("format") != "float8_e4m3fn":
            raise ValueError(f"{name}: unsupported comfy_quant {spec}")
        # `full_precision_matrix_mult` means weight-only FP8: no activation scale is stored
        # and none should be invented.
        input_scale = None if spec.get("full_precision_matrix_mult") else info.get("input_scale")
        _set_module(
            model,
            name,
            Fp8Linear(
                weight=tensors.pop(f"{name}.weight"),
                weight_scale=info["weight_scale"],
                input_scale=input_scale,
                bias=tensors.pop(f"{name}.bias", None),
                compute_dtype=compute_dtype,
                quantizer=quantizer,
            ),
        )

    # 3. With a fused QKV the three separate projections the config built are dead weight —
    #    and they are still on the meta device, so they have to go before the check below.
    if fuse_qkv:
        # The two token-refiner blocks ship unquantised, so step 2 never built them an
        # `Fp8Linear`; they still need their fused projection as a plain `Linear`.
        for key in [k for k in tensors if k.endswith(".attn.to_qkv.weight")]:
            weight = tensors.pop(key)
            linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            linear.weight = nn.Parameter(weight.to(compute_dtype), requires_grad=False)
            _set_module(model, key.removesuffix(".weight"), linear)

        for module in model.modules():
            if not hasattr(module, "to_qkv"):
                continue
            module.fused_projections = True
            if not keep_split_qkv:
                # Without `keep_split_qkv` the reader never produced weights for these, so
                # they are still on the meta device and would trip the check below.
                for dead in ("to_q", "to_k", "to_v"):
                    if hasattr(module, dead):
                        delattr(module, dead)

    # 4. The timestep path becomes a table lookup.
    model._adaln_t_table = table
    patch_pruned_adaln(model)

    # 5. Everything left is a plain tensor: norms, patch projections, output heads.
    fp32 = tuple(model._keep_in_fp32_modules or ())
    remaining = {
        key: value.float() if any(p in key for p in fp32) else value.to(compute_dtype)
        for key, value in tensors.items()
    }
    missing, unexpected = model.load_state_dict(remaining, strict=False, assign=True)
    # Three groups are absent from `remaining` by construction rather than by accident: the
    # recomputed `rope.inv_freq`, the table lookup's scalar dtype carrier, and the AdaLN and
    # FP8 modules that were assigned directly above. `still_meta` below is the real check.
    handled = tuple(f"{name}." for name in rebuilt | quant.keys())
    missing = [
        k for k in missing
        if not k.startswith(("rope.", "time_embedder.")) and not k.startswith(handled)
    ]
    if unexpected:
        raise ValueError(f"checkpoint has keys the model does not: {unexpected[:8]}")
    still_meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    still_meta += [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if still_meta:
        raise ValueError(f"parameters never received weights: {still_meta[:8]}")

    fused_blocks = patch_fused_adaln(model) if fuse_adaln else 0
    if fuse_rope:
        patch_fused_rope(model)
    if fuse_swiglu:
        patch_fused_swiglu(model)
    if sol_attn_tau is not None:
        patch_sol_attn(model, tau=sol_attn_tau)

    return model.to(device).eval(), {
        "missing": missing,
        "quantized_layers": len(quant),
        "fused_qkv": fuse_qkv,
        "quantizer": quantizer,
        "fused_adaln_blocks": fused_blocks,
        "fused_rope": fuse_rope,
        "fused_swiglu": fuse_swiglu,
        "sol_attn_tau": sol_attn_tau,
    }
