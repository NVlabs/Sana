"""LoRA fuse rule for NVFP4 cast / prequant policies."""

from __future__ import annotations

import torch
from ltx_kernels import nvfp4

from ltx_core.loader.fuse_loras import FuseRule, bf16_fuse_rule
from ltx_core.loader.primitives import StateDict
from ltx_core.quantization.nvfp4.convert import quantize_bf16_weight_to_nvfp4


def _nvfp4_fuse(
    key: str,
    weight: torch.Tensor,
    deltas: torch.Tensor,
    model_sd: StateDict,
) -> dict[str, torch.Tensor]:
    """Dequant NVFP4 ``.weight`` → BF16, add delta, re-quant; emit weight + scales.
    NVFP4 layers are identified by the ``.weight_scale`` + ``.weight_scale_2``
    companions. Other layers (plain BF16) fall back to ``bf16_fuse_rule``.
    ``input_scale`` is an activation scale and is left untouched.
    """
    scale_key = key.replace(".weight", ".weight_scale")
    scale_2_key = key.replace(".weight", ".weight_scale_2")
    if scale_key not in model_sd.sd or scale_2_key not in model_sd.sd:
        return bf16_fuse_rule(key, weight, deltas, model_sd)

    # ``weight`` is already on the fusion device; companions may still be CPU-retained.
    weight_scale = model_sd.sd[scale_key].to(device=weight.device)
    weight_scale_2 = model_sd.sd[scale_2_key].to(device=weight.device, dtype=torch.float32).reshape(())

    orig_device = weight.device
    packed = weight
    if packed.device.type != "cuda":
        packed = packed.to(device="cuda")
        weight_scale = weight_scale.to(device="cuda")
        weight_scale_2 = weight_scale_2.to(device="cuda")

    bf16_weight = nvfp4.dequantize_nvfp4(packed, weight_scale_2, weight_scale, out_dtype=torch.bfloat16)
    merged = bf16_weight + deltas.to(device=bf16_weight.device, dtype=torch.bfloat16)
    new_fp4, new_scale, new_scale_2 = quantize_bf16_weight_to_nvfp4(merged)
    return {
        key: new_fp4.to(device=orig_device),
        scale_key: new_scale.to(device=orig_device),
        scale_2_key: new_scale_2.to(device=orig_device),
    }


nvfp4_fuse_rule = FuseRule(aggregation_dtype=torch.bfloat16, fuse_fn=_nvfp4_fuse)
