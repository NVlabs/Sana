from collections.abc import Iterator

import torch

from diffusion.refiner.vendor.ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict


def _get_device(work_device: torch.device | None = None) -> torch.device:
    if work_device is not None:
        return work_device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def fuse_lora_weights(
    model_sd: StateDict,
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    dtype: torch.dtype | None = None,
    work_device: torch.device | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(key, fused_tensor)`` for each weight modified by at least one LoRA.
    For scaled-FP8 weights, this includes both the updated ``.weight`` tensor
    and its corresponding ``.weight_scale`` tensor.
    """
    target_device = _get_device(work_device)
    for key, original_weight in model_sd.sd.items():
        if original_weight is None or key.endswith(".weight_scale"):
            continue
        original_device = original_weight.device
        weight = original_weight.to(device=target_device)
        target_dtype = dtype if dtype is not None else weight.dtype
        deltas_dtype = target_dtype if target_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2] else torch.bfloat16

        deltas = _prepare_deltas(lora_sd_and_strengths, key, deltas_dtype, weight.device)
        if deltas is None:
            continue

        scale_key = key.replace(".weight", ".weight_scale") if key.endswith(".weight") else None
        is_scaled_fp8 = scale_key is not None and scale_key in model_sd.sd

        if weight.dtype == torch.float8_e4m3fn:
            if is_scaled_fp8:
                fused = _fuse_delta_with_scaled_fp8(deltas, weight, key, scale_key, model_sd)
            else:
                fused = _fuse_delta_with_cast_fp8(deltas, weight, key, target_dtype)
        elif weight.dtype == torch.bfloat16:
            fused = _fuse_delta_with_bfloat16(deltas, weight, key, target_dtype)
        else:
            raise ValueError(f"Unsupported dtype: {weight.dtype}")

        for k, v in fused.items():
            yield k, v.to(device=original_device)


def apply_loras(
    model_sd: StateDict,
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    dtype: torch.dtype | None = None,
    destination_sd: StateDict | None = None,
    work_device: torch.device | None = None,
) -> StateDict:
    if destination_sd is not None:
        sd = destination_sd.sd
        for key, tensor in fuse_lora_weights(
            model_sd,
            lora_sd_and_strengths,
            dtype=dtype,
            work_device=work_device,
        ):
            sd[key] = tensor
        return destination_sd

    fused = dict(
        fuse_lora_weights(
            model_sd,
            lora_sd_and_strengths,
            dtype=dtype,
            work_device=work_device,
        )
    )
    sd = {k: (fused[k] if k in fused else v.clone()) for k, v in model_sd.sd.items()}
    return StateDict(sd, model_sd.device, model_sd.size, model_sd.dtype)


def _prepare_deltas(
    lora_sd_and_strengths: list[LoraStateDictWithStrength], key: str, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    deltas = []
    prefix = key[: -len(".weight")]
    key_a = f"{prefix}.lora_A.weight"
    key_b = f"{prefix}.lora_B.weight"
    for lsd, coef in lora_sd_and_strengths:
        if key_a not in lsd.sd or key_b not in lsd.sd:
            continue
        a = lsd.sd[key_a].to(device=device)
        b = lsd.sd[key_b].to(device=device)
        product = torch.matmul(b * coef, a)
        del a, b
        deltas.append(product.to(dtype=dtype))
    if len(deltas) == 0:
        return None
    elif len(deltas) == 1:
        return deltas[0]
    return torch.sum(torch.stack(deltas, dim=0), dim=0)


def _fuse_delta_with_scaled_fp8(
    deltas: torch.Tensor,
    weight: torch.Tensor,
    key: str,
    scale_key: str,
    model_sd: StateDict,
) -> dict[str, torch.Tensor]:
    """Dequantize scaled FP8 weight, add LoRA delta, and re-quantize."""
    from diffusion.refiner.vendor.ltx_core.quantization.fp8_scaled_mm import (  # noqa: PLC0415
        quantize_weight_to_fp8_per_tensor,
    )

    weight_scale = model_sd.sd[scale_key]

    original_weight = weight.t().to(torch.float32) * weight_scale

    new_weight = original_weight + deltas.to(torch.float32)

    new_fp8_weight, new_weight_scale = quantize_weight_to_fp8_per_tensor(new_weight)
    return {key: new_fp8_weight, scale_key: new_weight_scale}


def _fuse_delta_with_cast_fp8(
    deltas: torch.Tensor,
    weight: torch.Tensor,
    key: str,
    target_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Fuse LoRA delta with cast-only FP8 weight (no scale factor)."""
    if str(weight.device).startswith("cuda"):
        from diffusion.refiner.vendor.ltx_core.quantization.fp8_cast import (  # noqa: PLC0415
            _fused_add_round_launch,
        )

        _fused_add_round_launch(deltas, weight, seed=0)
    else:
        deltas.add_(weight.to(dtype=deltas.dtype))
    return {key: deltas.to(dtype=target_dtype)}


def _fuse_delta_with_bfloat16(
    deltas: torch.Tensor,
    weight: torch.Tensor,
    key: str,
    target_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Fuse LoRA delta with bfloat16 weight."""
    deltas.add_(weight)
    return {key: deltas.to(dtype=target_dtype)}
