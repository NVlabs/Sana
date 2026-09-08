"""Install the mixed-precision compute profile on MiniMax-H3 transformer blocks."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch


FIRST_QUANTIZED_BLOCK = 2
LAST_QUANTIZED_BLOCK = 46
COMPUTE_QUANT_MODES = {"none", "mxfp8"}


@dataclass(frozen=True)
class ComputeQuantizationReport:
    mode: str
    first_block: int
    last_block: int
    quantized_blocks: int
    quantized_linears: int
    original_bytes: int
    quantized_bytes: int


def validate_platform() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MXFP8 compute quantization requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        raise RuntimeError(
            "MXFP8 compute quantization is enabled only for the validated "
            f"SM100-family path; detected SM{capability[0]}{capability[1]}"
        )
    required = ("ScalingType", "SwizzleType", "scaled_mm")
    missing = [name for name in required if not hasattr(torch.nn.functional, name)]
    if not hasattr(torch, "float8_e8m0fnu"):
        missing.append("torch.float8_e8m0fnu")
    if missing:
        raise RuntimeError(
            "the installed PyTorch build lacks MXFP8 support: " + ", ".join(missing)
        )


def _make_fused_qkv_source(attention) -> torch.nn.Linear:
    projections = (attention.to_q, attention.to_k, attention.to_v)
    if any(projection.bias is not None for projection in projections):
        raise ValueError("MiniMax-H3 attention projections must be bias-free")
    source = torch.nn.Linear(
        projections[0].in_features,
        sum(projection.out_features for projection in projections),
        bias=False,
        device="meta",
    )
    source.weight = torch.nn.Parameter(
        torch.cat([projection.weight for projection in projections], dim=0),
        requires_grad=False,
    )
    return source


def _replace(parent, attribute: str | int, replacement: torch.nn.Module) -> None:
    if isinstance(attribute, int):
        parent[attribute] = replacement
    else:
        setattr(parent, attribute, replacement)


def install(
    transformer,
    *,
    first_block: int = FIRST_QUANTIZED_BLOCK,
    last_block: int = LAST_QUANTIZED_BLOCK,
) -> ComputeQuantizationReport:
    """Replace attention and FFN linears in inclusive ``[first_block, last_block]``.

    Boundary blocks, AdaLN projections, refiners, VAE, and text/audio encoders stay
    in BF16.  Installation is intentionally one-way for a resident worker: keeping
    BF16 rollback tensors would defeat the memory reduction.
    """
    validate_platform()
    blocks = transformer.transformer_blocks
    if not (0 <= first_block <= last_block < len(blocks)):
        raise ValueError(
            f"invalid MXFP8 block range [{first_block}, {last_block}] for "
            f"{len(blocks)} transformer blocks"
        )

    from .mxfp8 import MXFP8Linear

    original_bytes = 0
    quantized_bytes = 0
    quantized_linears = 0

    for index in range(first_block, last_block + 1):
        block = blocks[index]
        attention = block.attn

        if getattr(attention, "fused_projections", False):
            qkv_source = attention.to_qkv
        else:
            qkv_source = _make_fused_qkv_source(attention)
        qkv = MXFP8Linear(qkv_source)
        original_bytes += qkv_source.weight.numel() * qkv_source.weight.element_size()
        quantized_bytes += qkv.storage_bytes
        quantized_linears += 1
        attention.to_qkv = qkv
        attention.fused_projections = True
        for name in ("to_q", "to_k", "to_v"):
            if hasattr(attention, name):
                delattr(attention, name)
        del qkv_source, qkv

        candidates = (
            (attention.to_out, 0),
            (block.ff.net[0], "proj"),
            (block.ff.net, 2),
        )
        for parent, attribute in candidates:
            source = (
                parent[attribute]
                if isinstance(attribute, int)
                else getattr(parent, attribute)
            )
            quantized = MXFP8Linear(source)
            original_bytes += source.weight.numel() * source.weight.element_size()
            quantized_bytes += quantized.storage_bytes
            quantized_linears += 1
            _replace(parent, attribute, quantized)
            del source, quantized

    gc.collect()
    torch.cuda.empty_cache()
    return ComputeQuantizationReport(
        mode="mxfp8",
        first_block=first_block,
        last_block=last_block,
        quantized_blocks=last_block - first_block + 1,
        quantized_linears=quantized_linears,
        original_bytes=original_bytes,
        quantized_bytes=quantized_bytes,
    )


__all__ = [
    "COMPUTE_QUANT_MODES",
    "ComputeQuantizationReport",
    "FIRST_QUANTIZED_BLOCK",
    "LAST_QUANTIZED_BLOCK",
    "install",
    "validate_platform",
]
