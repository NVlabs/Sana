# Adapted from: https://github.com/bghira/SimpleTuner
# With improvements from: https://github.com/ostris/ai-toolkit
from typing import Literal

import torch
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from ltx_trainer import logger

QuantizationOptions = Literal[
    "int8-quanto",
    "int4-quanto",
    "int2-quanto",
    "fp8-quanto",
    "fp8uz-quanto",
]

# Modules to exclude from quantization.
# These are glob patterns passed to quanto's `exclude` parameter.
# When quantizing the full model at once, these patterns match against full module paths.
# When quantizing block-by-block, we also use SKIP_ROOT_MODULES for top-level modules.
EXCLUDE_PATTERNS = [
    # Input/output projection layers
    "patchify_proj",
    "audio_patchify_proj",
    "proj_out",
    "audio_proj_out",
    # Timestep embedding layers - int4 tinygemm requires strict bfloat16 input
    # and these receive float32 sinusoidal embeddings that are cast to bfloat16
    "*adaln*",
    "time_proj",
    "timestep_embedder*",
    # Caption/text projection layers
    "caption_projection*",
    "audio_caption_projection*",
    # Normalization layers (usually excluded from quantization)
    "*norm*",
]

# Top-level modules to skip entirely during block-by-block quantization.
# These are exact matches against model.named_children() names.
# (Needed because quanto's exclude patterns don't work when calling quantize() directly on a module)
SKIP_ROOT_MODULES = {
    "patchify_proj",
    "audio_patchify_proj",
    "proj_out",
    "audio_proj_out",
    "audio_caption_projection",
}


def quantize_model(
    model: torch.nn.Module,
    precision: QuantizationOptions,
    quantize_activations: bool = False,
    device: torch.device | str | None = None,
) -> torch.nn.Module:
    """
    Quantize a model using optimum-quanto.
    For large models with transformer_blocks, this function quantizes block-by-block
    on GPU then moves back to CPU, which is much faster than quantizing on CPU and
    uses less peak VRAM than loading the entire model to GPU at once.
    Args:
        model: The model to quantize.
        precision: The quantization precision (e.g. "int8-quanto", "fp8-quanto").
        quantize_activations: Whether to quantize activations in addition to weights.
        device: Device to use for quantization. If None, uses CUDA if available, else CPU.
    Returns:
        The quantized model.
    """
    from optimum.quanto import freeze, quantize  # noqa: PLC0415

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    weight_quant = _get_quanto_dtype(precision)

    if quantize_activations:
        logger.debug("Quantizing model weights and activations")
        activations_quant = weight_quant
    else:
        activations_quant = None

    # Remember original device to restore after quantization
    original_device = next(model.parameters()).device

    # Check if model has transformer_blocks for block-by-block quantization
    if hasattr(model, "transformer_blocks"):
        logger.debug("Quantizing model using block-by-block approach for memory efficiency")
        _quantize_blockwise(
            model,
            weight_quant=weight_quant,
            activations_quant=activations_quant,
            device=device,
        )
    else:
        # Fallback: quantize entire model at once
        model.to(device)
        quantize(model, weights=weight_quant, activations=activations_quant, exclude=EXCLUDE_PATTERNS)
        freeze(model)

    # Restore model to original device
    model.to(original_device)

    return model


def _quantize_blockwise(
    model: torch.nn.Module,
    weight_quant: torch.dtype,
    activations_quant: torch.dtype | None,
    device: torch.device,
) -> None:
    """Quantize a model block-by-block using optimum-quanto.
    This approach:
    1. Moves each transformer block to GPU
    2. Quantizes on GPU (fast!)
    3. Freezes the quantized weights
    4. Moves back to CPU
    This is much faster than quantizing on CPU and uses less peak VRAM
    than loading the entire model to GPU.
    """
    from optimum.quanto import freeze, quantize  # noqa: PLC0415

    original_dtype = next(model.parameters()).dtype
    transformer_blocks = list(model.transformer_blocks)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Quantizing transformer blocks", total=len(transformer_blocks))

        for block in transformer_blocks:
            # Move block to GPU
            block.to(device, dtype=original_dtype, non_blocking=True)

            # Quantize on GPU
            quantize(block, weights=weight_quant, activations=activations_quant, exclude=EXCLUDE_PATTERNS)
            freeze(block)

            # Move back to CPU to free up VRAM for next block
            block.to("cpu", non_blocking=True)

            progress.advance(task)

    # Quantize remaining non-transformer-block modules (e.g., embeddings, timestep projections)
    # Skip modules that should not be quantized (patchify_proj, proj_out, etc.)
    logger.debug("Quantizing remaining model components")

    for name, module in model.named_children():
        if name == "transformer_blocks":
            continue  # Already quantized

        if name in SKIP_ROOT_MODULES:
            logger.debug(f"Skipping quantization for module: {name}")
            continue  # Don't quantize these modules

        # Move to device, quantize, freeze, move back
        module.to(device, dtype=original_dtype, non_blocking=True)
        quantize(module, weights=weight_quant, activations=activations_quant, exclude=EXCLUDE_PATTERNS)
        freeze(module)
        module.to("cpu", non_blocking=True)


def _get_quanto_dtype(precision: QuantizationOptions) -> torch.dtype:
    """Map precision string to quanto dtype."""
    from optimum.quanto import (  # noqa: PLC0415
        qfloat8,
        qfloat8_e4m3fnuz,
        qint2,
        qint4,
        qint8,
    )

    if precision == "int2-quanto":
        return qint2
    elif precision == "int4-quanto":
        return qint4
    elif precision == "int8-quanto":
        return qint8
    elif precision in ("fp8-quanto", "fp8uz-quanto"):
        if torch.backends.mps.is_available():
            raise ValueError("FP8 quantization is not supported on MPS devices. Use int2, int4, or int8 instead.")
        if precision == "fp8-quanto":
            return qfloat8
        elif precision == "fp8uz-quanto":
            return qfloat8_e4m3fnuz

    raise ValueError(f"Invalid quantization precision: {precision}")
