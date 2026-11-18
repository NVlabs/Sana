# ruff: noqa: PLC0415

"""
8-bit Gemma text encoder loading utilities.
This module provides functionality for loading the Gemma text encoder in 8-bit precision
using bitsandbytes, which significantly reduces GPU memory usage.
Example usage:
    from ltx_trainer.gemma_8bit import load_8bit_gemma
    text_encoder = load_8bit_gemma(gemma_model_path="/path/to/gemma")
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import torch

from ltx_core.text_encoders.gemma.encoders.base_encoder import LTXGemmaTextEncoder
from ltx_core.text_encoders.gemma.tokenizer import LTXGemmaTokenizer


def load_8bit_gemma(
    gemma_model_path: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | int | None = None,
) -> LTXGemmaTextEncoder:
    """Load the Gemma text encoder in 8-bit precision using bitsandbytes.
    Only the Gemma LLM backbone is loaded here.  The embeddings processor
    (feature extractor + connectors) should be loaded separately via
    :func:`ltx_trainer.model_loader.load_embeddings_processor`.
    Args:
        gemma_model_path: Path to Gemma model directory
        dtype: Data type for non-quantized model weights
        device: Device to place the quantized model on. When ``None`` (default),
            the device is inferred from ``LOCAL_RANK`` if CUDA is available, so
            multi-process launches put each rank's encoder on its own GPU
            instead of all colliding on ``cuda:0``.
    Returns:
        LTXGemmaTextEncoder with 8-bit quantized Gemma backbone
    Raises:
        ImportError: If bitsandbytes is not installed
        FileNotFoundError: If required model files are not found
    """
    try:
        from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
    except ImportError as e:
        raise ImportError(
            "8-bit text encoder loading requires bitsandbytes. Install it with: uv pip install bitsandbytes"
        ) from e

    gemma_path = _find_gemma_subpath(gemma_model_path, "model*.safetensors")
    # Resolve via tokenizer_config.json (loaded through AutoTokenizer), not the SentencePiece
    # tokenizer.model, matching the other loaders (multigpu/gemma/loader.py, test_gemma_matches_hf).
    tokenizer_path = _find_gemma_subpath(gemma_model_path, "tokenizer_config.json")

    # Pin the entire model to a single device. `device_map="auto"` collides on cuda:0
    # in multi-process launches because every rank picks the same default device.
    device_map: str | dict[str, int | str | torch.device]
    if device is not None:
        device_map = {"": device}
    elif torch.cuda.is_available():
        device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    else:
        device_map = "auto"

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    with _suppress_accelerate_memory_warnings():
        gemma_model = AutoModelForImageTextToText.from_pretrained(
            gemma_path,
            quantization_config=quantization_config,
            dtype=torch.bfloat16,
            device_map=device_map,
            local_files_only=True,
        )

    tokenizer = LTXGemmaTokenizer(tokenizer_path, 1024)

    return LTXGemmaTextEncoder(
        tokenizer=tokenizer,
        model=gemma_model,
        dtype=dtype,
    )


def _find_gemma_subpath(root_path: str | Path, pattern: str) -> str:
    """Find a file matching a glob pattern and return its parent directory."""
    matches = list(Path(root_path).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found under {root_path}")
    return str(matches[0].parent)


@contextmanager
def _suppress_accelerate_memory_warnings() -> Generator[None, None, None]:
    """Temporarily suppress INFO warnings from accelerate about memory allocation."""
    accelerate_logger = logging.getLogger("accelerate.utils.modeling")
    old_level = accelerate_logger.level
    accelerate_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        accelerate_logger.setLevel(old_level)
