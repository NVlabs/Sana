"""Load LTXGemmaTextEncoder with Accelerate ``device_map="auto"``.
The Gemma LLM backbone is spread across available CUDA devices using
HuggingFace Accelerate's automatic device placement.
Mirrors the ``PromptEncoder`` text-encoder loading in
``ltx_pipelines.utils.blocks`` but uses ``device_map="auto"`` instead of
placing the entire model on a single GPU.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForImageTextToText

from ltx_core.text_encoders.gemma.encoders.base_encoder import (
    LTXGemmaTextEncoder,
    build_gemma_processor,
    build_gemma_tokenizer,
    resolve_gemma_processor_config,
)
from ltx_core.utils import find_matching_file

logger = logging.getLogger(__name__)


def load_gemma_with_device_map(
    gemma_root_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> LTXGemmaTextEncoder:
    """Load LTXGemmaTextEncoder with the LLM backbone spread across GPUs.
    Uses ``AutoModelForImageTextToText.from_pretrained(device_map="auto")``
    to distribute layers across available CUDA devices.
    Args:
        gemma_root_path: Path to Gemma model directory.
        dtype: Data type for model weights.
    """
    model_folder = str(find_matching_file(gemma_root_path, "model*.safetensors").parent)
    tokenizer_root, processor_root, processor_class = resolve_gemma_processor_config(gemma_root_path)

    logger.info("Loading Gemma LLM with device_map='auto'...")
    gemma_model = AutoModelForImageTextToText.from_pretrained(
        model_folder,
        dtype=dtype,
        device_map="auto",
        local_files_only=True,
    )

    tokenizer = build_gemma_tokenizer(tokenizer_root)
    processor = build_gemma_processor(processor_class, processor_root, tokenizer.tokenizer)

    return LTXGemmaTextEncoder(
        model=gemma_model,
        tokenizer=tokenizer,
        processor=processor,
        dtype=dtype,
    )
