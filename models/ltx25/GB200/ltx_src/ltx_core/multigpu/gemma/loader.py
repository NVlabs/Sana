"""Load LTXGemmaTextEncoder with Accelerate ``device_map="auto"``.
The Gemma LLM backbone is spread across available CUDA devices using
HuggingFace Accelerate's automatic device placement.
Mirrors the ``PromptEncoder`` text-encoder loading in
``ltx_pipelines.utils.blocks`` but uses ``device_map="auto"`` instead of
placing the entire model on a single GPU.
"""

from __future__ import annotations

import logging

import safetensors
import torch
from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from accelerate.utils import get_balanced_memory
from transformers import AutoModelForImageTextToText

from ltx_core.text_encoders.gemma.encoders.base_encoder import LTXGemmaTextEncoder, build_gemma_tokenizer
from ltx_core.text_encoders.gemma.encoders.encoder_configurator import get_gemma_ops
from ltx_core.text_encoders.gemma.gemma_assets import (
    GemmaAssets,
    build_gemma_hf_config,
    build_gemma_processor,
    is_safetensors_file,
)
from ltx_core.utils import find_matching_file

logger = logging.getLogger(__name__)

# LTXGemmaTextEncoder nests the HuggingFace model at ``.model``, so the shared SDOps emit
# keys one level deeper than the raw HF module tree this loader builds.
_LTX_ENCODER_PREFIX = "model."


def load_gemma_with_device_map(
    gemma_root_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> LTXGemmaTextEncoder:
    """Load LTXGemmaTextEncoder with the LLM backbone spread across GPUs.
    Accepts a directory ``gemma_root`` (``from_pretrained(device_map="auto")``) or a packed
    single-file TE (meta init + SDOps load + balanced ``device_map`` via
    :func:`accelerate.utils.get_balanced_memory`).
    """
    assets = GemmaAssets.load(gemma_root_path)

    if is_safetensors_file(gemma_root_path):
        gemma_model = _load_single_file_with_device_map(assets, dtype)
    else:
        model_folder = str(find_matching_file(gemma_root_path, "model*.safetensors").parent)
        logger.info("Loading Gemma LLM with device_map='auto'...")
        gemma_model = AutoModelForImageTextToText.from_pretrained(
            model_folder,
            dtype=dtype,
            device_map="auto",
            local_files_only=True,
        )

    tokenizer = build_gemma_tokenizer(assets)
    processor = build_gemma_processor(assets, tokenizer.tokenizer)

    return LTXGemmaTextEncoder(
        model=gemma_model,
        tokenizer=tokenizer,
        processor=processor,
        dtype=dtype,
    )


def _hf_state_dict_from_single_file(assets: GemmaAssets, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Remap TE checkpoint keys onto the raw HF module tree (strip LTX wrapper prefix)."""
    sd_ops, _ = get_gemma_ops(assets.source)
    state_dict: dict[str, torch.Tensor] = {}
    for weight_path in assets.weight_paths:
        with safetensors.safe_open(weight_path, framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 -- safe_open is not a Mapping; keys() is the only iterator
                mapped = sd_ops.apply_to_key(key)
                if mapped is None:
                    continue
                for result in sd_ops.apply_to_key_value(mapped, f.get_tensor(key).to(dtype)):
                    if not result.new_key.startswith(_LTX_ENCODER_PREFIX):
                        raise ValueError(f"SDOps produced key {result.new_key!r} outside the encoder module tree.")
                    state_dict[result.new_key.removeprefix(_LTX_ENCODER_PREFIX)] = result.new_value
    if not state_dict:
        raise ValueError(
            f"No Gemma weights matched the text-encoder SDOps in {assets.source}. "
            f"Is this a split text-encoder checkpoint?"
        )
    return state_dict


def _load_single_file_with_device_map(assets: GemmaAssets, dtype: torch.dtype) -> torch.nn.Module:
    config = build_gemma_hf_config(assets)

    # include_buffers=False (default): keep rotary inv_freq etc. from __init__, not meta.
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(config)

    logger.info("Loading single-file Gemma TE from %s and dispatching across GPUs...", assets.source)
    state_dict = _hf_state_dict_from_single_file(assets, dtype)
    model.load_state_dict(state_dict, strict=True, assign=True)
    # After from_config under init_empty_weights, find_tied_parameters is empty; without this,
    # infer_auto_device_map can put lm_head and embed_tokens on different GPUs.
    model.tie_weights()

    # Match from_pretrained(device_map="auto"): bare infer_auto_device_map packs onto GPU 0.
    # Coerce to list — HF stores _no_split_modules as a set; Accelerate wraps non-list as [value].
    no_split = list(getattr(model, "_no_split_modules", None) or ())
    max_memory = get_balanced_memory(
        model,
        dtype=dtype,
        no_split_module_classes=no_split,
    )
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        dtype=dtype,
        no_split_module_classes=no_split,
    )
    return dispatch_model(model, device_map)
