"""Bridge the diffusers `minimax-h3` branch back to transformers 4.57.

The branch was developed against transformers 5.x (Sol-Engine's `model.toml` pins 5.8.1 for
its GB200 environment). Between the two, Qwen3-VL changed how it is told which tokens are
text and which are vision:

* **5.x** has the processor produce `mm_token_type_ids` (`0` text, `1` image, `2` video) and
  passes them into `Qwen3VLModel.forward`, which lays out the 3D rotary positions per
  modality run from them.
* **4.57** has no such argument. `Qwen3VLModel.forward` calls `get_rope_index(input_ids,
  image_grid_thw, video_grid_thw, attention_mask)`, which recovers the same runs by scanning
  `input_ids` for the vision pad ids the encoder block already inserted.

The two derive the same thing from the same information, so the shim is a translation and
not a reimplementation: it restores the processor method the block calls, and drops the
argument 4.57's forward does not accept rather than letting it fall through `**kwargs` into
the decoder layers.
"""

from __future__ import annotations

import torch


def _create_mm_token_type_ids(processor, batched_ids):
    """`0` text, `1` image, `2` video — recovered from the vision pad ids, as 5.x does."""
    config = processor.tokenizer
    image_pad = config.convert_tokens_to_ids("<|image_pad|>")
    video_pad = config.convert_tokens_to_ids("<|video_pad|>")
    out = []
    for ids in batched_ids:
        out.append([2 if i == video_pad else 1 if i == image_pad else 0 for i in ids])
    return out


def patch_processor(processor):
    """Give a 4.57 `Qwen3VLProcessor` the `create_mm_token_type_ids` the block expects."""
    if hasattr(processor, "create_mm_token_type_ids"):
        return processor
    processor.create_mm_token_type_ids = lambda batched_ids: _create_mm_token_type_ids(
        processor, batched_ids
    )
    return processor


def patch_text_encoder(text_encoder):
    """Drop `mm_token_type_ids` before it reaches a forward that cannot consume it.

    4.57 recomputes the equivalent inside `get_rope_index`, so dropping it changes nothing;
    forwarding it would push an unknown keyword down into the decoder layers.
    """
    inner = text_encoder.model
    if "mm_token_type_ids" in inner.forward.__code__.co_varnames:
        return text_encoder

    original = inner.forward

    def forward(*args, mm_token_type_ids=None, **kwargs):
        return original(*args, **kwargs)

    inner.forward = forward
    return text_encoder


def assert_text_only(processor, prompt_ids) -> None:
    """The equivalence above is exact for text-only prompts; flag anything else loudly."""
    types = torch.tensor(processor.create_mm_token_type_ids([prompt_ids]))
    if bool((types != 0).any()):
        raise ValueError(
            "this compatibility shim was validated for text-only (t2va) prompts; the prompt "
            "carries vision tokens, so verify the 4.57 rope layout before trusting the output"
        )
