"""Adapter for SageAttention."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "quantization/sageattention"
SOURCE_URLS = ["https://github.com/thu-ml/SageAttention"]
LOCAL_ENTRYPOINTS = [
    "sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn.SageAttentionBackend",
    "sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn3.SageAttention3Backend",
]


def dependency_status() -> dict[str, bool]:
    return _dependency_status(
        "sageattention",
        "sglang.multimodal_gen.runtime.layers.attention.backends.sage_attn",
    )
