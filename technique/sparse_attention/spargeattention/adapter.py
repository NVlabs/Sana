"""Adapter for SpargeAttention / Sage sparse-linear attention."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "sparse_attention/spargeattention"
SOURCE_URLS = ["https://github.com/thu-ml/SpargeAttn"]
LOCAL_ENTRYPOINTS = [
    "sglang.multimodal_gen.runtime.layers.attention.backends.sparse_linear_attn.SageSparseLinearAttentionBackend",
]


def dependency_status() -> dict[str, bool]:
    return _dependency_status(
        "spas_sage_attn",
        "sglang.multimodal_gen.runtime.layers.attention.backends.sparse_linear_attn",
    )
