"""Adapter for PISA / piecewise sparse attention."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "sparse_attention/pisa"
SOURCE_URLS = ["https://github.com/xie-lab-ml/piecewise-sparse-attention"]
LOCAL_ENTRYPOINTS = [
    "sglang.multimodal_gen.runtime.efficiency.transforms.sparse_attention.SparseAttention",
    "sglang.multimodal_gen.runtime.layers.attention.backends.piecewise_attn.PiecewiseAttentionBackend",
]


def dependency_status() -> dict[str, bool]:
    return _dependency_status(
        "sglang.multimodal_gen.runtime.efficiency.transforms.sparse_attention",
        "sglang.multimodal_gen.runtime.layers.attention.backends.piecewise_attn",
    )


def get_transform_class():
    from sglang.multimodal_gen.runtime.efficiency.transforms.sparse_attention import (
        SparseAttention,
    )

    return SparseAttention
