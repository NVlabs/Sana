"""Adapter for Video Sparse Attention."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "sparse_attention/vsa"
SOURCE_URLS = [
    "https://github.com/hao-ai-lab/FastVideo",
    "https://pypi.org/project/vsa/",
]
LOCAL_ENTRYPOINTS = [
    "sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn.VideoSparseAttentionBackend",
]


def dependency_status() -> dict[str, bool]:
    return _dependency_status(
        "vsa",
        "sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn",
    )
