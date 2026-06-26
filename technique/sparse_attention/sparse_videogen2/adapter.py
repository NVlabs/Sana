"""Adapter for Sparse VideoGen2."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "sparse_attention/sparse_videogen2"
SOURCE_URLS = ["https://github.com/svg-project/Sparse-VideoGen"]
LOCAL_ENTRYPOINTS = [
    "sglang.multimodal_gen.runtime.layers.attention.backends.sparse_video_gen_2_attn.SparseVideoGen2AttentionBackend",
]


def dependency_status() -> dict[str, bool]:
    return _dependency_status(
        "svg",
        "sglang.multimodal_gen.runtime.layers.attention.backends.sparse_video_gen_2_attn",
    )
