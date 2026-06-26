"""Adapter for Cache-DiT / DBCache."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "cache/cache_dit_dbcache"
SOURCE_URLS = ["https://github.com/vipshop/cache-dit"]
LOCAL_ENTRYPOINTS = [
    "sglang.multimodal_gen.runtime.cache.cache_dit_integration",
    "sglang.multimodal_gen.runtime.cache.ltx2_block_adapter",
]


def dependency_status() -> dict[str, bool]:
    return _dependency_status(
        "cache_dit",
        "sglang.multimodal_gen.runtime.cache.cache_dit_integration",
    )
