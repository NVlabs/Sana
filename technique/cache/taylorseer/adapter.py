"""Adapter for TaylorSeer through cache-dit."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "cache/taylorseer"
SOURCE_URLS = [
    "https://github.com/Shenyi-Z/TaylorSeer",
    "https://github.com/vipshop/cache-dit",
]
LOCAL_ENTRYPOINTS = [
    "sglang.multimodal_gen.runtime.cache.cache_dit_integration.CacheDitConfig.enable_taylorseer",
]


def dependency_status() -> dict[str, bool]:
    return _dependency_status(
        "cache_dit",
        "sglang.multimodal_gen.runtime.cache.cache_dit_integration",
    )
