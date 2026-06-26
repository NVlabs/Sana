"""Adapter for ToMe-SD."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "token_pruning/tome_sd"
SOURCE_URLS = [
    "https://github.com/dbolya/tomesd",
    "https://huggingface.co/docs/diffusers/en/optimization/tome",
]
LOCAL_ENTRYPOINTS = ["external package: tomesd.apply_patch"]


def dependency_status() -> dict[str, bool]:
    return _dependency_status("tomesd")


def apply_tome_patch(*args, **kwargs):
    import tomesd

    return tomesd.apply_patch(*args, **kwargs)
