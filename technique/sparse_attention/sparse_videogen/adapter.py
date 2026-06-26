"""Adapter for Sparse VideoGen."""

from __future__ import annotations

from technique.common import dependency_status as _dependency_status

METHOD_ID = "sparse_attention/sparse_videogen"
SOURCE_URLS = ["https://github.com/svg-project/Sparse-VideoGen"]
LOCAL_ENTRYPOINTS = ["external package: svg"]


def dependency_status() -> dict[str, bool]:
    return _dependency_status("svg")
