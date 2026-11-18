#!/usr/bin/env python3
"""Validate that the selected LTX-2.5 environment resolves local runtime code."""

from __future__ import annotations

import sys
from pathlib import Path

import ltx_core
import ltx_kernels
import ltx_pipelines
import torch
import triton


REPO = Path(__file__).resolve().parents[3]
FORBIDDEN = ("ltx25" + "-opt", "Sol-LTX" + "-Infer", "sol-engine" + "-ltx25")


def _module_path(module: object) -> Path:
    value = getattr(module, "__file__", None)
    if not value:
        raise RuntimeError(f"module has no __file__: {module!r}")
    return Path(value).resolve()


def main() -> int:
    modules = {
        "ltx_core": _module_path(ltx_core),
        "ltx_pipelines": _module_path(ltx_pipelines),
        "ltx_kernels": _module_path(ltx_kernels),
    }
    for name, path in modules.items():
        if REPO not in path.parents:
            raise RuntimeError(f"{name} resolved outside this repository: {path}")
        if any(value in str(path) for value in FORBIDDEN):
            raise RuntimeError(f"{name} retained a sibling-checkout path: {path}")

    print("ENVIRONMENT_PASS")
    print(f"python={sys.executable}")
    print(f"prefix={sys.prefix}")
    print(f"torch={torch.__version__}")
    print(f"triton={triton.__version__}")
    for name, path in modules.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
