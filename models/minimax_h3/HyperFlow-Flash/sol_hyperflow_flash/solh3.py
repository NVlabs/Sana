"""Import the Sol-H3 runtime modules this runner uses (../Sol-H3/h3_runtime of this Sana checkout) without executing the
package __init__ (which imports the whole H3 engine). Only these modules are used:

  sparse_attention  Sol-Attn routing + cuDNN block-sparse attention + Sol residual merge (backend ``sol_bsa``)
  fusions           fused per-head QK RMSNorm + partial RoPE kernel
  fusion_install    block-level fused RMSNorm / modulate / residual-gate / SwiGLU forward
  sol_residual      the residual merge kernel (tuned launch in `merge_tuned`)
  vae_parallel      parallel compiled video VAE decode / sharded encode
  ulysses, relayout, comm_quant   compressed Ulysses transport
  encoding          MP4 (+ audio) writer
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

VALIDATED_SOL_ENGINE = "781e8341fb3b29d3aea5d50265772e9619340b66"
SIBLING = Path(__file__).resolve().parents[2] / "Sol-H3"


def h3_runtime_dir(sol_h3_dir: str | None = None) -> str:
    d = sol_h3_dir or os.environ.get("SOL_H3_DIR") or str(SIBLING)
    p = Path(d)
    if (p / "h3_runtime").is_dir():
        p = p / "h3_runtime"
    if not (p / "sparse_attention.py").is_file():
        raise FileNotFoundError(f"{p} is not a Sol-H3 h3_runtime directory (set --sol-h3-dir or SOL_H3_DIR)")
    return str(p)


def stub_package(name: str, path: str):
    loaded = sys.modules.get(name)
    if loaded is not None:
        have = [str(Path(x).resolve()) for x in getattr(loaded, "__path__", [])]
        if str(Path(path).resolve()) not in have:
            raise RuntimeError(f"a different {name} package is already loaded from {have}")
        return loaded
    pkg = types.ModuleType(name)
    pkg.__path__ = [path]
    sys.modules[name] = pkg
    return pkg


def load(sol_h3_dir: str | None = None, modules=("sparse_attention", "fusions", "fusion_install")):
    rt = h3_runtime_dir(sol_h3_dir)
    stub_package("h3_runtime", rt)
    return types.SimpleNamespace(dir=rt, **{m: importlib.import_module(f"h3_runtime.{m}") for m in modules})
