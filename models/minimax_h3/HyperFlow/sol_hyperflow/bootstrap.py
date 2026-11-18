"""Locate the existing Sol-H3 package without copying its kernel sources."""
from pathlib import Path
import importlib
import sys


def ensure_sol_h3():
    root = Path(__file__).resolve().parents[2] / "Sol-H3"
    if not (root / "h3_runtime/lora_fusion.py").is_file():
        raise RuntimeError("HyperFlow requires the sibling Sol-H3 runtime including Sana PR #503.")
    loaded = sys.modules.get("h3_runtime")
    if loaded is not None and Path(loaded.__file__).resolve().parent != root / "h3_runtime":
        raise RuntimeError("A different h3_runtime package is already loaded; use this Sana checkout's Sol-H3.")
    if loaded is None:
        # Resolve the package here without permanently shadowing HyperFlow's
        # own infer.py with the sibling Sol-H3 command-line module.
        sys.path.insert(0, str(root))
        try:
            importlib.import_module("h3_runtime")
        finally:
            sys.path.remove(str(root))
    return root
