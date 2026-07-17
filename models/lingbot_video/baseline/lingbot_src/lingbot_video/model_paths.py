from __future__ import annotations

from pathlib import Path
from typing import Any


def model_component_dir(model_dir: str | Path, component: str) -> Path:
    """Return a named component directory under a public model root."""

    path = Path(model_dir) / component
    if not path.is_dir():
        raise FileNotFoundError(f"missing model component `{component}`: {path}")
    return path


def effective_refiner_model_dir(args: Any) -> Path | None:
    """Resolve the model root to use for refiner loading.

    Passing --run_refiner means the public runner should load `refiner/` from
    the same model root by default. --refiner_model_dir remains an override for
    nonstandard package layouts.
    """

    refiner_model_dir = getattr(args, "refiner_model_dir", None)
    requested = bool(getattr(args, "run_refiner", False) or refiner_model_dir)
    if not requested:
        return None
    return Path(refiner_model_dir or getattr(args, "model_dir"))
