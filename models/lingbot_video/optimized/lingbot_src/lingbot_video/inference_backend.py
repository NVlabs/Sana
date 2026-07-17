from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO


BACKEND_TO_ENGINE = {
    "diffusers": "diffusers",
    "sglang": "sglang-native",
}
ENGINE_CHOICES = frozenset(BACKEND_TO_ENGINE.values())


def _option_present(argv: Sequence[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def default_backend_argv(argv: Sequence[str]) -> list[str]:
    """Default the public CLI to direct diffusers when no backend is requested."""

    normalized = list(argv)
    if _option_present(normalized, "--backend") or _option_present(normalized, "--engine"):
        return normalized
    return ["--backend", "diffusers", *normalized]


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, KeyError, ValueError):
        return False


def sglang_native_available() -> bool:
    """Return whether the local SGLang-native adapter can be imported."""

    if not _module_available("sglang"):
        return False
    try:
        from lingbot_video.native_backend import LingBotVideoNativePipeline
    except Exception:
        return False
    return LingBotVideoNativePipeline is not None


def resolve_backend_engine(
    *,
    engine: str | None,
    backend: str | None,
    sglang_available: bool | None = None,
    stderr: TextIO | None = None,
) -> str:
    """Resolve public ``--backend`` and internal ``--engine`` into a runner engine."""

    explicit_engine = backend is None and engine is not None
    if backend is not None:
        if backend not in BACKEND_TO_ENGINE:
            choices = ", ".join(sorted(BACKEND_TO_ENGINE))
            raise ValueError(f"unsupported backend: {backend!r}; choices: {choices}")
        resolved = BACKEND_TO_ENGINE[backend]
    elif engine is not None:
        if engine not in ENGINE_CHOICES:
            choices = ", ".join(sorted(ENGINE_CHOICES))
            raise ValueError(f"unsupported engine: {engine!r}; choices: {choices}")
        resolved = engine
    else:
        resolved = "sglang-native"

    if resolved != "sglang-native":
        return resolved
    if explicit_engine:
        return resolved

    available = sglang_available
    if available is None:
        available = sglang_native_available()
    if available:
        return resolved

    stream = stderr if stderr is not None else sys.stderr
    print(
        "WARNING: SGLang backend requested but SGLang is not installed or its "
        "native diffusion API is unavailable; falling back to diffusers. "
        "Install requirements-sglang.txt to enable SGLang Diffusion.",
        file=stream,
        flush=True,
    )
    return "diffusers"


def load_negative_prompt_json(path: str | Path) -> str:
    """Load an auto-negative JSON file and serialize it as the prompt string."""

    with Path(path).open(encoding="utf-8") as f:
        payload = json.load(f)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def resolve_negative_prompt_arg(
    negative_prompt: str | None,
    negative_prompt_json: str | None,
) -> str | None:
    """Resolve manual and auto-negative prompt inputs for the runner."""

    if negative_prompt and negative_prompt_json:
        raise ValueError("Use either --negative_prompt or --negative_prompt_json, not both.")
    if negative_prompt_json:
        return load_negative_prompt_json(negative_prompt_json)
    return negative_prompt
