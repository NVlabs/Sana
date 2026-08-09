"""Where everything lives, resolved from this file's position in the repository.

The port was written against one machine and carried sixty absolute paths into a home
directory — the kind of thing that survives review and then fails on the first other box. Every
path here is derived from this module's own location instead, with environment overrides for
the things that legitimately live elsewhere.

The layout this assumes is the repository's own:

    models/minimax_h3/demo_prompt.json   the official cell's prompt
    models/minimax_h3/GB200/diffusers_src  the pinned diffusers (PR #14355)
    techniques/sparse_backends                      Sol-Attn's released kernel

`run_minimax_h3_gpu.sh` exports the same variables before launching, matching the
8xGB200 entrypoint next door, so a config run never depends on these fallbacks. They exist
so the entrypoint can also be run by hand, without a config manifest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT.parent
REPO_ROOT = MODEL_DIR.parent.parent

PROMPT_FILE = Path(os.environ.get("H3_PROMPT_FILE",
                                  MODEL_DIR / "demo_prompt.json"))
SPARSE_BACKENDS = Path(os.environ.get("H3_SOL_ATTN_ROOT",
                                      REPO_ROOT / "techniques" / "sparse_backends"))

# Artefacts a run produces, and the recorded inputs a benchmark needs. Kept beside the variant
# rather than in the repository's `runs/`, which `scripts/collect_run.py` owns.
CAPTURE_DIR = Path(os.environ.get("H3_CAPTURE_DIR", ROOT / "dit_inputs"))
REFERENCE_DIR = Path(os.environ.get("H3_REFERENCE_DIR", ROOT / "dit_reference"))
OUTPUT_DIR = Path(os.environ.get("H3_OUTPUT_DIR", ROOT / "outputs"))

HF_CACHE = Path(os.environ.get("HF_HOME", ROOT / "hf_cache"))


def _diffusers_src() -> Path:
    """The pinned diffusers. MiniMax-H3 is not in a release, so this is a hard dependency.

    GB200 vendors it at the commit its `SOURCE_SNAPSHOT.json` records; a checkout beside
    this variant is accepted as a fallback, because the tarball distribution of the
    repository does not carry `diffusers_src`.
    """
    override = os.environ.get("H3_DIFFUSERS_SRC")
    if override:
        return Path(override)
    for config in (MODEL_DIR / "GB200" / "diffusers_src" / "src",
                      ROOT / "diffusers_src" / "src"):
        if config.is_dir():
            return config
    return MODEL_DIR / "GB200" / "diffusers_src" / "src"


DIFFUSERS_SRC = _diffusers_src()


def _resolve(env: str, hf_repo: str, subpath: str = "") -> str:
    """Find a downloaded model, however it got here.

    `snapshot_download` from HuggingFace lands under `hub/models--org--name/snapshots/<sha>/`,
    ModelScope lands under a plain `org/name/` directory, and either may be somewhere else
    entirely. The environment variable wins, then the ModelScope layout, then the HF cache.
    Not hypothetical: huggingface.co does not resolve from the machine this was measured on,
    and all three repositories came from ModelScope.
    """
    override = os.environ.get(env)
    if override:
        return str(Path(override) / subpath) if subpath else override

    plain = HF_CACHE / hf_repo
    if plain.is_dir():
        return str(plain / subpath) if subpath else str(plain)

    cached = sorted(HF_CACHE.glob(f"hub/models--{hf_repo.replace('/', '--')}/snapshots/*"))
    if cached:
        return str(cached[-1] / subpath) if subpath else str(cached[-1])

    raise SystemExit(f"{hf_repo} not found under {HF_CACHE}. Set {env}, or see README.md.")


def h3_snapshot() -> str:
    """MiniMax-H3's own repository: configs, both VAEs, tokenizer, BF16 transformer."""
    return _resolve("H3_MODEL_ROOT", "MiniMaxAI/MiniMax-H3")


def qwen_fp8() -> str:
    """The FP8 conditioner. Byte-identical weights to the encoder MiniMax ships."""
    return _resolve("H3_QWEN_FP8", "Qwen/Qwen3-VL-32B-Instruct-FP8")


def dit_checkpoint() -> str:
    """ComfyUI's pruned FP8 DiT — the one this whole variant is built around."""
    override = os.environ.get("H3_DIT_CHECKPOINT")
    if override:
        return override
    return _resolve("H3_COMFY_ROOT", "Comfy-Org/MiniMax-H3",
                    "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors")


def setup(need_sol_engine: bool = False) -> None:
    """Put the pinned diffusers ahead of any installed one, and this variant on the path."""
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    for entry in (DIFFUSERS_SRC, ROOT):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    if need_sol_engine:
        if not SPARSE_BACKENDS.is_dir():
            raise SystemExit(
                f"Sol-Attn's kernel is not at {SPARSE_BACKENDS}.\n"
                f"Set H3_SOL_ATTN_ROOT, or run through run_minimax_h3_gpu.sh."
            )
        if str(SPARSE_BACKENDS) not in sys.path:
            sys.path.insert(0, str(SPARSE_BACKENDS))
