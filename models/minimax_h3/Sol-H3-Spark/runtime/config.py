"""Frozen recipe and portable paths. Importing this module does not load CUDA."""

import hashlib
import json
import os
from pathlib import Path
import re

PACKAGE = Path(__file__).resolve().parents[1]
FROZEN_RECIPE_SHA256 = "f3bec3ed9dee7bc92f6e936937dd1d5828852b4e485469c2e1d13727c4a41aab"
REQUIRED_PATHS = (
    "stage1_python", "stage2_python", "qwen_python", "h3_model", "vsa_lora",
    "fastvideo_root", "fa4_root", "ltx_root", "qwen_checkpoint", "comfy_root", "transformer",
    "refiner_lora", "output_video_vae", "audio_vae", "adapter_dir",
    "h3_upscaler_source", "h3_upscaler_checkpoint", "prompt_cache",
)


def load_recipe():
    recipe = json.loads((PACKAGE / "configs/default.json").read_text())
    # This is a record of the implemented recipe, not an ablation interface.
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != FROZEN_RECIPE_SHA256:
        raise ValueError("The frozen recipe record was modified; it would not describe this implementation")
    return recipe


def load_paths(filename):
    filename = Path(filename).resolve(strict=True)
    values = json.loads(filename.read_text())
    missing = [name for name in REQUIRED_PATHS if not values.get(name)]
    if missing:
        raise ValueError("Missing runtime paths: " + ", ".join(missing) + ". See prepare.py.")
    result = {}
    for name, value in values.items():
        if not isinstance(value, str):
            raise ValueError(f"Runtime path {name} must be a string")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = filename.parent / path
        if name.endswith("_python"):
            # Dereferencing venv/bin/python selects the base environment instead.
            if not path.is_file() or not os.access(path, os.X_OK):
                raise ValueError(f"{name} must name an executable interpreter or wrapper")
            result[name] = str(path.absolute())
        else:
            # Offline Gemma/connector files are optional once the cache exists.
            result[name] = str(path.resolve(strict=name in REQUIRED_PATHS))
    return result


def read_cases(filename):
    """JSONL with case_id, prompt, seed; reject ambiguous/unsafe output names."""
    cases, names = [], set()
    for number, line in enumerate(Path(filename).read_text().splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        name, prompt = case.get("case_id"), case.get("prompt")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", name):
            raise ValueError(f"Line {number}: case_id must be a simple file-safe name")
        if name in {"warmup", "stage1-worker", "stage2-worker", "qwen-worker", "warmup-qwen-worker"}:
            raise ValueError(f"Reserved case_id: {name}")
        if name in names:
            raise ValueError(f"Duplicate case_id: {name}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Line {number}: prompt must be nonempty text")
        seed = case.get("seed", 42)
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError(f"Line {number}: seed must be a nonnegative 63-bit integer")
        names.add(name)
        cases.append({"case_id": name, "prompt": prompt, "seed": seed})
    if not cases:
        raise ValueError("No prompts in input")
    return cases
