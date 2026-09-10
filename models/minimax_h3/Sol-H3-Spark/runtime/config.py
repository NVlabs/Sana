"""Frozen recipe and portable paths. Importing this module does not load CUDA."""

import hashlib
import json
import os
from pathlib import Path
import re

PACKAGE = Path(__file__).resolve().parents[1]
TASKS = ("t2va", "fl2va", "ref2va")
FROZEN_RECIPE_SHA256 = "f3bec3ed9dee7bc92f6e936937dd1d5828852b4e485469c2e1d13727c4a41aab"
REQUIRED_PATHS = (
    "stage1_python", "stage2_python", "qwen_python", "h3_model", "vsa_lora",
    "fastvideo_root", "fa4_root", "ltx_root", "qwen_checkpoint", "comfy_root", "transformer",
    "refiner_lora", "output_video_vae", "audio_vae", "adapter_dir",
    "h3_upscaler_source", "h3_upscaler_checkpoint", "prompt_cache",
)


def load_recipe(task="t2va"):
    recipe = json.loads((PACKAGE / "configs/default.json").read_text())
    # This is a record of the implemented recipe, not an ablation interface.
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != FROZEN_RECIPE_SHA256:
        raise ValueError("The frozen recipe record was modified; it would not describe this implementation")
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if task != "t2va":
        # Task inputs select a native model family, not an arbitrary tuning arm.
        recipe["stage1"]["task"] = task
        recipe["stage1"]["reference_encode"] = "native_H3_VAE"
    if task == "ref2va":
        recipe["stage1"].update(lora="LightX2V_Ref2VA_4step", attention="FA4_dense")
        recipe["stage1"].pop("sparsity")
        recipe["stage1"].pop("tile_size")
    return recipe


def load_paths(filename, *, task="t2va"):
    filename = Path(filename).resolve(strict=True)
    values = json.loads(filename.read_text())
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    required = set(REQUIRED_PATHS)
    if task == "ref2va":
        required.remove("vsa_lora")
        required.add("ref2va_lora")
    missing = sorted(name for name in required if not values.get(name))
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
            result[name] = str(path.resolve(strict=name in required))
    return result


def normalize_case(case, *, base_dir=None, default_task="t2va"):
    """Validate one request, retaining ordered local multimodal inputs."""
    if not isinstance(case, dict):
        raise ValueError("each case must be a JSON object")
    name, prompt = case.get("case_id"), case.get("prompt")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", name):
        raise ValueError("case_id must be a simple file-safe name")
    if name in {"warmup", "stage1-worker", "stage2-worker", "qwen-worker", "warmup-qwen-worker"}:
        raise ValueError(f"Reserved case_id: {name}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be nonempty text")
    seed = case.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    task = case.get("task", default_task)
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    root = Path(base_dir or Path.cwd())

    def media_path(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("media paths must be nonempty local filenames")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"media input must be a file: {path}")
        return str(path)

    result = {"case_id": name, "prompt": prompt, "seed": seed, "task": task}
    first, last, references = case.get("first_frame"), case.get("last_frame"), case.get("references", [])
    if not isinstance(references, list):
        raise ValueError("references must be an ordered list")
    if task == "t2va":
        if first is not None or last is not None or references:
            raise ValueError("T2VA does not accept frame or reference inputs; select fl2va or ref2va")
    elif task == "fl2va":
        if (first is None and last is None) or references:
            raise ValueError("FL2VA requires first_frame, last_frame, or both, and no references")
        for key in ("first_frame", "last_frame"):
            if case.get(key) is not None:
                result[key] = media_path(case[key])
    else:
        if first is not None or last is not None or not references:
            raise ValueError("Ref2VA requires references, not first_frame/last_frame")
        result["references"] = []
        for reference in references:
            if not isinstance(reference, dict) or reference.get("type") not in ("image", "video", "audio"):
                raise ValueError("each reference needs type=image|video|audio and path")
            result["references"].append({"type": reference["type"], "path": media_path(reference.get("path"))})
        if all(reference["type"] == "audio" for reference in result["references"]):
            raise ValueError("Ref2VA needs at least one image or video; audio alone is unsupported")
        kinds = [reference["type"] for reference in result["references"]]
        if len(kinds) > 12 or any(kinds.count(kind) > limit for kind, limit in
                                  (("image", 9), ("video", 3), ("audio", 3))):
            raise ValueError("Ref2VA allows at most 9 images, 3 videos, 3 audio inputs and 12 references total")
    return result


def read_cases(filename, *, default_task="t2va"):
    """Read a single-task JSONL batch; media paths are relative to the JSONL."""
    filename = Path(filename).resolve(strict=True)
    cases, names = [], set()
    for number, line in enumerate(filename.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = normalize_case(json.loads(line), base_dir=filename.parent, default_task=default_task)
        except (ValueError, OSError) as error:
            raise ValueError(f"Line {number}: {error}") from error
        name = case["case_id"]
        if name in names:
            raise ValueError(f"Duplicate case_id: {name}")
        names.add(name)
        if cases and case["task"] != cases[0]["task"]:
            raise ValueError("Use one task per batch; Ref2VA uses a different model partition")
        cases.append(case)
    if not cases:
        raise ValueError("No prompts in input")
    return cases
