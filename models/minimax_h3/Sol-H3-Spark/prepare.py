#!/usr/bin/env python3
"""Resolve existing Spark inputs and optionally fetch pinned public source trees.

This is a filesystem preparation command, not an environment installer or GPU
preflight. It imports no GPU frameworks. The inference sessions validate the
checkpoint, prompt-cache and attention contracts in their selected environments.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from download_checkpoints import ROOT, checkpoint_paths, sha256


def git(*args):
    return subprocess.check_output(["git", *map(str, args)], text=True).strip()


def fetch_source(entry, target):
    """Reuse a matching tree; never reset or overwrite an existing checkout."""
    if target.exists():
        if git("-C", target, "rev-parse", "HEAD") != entry["revision"]:
            raise ValueError(f"Existing source revision differs: {entry['key']} at {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none",
                    entry["repository"], str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", entry["revision"]], check=True)


def apply_source_patch(entry, source, changed_paths):
    """Apply the declared loader patch only to its exact pristine file set."""
    manifest_path = ROOT / entry["patch_manifest"]
    manifest = json.loads(manifest_path.read_text())
    patch = manifest_path.parent / manifest["patch"]
    if (manifest["revision"] != entry["revision"]
            or sha256(patch) != manifest["patch_sha256"]):
        raise ValueError(f"Declared source patch identity differs: {entry['key']}")
    files = manifest["files"]
    allowed = {item["path"] for item in files}
    if set(changed_paths) - allowed:
        raise ValueError(f"Undeclared tracked source modifications: {entry['key']}")
    actual = {item["path"]: sha256(source / item["path"]) for item in files}
    before = {item["path"]: item["before_sha256"] for item in files}
    after = {item["path"]: item["after_sha256"] for item in files}
    if actual == after:
        return "reused_exact_patch"
    if actual != before or changed_paths:
        raise ValueError(f"Partial or unrecognized source patch: {entry['key']}; files were preserved")
    subprocess.run(["git", "-C", str(source), "apply", "--check", str(patch)], check=True)
    subprocess.run(["git", "-C", str(source), "apply", str(patch)], check=True)
    if any(sha256(source / item["path"]) != item["after_sha256"] for item in files):
        raise RuntimeError(f"Source patch did not produce the declared hashes: {entry['key']}")
    return "applied_exact_patch"


def prepare_source(entry, source):
    if Path(git("-C", source, "rev-parse", "--show-toplevel")).resolve() != source.resolve():
        raise ValueError(f"Source path is not the checkout root: {entry['key']}")
    if git("-C", source, "rev-parse", "HEAD") != entry["revision"]:
        raise ValueError(f"Source revision differs for {entry['key']}")
    changed = set(git("-C", source, "diff", "--name-only", "HEAD").splitlines())
    if entry.get("patch_manifest"):
        return apply_source_patch(entry, source, changed)
    if changed:
        raise ValueError(f"Tracked source modifications are not pinned: {entry['key']}")
    return "reused_clean_source"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--sources", type=Path, default=ROOT / "dependencies")
    parser.add_argument("--paths", type=Path, help="Existing flat JSON path overrides, e.g. cached model locations")
    parser.add_argument("--output", type=Path, default=ROOT / "paths.json")
    parser.add_argument("--fetch-sources", action="store_true", help="Fetch only missing pinned public source trees")
    parser.add_argument("--plan", action="store_true", help="Print paths and missing inputs without writes or network")
    parser.add_argument("--python-stage1", type=Path)
    parser.add_argument("--python-stage2", type=Path)
    parser.add_argument("--python-qwen", type=Path)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--prompt-cache", type=Path)
    parser.add_argument("--cache-pending", action="store_true",
                        help="Write paths before one-time cache generation; requires --prompt-cache and offline weights")
    return parser.parse_args()


def prepare_fa4_namespace(paths):
    """Match the FA4-only namespace layout without importing upstream FA2."""
    runtime = Path(paths["fa4_root"])
    cute = runtime / "flash_attn/cute"
    if cute.is_dir():
        if (runtime / "flash_attn/__init__.py").exists():
            raise ValueError("fa4_root must be an isolated FA4 namespace, not the full FA2/FA4 repository")
        return
    source = Path(paths["fa4_source_root"]) / "flash_attn/cute"
    if not source.is_dir():
        raise ValueError("Missing pinned FA4 source; fetch it or supply an existing isolated fa4_root")
    cute.parent.mkdir(parents=True, exist_ok=True)
    cute.symlink_to(source, target_is_directory=True)


def main():
    args = arguments()
    manifest = json.loads((ROOT / "configs/dependencies.json").read_text())
    paths = checkpoint_paths(args.checkpoints)
    for entry in manifest["sources"]:
        paths[entry["key"]] = str((args.sources / entry["directory"]).resolve())
    paths["fa4_root"] = str((args.sources / "fa4-runtime").resolve())
    if args.paths:
        overrides = json.loads(args.paths.read_text())
        if not isinstance(overrides, dict) or any(not isinstance(v, str) for v in overrides.values()):
            raise ValueError("--paths must contain a flat JSON object of string paths")
        paths.update(overrides)
    for key, value in (("stage1_python", args.python_stage1), ("stage2_python", args.python_stage2),
                       ("qwen_python", args.python_qwen), ("adapter_dir", args.adapter_dir),
                       ("prompt_cache", args.prompt_cache)):
        if value is not None:
            paths[key] = str(value.expanduser().absolute())
    upscaler = next(item for item in manifest["sources"] if item["key"] == "h3_upscaler_root")
    paths.setdefault("h3_upscaler_source", str(Path(paths["h3_upscaler_root"]) / upscaler["source_file"]))
    # LTX reads tokenizer JSON and processor sidecars embedded in this exact TE
    # checkpoint. It does not require a second Gemma model download.
    paths["gemma_tokenizer"] = paths["offline_gemma"]
    reused_fa4 = (Path(paths["fa4_root"]) / "flash_attn/cute").is_dir()
    if args.fetch_sources and not args.plan:
        for entry in manifest["sources"]:
            if entry["key"] == "fa4_source_root" and reused_fa4:
                continue
            fetch_source(entry, Path(paths[entry["key"]]))
    if not args.plan:
        prepare_fa4_namespace(paths)

    required = ["h3_model", "vsa_lora", "qwen_checkpoint", "transformer", "refiner_lora",
                "output_video_vae", "audio_vae", "h3_upscaler_source", "h3_upscaler_checkpoint",
                "adapter_dir", "prompt_cache", "stage1_python", "stage2_python", "qwen_python"]
    required.append("fa4_root")
    required.extend(item["key"] for item in manifest["sources"]
                    if item["key"] != "fa4_source_root" or not reused_fa4)
    if args.cache_pending:
        if not paths.get("prompt_cache"):
            raise ValueError("--cache-pending requires --prompt-cache or an explicit prompt_cache path")
        required.remove("prompt_cache")
        required.extend(["offline_gemma", "offline_connector", "gemma_tokenizer"])
    missing = [key for key in required if key not in paths or not Path(paths[key]).exists()]
    if args.plan:
        print(json.dumps({"paths": paths, "missing_inputs": missing,
                          "source_patches": [item["patch_manifest"] for item in manifest["sources"]
                                             if item.get("patch_manifest")],
                          "unresolved": manifest["unresolved"],
                          "validation": "Plan only; no imports, downloads, writes or GPU validation"}, indent=2))
        return 0
    if missing:
        raise SystemExit("Missing required inputs: " + ", ".join(missing) + ". See docs/setup.md.")
    for key in ("stage1_python", "stage2_python", "qwen_python"):
        if not Path(paths[key]).is_file() or not os.access(paths[key], os.X_OK):
            raise ValueError(f"{key} must be an executable Python interpreter")
    for entry in manifest["sources"]:
        if entry["key"] == "fa4_source_root" and reused_fa4:
            continue
        source = Path(paths[entry["key"]])
        state = prepare_source(entry, source)
        print(f"source: {entry['key']} {state}")
    if sha256(paths["h3_upscaler_source"]) != upscaler["source_sha256"]:
        raise ValueError("H3 upscaler source checksum differs")
    for filename in ("config.json", "model.safetensors"):
        if not (Path(paths["adapter_dir"]) / filename).is_file():
            raise ValueError(f"Adapter directory is missing {filename}")
    cache_path = Path(paths["prompt_cache"])
    if not args.cache_pending and (not cache_path.is_file() or not 0 < cache_path.stat().st_size <= 64 * 1024**2):
        raise ValueError("Prompt cache must be a nonempty feature file of at most 64 MiB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(paths, indent=2) + "\n")
    print(args.output.resolve())
    if args.cache_pending:
        print("Prompt cache pending: run the offline cache builder before inference.")
    print("Filesystem inputs prepared. Model/cache semantics and GPU execution remain to be validated at startup.")
    return 0


if __name__ == "__main__":
    main()
