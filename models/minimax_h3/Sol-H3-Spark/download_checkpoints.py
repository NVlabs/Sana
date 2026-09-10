#!/usr/bin/env python3
"""Fetch the pinned public Spark checkpoint subset; never fetch at inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "configs/checkpoints.json"
TASKS = ("t2va", "fl2va", "ref2va")


def selected_entries(manifest, *, task="t2va", include_offline=True):
    if task not in TASKS:
        raise ValueError(f"Unknown checkpoint task: {task}")
    for entry in manifest["entries"]:
        if "tasks" in entry and task not in entry["tasks"]:
            continue
        if entry.get("offline_only") and not include_offline:
            continue
        entry = dict(entry)
        if "task_allow_patterns" in entry:
            entry["allow_patterns"] = entry["allow_patterns"] + entry["task_allow_patterns"][task]
        yield entry


def checkpoint_paths(output_dir, *, include_offline=True, task="t2va"):
    manifest = json.loads(MANIFEST.read_text())
    paths = {}
    for entry in selected_entries(manifest, task=task, include_offline=include_offline):
        path = Path(output_dir).resolve() / entry["directory"]
        if entry["kind"] == "file":
            path /= entry["filename"]
        paths[entry["key"]] = str(path)
    for alias, key in manifest["aliases"].items():
        if key in paths:
            paths[alias] = paths[key]
    return paths


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--task", choices=TASKS, default="t2va",
                        help="Select the H3 partition, input VAE and task-specific adapter")
    parser.add_argument("--include-offline", action="store_true",
                        help="Also fetch INT8 Gemma and the INT8 dev connector for one-time cache preparation")
    parser.add_argument("--plan", action="store_true", help="Print the exact download plan without network or writes")
    parser.add_argument("--verify-sha256", action="store_true",
                        help="Read complete single-file checkpoints and compare the manifest SHA-256 values")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    entries = list(selected_entries(manifest, task=args.task, include_offline=args.include_offline))
    paths = checkpoint_paths(args.output_dir, include_offline=args.include_offline, task=args.task)
    if args.plan:
        print(json.dumps({"task": args.task, "downloads": entries, "paths": paths,
                          "external_inputs": manifest["external"]}, indent=2))
        return 0

    # Uses the caller's normal Hugging Face authentication. Tokens are neither
    # read explicitly nor accepted in arguments, receipts or configuration.
    from huggingface_hub import hf_hub_download, snapshot_download
    for entry in entries:
        kwargs = dict(repo_id=entry["repo_id"], revision=entry["revision"],
                      local_dir=args.output_dir.resolve() / entry["directory"])
        if entry["kind"] == "snapshot":
            result = Path(snapshot_download(**kwargs, allow_patterns=entry["allow_patterns"]))
            for item in entry.get("verify_files", []):
                file = result / item["filename"]
                if file.stat().st_size != item["bytes"]:
                    raise ValueError(f"Checkpoint byte count differs: {entry['key']}/{item['filename']}")
                if args.verify_sha256 and sha256(file) != item["sha256"]:
                    raise ValueError(f"Checkpoint SHA-256 differs: {entry['key']}/{item['filename']}")
        else:
            result = Path(hf_hub_download(**kwargs, filename=entry["filename"]))
            if result.stat().st_size != entry["bytes"]:
                raise ValueError(f"Checkpoint byte count differs: {entry['key']}")
            if args.verify_sha256 and sha256(result) != entry["sha256"]:
                raise ValueError(f"Checkpoint SHA-256 differs: {entry['key']}")
        print(f"ready: {entry['key']} @ {entry['revision']}")
    output = args.output_dir.resolve() / "checkpoint-paths.json"
    output.write_text(json.dumps(paths, indent=2) + "\n")
    print(output)
    print("Generate or reuse the fixed prompt cache before inference; see docs/setup.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
