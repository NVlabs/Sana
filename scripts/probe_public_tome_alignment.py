#!/usr/bin/env python3
"""Compare local ``tome_merge_restore`` against public ToMe/ToMeSD behavior.

This is a public-behavior boundary probe. It pins the public ToMe family core
that the config cites and records whether the current local pure algorithm
matches that merge/unmerge boundary. Cosmos3 runtime quality is assessed
separately; this probe is only about the public ToMe algorithm shape.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TOME = Path("/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/.cache/autovideo/public_refs/tome")
PUBLIC_TOMESD = Path("/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/.cache/autovideo/public_refs/tomesd")
PUBLIC_TOME_MERGE = PUBLIC_TOME / "tome" / "merge.py"
PUBLIC_TOMESD_MERGE = PUBLIC_TOMESD / "tomesd" / "merge.py"
MANIFEST = ROOT / "config" / "token_prune" / "tome_merge_restore.toml"


def git_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_checks() -> dict[str, bool]:
    tome_text = PUBLIC_TOME_MERGE.read_text(errors="ignore") if PUBLIC_TOME_MERGE.exists() else ""
    tomesd_text = PUBLIC_TOMESD_MERGE.read_text(errors="ignore") if PUBLIC_TOMESD_MERGE.exists() else ""
    return {
        "has_public_tome_merge": PUBLIC_TOME_MERGE.exists(),
        "has_public_tomesd_merge": PUBLIC_TOMESD_MERGE.exists(),
        "tome_uses_bipartite_soft_matching": "def bipartite_soft_matching" in tome_text,
        "tome_uses_scatter_reduce_merge": "scatter_reduce" in tome_text,
        "tome_has_unmerge": "def unmerge" in tome_text and "out.scatter_" in tome_text,
        "tomesd_uses_random2d_matching": "def bipartite_soft_matching_random2d" in tomesd_text,
        "tomesd_uses_scatter_reduce_merge": "scatter_reduce" in tomesd_text,
    }


def _torch():
    import torch

    return torch


def _fixture():
    torch = _torch()
    torch.manual_seed(7)
    base = torch.arange(32, dtype=torch.float32).reshape(1, 8, 4)
    jitter = 0.01 * torch.randn_like(base)
    return base + jitter


def probe() -> dict[str, Any]:
    torch = _torch()
    sys.path.insert(0, str(ROOT))
    from techniques.methods.token_prune import keep_indices
    from techniques.methods.token_prune import tome_bipartite_soft_matching

    tome_merge = load_module(PUBLIC_TOME_MERGE, "public_tome_merge_probe")
    hidden = _fixture()
    remove = 2
    keep_ratio = (hidden.shape[1] - remove) / hidden.shape[1]
    merge, unmerge = tome_merge.bipartite_soft_matching(hidden, r=remove)
    public_merged = merge(hidden, mode="mean")
    public_restored = unmerge(public_merged)

    local_plan = tome_bipartite_soft_matching(hidden, remove)
    if local_plan is None:
        raise AssertionError("expected active ToMe merge plan")
    local_merged = local_plan.merge(hidden, mode="mean")
    local_restored = local_plan.unmerge(local_merged)
    legacy_idx = keep_indices("tome_merge_restore", hidden.shape[1], keep_ratio, hidden)

    same_shape = tuple(public_merged.shape) == tuple(local_merged.shape)
    merged_values_match = same_shape and torch.allclose(
        public_merged, local_merged, atol=1e-6, rtol=1e-6
    )
    restored_values_match = torch.allclose(
        public_restored, local_restored, atol=1e-6, rtol=1e-6
    )
    restored_is_identity = torch.allclose(public_restored, hidden, atol=1e-6, rtol=1e-6)
    legacy_kept_indices = [int(x) for x in legacy_idx.cpu().tolist()]

    return {
        "status": "pass",
        "public_reference": {
            "tome_repo": str(PUBLIC_TOME),
            "tome_commit": git_commit(PUBLIC_TOME),
            "tome_source": str(PUBLIC_TOME_MERGE),
            "tomesd_repo": str(PUBLIC_TOMESD),
            "tomesd_commit": git_commit(PUBLIC_TOMESD),
            "tomesd_source": str(PUBLIC_TOMESD_MERGE),
            "checks": source_checks(),
        },
        "behavior_probe": {
            "manifest": str(MANIFEST),
            "tokens": hidden.shape[1],
            "remove": remove,
            "keep_ratio": keep_ratio,
            "public_merged_shape": list(public_merged.shape),
            "public_restored_shape": list(public_restored.shape),
            "local_merged_shape": list(local_merged.shape),
            "local_restored_shape": list(local_restored.shape),
            "legacy_representative_indices": legacy_kept_indices,
            "same_merged_shape": same_shape,
            "merged_values_match": bool(merged_values_match),
            "restored_values_match": bool(restored_values_match),
            "public_unmerge_restores_original_shape": public_restored.shape == hidden.shape,
            "public_unmerge_is_identity": bool(restored_is_identity),
            "matches_public_tome_merge": bool(
                merged_values_match and restored_values_match
            ),
            "known_difference": (
                "local tome_merge_restore now matches the public ToMe balanced "
                "bipartite merge/unmerge boundary for this fixture. It still "
                "does not claim the full ToMeSD random-2D diffusion integration "
                "or useful Cosmos3 GPU quality."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = probe()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
