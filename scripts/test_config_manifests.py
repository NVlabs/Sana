#!/usr/bin/env python3
"""Validate every model-agnostic efficiency config manifest."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from techniques.config_manifest import dry_run_manifest, load_toml  # noqa: E402


EXPECTED_COUNTS = {
    "kwl_fusion": 6,
    "nvfp4_ffn": 5,
    "sparse_attention": 9,
    "step_cache": 5,
    "token_prune": 5,
}


def main() -> int:
    paths = sorted((ROOT / "config").glob("*/*.toml"))
    counts: Counter[str] = Counter()
    for path in paths:
        payload = dry_run_manifest(load_toml(path), ROOT)
        if payload is None:
            raise AssertionError(f"not an efficiency config: {path}")
        counts[payload["dimension"]] += 1
        if not payload["required_capabilities"]:
            raise AssertionError(f"missing capabilities: {path}")
        print(f"PASS {path.relative_to(ROOT)}")

    if dict(counts) != EXPECTED_COUNTS:
        raise AssertionError(f"config counts mismatch: {dict(counts)}")
    print(f"\n=== {sum(counts.values())} config manifests passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
