#!/usr/bin/env python3
"""Verify the OFF==baseline frame identity invariant for a transfeat run.

A transfeat that has no acceleration env (the OFF-identity check) must produce
frames byte/numerically-identical to the model baseline -- this is the framework
guarantee compose() rests on (an inactive technique is a no-op). PNG bytes can
differ if encoding metadata varies, so we compare RGB pixel arrays (exact equal
+ max-abs-diff on uint8).

Usage:
  python scripts/verify_off_identity.py <transfeat_run_dir> <baseline_run_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path


def _load_rgb(path: Path):
    import numpy as np
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    cand_dir, base_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    cand_frames = sorted((cand_dir / "outputs/frames").glob("*.png"))
    base_frames = sorted((base_dir / "outputs/frames").glob("*.png"))
    if not cand_frames:
        print(f"FAIL: no transfeat frames under {cand_dir}/outputs/frames", file=sys.stderr)
        return 1
    if not base_frames:
        print(f"FAIL: no baseline frames under {base_dir}/outputs/frames", file=sys.stderr)
        return 1
    if len(cand_frames) != len(base_frames):
        print(
            f"WARN: frame count differs: {len(cand_frames)} transfeat vs "
            f"{len(base_frames)} baseline",
            file=sys.stderr,
        )

    n = min(len(cand_frames), len(base_frames))
    max_abs_total = 0
    nonidentical = 0
    for c, b in zip(cand_frames[:n], base_frames[:n]):
        ca, ba = _load_rgb(c), _load_rgb(b)
        if ca.shape != ba.shape:
            print(f"FAIL: shape mismatch on {c.name}: {ca.shape} vs {ba.shape}",
                  file=sys.stderr)
            return 1
        m = int((ca.astype("int16") - ba.astype("int16")).max(initial=0))
        n_ = int(abs(ca.astype("int16") - ba.astype("int16")).max(initial=0))
        if n_ > max_abs_total:
            max_abs_total = n_
        if n_ != 0:
            nonidentical += 1
    if nonidentical == 0:
        print(f"OK: {n} frames byte/pixel-identical (max_abs_diff=0)")
        return 0
    print(
        f"DIVERGE: {nonidentical}/{n} frames differ from baseline, "
        f"max_abs_diff_uint8={max_abs_total}"
    )
    # Don't fail loudly -- the orchestrator may still accept small numerical
    # drift across submodule commits (e.g. unrelated kernels). The OFF=identity
    # contract is strict; the caller decides whether to reject.
    return 0 if max_abs_total == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
