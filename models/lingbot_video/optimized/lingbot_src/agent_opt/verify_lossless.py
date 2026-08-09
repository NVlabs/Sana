"""Lossless gate: compare a transfeat's refined video against the golden reference.

Parallelization changes only the ORDER of floating-point reductions, so frames must be
near-identical. We require mean PSNR >= threshold (default 45 dB). Prints a machine-readable
verdict line the agent can grep: `LOSSLESS_VERDICT PASS|FAIL psnr=<db> ...`.

Usage:
  python agent_opt/verify_lossless.py <golden.mp4> <transfeat.mp4> [--min-psnr 45]
Exit code 0 = PASS, 1 = FAIL, 2 = error.
"""
import argparse
import sys

import numpy as np
import imageio.v2 as imageio


def read_frames(path):
    r = imageio.get_reader(path, "ffmpeg")
    frames = [np.asarray(f).astype(np.float64) for f in r]
    r.close()
    return frames


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("transfeat")
    ap.add_argument("--min-psnr", type=float, default=45.0)
    args = ap.parse_args()

    try:
        g = read_frames(args.golden)
        c = read_frames(args.transfeat)
    except Exception as e:  # noqa: BLE001
        print(f"LOSSLESS_VERDICT ERROR reason=read_failed detail={e!r}")
        return 2

    if len(g) != len(c):
        print(f"LOSSLESS_VERDICT FAIL reason=frame_count golden={len(g)} transfeat={len(c)}")
        return 1
    if not g:
        print("LOSSLESS_VERDICT ERROR reason=empty")
        return 2
    if g[0].shape != c[0].shape:
        print(f"LOSSLESS_VERDICT FAIL reason=shape golden={g[0].shape} transfeat={c[0].shape}")
        return 1

    per = [psnr(gf, cf) for gf, cf in zip(g, c)]
    finite = [p for p in per if np.isfinite(p)]
    mean_psnr = float(np.mean(finite)) if finite else float("inf")
    min_psnr = float(np.min(per))
    ok = mean_psnr >= args.min_psnr
    print(
        f"LOSSLESS_VERDICT {'PASS' if ok else 'FAIL'} "
        f"psnr={mean_psnr:.2f} min_frame_psnr={min_psnr:.2f} "
        f"threshold={args.min_psnr} frames={len(g)}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
