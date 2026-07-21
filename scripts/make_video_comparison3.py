#!/usr/bin/env python3
"""Create a labeled 3-panel side-by-side MP4 from three frame directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def frames(path: Path) -> list[Path]:
    result = sorted(Path(path).glob("f_*.png"))
    if not result:
        raise SystemExit(f"no frames found: {path}")
    return result


def labeled_frame(path: Path, label: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    bar = Image.new("RGB", (image.width, 48), "black")
    draw = ImageDraw.Draw(bar)
    draw.text((16, 14), label, fill="white")
    output = Image.new("RGB", (image.width, image.height + bar.height))
    output.paste(bar, (0, 0))
    output.paste(image, (0, bar.height))
    return output


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs=3, type=Path, required=True)
    p.add_argument("--labels", nargs=3, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=int, default=24)
    a = p.parse_args()
    cols = [frames(d) for d in a.dirs]
    n = {len(c) for c in cols}
    if len(n) != 1:
        raise SystemExit(f"frame count mismatch: {[len(c) for c in cols]}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(a.output), fps=a.fps, codec="libx264",
                                macro_block_size=1, ffmpeg_log_level="error")
    try:
        for row in zip(*cols):
            imgs = [labeled_frame(f, lb) for f, lb in zip(row, a.labels)]
            w = sum(i.width for i in imgs)
            combined = Image.new("RGB", (w, imgs[0].height))
            x = 0
            for i in imgs:
                combined.paste(i, (x, 0))
                x += i.width
            writer.append_data(np.asarray(combined))
    finally:
        writer.close()
    print(a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
