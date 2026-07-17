#!/usr/bin/env python3
"""Create labeled side-by-side MP4 comparisons from two frame directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def frames(path: Path) -> list[Path]:
    result = sorted(path.glob("f_*.png"))
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


def make_comparison(left_dir: Path, right_dir: Path, left_label: str, right_label: str, output: Path, fps: int) -> None:
    left = frames(left_dir)
    right = frames(right_dir)
    if len(left) != len(right):
        raise SystemExit(f"frame count mismatch: {len(left)} vs {len(right)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    first_left = labeled_frame(left[0], left_label)
    first_right = labeled_frame(right[0], right_label)
    size = (first_left.width + first_right.width, first_left.height)
    writer = imageio.get_writer(
        str(output),
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    try:
        for left_path, right_path in zip(left, right):
            left_image = labeled_frame(left_path, left_label)
            right_image = labeled_frame(right_path, right_label)
            if (left_image.width + right_image.width, left_image.height) != size:
                raise SystemExit("frame dimensions changed within comparison")
            combined = Image.new("RGB", size)
            combined.paste(left_image, (0, 0))
            combined.paste(right_image, (left_image.width, 0))
            writer.append_data(np.asarray(combined))
    finally:
        writer.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()
    make_comparison(args.left, args.right, args.left_label, args.right_label, args.output, args.fps)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
