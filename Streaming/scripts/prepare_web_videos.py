#!/usr/bin/env python3
from pathlib import Path
import os
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "app.js"
DEST_ROOT = ROOT / "assets" / "media"
DEFAULT_SOURCE_ROOT = Path("/Users/yyzhao/Downloads/sana-streaming-example")
SOURCE_ROOT = Path(os.environ.get("SANA_STREAMING_SOURCE_ROOT", DEFAULT_SOURCE_ROOT))

LONG_WIDTH = 960
LONG_HEIGHT = 528
SHORT_WIDTH = 720
SHORT_HEIGHT = 405
FPS = 16


def run(cmd):
    print(" ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def ffprobe_size(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    width, height = result.stdout.strip().split("x")
    return int(width), int(height)


def referenced_videos():
    text = APP_JS.read_text(encoding="utf-8")
    paths = sorted(set(re.findall(r"ready-to-use/[^`]+?\.mp4", text)))
    if not paths:
        raise RuntimeError("No referenced videos found in app.js")
    return paths


def output_for_split(rel_path, half):
    return DEST_ROOT / rel_path.replace(".mp4", f"__{half}.mp4")


def transcode_split(src, rel_path, width, height):
    crop_h = height // 2
    for half, y in (("source", 0), ("edit", crop_h)):
      dest = output_for_split(rel_path, half)
      dest.parent.mkdir(parents=True, exist_ok=True)
      filters = (
          f"crop={width}:{crop_h}:0:{y},"
          f"scale={LONG_WIDTH}:{LONG_HEIGHT}:flags=lanczos,"
          f"fps={FPS},format=yuv420p"
      )
      run([
          "ffmpeg",
          "-y",
          "-i",
          str(src),
          "-vf",
          filters,
          "-an",
          "-c:v",
          "libx264",
          "-preset",
          "veryfast",
          "-crf",
          "30",
          "-movflags",
          "+faststart",
          str(dest),
      ])


def transcode_pair(src, rel_path):
    dest = DEST_ROOT / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    filters = (
        f"scale={SHORT_WIDTH}:{SHORT_HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={SHORT_WIDTH}:{SHORT_HEIGHT},"
        f"fps={FPS},format=yuv420p"
    )
    run([
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        filters,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "30",
        "-movflags",
        "+faststart",
        str(dest),
    ])


def main():
    if not SOURCE_ROOT.exists():
        raise RuntimeError(f"Source root does not exist: {SOURCE_ROOT}")

    shutil.rmtree(DEST_ROOT, ignore_errors=True)
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    for rel_path in referenced_videos():
        src = SOURCE_ROOT / rel_path
        if not src.exists():
            raise FileNotFoundError(src)
        width, height = ffprobe_size(src)
        if height >= width:
            transcode_split(src, rel_path, width, height)
        else:
            transcode_pair(src, rel_path)

    print(f"Wrote optimized videos to {DEST_ROOT}")


if __name__ == "__main__":
    main()
