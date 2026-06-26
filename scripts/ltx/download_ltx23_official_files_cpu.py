#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests

REPO_ID = "Lightricks/LTX-2.3"
DEFAULT_FILES = [
    "ltx-2.3-22b-dev.safetensors",
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
]


def parse_total(
    headers: requests.structures.CaseInsensitiveDict, downloaded_before: int
) -> int | None:
    content_range = headers.get("content-range")
    if content_range:
        match = re.search(r"/(\d+)$", content_range)
        if match:
            return int(match.group(1))
    content_length = headers.get("content-length")
    if content_length and content_length.isdigit():
        return downloaded_before + int(content_length)
    return None


def fmt_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}TB"


def download_file(
    filename: str, output_dir: Path, interval_s: float, chunk_size: int
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    partial = output_dir / f"{filename}.part"
    if target.exists() and target.stat().st_size > 0:
        print(
            f"[skip] {target} already exists ({fmt_bytes(target.stat().st_size)})",
            flush=True,
        )
        return target

    downloaded_before = partial.stat().st_size if partial.exists() else 0
    url = f"https://huggingface.co/{REPO_ID}/resolve/main/{quote(filename)}"
    headers = {}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if downloaded_before:
        headers["Range"] = f"bytes={downloaded_before}-"

    print(f"[download] {filename}", flush=True)
    if downloaded_before:
        print(f"[resume] existing partial={fmt_bytes(downloaded_before)}", flush=True)
    start = time.monotonic()
    last_t = start
    last_bytes = downloaded_before
    downloaded = downloaded_before

    with requests.get(
        url, headers=headers, stream=True, allow_redirects=True, timeout=(30, 180)
    ) as response:
        if response.status_code == 416 and partial.exists():
            partial.rename(target)
            print(f"[done] {target} ({fmt_bytes(target.stat().st_size)})", flush=True)
            return target
        response.raise_for_status()
        mode = "ab" if response.status_code == 206 and downloaded_before else "wb"
        if mode == "wb" and downloaded_before:
            print(
                "[warn] server did not honor Range request; restarting partial download",
                flush=True,
            )
            downloaded = 0
            last_bytes = 0
        total = parse_total(response.headers, downloaded if mode == "ab" else 0)
        print(
            f"[response] status={response.status_code} total={fmt_bytes(total) if total else 'unknown'} final_url_host={requests.utils.urlparse(response.url).netloc}",
            flush=True,
        )
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_t >= interval_s:
                    inst = (downloaded - last_bytes) / (now - last_t)
                    avg = (downloaded - downloaded_before) / max(now - start, 1e-6)
                    pct = f" {downloaded / total * 100:.2f}%" if total else ""
                    print(
                        f"[speed] file={filename} downloaded={fmt_bytes(downloaded)}{pct} inst={fmt_bytes(inst)}/s avg={fmt_bytes(avg)}/s",
                        flush=True,
                    )
                    last_t = now
                    last_bytes = downloaded
    partial.rename(target)
    elapsed = time.monotonic() - start
    new_bytes = target.stat().st_size - downloaded_before
    print(
        f"[done] {target} size={fmt_bytes(target.stat().st_size)} elapsed={elapsed:.1f}s avg={fmt_bytes(new_bytes / max(elapsed, 1e-6))}/s",
        flush=True,
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/LTX-2.3-official-files")
    parser.add_argument("--interval-s", type=float, default=15.0)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("files", nargs="*", default=DEFAULT_FILES)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    for filename in args.files:
        download_file(
            filename, output_dir, args.interval_s, args.chunk_mib * 1024 * 1024
        )


if __name__ == "__main__":
    main()
