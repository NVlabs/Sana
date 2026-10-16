"""Download this directory's media from the Sana-assets dataset on Hugging Face into ./assets/.

  assets/film/      the "Sol-H3 in ComfyUI" film and poster
  assets/demo/      earlier demo clips used by scripts/studio_preview.html
  assets/gallery/   Studio gallery clips (pass to configure.py --gallery)
  assets/promo/     sources to re-render the film (scripts/promo/)

Usage: python3 scripts/fetch_assets.py [film demo gallery promo]   (default: all)
Uses only the standard library; no token is needed (public dataset).
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = "https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets"
PREFIX = "Sol-Engine/Sol-H3-ComfyUI"
ROOT = Path(__file__).resolve().parents[1] / "assets"


def retry(action, attempts=5):
    for attempt in range(attempts):
        try:
            return action()
        except OSError as error:  # URLError and connection resets
            if attempt == attempts - 1:
                raise
            print("retrying after", error)
            time.sleep(2 * (attempt + 1))


def listing(folder):
    url = f"https://huggingface.co/api/datasets/Efficient-Large-Model/Sana-assets/tree/main/{PREFIX}/{folder}?recursive=true"

    def read():
        with urllib.request.urlopen(url, timeout=60) as response:
            return [row for row in json.load(response) if row["type"] == "file"]
    return retry(read)


def main():
    parts = sys.argv[1:] or ["film", "demo", "gallery", "promo"]
    if set(parts) - {"film", "demo", "gallery", "promo"}:
        sys.exit(__doc__)
    for part in parts:
        for row in listing(part):
            relative = row["path"][len(PREFIX) + 1:]
            target = ROOT / relative
            if target.is_file() and target.stat().st_size == row.get("size"):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            retry(lambda: urllib.request.urlretrieve(f"{REPO}/resolve/main/{row['path']}", temporary))
            temporary.replace(target)
            print("fetched", relative)
    print("assets in", ROOT)


if __name__ == "__main__":
    main()
