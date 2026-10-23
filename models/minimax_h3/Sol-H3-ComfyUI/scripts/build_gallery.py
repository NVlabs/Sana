"""Assemble the Studio gallery folder (config.json "gallery") from generated showcase clips.

Each entry becomes NAME.mp4 + NAME.json {title, title_zh, prompt, mode, image, pipeline, order} (+ NAME.jpg, the
first frame the clip was generated from, for first-frame entries). The manifest lists the entries:
[{"name", "video", "prompt", "title", "title_zh", "mode": "text|first_frame|reference", "image"?: path,
  "pipeline"?: "one_stage|two_stage"}] with video/image paths relative to the manifest. Clips are copied unchanged.
Usage: build_gallery.py MANIFEST.json OUT_DIR
"""
import json
import shutil
import sys
from pathlib import Path


def main():
    manifest, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for order, entry in enumerate(json.loads(manifest.read_text())):
        name = entry["name"]
        shutil.copyfile(manifest.parent / entry["video"], out / (name + ".mp4"))
        meta = {"title": entry["title"], "title_zh": entry.get("title_zh"), "prompt": entry["prompt"],
                "mode": entry.get("mode", "text"), "pipeline": entry.get("pipeline", "one_stage"), "order": order}
        if entry.get("image"):
            from PIL import Image
            image = Image.open(manifest.parent / entry["image"]).convert("RGB")
            image.thumbnail((1344, 1344))
            image.save(out / (name + ".jpg"), quality=90)
            meta["image"] = name + ".jpg"
        (out / (name + ".json")).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
        print("GALLERY", name, meta["mode"], meta["pipeline"])


if __name__ == "__main__":
    main()
