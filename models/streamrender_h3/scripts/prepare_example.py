"""Select an example and build a local runtime asset manifest."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = "B10_e-track-3_bluehour"


def requests():
    return [json.loads(line) for name in ("requests_A.jsonl", "requests_B.jsonl")
            for line in (ROOT / "examples" / name).read_text().splitlines() if line.strip()]


def prepare(asset_manifest, example_name, media_root, output):
    matches = [row for row in requests() if row["name"] == example_name]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous example: {example_name}")
    row = matches[0]
    source = Path(asset_manifest).resolve()
    document = json.loads(source.read_text())
    assets = document["assets"]
    # Moving the manifest must not change the meaning of relative weight paths.
    for key, value in assets.items():
        if key != "prompt":
            path = Path(value).expanduser()
            assets[key] = str(path.resolve() if path.is_absolute()
                              else (source.parent / path).resolve())
    picture = (Path(media_root).resolve() / row["picture"]).resolve()
    video = (Path(media_root).resolve() / row["reference_video_path"]).resolve()
    if not picture.is_file() or not video.is_file():
        raise FileNotFoundError("Download/extract the GitHub example media first.")
    assets["prompt"] = row["prompt"]
    assets["reference_image"] = str(picture)
    document["default_example"] = example_name
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    return row, video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--example", default=DEFAULT)
    parser.add_argument("--assets", help="Downloaded HF assets.json")
    parser.add_argument("--media-root", default=str(ROOT))
    parser.add_argument("--output", default=str(ROOT / "configs/example.local.json"))
    args = parser.parse_args()
    if args.list:
        for row in requests():
            print(row["name"])
    else:
        if not args.assets:
            parser.error("--assets is required")
        row, video = prepare(args.assets, args.example, args.media_root, args.output)
        print(json.dumps({"example": row["name"], "assets": str(Path(args.output).resolve()),
                          "reference_video": str(video)}, indent=2))
