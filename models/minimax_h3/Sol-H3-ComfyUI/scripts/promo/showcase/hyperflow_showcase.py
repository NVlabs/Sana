"""HyperFlow first-frame (fl2va) showcase clips on one RTX 5090.

Uses the plugin's HyperFlow engine directly with conditioning prepared by the official BF16
conditioner (official_conditioner.py, run on a larger GPU). Promotional content only, not a benchmark.
Usage: hyperflow_showcase.py OPTIONS.json PAYLOAD_DIR OUT_DIR [--seed 42]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[3] / "sol_h3_engine"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("options", type=Path)
    parser.add_argument("payloads", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sys.path.insert(0, str(ENGINE))
    from hyperflow_worker import HyperFlow
    import torch
    options = json.loads(args.options.read_text())
    options["hyperflow"]["workflows"] = ["fl2va"]
    options["work_dir"] = str(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    paths = {"h3_model": options["h3_model"]} if "h3_model" in options else \
        json.loads(Path(options["paths"]).read_text())
    engine = HyperFlow(options, paths)
    print("LOADED", round(engine.load_s, 1), flush=True)
    for path in sorted(args.payloads.glob("*.pt")):
        target = args.out / (path.stem + ".mp4")
        if target.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        work = args.out / path.stem
        work.mkdir(exist_ok=True)
        t0 = time.monotonic()
        row = engine.generate({"seed": args.seed, "task": "fl2va"}, payload, work, lambda *a: None)
        Path(row["output"]).replace(target)
        print("DONE", path.stem, round(time.monotonic() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
