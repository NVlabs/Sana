"""Promo keyframes with Qwen-Image-2.1 (official Diffusers QwenImage21Pipeline).

Native 16:9 2K (2752x1536), 40 steps, default guidance; saved full size and as the
1344x768 first frame HyperFlow consumes. Usage: qwen_keyframes.py keyframes.json OUT [--seeds 11,22]
"""
import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("keyframes", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--seeds", default="11,22")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    items = json.loads(args.keyframes.read_text())
    pipe = QwenImage21Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    report = []
    for item in items:
        for seed in (int(s) for s in args.seeds.split(",")):
            t0 = time.perf_counter()
            image = pipe(prompt=item["image"], width=2752, height=1536, num_inference_steps=40,
                         generator=torch.Generator("cuda").manual_seed(seed)).images[0].convert("RGB")
            full = args.out / ("%s-s%d-2k.png" % (item["id"], seed))
            image.save(full)
            image.resize((1344, 768), resample=3).save(args.out / ("%s-s%d.png" % (item["id"], seed)))
            row = {"id": item["id"], "seed": seed, "seconds": time.perf_counter() - t0}
            report.append(row)
            print("KEYFRAME", json.dumps(row), flush=True)
    (args.out / "keyframes-report.json").write_text(json.dumps(
        {"model": args.model, "size": [2752, 1536], "steps": 40, "items": report}, indent=2))
    print("KEYFRAMES_DONE", len(report), flush=True)


if __name__ == "__main__":
    main()
