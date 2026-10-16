"""official MiniMax-H3 conditioner for HyperFlow first-frame (fl2va) clips.

Runs only HyperFlow's conditioner blocks (`before_encode` + `text_encoder`, the official BF16
Qwen3-VL) — the same split HyperFlow's SplitPipeline uses — and saves the state the remaining
blocks read (prompt_embeds, text_token_tags, keyframes on the canvas, anchors, size). The 5090
then runs the rest (VAE encode of the keyframe, HyperFlow DiT, decoders) unchanged.
Usage: official_conditioner.py MODEL_DIR jobs.json OUT_DIR
jobs.json: [{"id", "prompt", "first_frame": "path.png"}]
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict
from hyperflow_h3 import hyperflow_blocks

CONDITIONER = ("before_encode", "text_encoder")


def main():
    model, jobs, out = sys.argv[1], json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    blocks = hyperflow_blocks("fl2va")
    head = InsertableDict()
    for name in CONDITIONER:
        head[name] = blocks.sub_blocks.pop(name)
    rest_inputs = {p.name for p in blocks.inputs if p.name}
    conditioner = SequentialPipelineBlocks.from_blocks_dict(head).init_pipeline(model)
    conditioner.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=model, local_files_only=True)
    conditioner.to("cuda")
    cond_inputs = {p.name for p in conditioner.blocks.inputs if p.name}
    for job in jobs:
        image = Image.open(job["first_frame"]).convert("RGB")
        kwargs = {"prompt": job["prompt"], "image": image, "height": 768, "width": 1344}
        state = conditioner(**{k: v for k, v in kwargs.items() if k in cond_inputs})
        keep = {k: v for k, v in state.values.items() if k in rest_inputs or k not in cond_inputs}
        payload = {}
        for key, value in keep.items():
            payload[key] = value.detach().cpu() if isinstance(value, torch.Tensor) else value
        payload["task"] = "fl2va"
        payload["prompt"] = job["prompt"]
        torch.save(payload, out / (job["id"] + ".pt"))
        shape = list(payload["prompt_embeds"].shape) if "prompt_embeds" in payload else None
        print("CONDITIONED", json.dumps({"id": job["id"], "keys": sorted(payload), "embeds": shape}), flush=True)
    print("CONDITIONER_DONE", len(jobs), flush=True)


if __name__ == "__main__":
    main()
