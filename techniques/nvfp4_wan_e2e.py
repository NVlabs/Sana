#!/usr/bin/env python3
"""End-to-end nvfp4 experiment on vanilla Wan (A14B) in the TE env.

Loads WanPipeline bf16, runs a baseline generation, then swaps selected dense
block linears to TransformerEngine te.Linear (NVFP4BlockScaling) and re-runs the
SAME seed under te.fp8_autocast. Reports end-to-end wall time + writes both videos
for offline PSNR/SSIM.
"""
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling

WEIGHTS = os.environ["WAN22_WEIGHTS"]
OUT = Path(os.environ["OUT_DIR"]); OUT.mkdir(parents=True, exist_ok=True)
H = int(os.environ.get("WAN_H", "480"))
W = int(os.environ.get("WAN_W", "832"))
F = int(os.environ.get("WAN_F", "81"))
STEPS = int(os.environ.get("WAN_STEPS", "20"))
G = float(os.environ.get("WAN_G", "4.0"))
G2 = float(os.environ.get("WAN_G2", "3.0"))
SHIFT = float(os.environ.get("WAN_SHIFT", "12.0"))
SEED = int(os.environ.get("WAN_SEED", "1024"))
# which block linears to quantize: "all" | "ffn" | "attn_ffn"
SCOPE = os.environ.get("NVFP4_SCOPE", "attn_ffn")
PROMPT = ("A lone hiker stands atop a towering cliff, silhouetted against the vast "
          "horizon. The rugged landscape stretches endlessly beneath. High angle, "
          "soft natural lighting emphasizing the grandeur of nature.")
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
       "整体发灰，最差质量，低质量")


def log(m):
    print(f"[nvfp4_wan] {m}", flush=True)


def load_pipe():
    vae = AutoencoderKLWan.from_pretrained(WEIGHTS, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(WEIGHTS, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    return pipe


def gen(pipe, use_nvfp4):
    g = torch.Generator("cuda").manual_seed(SEED)
    recipe = NVFP4BlockScaling() if use_nvfp4 else None
    torch.cuda.synchronize(); t = time.time()
    ctx = te.fp8_autocast(enabled=True, fp8_recipe=recipe) if use_nvfp4 else nullcontext()
    with torch.no_grad(), ctx:
        r = pipe(prompt=PROMPT, negative_prompt=NEG, height=H, width=W, num_frames=F,
                 num_inference_steps=STEPS, guidance_scale=G, guidance_scale_2=G2, generator=g)
    torch.cuda.synchronize()
    return time.time() - t, r.frames[0]


def want(parent_name, lin):
    if min(lin.in_features, lin.out_features) < 2048:
        return False
    if SCOPE == "all":
        return True
    if SCOPE == "ffn":
        return "ffn" in parent_name
    # attn_ffn: self-attn + ffn, skip cross-attn (attn2, small text-token M)
    return "ffn" in parent_name or "attn1" in parent_name


def swap(root):
    n = 0
    for mod_name, module in root.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and want(mod_name, child):
                tl = te.Linear(child.in_features, child.out_features,
                               bias=child.bias is not None, params_dtype=torch.bfloat16)
                with torch.no_grad():
                    tl.weight.copy_(child.weight)
                    if child.bias is not None:
                        tl.bias.copy_(child.bias)
                setattr(module, child_name, tl.to("cuda"))
                n += 1
    return n


def main():
    log(f"config H{H} W{W} F{F} steps{STEPS} scope={SCOPE} seed={SEED}")
    pipe = load_pipe()
    log("pipeline loaded")

    t_bf16, frames_bf16 = gen(pipe, use_nvfp4=False)
    export_to_video(frames_bf16, str(OUT / "bf16.mp4"), fps=16)
    log(f"bf16 end-to-end = {t_bf16:.2f}s")

    for name in ("transformer", "transformer_2"):
        m = getattr(pipe, name, None)
        if m is not None:
            k = swap(m)
            log(f"swapped {k} linears in {name} -> nvfp4")

    t_nvfp4, frames_nvfp4 = gen(pipe, use_nvfp4=True)
    export_to_video(frames_nvfp4, str(OUT / "nvfp4.mp4"), fps=16)
    log(f"nvfp4 end-to-end = {t_nvfp4:.2f}s")

    res = {
        "config": {"H": H, "W": W, "F": F, "steps": STEPS, "scope": SCOPE, "seed": SEED},
        "bf16_s": round(t_bf16, 3),
        "nvfp4_s": round(t_nvfp4, 3),
        "end_to_end_speedup": round(t_bf16 / t_nvfp4, 4),
    }
    (OUT / "nvfp4_result.json").write_text(json.dumps(res, indent=2))
    log(f"=== speedup {res['end_to_end_speedup']}x  ({t_bf16:.1f}s -> {t_nvfp4:.1f}s) ===")


if __name__ == "__main__":
    main()
