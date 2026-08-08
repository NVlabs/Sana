"""Is one DiT forward reproducible at all, run to run, within a process?

Every "lossless" verdict so far assumes it is. The compiled quantiser is bit-exact against
eager in every regime tested — no clamping, heavy clamping, at the boundary, across shapes
and scales — and every individual `Fp8Linear` returns bit-identical output, yet the full
forward moved 3.9%. Before hunting further, check the control that was never run.
"""
import sys

import paths

paths.setup(need_sol_engine=False)
import torch


import h3_fp8
from bench_dit import load_inputs, load_transformer

model, _ = load_transformer(fuse_qkv=True, quantizer="eager")
inputs = load_inputs(24)["tensors"]

runs = []
with torch.no_grad():
    for i in range(3):
        runs.append([t.float().cpu() for t in model(**inputs, return_dict=False)])

print("same model, same inputs, same quantiser, three consecutive forwards:")
for i in range(1, 3):
    for j, kind in enumerate(("video", "audio")):
        a, b = runs[0][j], runs[i][j]
        print(f"  run0 vs run{i}  {kind}: max|d|={(a - b).abs().max():.3e}  "
              f"mean|d|/mean|ref|={(a - b).abs().mean() / a.abs().mean():.3e}  "
              f"equal={torch.equal(a, b)}")
