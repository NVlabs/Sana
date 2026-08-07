"""Build the pruned FP8 DiT alone and check what it costs and whether it runs."""

import glob

import paths

paths.setup(need_sol_engine=False)
import json
import sys
import time

import torch


from h3_fp8 import build_pruned_fp8_transformer

SNAP = paths.h3_snapshot()
CKPT = paths.dit_checkpoint()

config = json.load(open(f"{SNAP}/transformer/config.json"))

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
model, info = build_pruned_fp8_transformer(CKPT, config, device="cuda")
torch.cuda.synchronize()
print(f"built in {time.perf_counter() - t0:.1f}s   quantized_layers={info['quantized_layers']}")
if info["missing"]:
    print(f"missing (first 8): {info['missing'][:8]}  total={len(info['missing'])}")

print(f"weights on device: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

from h3_fp8 import Fp8Linear

w8a8 = sum(1 for m in model.modules() if isinstance(m, Fp8Linear) and m.quantized_activations)
w8a16 = sum(1 for m in model.modules() if isinstance(m, Fp8Linear) and not m.quantized_activations)
print(f"Fp8Linear: {w8a8} w8a8 (_scaled_mm), {w8a16} w8a16 (dequantized weight)")
print("sample:", model.transformer_blocks[0].attn.to_q, "|", model.transformer_blocks[0].ff.net[2])
print("time_embedder ->", type(model.time_embedder).__name__,
      "table", tuple(model.time_embedder.table.shape))
print("adaln_proj.linear ->", model.transformer_blocks[0].adaln_proj.linear)
