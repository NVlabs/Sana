"""What exactly do the two quantisers disagree about, on block 39's real activation?

0.94% of that layer's elements differ, which is far too many for tie-breaking alone. The
candidates are the rounding mode of the FP32->FP8 downcast and the saturation behaviour at
the top of the E4M3 range, and they leave different fingerprints: ties differ by one code
anywhere in the range, saturation only at the boundary.
"""
import sys

import paths

paths.setup(need_sol_engine=False)
import torch


import h3_fp8
from bench_dit import load_inputs, load_transformer

model, _ = load_transformer(fuse_qkv=True, quantizer="eager")
inputs = load_inputs(24)["tensors"]

grabbed = {}
target = model.get_submodule("transformer_blocks.39.attn.to_out.0")
handle = target.register_forward_pre_hook(
    lambda mod, args: grabbed.setdefault("x", args[0].detach().clone())
)
with torch.no_grad():
    model(**inputs, return_dict=False)
handle.remove()

x = grabbed["x"]
flat = x.reshape(-1, x.shape[-1]).to(target.compute_dtype)
scale = target.input_scale
print(f"input {tuple(flat.shape)} {flat.dtype}, input_scale={scale.item():.6g}")

ratio = flat.float() / scale
print(f"x/scale: max {ratio.abs().max():.1f}, {(ratio.abs() > 448).float().mean():.3%} past the clamp")

a = h3_fp8._quantize_eager(flat, scale)
b = h3_fp8._quantize_compiled(flat, scale)
mask = a.view(torch.uint8) != b.view(torch.uint8)
print(f"differing: {mask.float().mean():.4%} ({int(mask.sum())} elements)")

idx = mask.nonzero()
sel = idx[torch.randperm(len(idx))[:10]]
print(f"\n{'x/scale':>14s} {'eager':>10s} {'compiled':>10s} {'codes':>12s}")
for i, j in sel.tolist():
    print(f"{ratio[i, j]:14.4f} {a[i, j].float():10.2f} {b[i, j].float():10.2f} "
          f"{a.view(torch.uint8)[i, j]:6d}/{b.view(torch.uint8)[i, j]:<6d}")

r = ratio[mask].abs()
print(f"\ndiffering elements' |x/scale|: min {r.min():.3f}, max {r.max():.3f}, "
      f"median {r.median():.3f}")
print(f"fraction of them at/over the clamp: {(r >= 448).float().mean():.2%}")
