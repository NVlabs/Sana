"""Compare the two quantisers on the activations the model actually produces.

Everything tested so far was synthetic: `randn`, scaled `randn`, values at and past the
clamp. All bit-exact. The real forward is the one input distribution left, so hook every
quantised linear and compare both quantisers on the tensor it is really handed.
"""
import sys

import paths

paths.setup(need_sol_engine=False)
import torch


import h3_fp8
from bench_dit import load_inputs, load_transformer

model, _ = load_transformer(fuse_qkv=True, quantizer="eager")
inputs = load_inputs(24)["tensors"]

stats = []

def make_hook(name, module):
    def hook(mod, args):
        x = args[0]
        flat = x.reshape(-1, x.shape[-1]).to(mod.compute_dtype)
        a = h3_fp8._quantize_eager(flat, mod.input_scale)
        c = h3_fp8._quantize_compiled(flat, mod.input_scale)
        t = h3_fp8._quantize_triton(flat, mod.input_scale)
        stats.append((
            name,
            float((c.view(torch.uint8) != a.view(torch.uint8)).float().mean()),
            float((t.view(torch.uint8) != a.view(torch.uint8)).float().mean()),
            float(flat.abs().max()),
        ))
    return hook

handles = []
for name, module in model.named_modules():
    if isinstance(module, h3_fp8.Fp8Linear) and module.quantized_activations:
        handles.append(module.register_forward_pre_hook(make_hook(name, module)))

with torch.no_grad():
    model(**inputs, return_dict=False)
for h in handles:
    h.remove()

print(f"quantised linears seen: {len(stats)}")
print(f"  vs eager, compiled disagrees on: {sum(1 for s in stats if s[1] > 0)} layers")
print(f"  vs eager, triton   disagrees on: {sum(1 for s in stats if s[2] > 0)} layers")
worst = sorted(stats, key=lambda s: -max(s[1], s[2]))[:10]
print(f"\n{'layer':44s} {'compiled':>10s} {'triton':>10s} {'max|x|':>9s}")
for name, comp, trit, mx in worst:
    print(f"{name:44s} {comp:9.4%} {trit:9.4%} {mx:9.1f}")
