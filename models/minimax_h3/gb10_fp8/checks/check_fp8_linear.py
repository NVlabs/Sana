"""Locate the 3.9% that appeared when the activation quantiser was compiled.

The quantiser itself is bit-exact against eager across every shape and scale the model uses,
so the discrepancy has to enter somewhere between it and `_scaled_mm`. This drives real
`Fp8Linear` modules out of the built model, on a real captured activation, and compares the
two quantiser paths layer by layer.
"""

import sys

import paths

paths.setup(need_sol_engine=False)

import torch


import h3_fp8
from bench_dit import load_inputs, load_transformer

model, _ = load_transformer(fuse_qkv=True, quantizer="triton")
inputs = load_inputs(24)["tensors"]

# A real activation of the width the block stack runs at.
torch.manual_seed(0)
probe = torch.randn(1, 38247, 5376, device="cuda", dtype=torch.bfloat16)

targets = [
    ("transformer_blocks.0.attn.to_qkv", probe),
    ("transformer_blocks.0.ff.net.0.proj", probe),
    ("transformer_blocks.25.attn.to_qkv", probe),
    ("transformer_blocks.49.attn.to_qkv", probe),
]

print(f"{'module':40s} {'quantised input':>16s} {'linear output':>16s}")
for name, x in targets:
    module = model.get_submodule(name)
    flat = x.reshape(-1, x.shape[-1])

    q_compiled = h3_fp8._quantize_compiled(flat, module.input_scale)
    q_eager = h3_fp8._quantize_eager(flat, module.input_scale)
    same_q = torch.equal(q_compiled.view(torch.uint8), q_eager.view(torch.uint8))

    module._quantize = h3_fp8._quantize_compiled
    out_compiled = module(x)
    module._quantize = h3_fp8._quantize_eager
    out_eager = module(x)
    same_out = torch.equal(out_compiled, out_eager)
    delta = (out_compiled.float() - out_eager.float()).abs().max()

    print(f"{name:40s} {str(same_q):>16s} {str(same_out) + f' ({delta:.2e})':>16s}")

print(f"\nquantised-input strides — compiled {q_compiled.stride()}, eager {q_eager.stride()}")
print(f"contiguous — compiled {q_compiled.is_contiguous()}, eager {q_eager.is_contiguous()}")

# Run the whole model both ways and see whether the gap is even reproducible.
print("\nfull forward, same process, both paths:")
outs = {}
for label, fn in (("eager", h3_fp8._quantize_eager), ("compiled", h3_fp8._quantize_compiled)):
    for module in model.modules():
        if isinstance(module, h3_fp8.Fp8Linear) and module.quantized_activations:
            module._quantize = fn
    with torch.no_grad():
        outs[label] = [t.float().cpu() for t in model(**inputs, return_dict=False)]

for i, kind in enumerate(("video", "audio")):
    a, b = outs["eager"][i], outs["compiled"][i]
    print(f"  {kind}: max|d|={(a - b).abs().max():.3e}  "
          f"mean|d|/mean|ref|={(a - b).abs().mean() / a.abs().mean():.3e}  "
          f"equal={torch.equal(a, b)}")
