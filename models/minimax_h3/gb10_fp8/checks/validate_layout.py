"""Which layout does the ComfyUI checkpoint actually store its QKV and SwiGLU weights in?

`convert_minimax_h3_to_diffusers.py` de-interleaves per-head QKV and swaps the SwiGLU halves
because that is what the *raw MiniMax shards* need. The ComfyUI file inherits MiniMax's key
names, so `h3_fp8` assumed it inherits their layout too — but ComfyUI may well have quantised
the already-converted tensors instead, in which case both transforms are one application too
many and attention comes out scrambled.

Block 0 settles it: the official repo ships it in diffusers layout, so every candidate
transform can be scored against ground truth. FP8 E4M3 has ~2 decimal digits, so the correct
transform should land near 1.0 and any wrong one near 0.
"""

import glob

import paths

paths.setup(need_sol_engine=False)
import sys

import torch
from safetensors.torch import safe_open

from h3_fp8 import _swap_swiglu_halves, reorder_interleaved_qkv

# Shard 1 of the released BF16 transformer holds block 0 in the diffusers layout, which is the
# ground truth both layout questions are settled against. It is 4.8 GiB and only this script
# needs it, so `bootstrap.sh` does not fetch it by default.
OFFICIAL = paths.h3_snapshot() + "/transformer/diffusion_pytorch_model-00001-of-00014.safetensors"
PRUNED = paths.dit_checkpoint()

HEADS, HEAD_DIM = 56, 128
INNER = HEADS * HEAD_DIM


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


with safe_open(OFFICIAL, framework="pt", device="cpu") as f:
    ref = {
        k: f.get_tensor(f"transformer_blocks.0.{k}")
        for k in ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight",
                  "attn.to_out.0.weight", "ff.net.0.proj.weight", "ff.net.2.weight")
    }

with safe_open(PRUNED, framework="pt", device="cpu") as f:
    qkv = f.get_tensor("blocks.0.attn.qkv_proj.weight")
    qkv_scale = f.get_tensor("blocks.0.attn.qkv_proj.weight_scale").float()
    out = f.get_tensor("blocks.0.attn.out_proj.weight")
    out_scale = f.get_tensor("blocks.0.attn.out_proj.weight_scale").float()
    fc1 = f.get_tensor("blocks.0.mlp.fc1.weight")
    fc1_scale = f.get_tensor("blocks.0.mlp.fc1.weight_scale").float()
    fc2 = f.get_tensor("blocks.0.mlp.fc2.weight")
    fc2_scale = f.get_tensor("blocks.0.mlp.fc2.weight_scale").float()

qkv = qkv.float() * qkv_scale
fc1 = fc1.float() * fc1_scale

# --- the controls: neither of these has an ambiguous layout ---------------------------
print("controls (pure renames, must be ~1.0):")
print(f"  out_proj -> attn.to_out.0   cos = {cosine(out.float() * out_scale, ref['attn.to_out.0.weight']):.6f}")
print(f"  mlp.fc2  -> ff.net.2        cos = {cosine(fc2.float() * fc2_scale, ref['ff.net.2.weight']):.6f}")

# --- QKV: de-interleave, or take contiguous thirds as they are? -----------------------
print("\nQKV layout:")
for label, tensor in (
    ("de-interleave then split (raw-shard layout)", reorder_interleaved_qkv(qkv, HEADS, HEAD_DIM)),
    ("split as-is (already [q;k;v])", qkv),
):
    q, k, v = tensor.split(INNER, dim=0)
    scores = [cosine(q, ref["attn.to_q.weight"]),
              cosine(k, ref["attn.to_k.weight"]),
              cosine(v, ref["attn.to_v.weight"])]
    print(f"  {label:44s} q={scores[0]:+.6f} k={scores[1]:+.6f} v={scores[2]:+.6f}")

# --- SwiGLU: swap the halves, or leave them? ------------------------------------------
print("\nSwiGLU layout:")
print(f"  swap halves ([gate;value] source)   cos = {cosine(_swap_swiglu_halves(fc1), ref['ff.net.0.proj.weight']):+.6f}")
print(f"  leave as-is ([value;gate] source)   cos = {cosine(fc1, ref['ff.net.0.proj.weight']):+.6f}")
