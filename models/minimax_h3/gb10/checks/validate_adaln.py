"""Does the pruned rank-8 AdaLN reproduce the released one, and how is its table indexed?

The pruned checkpoint replaces `t -> time_proj -> time_embedder -> silu -> Linear(2688 -> 96768)`
with `adaln_t_table[idx(t)] -> Linear(8 -> 96768)`. Nothing documents `idx`, and getting it
wrong produces a model that runs and returns noise. This compares both paths against the
released BF16 weights for block 0, over the whole unit interval, and scans the candidate
index conventions.
"""

import sys

import paths

paths.setup(need_sol_engine=False)

import torch
from safetensors.torch import safe_open


from diffusers.models.embeddings import TimestepEmbedding, Timesteps

OFFICIAL = (
    paths.h3_snapshot() + "/transformer/diffusion_pytorch_model-00001-of-00014.safetensors"
)
PRUNED = paths.dit_checkpoint()

dev = "cpu"

with safe_open(OFFICIAL, framework="pt", device="cpu") as f:
    have = set(f.keys())
    need = [
        "time_embedder.linear_1.weight", "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight", "time_embedder.linear_2.bias",
        "transformer_blocks.0.adaln_proj.linear.weight",
        "transformer_blocks.0.adaln_proj.linear.bias",
    ]
    missing = [k for k in need if k not in have]
    if missing:
        raise SystemExit(f"probe shard is missing {missing}")
    ref = {k: f.get_tensor(k) for k in need}

with safe_open(PRUNED, framework="pt", device="cpu") as f:
    table = f.get_tensor("adaln_t_table")
    w8 = f.get_tensor("blocks.0.adaln_proj.linear.weight")
    b8 = f.get_tensor("blocks.0.adaln_proj.linear.bias")

print(f"official adaln_proj.linear.weight {tuple(ref['transformer_blocks.0.adaln_proj.linear.weight'].shape)}"
      f" {ref['transformer_blocks.0.adaln_proj.linear.weight'].dtype}")
print(f"pruned   adaln_proj.linear.weight {tuple(w8.shape)} {w8.dtype}")
print(f"adaln_t_table {tuple(table.shape)} {table.dtype}  "
      f"range [{table.min():.4f}, {table.max():.4f}]")

# --- free check: the bias sits outside the factorisation, so it should survive untouched ---
b_ref = ref["transformer_blocks.0.adaln_proj.linear.bias"].float()
b_pru = b8.float()
print(f"\nbias  max|diff| = {(b_ref - b_pru).abs().max():.3e}   "
      f"rel = {((b_ref - b_pru).abs().max() / b_ref.abs().max()):.3e}   "
      f"exact = {torch.equal(b_ref, b_pru)}")

# --- reference path -----------------------------------------------------------------
time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
embedder = TimestepEmbedding(in_channels=256, time_embed_dim=5376, out_dim=2688)
embedder.linear_1.weight.data = ref["time_embedder.linear_1.weight"].float()
embedder.linear_1.bias.data = ref["time_embedder.linear_1.bias"].float()
embedder.linear_2.weight.data = ref["time_embedder.linear_2.weight"].float()
embedder.linear_2.bias.data = ref["time_embedder.linear_2.bias"].float()
embedder = embedder.to(dev).eval()

W_ref = ref["transformer_blocks.0.adaln_proj.linear.weight"].to(dev, torch.float32)
b_ref = b_ref.to(dev)
W8 = w8.to(dev, torch.float32)
b_pru = b_pru.to(dev)
table = table.to(dev, torch.float32)


@torch.no_grad()
def reference(t: torch.Tensor) -> torch.Tensor:
    temb = embedder(time_proj(t).to(dev, torch.float32))
    return torch.nn.functional.silu(temb) @ W_ref.T + b_ref


@torch.no_grad()
def pruned(rows: torch.Tensor) -> torch.Tensor:
    return rows @ W8.T + b_pru


# --- scan the index conventions -----------------------------------------------------
rows = table.shape[0] - 1
ts = torch.linspace(0, 1, 41)

candidates = {
    "round(t*1024)": lambda t: (t * rows).round().long().clamp(0, rows),
    "floor(t*1024)": lambda t: (t * rows).floor().long().clamp(0, rows),
    "round(t*1000)": lambda t: (t * 1000).round().long().clamp(0, rows),
    "round((1-t)*1024)": lambda t: ((1 - t) * rows).round().long().clamp(0, rows),
}

print(f"\n{'convention':22s} {'mean rel err':>14s} {'max rel err':>14s}")
best = None
for name, fn in candidates.items():
    ref_out = reference(ts)
    pru_out = pruned(table.index_select(0, fn(ts).to(dev)))
    denom = ref_out.abs().mean()
    mean_err = (ref_out - pru_out).abs().mean() / denom
    max_err = (ref_out - pru_out).abs().max() / ref_out.abs().max()
    print(f"{name:22s} {mean_err:14.3e} {max_err:14.3e}")
    if best is None or mean_err < best[1]:
        best = (name, mean_err)

print(f"\nbest: {best[0]}  (mean rel err {best[1]:.3e})")

# --- how good is it on the grid points themselves? ----------------------------------
fn = candidates[best[0]]
exact_ts = torch.arange(0, rows + 1, 64).float() / rows
ref_out = reference(exact_ts)
pru_out = pruned(table.index_select(0, fn(exact_ts).to(dev)))
print(f"\non-grid t: mean rel err {(ref_out - pru_out).abs().mean() / ref_out.abs().mean():.3e}")

# --- is rank 8 actually enough? SVD of the true map --------------------------------
with torch.no_grad():
    dense = reference(torch.linspace(0, 1, 257)) - b_ref
    s = torch.linalg.svdvals(dense.double())
    energy = (s**2).cumsum(0) / (s**2).sum()
print(f"\nSVD of the true modulation map over t (rank-8 hypothesis):")
print(f"   singular values 1..10: {[f'{v:.3e}' for v in s[:10].tolist()]}")
print(f"   energy captured by rank 8: {energy[7]:.10f}")
print(f"   residual after rank 8:     {1 - energy[7]:.3e}")
