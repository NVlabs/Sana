"""GPU correctness test for the HunyuanVideo SOL split-merge (v2 + v1).

The colmask kernel's route saturates at ~0.5-0.6 density even at tau=0, so the
old "density->1.0 must equal dense" diagnostic is unachievable through the real
kernel. Instead this splits the validation:

  A. PLUMBING/MERGE — inject a pure-torch dense "kernel" (same (out, lse)
     contract: natural-log LSE of q.k*scale) into the v2 and v1 paths. The
     full split-merge must then equal full dense SDPA over [video, text] with
     the padding mask, up to bf16 error. This isolates the LSE merge
     convention, Morton reorder round-trip, masking, and concat exactly.
  B. KERNEL LSE SANITY — the real kernel's LSE is a routed-subset LSE, so
     exp(lse_kernel - lse_dense_video) must lie in (0, 1]. A wrong base
     (log2 vs ln) or missing scale would blow this ratio up or crush it.
  C. DEGENERACY — with ALL text keys padded away, the v2 video rows must
     equal the raw kernel output exactly (merge weight on text -> 0).

Run on an SM100 (GB200) node with the sparse_attn_training venv.
"""
import os
import sys

import torch

sys.path.insert(0, ".")
os.environ["SOL_ATTN_STRICT"] = "1"
os.environ.setdefault("SOL_ATTN_ALLOW_LOW_TAU", "1")

from techniques.sparse_backends import sol_attn_backend as v1  # noqa: E402
from techniques.sparse_backends import sol_attn_hunyuan_v2 as v2  # noqa: E402

dev = "cuda"
assert torch.cuda.is_available(), "needs an SM100 GPU"
print("device cap:", torch.cuda.get_device_capability())
SCALE = 128 ** -0.5


def rel_l2(a, b):
    return ((a - b).float().norm() / b.float().norm()).item()


def make_case(grid, tl, n_pad, B=1, H=8, D=128, seed=0):
    vl = grid[0] * grid[1] * grid[2]
    S = vl + tl
    torch.manual_seed(seed)
    q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    key_valid = torch.ones(B, S, dtype=torch.bool, device=dev)
    if n_pad:
        key_valid[:, S - n_pad:] = False
    am = torch.zeros(B, 1, 1, S, device=dev, dtype=torch.bfloat16).masked_fill(
        ~key_valid[:, None, None, :], float("-inf"))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=am)
    return q, k, v, key_valid, vl, ref


def dense_lse_attn(q, k, v):
    """Pure-torch dense attention returning (out bf16, lse fp32) — the exact
    contract of integrations.wan.run(return_lse=True)."""
    s = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * SCALE
    lse = torch.logsumexp(s, dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", torch.softmax(s, dim=-1), v.float())
    return out.to(torch.bfloat16), lse.float()


FAKE = {
    "run": lambda q, k, v, *, tau, block_size, return_lse=False:
        dense_lse_attn(q, k, v) if return_lse else dense_lse_attn(q, k, v)[0],
    "calibrate_tau": lambda *a, **kw: {"threshold": 0.0},
}

# ---------------------------------------------------------------- A. plumbing
print("\n[A] merge/plumbing with dense fake kernel (must match dense SDPA)")
_orig_v2_loader = v2._load_colmask_v2
_orig_v1_loader = v1._load_colmask
v2._load_colmask_v2 = lambda: FAKE
v1._load_colmask = lambda: FAKE
try:
    for name, grid, tl, n_pad in (
        ("padded-text", (8, 16, 16), 128, 48),
        ("no-padding", (8, 16, 16), 128, 0),
        ("mostly-pad", (8, 16, 16), 256, 224),
        ("ragged-grid", (5, 9, 16), 128, 48),
    ):
        q, k, v, key_valid, vl, ref = make_case(grid, tl, n_pad)
        o2 = v2.sol_attn_hunyuan_v2(
            q, k, v, video_len=vl, key_valid=key_valid, grid=grid,
            target_density=0.15, block_size=64)
        o1 = v1.sol_attn_attention_hunyuan(
            q, k, v, video_len=vl, key_valid=key_valid, grid=grid,
            target_density=0.15, block_size=64)
        r2, r1 = rel_l2(o2, ref), rel_l2(o1, ref)
        ok2 = "ok" if r2 < 5e-3 else "FAIL <-- merge convention broken"
        ok1 = "ok" if r1 < 5e-3 else "FAIL <-- merge convention broken"
        print(f"  {name:<12} v2: rel_l2={r2:.6f} [{ok2}]   v1: rel_l2={r1:.6f} [{ok1}]")
finally:
    v2._load_colmask_v2 = _orig_v2_loader
    v1._load_colmask = _orig_v1_loader

# ------------------------------------------------------- B. kernel LSE sanity
print("\n[B] real-kernel LSE sanity: exp(lse_kernel - lse_dense_video) in (0, 1]")
v2._TAU_CACHE_V2.clear()
q, k, v, key_valid, vl, ref = make_case((8, 16, 16), 128, 48)
cm = v2._load_colmask_v2()
perm, inv = v2._morton3d_perm_v2((8, 16, 16), dev)
qv = q[:, :, :vl][:, :, perm].contiguous()
kv = k[:, :, :vl][:, :, perm].contiguous()
vv = v[:, :, :vl][:, :, perm].contiguous()
tau = float(cm["calibrate_tau"](qv, kv, vv, target_density=0.15,
                                block_size=64)["threshold"])
out_k, lse_k = cm["run"](qv, kv, vv, tau=tau, block_size=64, return_lse=True)
_, lse_dense = dense_lse_attn(qv, kv, vv)
ratio = torch.exp(lse_k.float() - lse_dense)
print(f"  tau={tau:.4f} mass-fraction: min={ratio.min().item():.4f} "
      f"mean={ratio.mean().item():.4f} max={ratio.max().item():.4f} "
      f"[{'ok' if 0 < ratio.min().item() and ratio.max().item() < 1.02 else 'FAIL <-- LSE units wrong'}]")

# ------------------------------------------------------------- C. degeneracy
print("\n[C] all-text-padded: v2 video rows must equal raw kernel output")
key_valid_none = key_valid.clone()
key_valid_none[:, vl:] = False
o2 = v2.sol_attn_hunyuan_v2(
    q, k, v, video_len=vl, key_valid=key_valid_none, grid=(8, 16, 16),
    target_density=0.15, block_size=64)
kernel_back = out_k[:, :, inv, :]
d = rel_l2(o2[:, :, :vl], kernel_back)
print(f"  rel_l2(v2 video rows, kernel)={d:.6f} [{'ok' if d < 1e-3 else 'FAIL'}]")

# --------------------------------------------- reference: realized densities
print("\n[info] realized sparse error on random tensors at density 0.15 is large "
      "by construction (near-uniform softmax); real-model quality is validated "
      "by the end-to-end HunyuanVideo run + frame metrics, not here.")
print("DONE")
