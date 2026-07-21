"""GPU correctness test for SOL v3 (PISA hyvideo piecewise, aligned upstream).

Unlike the colmask kernel (route saturates ~0.53), the top-k piecewise kernel
CAN reach density 1.0 — top_k == all KV blocks makes every block exact — so
"density->1.0 must equal dense SDPA" is a real identity check here (bf16 tol).
At lower densities non-selected blocks contribute via centroids, so random-
tensor error is bounded and much smaller than hard-drop sparsity.

Run on a GB200 node with the sparse_attn_training venv.
"""
import os
import sys

import torch

sys.path.insert(0, ".")
os.environ["SOL_ATTN_STRICT"] = "1"

from techniques.sparse_backends import sol_attn_hunyuan_v2 as v2  # noqa: E402
from techniques.sparse_backends import sol_attn_hunyuan_v3 as v3  # noqa: E402

dev = "cuda"
assert torch.cuda.is_available(), "needs a CUDA GPU (SM90+)"
print("device cap:", torch.cuda.get_device_capability())


def rel_l2(a, b):
    return ((a - b).float().norm() / b.float().norm()).item()


def case(name, grid, tl, n_pad, B=1, H=8, D=128,
         densities=(1.0, 0.5, 0.15), with_v2=False):
    vl = grid[0] * grid[1] * grid[2]
    S = vl + tl
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    key_valid = torch.ones(B, S, dtype=torch.bool, device=dev)
    if n_pad:
        key_valid[:, S - n_pad:] = False
    am = torch.zeros(B, 1, 1, S, device=dev, dtype=torch.bfloat16).masked_fill(
        ~key_valid[:, None, None, :], float("-inf"))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=am)
    s_eff = S - n_pad

    print(f"\n=== {name}: grid={grid} video={vl} text={tl} pad={n_pad} S={S}")
    for dens in densities:
        try:
            o3 = v3.sol_attn_hunyuan_v3(
                q, k, v, video_len=vl, key_valid=key_valid,
                target_density=dens, block_size=64)
            # Compare only the valid (non-padded) rows; padded rows are zeros
            # by design in v3 and garbage-but-masked in the reference.
            r = rel_l2(o3[:, :, :s_eff], ref[:, :, :s_eff])
            mx = (o3[:, :, :s_eff] - ref[:, :, :s_eff]).abs().max().item()
            flag = ""
            if dens >= 1.0:
                flag = "  [ok: identity]" if r < 5e-3 else "  <-- FAIL (must equal dense at density 1.0)"
            print(f"  density={dens:<5} v3: rel_l2={r:.5f} max_abs={mx:.5f}{flag}")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  density={dens:<5} v3: FAILED {type(exc).__name__}: {exc}")
        if with_v2 and dens < 1.0:
            v2._TAU_CACHE_V2.clear()
            try:
                o2 = v2.sol_attn_hunyuan_v2(
                    q, k, v, video_len=vl, key_valid=key_valid, grid=grid,
                    target_density=dens, block_size=64)
                r2 = rel_l2(o2[:, :, :s_eff], ref[:, :, :s_eff])
                print(f"  density={dens:<5} v2(colmask, hard-drop): rel_l2={r2:.5f}")
            except Exception as exc:
                print(f"  density={dens:<5} v2: FAILED {type(exc).__name__}: {exc}")


def case_morton(name, grid, tl, n_pad, B=1, H=8, D=128,
                densities=(1.0, 0.15)):
    """Morton variant: identity at d=1.0 must still hold (permutation-exact);
    at low density print raster vs morton side by side."""
    vl = grid[0] * grid[1] * grid[2]
    S = vl + tl
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    key_valid = torch.ones(B, S, dtype=torch.bool, device=dev)
    if n_pad:
        key_valid[:, S - n_pad:] = False
    am = torch.zeros(B, 1, 1, S, device=dev, dtype=torch.bfloat16).masked_fill(
        ~key_valid[:, None, None, :], float("-inf"))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=am)
    s_eff = S - n_pad
    print(f"\n=== {name} (morton): grid={grid} video={vl} text={tl} pad={n_pad}")
    for dens in densities:
        for tag, mort in (("morton", True), ("raster", False)):
            o3 = v3.sol_attn_hunyuan_v3(
                q, k, v, video_len=vl, key_valid=key_valid,
                target_density=dens, block_size=64, grid=grid, morton=mort)
            r = rel_l2(o3[:, :, :s_eff], ref[:, :, :s_eff])
            flag = ""
            if dens >= 1.0:
                flag = "  [ok: identity]" if r < 5e-3 else "  <-- FAIL"
            print(f"  density={dens:<5} v3-{tag}: rel_l2={r:.5f}{flag}")


case_morton("padded-text", (8, 16, 16), tl=128, n_pad=48)
case("padded-text", (8, 16, 16), tl=128, n_pad=48, with_v2=True)
case("no-padding", (8, 16, 16), tl=128, n_pad=0, densities=(1.0, 0.15))
case("mostly-pad", (8, 16, 16), tl=256, n_pad=224, densities=(1.0, 0.15))
case("ragged", (5, 9, 16), tl=128, n_pad=48, densities=(1.0, 0.15))
# Hunyuan-scale sanity at reduced heads (118800 video tokens is the real shape;
# use a 16k-token slice shape to keep the test fast but block-count realistic).
case("mid-scale", (16, 32, 32), tl=256, n_pad=176, densities=(0.15,), with_v2=True)

print("\nDONE")
