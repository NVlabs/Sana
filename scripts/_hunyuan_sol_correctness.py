"""GPU correctness test for the HunyuanVideo SOL split-merge path.

At density->1.0 the sparse video x video degenerates to FULL video x video, so
sol_attn_attention_hunyuan must equal full dense attention over [video, text]
(with the padding mask) up to bf16 error. A large error there means the LSE
merge convention is wrong. At lower density the error is the sparse approximation.
Run on an SM100 (GB200) node with the sparse_attn_training venv.
"""
import sys
import torch

sys.path.insert(0, ".")
from techniques.sparse_backends import sol_attn_backend as sb  # noqa: E402

dev = "cuda"
assert torch.cuda.is_available(), "needs SM100 GPU"
print("device cap:", torch.cuda.get_device_capability())

B, H, D = 1, 8, 128
grid = (8, 16, 16)
vl = grid[0] * grid[1] * grid[2]      # 2048 video tokens
tl = 128                               # text (with padding)
S = vl + tl
torch.manual_seed(0)
q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
key_valid = torch.ones(B, S, dtype=torch.bool, device=dev)
key_valid[:, vl + 80:] = False         # last 48 text tokens are padding

am = torch.zeros(B, 1, 1, S, device=dev, dtype=torch.bfloat16).masked_fill(
    ~key_valid[:, None, None, :], float("-inf"))
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=am)


def rel_l2(a, b):
    return ((a - b).float().norm() / b.float().norm()).item()


print(f"shape B={B} H={H} video={vl} text={tl} (pad {int((~key_valid).sum().item())})")
for dens in [1.0, 0.99, 0.5, 0.15]:
    sb._TAU_CACHE.clear()
    try:
        out = sb.sol_attn_attention_hunyuan(
            q, k, v, video_len=vl, key_valid=key_valid, grid=grid,
            target_density=dens, block_size=64)
        print(f"density={dens:<4}: rel_l2_vs_dense={rel_l2(out, ref):.4f}  "
              f"max_abs={(out - ref).abs().max().item():.4f}  shape={tuple(out.shape)}")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"density={dens}: FAILED {type(exc).__name__}: {exc}")

# Isolate the merge: compare against a manual full[video+text] using the same
# dense path for BOTH parts (density irrelevant) — this checks the SDPA text
# tail + concat plumbing exactly.
print("[plumbing] full-dense via hunyuan fallback vs SDPA:",
      f"{rel_l2(sb.sol_attn_attention_hunyuan(q, k, v, video_len=vl, key_valid=key_valid, grid=grid, target_density=1.0, block_size=64), ref):.4f}")
print("DONE")
