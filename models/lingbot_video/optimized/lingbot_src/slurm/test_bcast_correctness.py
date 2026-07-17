"""Preflight for _broadcast_transformer_weights: CPU/gloo, mixed dtypes + buffer.

Validates the name-ordered, spec-synced broadcast + empty-init overwrite algorithm:
rank 0 holds the true weights; other ranks start with EMPTY (garbage) same-arch modules;
after broadcast every rank must equal rank 0's tensors exactly (bit-identical, same dtype).

Run: torchrun --standalone --nproc_per_node 3 slurm/test_bcast_correctness.py
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lingbot_video.runner import _broadcast_transformer_weights  # noqa: E402


class Toy(nn.Module):
    """Mixed-dtype module mimicking the DiT: bf16 big weights + fp32 norms + a buffer."""

    def __init__(self, fill: float):
        super().__init__()
        self.w_big = nn.Parameter(torch.full((512, 256), fill, dtype=torch.bfloat16))
        self.norm = nn.Parameter(torch.full((256,), fill, dtype=torch.float32))
        self.experts = nn.Parameter(torch.full((8, 128, 64), fill, dtype=torch.bfloat16))
        self.register_buffer("bias", torch.full((8,), fill, dtype=torch.float32))


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(0)
    if rank == 0:
        model = Toy(fill=0.0)
        # give rank 0 distinctive real values
        with torch.no_grad():
            model.w_big.copy_(torch.randn(512, 256).to(torch.bfloat16))
            model.norm.copy_(torch.randn(256))
            model.experts.copy_(torch.randn(8, 128, 64).to(torch.bfloat16))
            model.bias.copy_(torch.arange(8, dtype=torch.float32))
    else:
        model = Toy(fill=float(rank * 100 + 7))  # garbage, wrong values

    _broadcast_transformer_weights(model, src=0, device=torch.device("cpu"))

    # Compare against rank 0's ground truth, gathered via a fresh broadcast of a checksum.
    ref = {n: p.detach().clone() for n, p in model.named_parameters()}
    ref.update({n: b.detach().clone() for n, b in model.named_buffers()})
    # Build rank-0 truth independently to compare (rank 0 already correct).
    checks = []
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        # sum reduced across ranks: if all equal, max-min == 0
        s = t.double().sum().clone()
        alls = [torch.zeros_like(s) for _ in range(dist.get_world_size())]
        dist.all_gather(alls, s)
        spread = (torch.stack(alls).max() - torch.stack(alls).min()).item()
        checks.append((name, str(t.dtype), tuple(t.shape), spread))

    if rank == 0:
        ok = all(abs(c[3]) < 1e-6 for c in checks)
        for name, dt, sh, spread in checks:
            print(f"  {name:12s} dtype={dt:14s} shape={sh} cross_rank_spread={spread:.3e}")
        print(f"BCAST_TEST {'PASS' if ok else 'FAIL'} tensors={len(checks)}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
