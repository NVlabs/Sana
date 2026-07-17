"""Correctness test for the EP MoE path: each rank compares its own tokens' output
from the expert-parallel path against the reference full-local MoE (all experts on
one rank). They must match — EP only changes WHERE experts run, not the math."""
import os
import torch
import torch.distributed as dist

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lingbot_video.transformer_lingbot_video import LingBotVideoSparseMoeBlock

dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)

H, MOE_I, E, K = 64, 32, 8, 2   # E divisible by world (8/4=2)
torch.manual_seed(0)
block = LingBotVideoSparseMoeBlock(
    hidden_size=H, intermediate_size=H * 4, num_experts=E, top_k=K,
    moe_intermediate_size=MOE_I, score_func="sigmoid", norm_topk_prob=True,
    n_group=1, topk_group=1, routed_scaling_factor=1.0, n_shared_experts=0,
).to(dev).to(torch.bfloat16)
# real random weights (torch.empty may land on zeroed pages -> trivial all-zero output)
with torch.no_grad():
    for p in block.parameters():
        p.data.normal_(0.0, 0.1)
    for b in block.buffers():
        b.data.zero_()
# identical weights/buffers across ranks
for p in block.parameters():
    dist.broadcast(p.data, src=0)
for b in block.buffers():
    dist.broadcast(b.data, src=0)

torch.manual_seed(100 + rank)
tok = torch.randn(1, 16, H, device=dev, dtype=torch.bfloat16)

# reference: full local MoE (EP off), weights intact
os.environ["LINGBOT_MOE_EP"] = "0"
with torch.no_grad():
    out_ref = block(tok.clone())

# EP path (shards weights in place afterwards; that's fine, ref already computed)
os.environ["LINGBOT_MOE_EP"] = "1"
with torch.no_grad():
    out_ep = block(tok.clone())

diff = (out_ref.float() - out_ep.float()).abs()
maxd = diff.max().item()
meand = diff.mean().item()
ref_scale = out_ref.float().abs().mean().item()
ok = maxd < 1e-2 * max(ref_scale, 1e-3) + 5e-3
print(f"[rank {rank}/{world}] max_abs_diff={maxd:.3e} mean_abs_diff={meand:.3e} "
      f"ref_scale={ref_scale:.3e} PASS={ok}", flush=True)

flag = torch.tensor([1 if ok else 0], device=dev)
dist.all_reduce(flag, op=dist.ReduceOp.MIN)
if rank == 0:
    print("EP CORRECTNESS:", "PASS" if flag.item() == 1 else "FAIL", flush=True)
dist.destroy_process_group()
