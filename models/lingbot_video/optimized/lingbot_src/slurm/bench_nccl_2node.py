"""NCCL all-reduce bus bandwidth: intra-node (4 GPUs) vs inter-node (8 GPUs, 2 trays).
Also prints CliqueId per rank (same across nodes => one NVLink domain / NVL72).
Correct subgroup creation: EVERY rank calls new_group for EVERY node's rank-range.
"""
import os
import subprocess
from datetime import timedelta
import torch
import torch.distributed as dist

dist.init_process_group("nccl", timeout=timedelta(seconds=90))
rank = dist.get_rank()
world = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
dev = torch.device("cuda", local_rank)
host = os.uname().nodename
gpn = int(os.environ.get("SLURM_GPUS_ON_NODE", "4"))

clique = "?"
try:
    out = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, timeout=20).stdout
    for line in out.splitlines():
        if "CliqueId" in line:
            clique = line.split(":")[-1].strip(); break
except Exception:
    pass
print(f"[rank {rank}/{world}] host={host} local_rank={local_rank} CliqueId={clique}", flush=True)

# every rank builds every node's subgroup (collective, same order on all ranks)
num_nodes = world // gpn
my_intra = None
for nid in range(num_nodes):
    ranks = list(range(nid * gpn, nid * gpn + gpn))
    g = dist.new_group(ranks=ranks)
    if rank in ranks:
        my_intra = g


def busbw(group, n_ranks, nbytes=512 * 1024 * 1024, iters=20):
    x = torch.ones(nbytes // 2, device=dev, dtype=torch.bfloat16)
    for _ in range(5):
        dist.all_reduce(x, group=group)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        dist.all_reduce(x, group=group)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    algbw = nbytes / (ms * 1e-3) / 1e9
    return ms, algbw, algbw * 2 * (n_ranks - 1) / n_ranks


ms_i, alg_i, bus_i = busbw(my_intra, gpn)
dist.barrier()
ms_g, alg_g, bus_g = busbw(dist.group.WORLD, world)
if rank == 0:
    print(f"\n[INTRA-NODE {gpn} GPU]  {ms_i:6.2f} ms  algbw {alg_i:6.1f} GB/s  busbw {bus_i:6.1f} GB/s", flush=True)
    print(f"[INTER-NODE {world} GPU]  {ms_g:6.2f} ms  algbw {alg_g:6.1f} GB/s  busbw {bus_g:6.1f} GB/s", flush=True)
    print(f"\n=> inter/intra busbw = {bus_g/bus_i:.2f}  "
          f"({'NVLink spans nodes (NVL72)' if bus_g/bus_i > 0.5 else 'inter-node IB-limited (much slower)'})", flush=True)
dist.destroy_process_group()
