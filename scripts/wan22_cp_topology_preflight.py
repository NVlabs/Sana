"""Four-GPU preflight for Wan's ring/Ulysses device-mesh topology."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from diffusers import ContextParallelConfig


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", mesh_shape=(2, 2), mesh_dim_names=("ring", "ulysses")
    )
    config = ContextParallelConfig(ring_degree=2, ulysses_degree=2)
    config.setup(rank, world, torch.device("cuda", local_rank), mesh=mesh)
    dist.barrier()
    print(
        f"rank={rank} world={world} mesh={config.mesh_shape} "
        f"ring_local={config._ring_local_rank} ulysses_local={config._ulysses_local_rank}",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
