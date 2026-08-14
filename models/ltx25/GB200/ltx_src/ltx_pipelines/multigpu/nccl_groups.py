"""NCCL process group management."""

from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist


@dataclass
class NCCLGroups:
    """Container for the NCCL process groups used by each pipeline component."""

    gemma_group: dist.ProcessGroup
    transformer_group: dist.ProcessGroup
    vae_group: dist.ProcessGroup


def create_local_nccl_groups() -> NCCLGroups:
    """Create NCCL process groups for each pipeline component.
    All ranks must call this collectively because ``dist.new_group`` is a
    collective operation.  All ranks participate in every group.
    Returns:
        NCCLGroups with one process group per component.
    """
    all_ranks = list(range(dist.get_world_size()))
    return NCCLGroups(
        gemma_group=dist.new_group(ranks=all_ranks),
        transformer_group=dist.new_group(ranks=all_ranks),
        vae_group=dist.new_group(ranks=all_ranks),
    )
