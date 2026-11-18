"""Sharded state dict with distributed weight backup.
Each rank stores ~1/N of the model weights. Bucketed broadcasts
restore weights into a target state dict using only a small,
caller-provided staging buffer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.distributed as dist


def _stable_owner(key: str, world: int) -> int:
    """Deterministic rank assignment (same across all processes)."""
    h = hashlib.md5(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % world


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


@dataclass
class ShardedSD:
    """Sharded state dict with distributed weight backup.
    Distributes model weights across ranks for memory-efficient backup
    and restoration.  Can be used for any scenario where a full state dict
    needs to be restored from sharded storage (e.g. LoRA hot-swap, weight
    rollback, checkpoint recovery).
    - Deterministic ownership: ``MD5(key) % world_size``
    - Local storage only for owned keys (VRAM ≈ 1/world_size of model)
    - Bucketed broadcast using a single small staging buffer
    Usage::
        backup = ShardedSD.from_state_dict(model.state_dict(), group)
        staging = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
        backup.broadcast_shards_into(target_sd, staging)    # cooperative: all ranks must call
    """

    keys: tuple[str, ...]
    """All parameter keys in the original state dict, in insertion order."""
    key_sizes: dict[str, int]
    """Byte size of each parameter tensor (numel * element_size)."""
    owner_of: dict[str, int]
    """Maps each key to the rank that stores it."""
    local_shard: dict[str, torch.Tensor]
    """Tensors owned by this rank (subset of the full state dict)."""
    rank: int
    """This process's rank within the group."""
    world: int
    """Total number of ranks in the group."""
    group: dist.ProcessGroup
    """NCCL process group used for broadcast operations."""
    _owner_groups: dict[int, list[str]]
    """Keys grouped by owning rank, sorted by descending tensor size."""

    @classmethod
    def from_state_dict(
        cls,
        sd: dict[str, torch.Tensor],
        group: dist.ProcessGroup,
        clone: bool = True,
    ) -> ShardedSD:
        rank = dist.get_rank(group)
        world = dist.get_world_size(group)

        keys = tuple(sd.keys())
        # Co-locate .weight_scale with its .weight on the same rank.
        owner_of: dict[str, int] = {}
        for k in keys:
            if k.endswith(".weight_scale"):
                parent = k.replace(".weight_scale", ".weight")
                if parent in sd:
                    owner_of[k] = _stable_owner(parent, world)
                    continue
            owner_of[k] = _stable_owner(k, world)

        key_sizes = {k: _nbytes(v) for k, v in sd.items()}

        local_shard: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if owner_of[k] == rank:
                local_shard[k] = v.clone() if clone else v

        owner_groups: dict[int, list[str]] = {r: [] for r in range(world)}
        for k in keys:
            owner_groups[owner_of[k]].append(k)
        for r in range(world):
            owner_groups[r].sort(key=lambda kk: key_sizes[kk], reverse=True)

        return cls(
            keys=keys,
            key_sizes=key_sizes,
            owner_of=owner_of,
            local_shard=local_shard,
            rank=rank,
            world=world,
            group=group,
            _owner_groups=owner_groups,
        )

    def broadcast_shards_into(
        self,
        target_sd: dict[str, torch.Tensor],
        staging: torch.Tensor,
    ) -> None:
        """Broadcast stored weights from sharded backup into *target_sd*.
        This is a **cooperative operation** — all ranks in the process group
        must call it simultaneously.
        *staging* is a caller-owned ``uint8`` scratch buffer; its size sets the
        broadcast granularity (tensors larger than it split across rounds). It
        may be shared by instances that never broadcast at the same time. Writes
        directly into existing tensors in *target_sd*.
        """
        if staging.dtype != torch.uint8 or staging.numel() == 0:
            raise ValueError("staging must be a non-empty uint8 buffer")
        for owner, klist in self._owner_groups.items():
            if klist:
                self._broadcast_group(owner, klist, target_sd, staging)

    def _broadcast_group(
        self,
        owner: int,
        keys: list[str],
        target_sd: dict[str, torch.Tensor],
        staging: torch.Tensor,
    ) -> None:
        """Pack & broadcast params from *owner*, splitting tensors across rounds."""
        rounds = self._plan_rounds(keys, staging.numel())

        for round_chunks in rounds:
            filled = 0
            if self.rank == owner:
                for k, offset, chunk_size in round_chunks:
                    src = self.local_shard[k]
                    if not src.is_contiguous():
                        raise RuntimeError(f"ShardedSD: local shard tensor '{k}' is not contiguous")
                    src_bytes = src.view(torch.uint8).view(-1)
                    staging[filled : filled + chunk_size].copy_(
                        src_bytes[offset : offset + chunk_size], non_blocking=True
                    )
                    filled += chunk_size
            else:
                filled = sum(chunk_size for (_, _, chunk_size) in round_chunks)

            if filled == 0:
                continue
            view = staging[:filled]
            dist.broadcast(view, src=owner, group=self.group)

            cursor = 0
            for k, offset, chunk_size in round_chunks:
                dst = target_sd[k]
                if not dst.is_contiguous():
                    raise RuntimeError(f"ShardedSD: target tensor '{k}' is not contiguous")
                dst_bytes = dst.view(torch.uint8).view(-1)
                dst_bytes[offset : offset + chunk_size].copy_(staging[cursor : cursor + chunk_size], non_blocking=True)
                cursor += chunk_size

    def _plan_rounds(self, keys: list[str], capacity: int) -> list[list[tuple[str, int, int]]]:
        """Build rounds that pack a *capacity*-byte buffer, splitting tensors if needed.
        Returns a list of rounds, each containing ``(key, byte_offset, chunk_bytes)`` tuples.
        """
        rounds: list[list[tuple[str, int, int]]] = []
        current: list[tuple[str, int, int]] = []
        used = 0

        for k in keys:
            remaining = self.key_sizes[k]
            offset = 0

            while remaining > 0:
                space = capacity - used
                if space == 0:
                    rounds.append(current)
                    current = []
                    used = 0
                    space = capacity

                chunk = min(remaining, space)
                current.append((k, offset, chunk))
                used += chunk
                offset += chunk
                remaining -= chunk

        if current:
            rounds.append(current)

        return rounds
