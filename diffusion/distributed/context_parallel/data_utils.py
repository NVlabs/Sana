"""Data utilities for Context Parallel training.

Provides helpers for:
  - Broadcasting tensors across CP ranks.
  - Splitting temporal tensors by CP rank.
  - Handling non-divisible temporal lengths via right-padding.
  - Building frame-valid masks for padded temporal tails.
  - Reducing loss scalars across CP ranks.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup


def _cp_src_global_rank(group: ProcessGroup) -> int:
    """Global rank of CP-rank-0 in the given process group."""
    return dist.get_global_rank(group, 0)


def cp_broadcast_tensor(
    tensor: Tensor,
    group: ProcessGroup,
) -> Tensor:
    """In-place broadcast *tensor* from CP-rank-0 to all ranks in *group*."""
    src = _cp_src_global_rank(group)
    dist.broadcast(tensor, src=src, group=group)
    return tensor


def cp_split_temporal(
    tensor: Tensor,
    dim: int,
    group: ProcessGroup,
) -> Tensor:
    """Slice *tensor* along *dim* to keep only this rank's temporal chunk."""
    cp_rank = dist.get_rank(group)
    cp_world = dist.get_world_size(group)
    T = tensor.shape[dim]
    assert T % cp_world == 0, f"Temporal size {T} (dim={dim}) must be divisible by cp_size={cp_world}"
    chunk = T // cp_world
    return tensor.narrow(dim, cp_rank * chunk, chunk).contiguous()


def cp_right_pad_size(length: int, multiple: int) -> int:
    """Return right-pad size needed to make ``length`` divisible by ``multiple``."""
    if multiple <= 0:
        raise ValueError(f"multiple must be > 0, got {multiple}")
    return (-length) % multiple


def cp_right_pad_temporal(
    tensor: Tensor,
    dim: int,
    pad_size: int,
    value: float = 0.0,
) -> Tensor:
    """Right-pad ``tensor`` along temporal ``dim`` by ``pad_size``."""
    if pad_size <= 0:
        return tensor
    if dim < 0:
        dim = tensor.ndim + dim
    if dim < 0 or dim >= tensor.ndim:
        raise ValueError(f"Invalid dim={dim} for tensor with ndim={tensor.ndim}")

    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_size
    pad_tensor = torch.full(
        pad_shape,
        fill_value=value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad_tensor], dim=dim)


def cp_build_frame_valid_mask(clean_images: Tensor, pad_frames: int) -> Tensor:
    """Build ``(B, 1, T, 1, 1)`` frame-valid mask after temporal right-padding."""
    if clean_images.ndim < 3:
        raise ValueError(f"clean_images must have at least 3 dims (B, C, T, ...), got shape={list(clean_images.shape)}")

    B = clean_images.shape[0]
    T = clean_images.shape[2]
    if pad_frames < 0 or pad_frames > T:
        raise ValueError(f"pad_frames must satisfy 0 <= pad_frames <= T, got pad_frames={pad_frames}, T={T}")

    mask = torch.ones((B, 1, T, 1, 1), device=clean_images.device, dtype=clean_images.dtype)
    if pad_frames > 0:
        mask[:, :, T - pad_frames :, :, :] = 0
    return mask


def cp_reduce_loss(
    loss: Tensor,
    group: ProcessGroup,
    num_valid_tokens: Tensor | int | float | None = None,
) -> Tensor:
    """Reduce CP-local loss to a global scalar with correct gradient scaling.

    This function is autograd-safe for CP: it returns a forward value equal to
    the CP-global reduced loss while preserving backward gradients scaled by the
    local contribution ratio.

    Args:
        loss: Local scalar loss.
        group: CP process group.
        num_valid_tokens: Optional local token count for weighted reduction.
            If omitted, all ranks are weighted equally.
    """
    if num_valid_tokens is None:
        loss_avg_detached = loss.detach().clone()
        dist.all_reduce(loss_avg_detached, op=dist.ReduceOp.SUM, group=group)
        loss_avg_detached = loss_avg_detached / dist.get_world_size(group)
        # Keep local backward unchanged; only replace forward scalar for logging.
        return loss + (loss_avg_detached - loss.detach())

    if torch.is_tensor(num_valid_tokens):
        local_tokens = num_valid_tokens.to(device=loss.device, dtype=loss.dtype)
    else:
        local_tokens = torch.tensor(float(num_valid_tokens), device=loss.device, dtype=loss.dtype)

    world = dist.get_world_size(group)
    total_tokens = local_tokens.detach().clone()
    dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM, group=group)
    total_tokens = total_tokens.clamp_min(1.0)

    # FSDP2 already averages grads across the CP-enabled sharding mesh.
    # To obtain weighted global-token gradients, scale by n_i / mean(n).
    mean_tokens = (total_tokens / world).clamp_min(1.0)
    loss_for_backward = loss * (local_tokens / mean_tokens)

    weighted_loss_detached = loss.detach() * local_tokens.detach()
    dist.all_reduce(weighted_loss_detached, op=dist.ReduceOp.SUM, group=group)
    loss_avg_detached = weighted_loss_detached / total_tokens
    return loss_for_backward + (loss_avg_detached - loss_for_backward.detach())
