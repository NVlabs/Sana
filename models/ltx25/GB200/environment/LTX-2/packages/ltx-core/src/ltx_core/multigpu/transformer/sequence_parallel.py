"""
Multi-GPU inference wrapper for LTX transformer models.
This module provides utilities for running LTX model inference across multiple GPUs
using sequence parallelism. It:
- Tiles the video inputs across GPUs in the sequence (token) dimension
- Patches video self-attention operations with all2all attention
- Runs the model forward pass on each GPU with its local tile
- Gathers all tokens back to all GPUs after the forward pass
"""

from dataclasses import replace
from itertools import accumulate

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
from ltx_core.multigpu.transformer.attention import (
    All2AllAttention,
    AttentionManager,
    AudioAll2AllAttention,
    MaskedAll2AllAttention,
    MaskedAudioAll2AllAttention,
)


def compute_sequence_partition(
    total_tokens: int,
    world_size: int,
) -> list[int]:
    """
    Compute uniform per-rank token counts.
    Requires ``total_tokens % world_size == 0`` — callers must pad up-front via
    :func:`pad_modality_for_uniform_sharding`. Uniform sharding lets the
    All2All custom-op fakes derive output shapes symbolically from input shapes
    (``x.shape[1] * world_size`` / ``x.shape[1] // world_size``) instead of from
    Python int args.
    """
    if total_tokens % world_size != 0:
        raise ValueError(
            f"compute_sequence_partition expects uniform sharding: total_tokens "
            f"({total_tokens}) must be divisible by world_size ({world_size}). "
            f"Pad the modality up-front."
        )
    per_rank = total_tokens // world_size
    return [per_rank] * world_size


def pad_modality_for_uniform_sharding(
    modality: Modality,
    world_size: int,
) -> tuple[Modality, int]:
    """Pad the seq dim up to the next multiple of ``world_size`` and attach a
    padding-aware attention bias so the padded keys are ignored.
    Returns ``(padded_modality, original_seq_len)``. If no padding is needed
    the original modality is returned unchanged.
    """
    t_orig = modality.latent.shape[1]
    pad = (-t_orig) % world_size
    if pad == 0:
        return modality, t_orig

    t_padded = t_orig + pad
    b = modality.latent.shape[0]
    device = modality.latent.device
    dtype = modality.latent.dtype

    latent_pad = torch.zeros(b, pad, modality.latent.shape[2], dtype=dtype, device=device)
    latent = torch.cat([modality.latent, latent_pad], dim=1)

    timesteps_pad_shape = list(modality.timesteps.shape)
    timesteps_pad_shape[1] = pad
    timesteps_pad = torch.zeros(timesteps_pad_shape, dtype=modality.timesteps.dtype, device=modality.timesteps.device)
    timesteps = torch.cat([modality.timesteps, timesteps_pad], dim=1)

    positions_pad_shape = list(modality.positions.shape)
    positions_pad_shape[2] = pad
    positions_pad = torch.zeros(positions_pad_shape, dtype=modality.positions.dtype, device=modality.positions.device)
    positions = torch.cat([modality.positions, positions_pad], dim=2)

    if modality.attention_mask is None:
        # Key-only padding mask in the canonical [0, 1] form: 1 on valid keys,
        # 0 on padded keys. Shape (1, 1, T_padded) broadcasts across batch and
        # queries -- O(T) memory instead of materialising a dense (B, T, T)
        # matrix just to mask `pad` (< world_size) keys.
        # `_prepare_self_attention_mask` does the standard 3D -> 4D log-space
        # conversion and produces a (1, 1, 1, T_padded) bias.
        attention_mask = torch.ones(1, 1, t_padded, dtype=torch.float32, device=device)
        attention_mask[:, :, t_orig:] = 0.0
    else:
        # User-supplied (B, T, T) [0, 1] mask: extend with padded rows/cols.
        # Padded query rows attend to all valid keys so their softmax stays
        # well-defined (the outputs are sliced off after the gather, but a
        # fully-masked row would produce NaN).
        old = modality.attention_mask
        attention_mask = torch.zeros(b, t_padded, t_padded, dtype=old.dtype, device=old.device)
        attention_mask[:, :t_orig, :t_orig] = old
        attention_mask[:, t_orig:, :t_orig] = 1.0

    padded = replace(
        modality,
        latent=latent,
        timesteps=timesteps,
        positions=positions,
        attention_mask=attention_mask,
    )
    return padded, t_orig


def compute_sequence_offsets(token_counts: list[int]) -> list[int]:
    """
    Compute the starting offset for each rank's token partition.
    Args:
        token_counts: List of token counts per rank.
    Returns:
        List of starting offsets for each rank.
    """
    return [0, *accumulate(token_counts[:-1])]


def tile_modality_for_rank(
    modality: Modality,
    rank: int,
    world_size: int,
) -> tuple[Modality, list[int]]:
    """
    Tile a modality's tensors for a specific GPU rank.
    Splits the sequence dimension (dim 1 for latent/timesteps, dim 2 for positions)
    across GPUs, returning the local tile for the given rank.
    Args:
        modality: The modality to tile.
        rank: Current GPU rank.
        world_size: Total number of GPUs.
    Returns:
        Tuple of (tiled_modality, token_counts_per_rank).
    """
    total_tokens = modality.latent.shape[1]
    token_counts = compute_sequence_partition(total_tokens, world_size)
    offsets = compute_sequence_offsets(token_counts)

    start = offsets[rank]
    end = start + token_counts[rank]

    # Tile latent: (B, T, D) -> (B, T_local, D)
    tiled_latent = modality.latent[:, start:end, :]

    # Tile timesteps: (B, T) -> (B, T_local)
    tiled_timesteps = modality.timesteps[:, start:end]

    # Tile positions: (B, 3, T, 2) -> (B, 3, T_local, 2)
    tiled_positions = modality.positions[:, :, start:end, :]

    tiled_modality = replace(
        modality,
        latent=tiled_latent,
        timesteps=tiled_timesteps,
        positions=tiled_positions,
    )

    return tiled_modality, token_counts


def gather_output_tokens(
    local_output: torch.Tensor,
    token_counts: list[int],
    group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """
    Gather output tokens from all GPUs back into a single tensor.
    Args:
        local_output: Local output tensor of shape (B, T_local, D).
        token_counts: Number of tokens on each rank.
        group: Process group for communication. If None, uses default group.
    Returns:
        Gathered tensor of shape (B, T_total, D) on all ranks.
    """
    world_size = len(token_counts)
    batch_size = local_output.shape[0]
    hidden_dim = local_output.shape[2]

    # Prepare output tensors for all_gather
    max_tokens = max(token_counts)

    # Pad local output to max size for uniform all_gather
    padded_local = torch.zeros(
        batch_size,
        max_tokens,
        hidden_dim,
        dtype=local_output.dtype,
        device=local_output.device,
    )
    padded_local[:, : local_output.shape[1], :] = local_output

    # All gather padded outputs
    gathered_list = [torch.zeros_like(padded_local) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_list, padded_local, group=group)

    # Extract actual tokens (remove padding) and concatenate
    outputs = []
    for i, count in enumerate(token_counts):
        outputs.append(gathered_list[i][:, :count, :])

    return torch.cat(outputs, dim=1)


def create_video_self_attention_module_ops(
    attention_manager: AttentionManager,
) -> ModuleOps:
    """
    Create ModuleOps for patching video self-attention with all2all attention.
    This patches the `attn1` attribute on BasicAVTransformerBlock instances,
    which is the video self-attention module.
    Args:
        attention_manager: The AttentionManager instance for all2all communication.
    Returns:
        ModuleOps that can be used to patch the model.
    """

    def mutator(module: torch.nn.Module) -> torch.nn.Module:
        for block in module.transformer_blocks:
            if not isinstance(block, BasicAVTransformerBlock):
                continue

            # Video self-attention: ``Attention.forward`` may receive a non-None
            # ``mask`` (``video.self_attention_mask``), so wrap both slots; the
            # branch in ``Attention.forward`` then routes to whichever wrapper
            # corresponds to the actual call.
            if hasattr(block, "attn1"):
                attn1 = block.attn1
                if isinstance(attn1, Attention):
                    attn1.attention_function = All2AllAttention(attention_manager, attn1.attention_function)
                    attn1.masked_attention_function = MaskedAll2AllAttention(
                        attention_manager, attn1.masked_attention_function
                    )
            # video_to_audio cross-attention: no current caller passes a mask
            # (see ``BasicAVTransformerBlock.forward``), so the masked branch
            # is dead code today. Wrap both slots anyway so that if a future
            # caller adds a mask, the SP plumbing is already in place rather
            # than silently bypassing All2All on that path.
            if hasattr(block, "video_to_audio_attn"):
                video_to_audio_attn = block.video_to_audio_attn
                if isinstance(video_to_audio_attn, Attention):
                    video_to_audio_attn.attention_function = AudioAll2AllAttention(
                        attention_manager, video_to_audio_attn.attention_function
                    )
                    video_to_audio_attn.masked_attention_function = MaskedAudioAll2AllAttention(
                        attention_manager, video_to_audio_attn.masked_attention_function
                    )
        return module

    return ModuleOps(
        name="video_self_attention_all2all",
        matcher=lambda module: isinstance(module, LTXModel),
        mutator=mutator,
    )


class SequenceParallelModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, attention_manager: AttentionManager):
        super().__init__()
        self.model = model
        self.attention_manager = attention_manager

    @property
    def num_blocks(self) -> int:
        return self.model.num_blocks

    def forward(
        self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if video is None:
            return self.model(video, audio, perturbations)

        # Pad the video seq dim up to a multiple of world_size so all ranks get
        # equal shards. The attention mask we attach makes the padded keys
        # invisible to attention; padded rows are sliced off after the gather.
        video, t_orig = pad_modality_for_uniform_sharding(video, self.attention_manager.world_size)

        video_tile, token_counts = tile_modality_for_rank(
            video, self.attention_manager.rank, self.attention_manager.world_size
        )
        total_tokens = sum(token_counts)
        if total_tokens > self.attention_manager.max_tokens:
            raise ValueError(
                f"Total video token count ({total_tokens}) exceeds attention_manager max_tokens "
                f"({self.attention_manager.max_tokens}). Use a smaller resolution or fewer frames."
            )
        self.attention_manager.set_seqlen_all2all(token_counts)
        torch.distributed.barrier(self.attention_manager.group)
        video, audio = self.model(video_tile, audio, perturbations)
        video = gather_output_tokens(video, token_counts, self.attention_manager.group)
        # Unpad: drop the rows we added in `pad_modality_for_uniform_sharding` to make
        # the seq dim divisible by world_size, restoring the caller's original length.
        if video.shape[1] != t_orig:
            video = video[:, :t_orig, :]
        return video, audio
