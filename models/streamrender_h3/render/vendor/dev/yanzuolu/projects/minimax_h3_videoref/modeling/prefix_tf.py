# SPDX-License-Identifier: Apache-2.0
"""Native reference prefixes followed by dataset-defined causal target blocks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from dev.yanzuolu.projects.minimax_h3.data.causal_text_only import (
    _stereo_chunk_indices,
)
from dev.yanzuolu.utils.flex_attn import _prepare_flex_attention_mask

_PREFIX = 0
_NOISY = 2


def build_video_ref_prefix_tf_layout(
    *,
    native: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize dataset-defined logical blocks using native Ref2VA rows.

    ``source_row_indices`` maps each packed row to the native pack, preserving
    its RoPE and token tag even when a target row is copied. ``row_roles`` uses
    0 for prefix, 1 for clean target, and 2 for noisy target. ``noisy_img_pos``
    selects packed video rows. ``audio_noisy_sel`` selects entries in the
    ``audio_pos`` output. The two ``noisy_*_native_sel`` tensors index the
    corresponding target-only native output, including its stereo ordering.
    Padding and offsets between samples are supplied by the caller.
    """
    video_shape = tuple(int(value) for value in plan["target_video_shape"])
    audio_shape = tuple(int(value) for value in plan["target_audio_shape"])
    _, latent_t, latent_h, latent_w = video_shape
    audio_t = audio_shape[2]
    video_ranges = plan["video_chunk_ranges"]
    audio_ranges = plan["audio_chunk_ranges"]

    device = native["img_position_ids"].device
    native_img_pos = native["img_pos"].to(device=device, dtype=torch.long)
    native_audio_pos = native["audio_pos"].to(device=device, dtype=torch.long)
    native_text_pos = native["text_pos"].to(device=device, dtype=torch.long)
    target_img_pos = native_img_pos[native["update_mask"].to(device)]
    target_audio_pos = native_audio_pos[native["audio_update_mask"].to(device)]
    frame_rows = (latent_h // 2) * (latent_w // 2)
    if target_img_pos.numel() != latent_t * frame_rows:
        raise ValueError("native target video rows do not match target_video_shape")
    if target_audio_pos.numel() != 2 * audio_t:
        raise ValueError("native target audio rows do not match target_audio_shape")

    prefix_len = (
        native_text_pos.numel()
        + native_img_pos.numel() - target_img_pos.numel()
        + native_audio_pos.numel() - target_audio_pos.numel()
    )
    source_parts, role_parts, split_lens = [], [], []
    attn_modes = list(plan["attn_modes"])
    chunks: dict[int, dict[str, Any]] = {}
    chunk_sources: dict[int, torch.Tensor] = {}
    cursor = 0
    for role, index in zip(plan["block_roles"], plan["block_chunk_indices"], strict=True):
        if role == _PREFIX:
            source_rows = torch.arange(prefix_len, dtype=torch.long, device=device)
        else:
            if index not in chunks:
                video_start, video_stop = video_ranges[index]
                audio_start, audio_stop = audio_ranges[index]
                chunk_sources[index] = torch.cat(
                    (
                        target_audio_pos.index_select(
                            0, _stereo_chunk_indices(audio_t, audio_start, audio_stop).to(device)
                        ),
                        target_img_pos[video_start * frame_rows : video_stop * frame_rows],
                    )
                )
                chunks[index] = {
                    "chunk_index": index,
                    "video_start": video_start,
                    "video_stop": video_stop,
                    "audio_start": audio_start,
                    "audio_stop": audio_stop,
                    "noise_start": None,
                    "clean_start": None,
                }
            source_rows = chunk_sources[index]
            chunks[index]["noise_start" if role == _NOISY else "clean_start"] = cursor
        source_parts.append(source_rows)
        role_parts.append(torch.full_like(source_rows, role))
        split_lens.append(source_rows.numel())
        cursor += source_rows.numel()

    source_row_indices = torch.cat(source_parts)
    row_roles = torch.cat(role_parts)
    native_row_kind = torch.zeros(int(native["seq_len"]), dtype=torch.int8, device=device)
    native_row_kind[native_img_pos] = 1
    native_row_kind[native_audio_pos] = 2
    native_row_kind[native_text_pos] = 3
    packed_row_kind = native_row_kind.index_select(0, source_row_indices)
    img_pos = torch.nonzero(packed_row_kind == 1, as_tuple=False).flatten()
    audio_pos = torch.nonzero(packed_row_kind == 2, as_tuple=False).flatten()
    text_pos = torch.nonzero(packed_row_kind == 3, as_tuple=False).flatten()
    noisy_img_pos = img_pos[row_roles[img_pos] == _NOISY]
    audio_noisy_sel = torch.nonzero(row_roles[audio_pos] == _NOISY, as_tuple=False).flatten()

    native_target_indices = torch.full_like(native_row_kind, -1, dtype=torch.long)
    native_target_indices[target_img_pos] = torch.arange(target_img_pos.numel(), device=device)
    native_target_indices[target_audio_pos] = torch.arange(target_audio_pos.numel(), device=device)
    q_ranges, k_ranges, attn_type_map, attn_workloads = _prepare_flex_attention_mask(
        split_lens, attn_modes, sink=plan["sink"], window_size=plan["window_size"], device=device
    )
    return {
        "source_row_indices": source_row_indices,
        "row_roles": row_roles,
        "prefix_len": prefix_len,
        "position_ids": native["img_position_ids"].index_select(0, source_row_indices),
        "token_tags": native["token_tags"].to(device).index_select(0, source_row_indices),
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "text_pos": text_pos,
        "noisy_img_pos": noisy_img_pos,
        "audio_noisy_sel": audio_noisy_sel,
        "noisy_video_native_sel": native_target_indices[source_row_indices[noisy_img_pos]],
        "noisy_audio_native_sel": native_target_indices[
            source_row_indices[audio_pos[audio_noisy_sel]]
        ],
        "video_chunk_ranges": video_ranges,
        "audio_chunk_ranges": audio_ranges,
        "chunks": tuple(chunks.values()),
        "sample_lens": cursor,
        "split_lens": split_lens,
        "attn_modes": attn_modes,
        "q_ranges": q_ranges,
        "k_ranges": k_ranges,
        "attn_type_map": attn_type_map,
        "attn_workloads": attn_workloads,
        "sink": plan["sink"],
        "window_size": plan["window_size"],
    }


__all__ = ["build_video_ref_prefix_tf_layout"]
