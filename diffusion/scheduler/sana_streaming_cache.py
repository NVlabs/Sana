# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixed-RoPE cache operations for SANA streaming training and inference."""

import torch


def accumulate_fixed_rope_kv_cache(
    kv_cache: list,
    chunk_idx: int,
    *,
    block_is_state_cached: list[bool],
    num_cached_blocks: int,
    sink_token: bool,
    full_history_softmax_cache: bool,
    chunk_indices: list[int],
    spatial_hw: int,
    cache_slots: int = 6,
) -> tuple[list, int, int, int]:
    """Prepare the cache window for one fixed-RoPE streaming chunk."""

    if chunk_idx == 0:
        return kv_cache[0], 0, 0, 0

    cur_kv_cache = kv_cache[chunk_idx]
    start_chunk_idx = max(chunk_idx - num_cached_blocks, 0) if num_cached_blocks > 0 else 0
    num_cached_frames = 0
    sink_num = 0

    for block_id, is_state_cached in enumerate(block_is_state_cached):
        if is_state_cached:
            prev = kv_cache[chunk_idx - 1][block_id]
            cur_kv_cache[block_id][0] = prev[0]
            cur_kv_cache[block_id][1] = prev[1]
            cur_kv_cache[block_id][-1] = prev[-1]
            continue

        if full_history_softmax_cache:
            prev = kv_cache[chunk_idx - 1][block_id]
            cur_kv_cache[block_id] = [prev[0], prev[1], prev[2], None, None, prev[-1]]
            if prev[0] is not None and spatial_hw > 0:
                num_cached_frames = prev[0].shape[-1] // spatial_hw
            continue

        previous_q = previous_k = previous_v = previous_tconv = None
        valid_cached_chunks = list(range(start_chunk_idx, chunk_idx))
        if num_cached_blocks > 0 and sink_token:
            window_start_chunk = max(chunk_idx - num_cached_blocks + 1, 0)
            if window_start_chunk > 0:
                valid_cached_chunks = [0] + list(range(window_start_chunk, chunk_idx))
                if sink_num == 0:
                    sink_num = chunk_indices[1] - chunk_indices[0]

        for cache_idx in range(chunk_idx):
            if cache_idx not in valid_cached_chunks:
                kv_cache[cache_idx][block_id] = [None] * cache_slots
                continue

            prev = kv_cache[cache_idx][block_id]
            if prev[0] is not None:
                if previous_q is None:
                    previous_q = prev[0].clone()
                    previous_k = prev[1].clone()
                    previous_v = prev[2].clone()
                else:
                    previous_q = torch.cat([previous_q, prev[0]], dim=-1)
                    previous_k = torch.cat([previous_k, prev[1]], dim=-1)
                    previous_v = torch.cat([previous_v, prev[2]], dim=-1)
            if prev[-1] is not None:
                previous_tconv = (
                    prev[-1].clone() if previous_tconv is None else torch.cat([previous_tconv, prev[-1]], dim=2)
                )

        cur_kv_cache[block_id] = [previous_q, previous_k, previous_v, None, None, previous_tconv]
        if previous_q is not None and spatial_hw > 0:
            num_cached_frames = previous_q.shape[-1] // spatial_hw

    return cur_kv_cache, chunk_idx - start_chunk_idx, sink_num, num_cached_frames


def promote_fixed_rope_full_history_cache(
    kv_cache: list,
    chunk_idx: int,
    *,
    block_is_state_cached: list[bool],
) -> None:
    """Promote the current chunk to a full-history softmax cache."""

    if chunk_idx == 0:
        return

    for block_id, is_state_cached in enumerate(block_is_state_cached):
        if is_state_cached:
            continue

        prev = kv_cache[chunk_idx - 1][block_id]
        cur = kv_cache[chunk_idx][block_id]
        if prev[0] is not None and cur[0] is not None:
            cur[0] = torch.cat([prev[0], cur[0]], dim=-1)
            cur[1] = torch.cat([prev[1], cur[1]], dim=-1)
            cur[2] = torch.cat([prev[2], cur[2]], dim=-1)
        elif prev[0] is not None:
            cur[0], cur[1], cur[2] = prev[0], prev[1], prev[2]

        if prev[-1] is not None and cur[-1] is not None:
            cur[-1] = torch.cat([prev[-1], cur[-1]], dim=2)
        elif prev[-1] is not None:
            cur[-1] = prev[-1]
        kv_cache[chunk_idx - 1][block_id] = [None] * len(prev)
