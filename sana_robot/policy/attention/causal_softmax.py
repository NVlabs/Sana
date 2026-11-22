# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from sana_robot.policy.attention.softmax import GatedSoftmaxAttention, rotate
from sana_robot.policy.layout.teacher_forcing import teacher_forcing_observation_index


def teacher_forcing_reads(bounds: tuple, clean_segments: int, window: int, sink: int) -> list:
    """Return, per segment of the teacher-forcing sequence, the segments it reads and the window start of the rotated sink copy it reads (None when it reads the sink in place). A segment reads the sink, the last ``window`` clean chunks and itself; once its window has moved past the first chunk it reads the sink through a copy rotated to that window."""
    sink_end = bounds[sink]
    reads = []
    for segment in range(len(bounds) - 1):
        if segment < clean_segments:
            first = sink if segment < sink else max(sink, segment - window)
            visible = list(range(0 if first == sink else first, segment + 1))
        else:
            chunk = segment - clean_segments
            first = sink + max(0, chunk - window)
            visible = list(range(0 if first == sink else first, sink + chunk)) + [segment]
        evicted = bool(sink) and first != sink and (bounds[first] > sink_end)
        reads.append((visible, first - sink if evicted else None))
    return reads


class TeacherForcingKeyPlan(NamedTuple):
    """Keys of the single-call teacher-forcing attention, [sequence | one rotated sink copy per window start]: the block mask over them and the window starts of the copies, in key order."""

    block_mask: BlockMask
    sink_starts: torch.Tensor


@lru_cache(maxsize=256)
def teacher_forcing_key_plan(
    bounds: tuple,
    clean_segments: int,
    window: int,
    sink: int,
    device: torch.device,
    observation_spans: tuple = (),
) -> TeacherForcingKeyPlan:
    """Return the ``TeacherForcingKeyPlan`` of a teacher-forcing layout, cached per layout and device. The observation tokens of a noisy chunk read their own segment's observation tokens only, never its targets or action rows."""
    reads = teacher_forcing_reads(bounds, clean_segments, window, sink)
    starts = sorted({start for (_visible, start) in reads if start is not None})
    num_segments = len(bounds) - 1
    sizes = torch.tensor(bounds[1:]) - torch.tensor(bounds[:-1])
    query_segments = torch.repeat_interleave(torch.arange(num_segments), sizes)
    copy_block = torch.arange(num_segments, num_segments + len(starts)).repeat_interleave(bounds[sink])
    key_blocks = torch.cat([query_segments, copy_block])
    table = torch.zeros(num_segments, num_segments + len(starts), dtype=torch.bool)
    for segment, (visible, start) in enumerate(reads):
        table[segment, visible] = True
        if start is not None:
            table[segment, num_segments + starts.index(start)] = True
    (query_segments, key_blocks, table) = (query_segments.to(device), key_blocks.to(device), table.to(device))
    q_observation = torch.zeros(query_segments.numel(), dtype=torch.bool)
    q_observation[teacher_forcing_observation_index(bounds, clean_segments, observation_spans)[0]] = True
    k_observation = torch.cat([q_observation, torch.zeros(copy_block.numel(), dtype=torch.bool)]).to(device)
    q_observation = q_observation.to(device)

    def mask_mod(_batch, _head, q_index, kv_index):
        (query_segment, key_block) = (query_segments[q_index], key_blocks[kv_index])
        own_targets = q_observation[q_index] & (key_block == query_segment) & ~k_observation[kv_index]
        return table[query_segment, key_block] & ~own_targets

    block_mask = create_block_mask(mask_mod, None, None, query_segments.numel(), key_blocks.numel(), device=device)
    return TeacherForcingKeyPlan(block_mask, torch.tensor(starts, dtype=torch.long, device=device))


def rotated_sink_keys(sink_keys, sink_phases, starts):
    """Return the sink keys [B, S, heads, head_dim] rotated to every window start in ``starts``, [B, len(starts) * S, heads, head_dim] in start order."""
    keys = torch.view_as_complex(sink_keys.to(torch.float64).unflatten(-1, (-1, 2)))
    phases = sink_phases[:, 0].index_select(1, starts)
    rotated = keys[:, None] * phases[:, :, None, None, :]
    return torch.view_as_real(rotated).flatten(-2).flatten(1, 2).type_as(sink_keys)


class CausalSoftmaxAttention(GatedSoftmaxAttention):
    """Softmax attention of SANA-Robot-AR. On the teacher-forcing sequence every segment reads the sink, the clean chunks inside its sliding window and itself. A streaming window reads the keys and values that the layer's cache manager (the layer cache's ``"kv"`` entry) selects, rotated at the positions the manager assigns."""

    def forward(
        self,
        x,
        rotary_emb,
        chunk_bounds=None,
        teacher_forcing_clean_segments=None,
        observation_spans=(),
        kv_cache=None,
        save_kv_cache: bool = False,
        observation_prefill: bool = False,
        window_chunks: int = 1,
        sink_phases=None,
        windows=None,
    ):
        (q, k, v) = self.project(x)
        dtype = v.dtype
        if windows is not None:
            (q, k) = (rotate(q, rotary_emb), rotate(k, rotary_emb))
            out = self._packed_row_attention(q, k, v, windows, window_chunks, tuple(observation_spans))
            return self.gate_and_project(out, x, dtype)
        if kv_cache is None:
            (q, k) = (rotate(q, rotary_emb), rotate(k, rotary_emb))
            bounds = [int(bound) for bound in chunk_bounds]
            out = self._teacher_forcing_attention(
                q,
                k,
                v,
                bounds,
                int(teacher_forcing_clean_segments),
                window_chunks,
                sink_phases,
                tuple(observation_spans),
            )
            return self.gate_and_project(out, x, dtype)
        manager = kv_cache["kv"]
        if observation_prefill:
            (keys, values, valid, phases) = manager.observe(k.detach(), v.detach(), q)
        elif save_kv_cache:
            (keys, values, valid, phases) = manager.commit(k.detach(), v.detach(), q)
        else:
            (keys, values, valid, phases) = manager.read(k, v, q)
        if phases is not None:
            query_phases = phases.narrow(2, keys.shape[1] - q.shape[1], q.shape[1])
            (q, keys) = (rotate(q, query_phases), rotate(keys, phases))
        if valid is None:
            out = self.attend(q, keys, values)
        else:
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), attn_mask=valid
            ).transpose(1, 2)
        return self.gate_and_project(out, x, dtype)

    def _packed_row_attention(self, q, k, v, windows, window, observation_spans):
        """Attend a packed row of teacher-forcing windows, every window on its own tokens: one FlashAttention call for the whole row on CUDA, one call per window elsewhere."""
        if q.is_cuda:
            from sana_robot.policy.attention.causal_softmax_varlen import varlen_packed_row_attention

            return varlen_packed_row_attention(
                q, k, v, windows, window, 1, observation_spans, rotated_sink_keys=rotated_sink_keys
            )
        outputs = [
            self._teacher_forcing_attention(
                *(tensor[:, row.start : row.end] for tensor in (q, k, v)),
                list(row.bounds),
                row.num_chunks,
                window,
                row.sink_phases,
                observation_spans,
            )
            for row in windows
        ]
        return torch.cat(outputs, dim=1)

    def _teacher_forcing_attention(self, q, k, v, bounds, clean_segments, window, sink_phases, observation_spans):
        """Attend every segment of the teacher-forcing sequence to the segments it reads in one call: FlashAttention over variable-length groups on CUDA, FlexAttention under a block mask elsewhere."""
        if q.is_cuda:
            from sana_robot.policy.attention.causal_softmax_varlen import varlen_teacher_forcing_attention

            return varlen_teacher_forcing_attention(
                q,
                k,
                v,
                bounds,
                clean_segments,
                window=window,
                sink=1,
                sink_phases=sink_phases,
                observation_spans=observation_spans,
                rotated_sink_keys=rotated_sink_keys,
                deterministic=True,
            )
        plan = teacher_forcing_key_plan(tuple(bounds), clean_segments, int(window), 1, q.device, observation_spans)
        copies = plan.sink_starts.numel()
        if copies:
            sink_end = bounds[1]
            k = torch.cat([k, rotated_sink_keys(k[:, :sink_end], sink_phases, plan.sink_starts)], dim=1)
            v = torch.cat([v, v[:, :sink_end].repeat(1, copies, 1, 1)], dim=1)
        out = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=plan.block_mask,
            kernel_options={"FLOAT32_PRECISION": "'tf32'"},
        )
        return out.transpose(1, 2)
