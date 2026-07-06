# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal parallel state: data parallel + Ulysses sequence parallel.

Single-GPU inference uses the default state (Ulysses disabled) and pulls in no
extra dependency. Multi-GPU sequence parallel delegates the actual collective
communication to Open-VeOmni, which is imported lazily only when enabled.
"""

import os

import torch.distributed as dist


class ParallelState:
    """Process-group layout for a single inference run.

    Rank layout: consecutive blocks of ``ulysses_size`` ranks form one Ulysses
    sequence-parallel group; ranks sharing the same in-block offset form one
    data-parallel group.
    """

    def __init__(self, ulysses_size: int = 1):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        if self.world_size % ulysses_size != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by "
                f"ulysses_size ({ulysses_size})"
            )
        self.ulysses_size = ulysses_size
        self.dp_size = self.world_size // ulysses_size
        self.ulysses_rank = self.rank % ulysses_size
        self.dp_rank = self.rank // ulysses_size
        self.ulysses_group = None
        self.dp_group = None

    @property
    def ulysses_enabled(self) -> bool:
        return self.ulysses_size > 1

    @property
    def sp_enabled(self) -> bool:
        return self.ulysses_enabled

    @property
    def sp_size(self) -> int:
        return self.ulysses_size

    @property
    def sp_group(self):
        return self.ulysses_group


_PARALLEL_STATE = ParallelState(ulysses_size=1)


def get_parallel_state() -> ParallelState:
    return _PARALLEL_STATE


def init_parallel_state(ulysses_size: int = 1) -> ParallelState:
    """Build the data-parallel / Ulysses process groups.

    ``torch.distributed`` must already be initialized when ``ulysses_size > 1``.
    """
    global _PARALLEL_STATE
    ps = ParallelState(ulysses_size)

    if ps.ulysses_enabled:
        # Ulysses groups: consecutive blocks of `ulysses_size` ranks.
        for i in range(ps.dp_size):
            ranks = list(range(i * ulysses_size, (i + 1) * ulysses_size))
            group = dist.new_group(ranks)
            if ps.rank in ranks:
                ps.ulysses_group = group
        # Data-parallel groups: one rank from each Ulysses group.
        for i in range(ulysses_size):
            ranks = list(range(i, ps.world_size, ulysses_size))
            group = dist.new_group(ranks)
            if ps.rank in ranks:
                ps.dp_group = group
        # Register the Ulysses group with Open-VeOmni so its native sequence-
        # parallel primitives use the exact same communication semantics as
        # veomni_editing Gradio inference.
        from veomni.distributed.parallel_state import init_parallel_state as init_veomni_parallel_state
        from veomni.distributed.sequence_parallel.comm import (
            set_ulysses_sequence_parallel_group,
            set_unified_sequence_parallel_group,
        )

        init_veomni_parallel_state(dp_size=ps.dp_size, ulysses_size=ulysses_size)
        set_ulysses_sequence_parallel_group(ps.ulysses_group)
        set_unified_sequence_parallel_group(ps.ulysses_group)
    _PARALLEL_STATE = ps
    return ps
