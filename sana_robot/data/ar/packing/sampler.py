# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

from sana_robot.data.ar.packing.ordering import balanced_order
from sana_robot.data.ar.packing.packing import choose_min_gap, pack_steps


class PackedWindowSampler(torch.utils.data.Sampler):
    """Batch sampler of whole-episode windows: task-balanced ordering, then packing to a chunk budget per GPU.

    Every epoch orders all windows into one queue that keeps the task mix of the data in every stretch
    (``balanced_order``, the weight of a window being its chunks) and cuts the queue into steps of one row of
    windows per rank, each row holding at most ``chunk_budget`` chunks (``pack_steps``). The plan is a function of
    the windows, the seed, the epoch and the number of ranks, so every rank builds the same plan and takes its own
    row of every step, and a run resumes at any step of an epoch.

    A step first pools the windows the previous step deferred and the next windows of the queue, up to 95% of its
    capacity. The windows deferred twice go first, each to the least-loaded rank where it fits. The other pooled
    windows follow from the longest to the shortest, each to the least-loaded rank if that leaves the rank a gap
    of 0 or of at least ``min_gap`` chunks, else to a rank it fills exactly, else to the next step. The ranks are
    then topped up, from the lightest one, with unplaced windows among the next 1,024 queue positions: a window of
    exactly the gap, else the longest that leaves at least ``min_gap`` chunks, else the longest that fits. Once
    the queue is used up, a step that cannot bring every rank to 95% of the budget ends the epoch; its windows
    are left out of this epoch.

    Args:
        window_chunks: The chunks of every window.
        window_tasks: The task id of every window.
        chunk_budget: The most chunks one rank holds in a step.
        world_size: The number of ranks.
        rank: The rank this sampler yields the rows of.
        seed: The seed of the orders of all epochs.
    """

    def __init__(
        self, window_chunks, window_tasks, chunk_budget: int, world_size: int = 1, rank: int = 0, seed: int = 0
    ):
        self.window_chunks = np.asarray(window_chunks, dtype=np.int64)
        self.window_tasks = np.asarray(window_tasks, dtype=np.int64)
        self.chunk_budget = int(chunk_budget)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.min_gap = choose_min_gap(self.window_chunks, self.world_size, self.chunk_budget)
        self.epoch = 0
        self.first_step = 0
        self.planned_epoch = None
        self.planned_steps = None

    def set_epoch(self, epoch: int, first_step: int = 0) -> None:
        """Select the epoch to iterate and the step of it to start from."""
        self.epoch = int(epoch)
        self.first_step = int(first_step)

    def steps(self) -> list:
        """Return the steps of the current epoch, each a list of one row of window indices per rank."""
        if self.planned_epoch != self.epoch:
            generator = np.random.default_rng([self.seed & 0xFFFFFFFF, self.epoch & 0xFFFFFFFF])
            order = balanced_order(self.window_tasks, self.window_chunks, generator)
            self.planned_steps = pack_steps(order, self.window_chunks, self.world_size, self.chunk_budget, self.min_gap)
            self.planned_epoch = self.epoch
        return self.planned_steps

    def __len__(self) -> int:
        return len(self.steps()) - self.first_step

    def __iter__(self):
        for step in self.steps()[self.first_step :]:
            yield list(step[self.rank])
