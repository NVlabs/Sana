# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random

import torch

SEED_MASK = (1 << 64) - 1


def epoch_seed(seed: int, epoch: int) -> int:
    """Return the 64-bit seed of an epoch's order, the run seed and the epoch number mixed so that consecutive epochs take unrelated orders."""
    value = ((seed & SEED_MASK) ^ ((epoch + 1) * 0x9E3779B97F4A7C15)) & SEED_MASK
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & SEED_MASK
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & SEED_MASK
    return value ^ (value >> 31)


class ChunkBatchSampler(torch.utils.data.Sampler):
    """Batch sampler of one-chunk windows: every epoch shuffles all windows and cuts the order into global batches, one slice of each per rank.

    A global batch holds ``batch_size`` windows for every rank; rank r takes windows ``r * batch_size`` to
    ``(r + 1) * batch_size`` of it. The order of an epoch is a function of the seed and the epoch, so every rank
    builds the same order and a run resumes at any step of an epoch. The windows that do not fill a last global
    batch are left out of the epoch.

    Args:
        windows: The number of windows.
        batch_size: The windows of one batch on one rank.
        world_size: The number of ranks.
        rank: The rank this sampler yields the batches of.
        seed: The seed of the orders of all epochs.
    """

    def __init__(self, windows: int, batch_size: int, world_size: int = 1, rank: int = 0, seed: int = 0):
        self.windows = int(windows)
        self.batch_size = int(batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.first_step = 0

    def set_epoch(self, epoch: int, first_step: int = 0) -> None:
        """Select the epoch to iterate and the step of it to start from."""
        self.epoch = int(epoch)
        self.first_step = int(first_step)

    def steps(self) -> int:
        """Return the number of steps of an epoch."""
        return self.windows // (self.batch_size * self.world_size)

    def __len__(self) -> int:
        return max(self.steps() - self.first_step, 0)

    def __iter__(self):
        order = list(range(self.windows))
        random.Random(epoch_seed(self.seed, self.epoch)).shuffle(order)
        global_batch = self.batch_size * self.world_size
        for step in range(self.first_step, self.steps()):
            first = step * global_batch + self.rank * self.batch_size
            yield order[first : first + self.batch_size]
