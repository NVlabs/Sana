# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np


def balanced_order(tasks, weights, generator: np.random.Generator) -> np.ndarray:
    """Return an order of the windows in which every stretch holds each task near its share of the weights. The windows of a task take a random order and lie end to end on a line from 0 to 1, each spanning its share of the task's weight; sorting all windows by the middle of their spans advances every task along its line at the same rate."""
    tasks = np.asarray(tasks, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if not len(tasks):
        return np.zeros(0, dtype=np.int64)
    order = generator.permutation(len(tasks))
    order = order[np.argsort(tasks[order], kind="stable")]
    task, weight = tasks[order], weights[order]
    first = np.flatnonzero(np.r_[True, task[1:] != task[:-1]])
    sizes = np.diff(np.r_[first, len(task)])
    end = np.cumsum(weight)
    before = end - weight - np.repeat(end[first] - weight[first], sizes)
    key = (before + 0.5 * weight) / np.repeat(np.add.reduceat(weight, first), sizes)
    return order[np.argsort(key, kind="stable")]
