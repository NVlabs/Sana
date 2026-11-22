# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class View:
    """One camera of a dataset: its name, its slot in the spatial layout and its description in the prompt."""

    name: str
    slot: int
    description: str


@dataclass(frozen=True)
class DatasetInfo:
    """What every episode of a dataset shares: the control frequency, the prompt fields, the cameras, the robot slots in use and how its frames reach a view's size (``stretch`` or ``crop``)."""

    fps: float
    embodiment: str
    action_mode: str
    views: tuple
    state_mask: np.ndarray
    action_mask: np.ndarray
    resize: str = "stretch"


@dataclass(frozen=True)
class EpisodeRecord:
    """One episode in the index of a dataset: its name, task, number of rows, instructions and where the storage keeps it."""

    name: str
    task: str
    rows: int
    instructions: tuple
    location: dict


def task_ids(records) -> np.ndarray:
    """Return one integer per episode, equal for the episodes of one task."""
    names = sorted({record.task for record in records})
    lookup = {name: index for index, name in enumerate(names)}
    return np.asarray([lookup[record.task] for record in records], dtype=np.int64)
