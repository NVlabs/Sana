# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from sana_robot.data.format.robot80 import PAIRED_FIELDS, ROBOT_DIM, ROBOT_FIELDS, ROTATION_FIELDS

SCALE_FLOOR = 1.0e-6
QUANTILES = (0.01, 0.99)


def load_normalization(path) -> dict:
    """Return the ``center``, ``scale`` and ``mask`` arrays [80] of a normalization file."""
    content = json.loads(Path(path).read_text())
    return {
        "center": np.asarray(content["center"], dtype=np.float32),
        "scale": np.asarray(content["scale"], dtype=np.float32),
        "mask": np.asarray(content["mask"], dtype=bool),
    }


def save_normalization(path, normalization: dict) -> None:
    """Write the ``center``, ``scale`` and ``mask`` arrays [80] as a normalization file."""
    content = {
        "formula": "normalized = (value - center) / scale on the slots of mask, no clipping",
        "center": np.asarray(normalization["center"], dtype=np.float64).tolist(),
        "scale": np.asarray(normalization["scale"], dtype=np.float64).tolist(),
        "mask": np.asarray(normalization["mask"], dtype=bool).tolist(),
    }
    Path(path).write_text(json.dumps(content, indent=2) + "\n")


def normalize_rows(rows: torch.Tensor, valid_mask: torch.Tensor, normalization: dict) -> torch.Tensor:
    """Return robot rows [..., 80] mapped to (value - center) / scale on the normalized slots and set to zero where ``valid_mask`` is false."""
    center = torch.as_tensor(normalization["center"], dtype=rows.dtype, device=rows.device)
    scale = torch.as_tensor(normalization["scale"], dtype=rows.dtype, device=rows.device)
    mask = torch.as_tensor(normalization["mask"], dtype=torch.bool, device=rows.device)
    normalized = torch.where(mask, (rows - center) / scale, rows)
    return torch.where(valid_mask, normalized, torch.zeros_like(rows))


def denormalize_rows(rows: torch.Tensor, valid_mask: torch.Tensor, normalization: dict) -> torch.Tensor:
    """Return normalized robot rows [..., 80] mapped back to value * scale + center, zero where ``valid_mask`` is false."""
    center = torch.as_tensor(normalization["center"], dtype=rows.dtype, device=rows.device)
    scale = torch.as_tensor(normalization["scale"], dtype=rows.dtype, device=rows.device)
    mask = torch.as_tensor(normalization["mask"], dtype=torch.bool, device=rows.device)
    restored = torch.where(mask, rows * scale + center, rows)
    return torch.where(valid_mask, restored, torch.zeros_like(rows))


def weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the 1% and 99% quantiles of weighted values: every value sits at the middle of its share of the total weight, with linear interpolation between neighbours."""
    order = np.argsort(values, kind="stable")
    (values, weights) = (values[order].astype(np.float64), weights[order])
    positions = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(QUANTILES, positions, values)


def quantile_normalization(rows: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> dict:
    """Return the normalization of state rows [N, 80] with one weight per row on the slots of ``mask``: the 1% and 99% quantiles map to -1 and 1, the left and the right arm share their quantiles, and a rotation keeps its value."""
    mask = np.asarray(mask, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    low = np.zeros(ROBOT_DIM, dtype=np.float64)
    high = np.zeros(ROBOT_DIM, dtype=np.float64)
    for slot in np.flatnonzero(mask):
        (low[slot], high[slot]) = weighted_quantiles(rows[:, slot], weights)
    for left_name, right_name in PAIRED_FIELDS:
        (left, right) = (ROBOT_FIELDS[left_name], ROBOT_FIELDS[right_name])
        for left_slot, right_slot in zip(range(left.start, left.stop), range(right.start, right.stop)):
            if mask[left_slot] and mask[right_slot]:
                pooled = np.concatenate((rows[:, left_slot], rows[:, right_slot]))
                (shared_low, shared_high) = weighted_quantiles(pooled, np.concatenate((weights, weights)))
                (low[[left_slot, right_slot]], high[[left_slot, right_slot]]) = (shared_low, shared_high)
    for name in ROTATION_FIELDS:
        (low[ROBOT_FIELDS[name]], high[ROBOT_FIELDS[name]]) = (-1.0, 1.0)
    center = np.where(mask, (low + high) / 2.0, 0.0)
    scale = np.where(mask, np.maximum((high - low) / 2.0, SCALE_FLOOR), 1.0)
    return {"center": center.astype(np.float32), "scale": scale.astype(np.float32), "mask": mask}
