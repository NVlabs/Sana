# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

import numpy as np

from sana_robot.data.format.normalization import quantile_normalization, save_normalization
from sana_robot.data.storage import open_storage


def compute_normalization(dataset, output, lerobot_mapping=None) -> dict:
    """Compute the normalization of a dataset's robot rows from the state rows of all its episodes.

    Every episode weighs the same, whatever its length, as in the normalization of the released checkpoints. The
    absolute action targets share the statistics of the states they command.

    Args:
        dataset: The dataset directory, zip archives or LeRobot.
        output: The normalization file to write.
        lerobot_mapping: The column mapping of a LeRobot dataset, or its JSON file.

    Returns:
        The normalization: ``center``, ``scale`` and ``mask`` [80].
    """
    storage = open_storage(dataset, lerobot_mapping)
    rows = np.concatenate([storage.read_rows(record)[0] for record in storage.episodes])
    weights = np.concatenate([np.full(record.rows, 1.0 / record.rows) for record in storage.episodes])
    normalization = quantile_normalization(rows, weights, storage.info.state_mask | storage.info.action_mask)
    save_normalization(output, normalization)
    return normalization


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lerobot-mapping", default=None)
    arguments = parser.parse_args()
    compute_normalization(arguments.dataset, arguments.output, arguments.lerobot_mapping)
