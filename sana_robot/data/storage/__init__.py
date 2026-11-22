# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from sana_robot.data.storage.lerobot import LeRobotStorage
from sana_robot.data.storage.zip_archive import INFO_FILE, ZipArchiveStorage


def open_storage(root, lerobot_mapping=None):
    """Return the reader of a dataset directory: a zip archive dataset, or a LeRobot dataset read through ``lerobot_mapping``."""
    if Path(root, INFO_FILE).exists():
        return ZipArchiveStorage(root)
    return LeRobotStorage(root, lerobot_mapping)
