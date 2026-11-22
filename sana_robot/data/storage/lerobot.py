# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from sana_robot.data.format.episode import DatasetInfo, EpisodeRecord, View
from sana_robot.data.format.robot80 import GRIPPER_FIELDS, gripper_closedness, pack_robot_rows
from sana_robot.data.format.video import decode_frames

VIDEO_LOCATION_FIELDS = ("chunk_index", "file_index", "from_timestamp", "to_timestamp")


def column_matrix(table, name: str) -> np.ndarray:
    """Return a table column of scalars or of fixed-length vectors as a [T, k] array."""
    column = table[name].combine_chunks()
    values = column.flatten() if hasattr(column, "flatten") else column
    return np.asarray(values).reshape(len(column), -1)


def map_robot_rows(table, fields: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the [T, 80] rows and the [80] slot mask that a field mapping reads from a table; a gripper entry with ``open`` and ``closed`` becomes closedness."""
    values = {}
    for name, source in fields.items():
        selected = column_matrix(table, source["column"])[:, source["index"]]
        if name in GRIPPER_FIELDS and "open" in source:
            selected = gripper_closedness(selected, source["open"], source["closed"])
        values[name] = selected
    return pack_robot_rows(values)


def read_episodes_v3(root: Path, view_columns) -> list:
    """Return the episode entries of a LeRobot v3.0 dataset: index, tasks, rows and where the rows and each video start."""
    columns = ["episode_index", "tasks", "length", "data/chunk_index", "data/file_index"]
    for column in view_columns:
        columns += [f"videos/{column}/{name}" for name in VIDEO_LOCATION_FIELDS]
    entries = []
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        entries += pq.read_table(path, columns=columns).to_pylist()
    return sorted(entries, key=lambda entry: entry["episode_index"])


def read_episodes_v2(root: Path) -> list:
    """Return the episode entries of a LeRobot v2 dataset: index, tasks and rows."""
    lines = (root / "meta" / "episodes.jsonl").read_text().splitlines()
    return sorted((json.loads(line) for line in lines if line), key=lambda entry: entry["episode_index"])


class LeRobotStorage:
    """Read a LeRobot dataset (v2.x or v3.0) in place through a mapping from its columns to the robot rows.

    The mapping names what the dataset cannot say itself::

        {"embodiment": "dual-arm ... robot with parallel grippers",
         "action_mode": "joint_only",
         "views": [{"name": "cam_high", "column": "observation.images.cam_high", "slot": 0,
                    "description": "overhead external camera"}, ...],
         "state": {"left_joints": {"column": "observation.state", "index": [0, 1, 2, 3, 4, 5]},
                   "left_gripper": {"column": "observation.state", "index": [6], "open": 1.0, "closed": 0.0}, ...},
         "action": {...}}

    ``state`` and ``action`` map field names of the robot rows to the entries of a column; the action column must
    hold the absolute target commanded at each row. The dataset must already hold every field the policy is
    trained on, an end-effector pose included.

    Args:
        root: The dataset directory.
        mapping: The mapping, or the path of its JSON file.
    """

    def __init__(self, root, mapping):
        self.root = Path(root)
        self.mapping = mapping if isinstance(mapping, dict) else json.loads(Path(mapping).read_text())
        self.layout = json.loads((self.root / "meta" / "info.json").read_text())
        self.is_v3 = str(self.layout["codebase_version"]).startswith("v3")
        view_columns = [view["column"] for view in self.mapping["views"]]
        entries = read_episodes_v3(self.root, view_columns) if self.is_v3 else read_episodes_v2(self.root)
        self.episodes = [
            EpisodeRecord(
                name=f"episode_{int(entry['episode_index']):06d}",
                task=entry["tasks"][0],
                rows=int(entry["length"]),
                instructions=tuple(entry["tasks"]),
                location=entry,
            )
            for entry in entries
        ]
        self.columns = sorted(
            {
                source["column"]
                for fields in (self.mapping["state"], self.mapping["action"])
                for source in fields.values()
            }
        )
        first_rows = self.read_table(self.episodes[0])
        self.info = DatasetInfo(
            fps=float(self.layout["fps"]),
            embodiment=self.mapping["embodiment"],
            action_mode=self.mapping["action_mode"],
            views=tuple(View(view["name"], int(view["slot"]), view["description"]) for view in self.mapping["views"]),
            state_mask=map_robot_rows(first_rows, self.mapping["state"])[1],
            action_mask=map_robot_rows(first_rows, self.mapping["action"])[1],
            resize=self.mapping.get("resize", "stretch"),
        )
        self.view_columns = {view["name"]: view["column"] for view in self.mapping["views"]}

    def read_table(self, record: EpisodeRecord):
        """Return the mapped columns of an episode's rows."""
        index = int(record.location["episode_index"])
        if self.is_v3:
            path = self.layout["data_path"].format(
                chunk_index=record.location["data/chunk_index"], file_index=record.location["data/file_index"]
            )
            return pq.read_table(self.root / path, columns=self.columns, filters=[("episode_index", "=", index)])
        path = self.layout["data_path"].format(
            episode_chunk=index // int(self.layout["chunks_size"]), episode_index=index
        )
        return pq.read_table(self.root / path, columns=self.columns)

    def read_rows(self, record: EpisodeRecord) -> tuple[np.ndarray, np.ndarray]:
        """Return the state rows and the action rows [T, 80] of an episode."""
        table = self.read_table(record)
        return map_robot_rows(table, self.mapping["state"])[0], map_robot_rows(table, self.mapping["action"])[0]

    def read_frames(self, record: EpisodeRecord, view_name: str, frame_indices) -> np.ndarray:
        """Return the uint8 frames [n, H, W, 3] of one view of an episode at the given rows."""
        column = self.view_columns[view_name]
        index = int(record.location["episode_index"])
        if self.is_v3:
            path = self.layout["video_path"].format(
                video_key=column,
                chunk_index=record.location[f"videos/{column}/chunk_index"],
                file_index=record.location[f"videos/{column}/file_index"],
            )
            start_time = float(record.location[f"videos/{column}/from_timestamp"])
            end_time = float(record.location[f"videos/{column}/to_timestamp"])
            last_frame = int(round((end_time - start_time) * self.info.fps)) - 1
            return decode_frames(self.root / path, np.minimum(frame_indices, last_frame), self.info.fps, start_time)
        path = self.layout["video_path"].format(
            episode_chunk=index // int(self.layout["chunks_size"]), video_key=column, episode_index=index
        )
        return decode_frames(self.root / path, frame_indices, self.info.fps)
