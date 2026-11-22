# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np

from sana_robot.data.format.episode import DatasetInfo, EpisodeRecord, View
from sana_robot.data.format.robot80 import slot_mask
from sana_robot.data.format.video import decode_frames

ZIP_FORMAT = "sana_robot_zip_v1"
INFO_FILE = "meta.json"
INDEX_FILE = "episodes.jsonl"


def write_dataset_info(root, info: DatasetInfo) -> None:
    """Write the dataset description of a zip archive dataset."""
    content = {
        "format": ZIP_FORMAT,
        "fps": float(info.fps),
        "embodiment": info.embodiment,
        "action_mode": info.action_mode,
        "views": [{"name": view.name, "slot": int(view.slot), "description": view.description} for view in info.views],
        "state_slots": np.flatnonzero(info.state_mask).tolist(),
        "action_slots": np.flatnonzero(info.action_mask).tolist(),
        "resize": info.resize,
    }
    Path(root, INFO_FILE).write_text(json.dumps(content, indent=2) + "\n")


def write_episode_index(root, records) -> None:
    """Write the episode index of a zip archive dataset, one line per episode."""
    lines = [
        json.dumps(
            {
                "name": record.name,
                "task": record.task,
                "rows": int(record.rows),
                "instructions": list(record.instructions),
                "archive": record.location["archive"],
            }
        )
        for record in records
    ]
    Path(root, INDEX_FILE).write_text("\n".join(lines) + "\n")


def write_episode(archive: zipfile.ZipFile, name: str, state: np.ndarray, action: np.ndarray, videos: dict) -> None:
    """Store one episode in an open archive: its state and action rows [T, 80] and one encoded video per view name."""
    rows = io.BytesIO()
    np.savez(rows, state=np.asarray(state, dtype=np.float32), action=np.asarray(action, dtype=np.float32))
    archive.writestr(zipfile.ZipInfo(f"{name}/rows.npz"), rows.getvalue(), compress_type=zipfile.ZIP_STORED)
    for view_name, encoded in videos.items():
        member = zipfile.ZipInfo(f"{name}/videos/{view_name}.mp4")
        archive.writestr(member, encoded, compress_type=zipfile.ZIP_STORED)


class ZipArchiveStorage:
    """Read a dataset stored as zip archives of episodes.

    The dataset directory holds ``meta.json`` (the control frequency, the prompt fields, the cameras, the robot
    slots in use and the resize rule of the frames), ``episodes.jsonl`` (one episode per line: name, task, rows,
    instructions and archive) and the archives. An archive stores every episode uncompressed as
    ``<name>/rows.npz`` (``state`` and ``action`` [T, 80]: the observed state of every row and the absolute target
    commanded at it) and ``<name>/videos/<view>.mp4`` (one frame per row).

    Args:
        root: The dataset directory.
    """

    def __init__(self, root):
        self.root = Path(root)
        content = json.loads((self.root / INFO_FILE).read_text())
        self.info = DatasetInfo(
            fps=float(content["fps"]),
            embodiment=content["embodiment"],
            action_mode=content["action_mode"],
            views=tuple(View(view["name"], int(view["slot"]), view["description"]) for view in content["views"]),
            state_mask=slot_mask(content["state_slots"]),
            action_mask=slot_mask(content["action_slots"]),
            resize=content.get("resize", "stretch"),
        )
        index = self.root / INDEX_FILE
        entries = [json.loads(line) for line in index.read_text().splitlines() if line] if index.exists() else []
        self.episodes = [
            EpisodeRecord(
                name=entry["name"],
                task=entry["task"],
                rows=int(entry["rows"]),
                instructions=tuple(entry["instructions"]),
                location={"archive": entry["archive"]},
            )
            for entry in entries
        ]
        self.open_archives = {}

    def archive(self, record: EpisodeRecord) -> zipfile.ZipFile:
        """Return the open archive of an episode, opened once per process."""
        key = (os.getpid(), record.location["archive"])
        if key not in self.open_archives:
            self.open_archives[key] = zipfile.ZipFile(self.root / record.location["archive"])
        return self.open_archives[key]

    def read_rows(self, record: EpisodeRecord) -> tuple[np.ndarray, np.ndarray]:
        """Return the state rows and the action rows [T, 80] of an episode."""
        with np.load(io.BytesIO(self.archive(record).read(f"{record.name}/rows.npz"))) as rows:
            return rows["state"], rows["action"]

    def read_frames(self, record: EpisodeRecord, view_name: str, frame_indices) -> np.ndarray:
        """Return the uint8 frames [n, H, W, 3] of one view of an episode at the given rows."""
        encoded = self.archive(record).read(f"{record.name}/videos/{view_name}.mp4")
        return decode_frames(encoded, frame_indices, self.info.fps)
