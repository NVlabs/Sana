# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from sana_robot.data.format.episode import DatasetInfo, EpisodeRecord, View
from sana_robot.data.format.normalization import quantile_normalization, save_normalization
from sana_robot.data.format.robot80 import ACTION_MODE_EEF, field_slots, slot_mask
from sana_robot.data.format.video import RESIZE_CROP
from sana_robot.data.storage.lerobot import column_matrix, read_episodes_v2
from sana_robot.data.storage.zip_archive import write_dataset_info, write_episode, write_episode_index
from sana_robot.data.utils.kinematics import dual_arm_rows

EMBODIMENT = "dual-arm RoboTwin2 Aloha-AgileX robot with parallel grippers"
VIEWS = (
    View("head_camera", 0, "overhead external camera"),
    View("left_camera", 2, "left wrist-mounted camera"),
    View("right_camera", 3, "right wrist-mounted camera"),
)
VIDEO_COLUMNS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
GRIPPER_OPEN, GRIPPER_CLOSED = 1.0, 0.0
EPISODES_PER_TASK = 550
CLEAN_EPISODES_PER_TASK = 50
NORMALIZED_FIELDS = (("left_joints", 6), ("left_eef_position", 3), ("right_joints", 6), ("right_eef_position", 3))

ALOHA_AGILEX_JOINTS = (
    ((0.0, 0.0, 0.058), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.025013, 0.00060169, 0.042), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((-0.26396, 0.0044548, 0.0), (-3.1416, 0.0, -0.015928), (0.0, 1.0, 0.0)),
    ((0.246, -0.00025, -0.06), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.06775, 0.0015, -0.0855), (0.0, 0.0, -0.015928), (0.0, 0.0, 1.0)),
    ((0.03095, 0.0, 0.0855), (-3.1416, 0.0, 0.0), (1.0, 0.0, 0.0)),
)


def episode_rows(source: Path, layout: dict, index: int, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the [T, 80] robot rows and their slot mask that one column of an episode of the release gives: ``observation.state`` or ``action``."""
    place = {"episode_chunk": index // int(layout["chunks_size"]), "episode_index": index}
    table = pq.read_table(source / layout["data_path"].format(**place), columns=[column])
    return dual_arm_rows(column_matrix(table, column), ALOHA_AGILEX_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSED)


def state_rows(job: tuple) -> np.ndarray:
    """Return the state rows [T, 80] of one episode for a job of (release directory, its layout, episode index)."""
    return episode_rows(*job, "observation.state")[0]


def release_normalization(source) -> dict:
    """Return the normalization of the released RoboTwin checkpoint: from the state rows of every episode of the release, clean and randomized, each row weighing the same; the joints and the flange positions are normalized, with one set of statistics for both arms, and the rotations and the grippers keep their values."""
    source = Path(source)
    layout = json.loads((source / "meta" / "info.json").read_text())
    jobs = [(source, layout, int(entry["episode_index"])) for entry in read_episodes_v2(source)]
    with ProcessPoolExecutor() as pool:
        rows = np.concatenate(list(pool.map(state_rows, jobs, chunksize=64)))
    slots = [slot for name, width in NORMALIZED_FIELDS for slot in field_slots(name, width)]
    return quantile_normalization(rows, np.ones(len(rows)), slot_mask(slots))


def convert(
    source,
    output,
    clean_only: bool = True,
    episodes_per_archive: int = 500,
    episode_count: int | None = None,
    normalization=None,
):
    """Build a zip archive dataset from the LeRobot v2.1 release of RoboTwin 2.0 (``yuanty/robotwin2.0-fastwam``).

    Every episode keeps its rows and frames as they are: the state of a row is its joint angles, the pose of each
    flange in its arm's base frame from the forward kinematics of the Aloha-AgileX arm and the gripper closedness;
    the action of a row is the same fields of the commanded joint angles. The videos are copied as they are. The
    release holds 550 episodes per task, the first 50 of them recorded in the clean scenes and the others in the
    randomized ones.

    Args:
        source: The directory of the release (``robotwin2.0``).
        output: The directory of the zip archive dataset to create.
        clean_only: Keep only the episodes of the clean scenes.
        episodes_per_archive: Episodes stored in one archive.
        episode_count: Number of episodes to convert, all the selected ones by default.
        normalization: The normalization file to write next to the dataset, from every episode of the release
            (``release_normalization``); none by default.

    Returns:
        The episode records of the new dataset.
    """
    source, output = Path(source), Path(output)
    layout = json.loads((source / "meta" / "info.json").read_text())
    entries = [
        entry
        for entry in read_episodes_v2(source)
        if not clean_only or int(entry["episode_index"]) % EPISODES_PER_TASK < CLEAN_EPISODES_PER_TASK
    ][:episode_count]
    (output / "chunks").mkdir(parents=True, exist_ok=True)
    records, mask = [], None
    for start in range(0, len(entries), episodes_per_archive):
        archive_name = f"chunks/chunk-{start // episodes_per_archive:06d}.zip"
        with zipfile.ZipFile(output / archive_name, "w") as archive:
            for entry in entries[start : start + episodes_per_archive]:
                index = int(entry["episode_index"])
                place = {"episode_chunk": index // int(layout["chunks_size"]), "episode_index": index}
                state, mask = episode_rows(source, layout, index, "observation.state")
                action, _ = episode_rows(source, layout, index, "action")
                videos = {
                    view.name: (source / layout["video_path"].format(video_key=column, **place)).read_bytes()
                    for view, column in zip(VIEWS, VIDEO_COLUMNS)
                }
                name = f"episode_{index:06d}"
                write_episode(archive, name, state, action, videos)
                records.append(
                    EpisodeRecord(
                        name=name,
                        task=f"task_{index // EPISODES_PER_TASK:02d}",
                        rows=len(state),
                        instructions=tuple(text.strip() for text in entry["tasks"]),
                        location={"archive": archive_name},
                    )
                )
        print(f"{archive_name}: {len(records)} episodes done", flush=True)
    info = DatasetInfo(float(layout["fps"]), EMBODIMENT, ACTION_MODE_EEF, VIEWS, mask, mask, RESIZE_CROP)
    write_dataset_info(output, info)
    write_episode_index(output, records)
    if normalization:
        save_normalization(normalization, release_normalization(source))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument("--episodes-per-archive", type=int, default=500)
    parser.add_argument("--episode-count", type=int, default=None)
    parser.add_argument("--normalization", default=None)
    arguments = parser.parse_args()
    convert(
        arguments.source,
        arguments.output,
        not arguments.all_episodes,
        arguments.episodes_per_archive,
        arguments.episode_count,
        arguments.normalization,
    )
