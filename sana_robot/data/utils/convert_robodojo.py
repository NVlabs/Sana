# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import av
import pyarrow.parquet as pq

from sana_robot.data.format.episode import DatasetInfo, EpisodeRecord, View
from sana_robot.data.format.robot80 import ACTION_MODE_EEF
from sana_robot.data.storage.lerobot import column_matrix, read_episodes_v3
from sana_robot.data.storage.zip_archive import write_dataset_info, write_episode, write_episode_index
from sana_robot.data.utils.kinematics import dual_arm_rows

EMBODIMENT = "dual-arm RoboDojo ARX-X5 robot with parallel grippers"
VIEWS = (
    View("cam_high", 0, "overhead external camera"),
    View("cam_left_wrist", 2, "left wrist-mounted camera"),
    View("cam_right_wrist", 3, "right wrist-mounted camera"),
)
GRIPPER_OPEN, GRIPPER_CLOSED = 1.0, 0.0

ARX_X5_JOINTS = (
    ((0.0, 0.0, 0.0605), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.02, 0.0, 0.04), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((-0.264, 0.0, 0.0), (3.1416, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.245, 0.0, -0.056), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.06775, 0.0005, -0.0865), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.02895, 0.0, 0.0865), (-3.1416, 0.0, 0.0), (1.0, 0.0, 0.0)),
)


def cut_video(path, start_time: float, end_time: float) -> bytes:
    """Return the frames shown in [``start_time``, ``end_time``) of a video file as an MP4 of its own, copied without re-encoding."""
    cut = io.BytesIO()
    with av.open(str(path)) as source, av.open(cut, "w", format="mp4") as output:
        stream = source.streams.video[0]
        context = stream.codec_context
        copy = output.add_mux_stream(
            context.codec.canonical_name,
            rate=stream.average_rate,
            width=context.width,
            height=context.height,
            time_base=stream.time_base,
        )
        first_tick = int(round(start_time / stream.time_base))
        end_tick = int(round(end_time / stream.time_base))
        source.seek(first_tick, stream=stream, backward=True)
        for packet in source.demux(stream):
            if packet.dts is None or packet.pts < first_tick:
                continue
            if packet.pts >= end_tick:
                break
            packet.pts -= first_tick
            packet.dts -= first_tick
            packet.stream = copy
            output.mux(packet)
    return cut.getvalue()


def convert(source, output, episodes_per_archive: int = 1000, first_episode: int = 0, episode_count: int | None = None):
    """Build a zip archive dataset from the official RoboDojo LeRobot v3.0 dataset of joint states.

    Every episode keeps its rows and frames as they are: the state of a row is its joint angles, the pose of each
    flange from the forward kinematics of the ARX-X5 arm and the gripper closedness; the action of a row is the
    same fields of the commanded joint angles. The videos are copied without re-encoding.

    Args:
        source: The directory of ``RoboDojo_lerobot_v30_video``.
        output: The directory of the zip archive dataset to create.
        episodes_per_archive: Episodes stored in one archive.
        first_episode: Index of the first episode to convert.
        episode_count: Number of episodes to convert, all the remaining ones by default.

    Returns:
        The episode records of the new dataset.
    """
    source, output = Path(source), Path(output)
    layout = json.loads((source / "meta" / "info.json").read_text())
    fps = float(layout["fps"])
    columns = [f"observation.images.{view.name}" for view in VIEWS]
    entries = read_episodes_v3(source, columns)
    entries = entries[first_episode : None if episode_count is None else first_episode + episode_count]
    (output / "chunks").mkdir(parents=True, exist_ok=True)
    records, mask = [], None
    for start in range(0, len(entries), episodes_per_archive):
        archive_name = f"chunks/chunk-{(first_episode + start) // episodes_per_archive:06d}.zip"
        with zipfile.ZipFile(output / archive_name, "w") as archive:
            for entry in entries[start : start + episodes_per_archive]:
                index = int(entry["episode_index"])
                data_path = layout["data_path"].format(
                    chunk_index=entry["data/chunk_index"], file_index=entry["data/file_index"]
                )
                table = pq.read_table(
                    source / data_path, columns=["observation.state", "action"], filters=[("episode_index", "=", index)]
                )
                state, mask = dual_arm_rows(
                    column_matrix(table, "observation.state"), ARX_X5_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSED
                )
                action, _ = dual_arm_rows(column_matrix(table, "action"), ARX_X5_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSED)
                videos = {}
                for view, column in zip(VIEWS, columns):
                    video_path = layout["video_path"].format(
                        video_key=column,
                        chunk_index=entry[f"videos/{column}/chunk_index"],
                        file_index=entry[f"videos/{column}/file_index"],
                    )
                    start_time = float(entry[f"videos/{column}/from_timestamp"])
                    end_time = float(entry[f"videos/{column}/to_timestamp"])
                    if int(round((end_time - start_time) * fps)) != len(state):
                        print(
                            f"episode {index} {view.name}: {(end_time - start_time) * fps:.0f} frames for {len(state)} rows"
                        )
                    videos[view.name] = cut_video(source / video_path, start_time, end_time)
                name = f"episode_{index:06d}"
                write_episode(archive, name, state, action, videos)
                records.append(
                    EpisodeRecord(
                        name=name,
                        task=entry["tasks"][0],
                        rows=len(state),
                        instructions=tuple(entry["tasks"]),
                        location={"archive": archive_name},
                    )
                )
        print(f"{archive_name}: {len(records)} episodes done", flush=True)
    write_dataset_info(output, DatasetInfo(fps, EMBODIMENT, ACTION_MODE_EEF, VIEWS, mask, mask))
    write_episode_index(output, records)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes-per-archive", type=int, default=1000)
    parser.add_argument("--first-episode", type=int, default=0)
    parser.add_argument("--episode-count", type=int, default=None)
    arguments = parser.parse_args()
    convert(
        arguments.source,
        arguments.output,
        arguments.episodes_per_archive,
        arguments.first_episode,
        arguments.episode_count,
    )
