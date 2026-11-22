# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import struct
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np
from PIL import Image

from sana_robot.data.format.episode import DatasetInfo, EpisodeRecord, View
from sana_robot.data.format.normalization import save_normalization
from sana_robot.data.format.robot80 import (
    ACTION_MODE_EEF,
    ROBOT_DIM,
    field_slots,
    gripper_closedness,
    pack_robot_rows,
    rotation_to_rot6d,
    slot_mask,
)
from sana_robot.data.format.video import RESIZE_CROP
from sana_robot.data.storage.zip_archive import write_dataset_info, write_episode, write_episode_index
from sana_robot.data.utils.kinematics import flange_poses

EMBODIMENT = "single-arm Franka Panda robot with a parallel gripper"
VIEWS = (
    View("image", 0, "front external camera"),
    View("wrist_image", 2, "wrist-mounted camera on the single controlled arm"),
)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
FPS = 20
GRIPPER_OPEN, GRIPPER_CLOSED = 0.08, 0.0
STATE_FIELDS = ("right_joints", "right_eef_position", "right_eef_rotation", "right_gripper")
ACTION_FIELDS = ("right_eef_position", "right_eef_rotation", "right_gripper")
NORMALIZED_FIELDS = ("right_joints", "right_eef_position")
PRETRAIN_CENTER = (
    -0.033311367,
    0.18197846,
    0.00015175343,
    -1.8144687,
    0.027134597,
    2.2185416,
    0.07553089,
    0.5068566,
    0.009417623,
    0.3766749,
)
PRETRAIN_SCALE = (
    1.090375,
    0.9845456,
    0.95905024,
    0.9170548,
    1.5745702,
    0.99462724,
    2.0304637,
    0.2215178,
    0.4880964,
    0.3214174,
)

FRANKA_JOINTS = (
    ((0.0, 0.0, 0.333), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 0.0), (-1.57079632679, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, -0.316, 0.0), (1.57079632679, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0825, 0.0, 0.0), (1.57079632679, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((-0.0825, 0.384, 0.0), (-1.57079632679, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 0.0), (1.57079632679, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.088, 0.0, 0.0), (1.57079632679, 0.0, 0.0), (0.0, 0.0, 1.0)),
)
FRANKA_FLANGE = ((0.0, 0.0, 0.107), (0.0, 0.0, 0.0))
FRANKA_FLANGE_AXES = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
FRANKA_RESET_JOINTS = (0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.22675220, math.pi / 4)

HAND_HEIGHT, HAND_QUATERNION = 0.1065, (0.0, 0.0, -0.383, 0.924)
GRIP_SITE_HEIGHT, GRIP_SITE_TURN = 0.097, -math.pi / 2
POSITION_STEP, ROTATION_STEP = 0.05, 0.5


def read_integer(data, position: int) -> tuple[int, int]:
    """Return the variable-length integer at ``position`` of a protobuf message and the position after it."""
    value = shift = 0
    while True:
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, position
        shift += 7


def message_fields(data):
    """Yield the (number, content) of every field of a protobuf message whose fields are all length-delimited."""
    position = 0
    while position < len(data):
        tag, position = read_integer(data, position)
        length, position = read_integer(data, position)
        yield tag >> 3, data[position : position + length]
        position += length


def read_episodes(path):
    """Yield every episode of a TFRecord file of the release as a dict from a feature name to its values, a list of byte strings or a float32 array; integer features are left out."""
    with open(path, "rb") as file:
        while header := file.read(12):
            record = memoryview(file.read(struct.unpack("<Q", header[:8])[0] + 4))[:-4]
            episode = {}
            for _, features in message_fields(record):
                for _, entry in message_fields(features):
                    parts = dict(message_fields(entry))
                    for value_type, values in message_fields(parts[2]):
                        contents = [bytes(content) for _, content in message_fields(values)]
                        if value_type == 1:
                            episode[bytes(parts[1]).decode()] = contents
                        elif value_type == 2:
                            episode[bytes(parts[1]).decode()] = np.frombuffer(b"".join(contents), dtype="<f4")
            yield episode


def encode_video(images) -> bytes:
    """Return the encoded images of an episode, one per row, as an MP4 of lossless H.264 RGB frames at the control frequency."""
    video = io.BytesIO()
    with av.open(video, "w", format="mp4") as container:
        stream = container.add_stream("libx264rgb", rate=FPS)
        stream.pix_fmt = "rgb24"
        stream.options = {"crf": "0", "preset": "fast"}
        stream.codec_context.thread_count = 1
        for index, image in enumerate(images):
            pixels = np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))
            if index == 0:
                stream.height, stream.width = pixels.shape[:2]
            container.mux(stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")))
        container.mux(stream.encode())
    return video.getvalue()


def quaternion_to_rotation(quaternion) -> np.ndarray:
    """Return the rotation matrices [..., 3, 3] of unit quaternions [..., 4] in the order (x, y, z, w)."""
    x, y, z, w = np.moveaxis(np.asarray(quaternion, dtype=np.float64), -1, 0)
    rows = (
        (x * x - y * y - z * z + w * w, 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), -x * x + y * y - z * z + w * w, 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), -x * x - y * y + z * z + w * w),
    )
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def rotation_vector_to_rotation(vector) -> np.ndarray:
    """Return the rotation matrices [..., 3, 3] of rotation vectors [..., 3]: the axis of a turn times its angle in radians."""
    x, y, z = np.moveaxis(np.asarray(vector, dtype=np.float64), -1, 0)
    angle = np.sqrt(x * x + y * y + z * z)
    small = angle <= 1.0e-3
    series = 0.5 - angle * angle / 48 + angle * angle * angle * angle / 3840
    scale = np.where(small, series, np.sin(angle / 2) / np.where(small, 1.0, angle))
    return quaternion_to_rotation(np.stack((scale * x, scale * y, scale * z, np.cos(angle / 2)), axis=-1))


def grip_site_pose() -> np.ndarray:
    """Return the fixed [4, 4] pose of the grip site, the point LIBERO's controller moves, in the flange frame of ``robot_rows``: the hand of the simulated Panda sits 0.1065 m after the last joint and the grip site 0.097 m after the hand."""
    axes, hand, site = np.eye(4), np.eye(4), np.eye(4)
    axes[:3, :3] = FRANKA_FLANGE_AXES
    hand[:3, :3] = quaternion_to_rotation(np.asarray(HAND_QUATERNION) / np.linalg.norm(HAND_QUATERNION))
    hand[2, 3] = HAND_HEIGHT - FRANKA_FLANGE[0][2]
    site[:3, :3] = quaternion_to_rotation((0.0, 0.0, math.sin(GRIP_SITE_TURN / 2), math.cos(GRIP_SITE_TURN / 2)))
    site[2, 3] = GRIP_SITE_HEIGHT
    return axes.T @ hand @ site


def commanded_poses(pose: np.ndarray, command: np.ndarray) -> np.ndarray:
    """Return the absolute flange poses [T, 4, 4] that the controller commands [T, 7] of a whole episode ask for, from the observed flange poses [T, 4, 4] of the same rows. A command moves the grip site from its observed position by up to 5 cm per axis and turns it from its observed rotation by a rotation vector of up to 0.5 rad per axis, in the axes of the robot base; a command without a turn keeps the rotation goal of the row before it, at first that of the reset pose."""
    site_pose = grip_site_pose()
    site = pose @ site_pose
    change = np.clip(np.asarray(command[:, :6], dtype=np.float64), -1.0, 1.0)
    goal = site.copy()
    goal[:, :3, 3] += change[:, :3] * POSITION_STEP
    turned = rotation_vector_to_rotation(change[:, 3:] * ROTATION_STEP) @ site[:, :3, :3]
    reset = flange_poses(FRANKA_JOINTS, np.asarray(FRANKA_RESET_JOINTS)[None], FRANKA_FLANGE, FRANKA_FLANGE_AXES)
    rotation = (reset @ site_pose)[0, :3, :3]
    for row, turns in enumerate(change[:, 3:].any(axis=1)):
        rotation = turned[row] if turns else rotation
        goal[row, :3, :3] = rotation
    flange_in_site = np.eye(4)
    flange_in_site[:3, :3] = site_pose[:3, :3].T
    flange_in_site[:3, 3] = -(site_pose[:3, :3].T @ site_pose[:3, 3])
    return goal @ flange_in_site


def state_rows(joints: np.ndarray, fingers: np.ndarray) -> np.ndarray:
    """Return the state rows [T, 80] of joint angles [T, 7] and two finger positions [T, 2] in metres: the joints, the flange pose in the arm's base frame and the gripper closedness."""
    pose = flange_poses(FRANKA_JOINTS, joints, FRANKA_FLANGE, FRANKA_FLANGE_AXES)
    fingers = np.asarray(fingers, dtype=np.float64)
    state, _ = pack_robot_rows(
        {
            "right_joints": joints,
            "right_eef_position": pose[:, :3, 3],
            "right_eef_rotation": rotation_to_rot6d(pose[:, :3, :3]),
            "right_gripper": gripper_closedness(fingers[:, 0] - fingers[:, 1], GRIPPER_OPEN, GRIPPER_CLOSED),
        }
    )
    return state


def robot_rows(joints: np.ndarray, fingers: np.ndarray, command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the state rows and the action rows [T, 80] of an episode from its joint angles [T, 7], its two finger positions [T, 2] in metres and its controller commands [T, 7]. The state is the joints, the pose of the flange (``panda_link8``, with +X along its approach axis) and the gripper closedness from the finger distance; the action is the commanded flange pose and the commanded closedness."""
    pose = flange_poses(FRANKA_JOINTS, joints, FRANKA_FLANGE, FRANKA_FLANGE_AXES)
    goal = commanded_poses(pose, command)
    state = state_rows(joints, fingers)
    action, _ = pack_robot_rows(
        {
            "right_eef_position": goal[:, :3, 3],
            "right_eef_rotation": rotation_to_rot6d(goal[:, :3, :3]),
            "right_gripper": (np.clip(np.asarray(command[:, 6], dtype=np.float64), -1.0, 1.0) + 1.0) / 2.0,
        }
    )
    return state, action


def suite_jobs(source: Path, suite: str, episode_count: int | None) -> list:
    """Return the conversion jobs of one suite, one per TFRecord file in the order of the release: (file, suite, index of its first episode in the suite, number of its episodes to convert)."""
    folder = source / f"{suite}_no_noops" / "1.0.0"
    description = json.loads((folder / "dataset_info.json").read_text())
    lengths = [
        int(length) for split in description["splits"] if split["name"] == "train" for length in split["shardLengths"]
    ]
    jobs, first = [], 0
    for path, length in zip(sorted(folder.glob("*.tfrecord-*")), lengths):
        count = length if episode_count is None else min(length, episode_count - first)
        if count > 0:
            jobs.append((path, suite, first, count))
        first += length
    return jobs


def convert_file(job: tuple) -> list:
    """Return the episodes of one job of ``suite_jobs`` as (name, instruction, state rows, action rows, encoded video per view name)."""
    path, suite, first, count = job
    episodes = []
    for index, features in zip(range(first, first + count), read_episodes(path)):
        state, action = robot_rows(
            features["steps/observation/joint_state"].reshape(-1, 7),
            features["steps/observation/state"].reshape(-1, 8)[:, 6:],
            features["steps/action"].reshape(-1, 7),
        )
        videos = {view.name: encode_video(features[f"steps/observation/{view.name}"]) for view in VIEWS}
        instruction = features["steps/language_instruction"][0].decode().strip()
        episodes.append((f"{suite}_{index:06d}", instruction, state, action, videos))
    return episodes


def pretrain_normalization() -> dict:
    """Return the normalization of the released LIBERO checkpoint, which keeps the statistics of the Franka arms of the pretraining data: the joints and the flange position are normalized, and the rotation and the gripper keep their values."""
    slots = [slot for name in NORMALIZED_FIELDS for slot in field_slots(name)]
    center, scale = np.zeros(ROBOT_DIM, dtype=np.float32), np.ones(ROBOT_DIM, dtype=np.float32)
    center[slots], scale[slots] = PRETRAIN_CENTER, PRETRAIN_SCALE
    return {"center": center, "scale": scale, "mask": slot_mask(slots)}


def convert(
    source,
    output,
    episodes_per_archive: int = 500,
    episode_count: int | None = None,
    workers: int | None = None,
    normalization=None,
):
    """Build a zip archive dataset from the RLDS release of LIBERO without no-op rows (``openvla/modified_libero_rlds``).

    The four suites follow each other (spatial, object, goal, 10), every episode keeps its rows and frames as they
    are. The state of a row is its joint angles, the pose of the flange in the arm's base frame from the forward
    kinematics of the Franka Panda arm and the gripper closedness. The action of a row is the absolute flange pose
    its controller command asks for and the commanded closedness. The frames are stored as lossless videos.

    Args:
        source: The directory of the release that holds ``libero_spatial_no_noops`` and the three other suites.
        output: The directory of the zip archive dataset to create.
        episodes_per_archive: Episodes stored in one archive.
        episode_count: Number of episodes to convert per suite, all of them by default.
        workers: Number of processes that convert the files, one per processor by default.
        normalization: The normalization file to write next to the dataset (``pretrain_normalization``); none by
            default.

    Returns:
        The episode records of the new dataset.
    """
    source, output = Path(source), Path(output)
    jobs = [job for suite in SUITES for job in suite_jobs(source, suite, episode_count)]
    (output / "chunks").mkdir(parents=True, exist_ok=True)
    records = []
    with ProcessPoolExecutor(workers) as pool:
        episodes = itertools.chain.from_iterable(pool.map(convert_file, jobs))
        for start in range(0, sum(job[3] for job in jobs), episodes_per_archive):
            archive_name = f"chunks/chunk-{start // episodes_per_archive:06d}.zip"
            with zipfile.ZipFile(output / archive_name, "w") as archive:
                for name, instruction, state, action, videos in itertools.islice(episodes, episodes_per_archive):
                    write_episode(archive, name, state, action, videos)
                    records.append(
                        EpisodeRecord(
                            name=name,
                            task=instruction,
                            rows=len(state),
                            instructions=(instruction,),
                            location={"archive": archive_name},
                        )
                    )
            print(f"{archive_name}: {len(records)} episodes done", flush=True)
    state_mask = slot_mask(slot for name in STATE_FIELDS for slot in field_slots(name))
    action_mask = slot_mask(slot for name in ACTION_FIELDS for slot in field_slots(name))
    write_dataset_info(
        output, DatasetInfo(float(FPS), EMBODIMENT, ACTION_MODE_EEF, VIEWS, state_mask, action_mask, RESIZE_CROP)
    )
    write_episode_index(output, records)
    if normalization:
        save_normalization(normalization, pretrain_normalization())
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes-per-archive", type=int, default=500)
    parser.add_argument("--episode-count", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--normalization", default=None)
    arguments = parser.parse_args()
    convert(
        arguments.source,
        arguments.output,
        arguments.episodes_per_archive,
        arguments.episode_count,
        arguments.workers,
        arguments.normalization,
    )
