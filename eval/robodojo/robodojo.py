# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""The RoboDojo ARX-X5 side of the adapter: observations into SANA-Robot robot rows and robot rows into actions.

A SANA-Robot policy reads and writes robot rows of 80 slots in the units of its training dataset. For RoboDojo
that dataset is the one ``sana_robot.data.utils.convert_robodojo`` builds from the official LeRobot export: per arm
six joint angles, the pose of the flange in the arm's base frame (from the forward kinematics of the joints, with
the axes +X approach, +Y down, +Z left) and the gripper closedness. The functions here apply the same conversion
to an observation and undo it for the actions the environment client takes.

``CAMERA_OF_VIEW`` names the camera of an observation that each view of the dataset was recorded from: the official
LeRobot export stores ``vision/cam_head`` as ``observation.images.cam_high`` and keeps the names of the wrist
cameras. ``ARX_X5_ROOT_POSES`` holds the pose [x, y, z, qw, qx, qy, qz] of each arm's base in the frame the
environment reports ``*_ee_pose`` in: ``default_root_pos`` and ``default_root_rot`` of RoboDojo's
``env_cfg/robot/dual_x5.yml``.
"""

from __future__ import annotations

import numpy as np

from sana_robot.data.format.robot80 import ROBOT_FIELDS, rot6d_to_rotation
from sana_robot.data.utils.convert_robodojo import ARX_X5_JOINTS, GRIPPER_CLOSED, GRIPPER_OPEN
from sana_robot.data.utils.kinematics import FLANGE_AXES, dual_arm_rows

ARM_JOINTS = len(ARX_X5_JOINTS)

SIDES = ("left", "right")
ACTION_TYPE_JOINT = "joint"
ACTION_TYPE_EE = "ee"
ACTION_TYPES = (ACTION_TYPE_JOINT, ACTION_TYPE_EE)

CAMERA_OF_VIEW = {"cam_high": "cam_head", "cam_left_wrist": "cam_left_wrist", "cam_right_wrist": "cam_right_wrist"}

ARX_X5_ROOT_POSES = {
    "left": (-0.3, -0.45, 0.765, 0.707, 0.0, 0.0, 0.707),
    "right": (0.3, -0.45, 0.765, 0.707, 0.0, 0.0, 0.707),
}


def quaternion_to_rotation(quaternion) -> np.ndarray:
    """Return the rotation matrix [3, 3] of a quaternion [qw, qx, qy, qz], which need not have unit length."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4) / np.linalg.norm(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def rotation_to_quaternion(rotation) -> np.ndarray:
    """Return the unit quaternion [qw, qx, qy, qz] of a rotation matrix [3, 3], with qw >= 0."""
    m = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    candidates = np.array(
        [
            [1.0 + m[0, 0] + m[1, 1] + m[2, 2], m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]],
            [m[2, 1] - m[1, 2], 1.0 + m[0, 0] - m[1, 1] - m[2, 2], m[0, 1] + m[1, 0], m[0, 2] + m[2, 0]],
            [m[0, 2] - m[2, 0], m[0, 1] + m[1, 0], 1.0 - m[0, 0] + m[1, 1] - m[2, 2], m[1, 2] + m[2, 1]],
            [m[1, 0] - m[0, 1], m[0, 2] + m[2, 0], m[1, 2] + m[2, 1], 1.0 - m[0, 0] - m[1, 1] + m[2, 2]],
        ]
    )
    quaternion = candidates[int(np.argmax(np.diag(candidates)))]
    quaternion = quaternion / np.linalg.norm(quaternion)
    return -quaternion if quaternion[0] < 0 else quaternion


def pose_to_transform(pose) -> np.ndarray:
    """Return the transform [4, 4] of a pose [x, y, z, qw, qx, qy, qz]."""
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    transform = np.eye(4)
    transform[:3, :3] = quaternion_to_rotation(pose[3:])
    transform[:3, 3] = pose[:3]
    return transform


def transform_to_pose(transform) -> np.ndarray:
    """Return the pose [x, y, z, qw, qx, qy, qz] of a transform [4, 4]."""
    transform = np.asarray(transform, dtype=np.float64)
    return np.concatenate((transform[:3, 3], rotation_to_quaternion(transform[:3, :3])))


def root_transforms(root_poses=None) -> dict:
    """Return the transform [4, 4] of each arm's base from ``{side: [x, y, z, qw, qx, qy, qz]}``; the ARX-X5 poses of RoboDojo by default."""
    poses = ARX_X5_ROOT_POSES if root_poses is None else root_poses
    return {side: pose_to_transform(poses[side]) for side in SIDES}


def check_robot(action_dim_info: dict) -> None:
    """Refuse a robot other than two six-joint arms with one gripper value each, the robot the conversion is written for."""
    arms, grippers = list(action_dim_info["arm_dim"]), list(action_dim_info["ee_dim"])
    if arms != [ARM_JOINTS] * len(SIDES) or grippers != [1] * len(SIDES):
        raise ValueError(
            f"SANA_Robot serves the RoboDojo ARX-X5 robot (arm_dim {[ARM_JOINTS] * len(SIDES)}, ee_dim "
            f"{[1] * len(SIDES)}), got arm_dim {arms} and ee_dim {grippers}"
        )


def camera_images(vision: dict, cameras) -> list:
    """Return the RGB frame uint8 [H, W, 3] of every camera in ``cameras`` as a writable copy of what the server decoded."""
    images = []
    for camera in cameras:
        image = np.asarray(vision[camera]["color"])
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{camera}: expected a uint8 RGB frame [H, W, 3], got {image.dtype} {image.shape}")
        images.append(np.array(image, copy=True, order="C"))
    return images


def state_row(state: dict) -> np.ndarray:
    """Return the robot row [80] of an observation's ``state``: the joint angles and the gripper opening of both arms, converted as the training dataset was."""
    values = []
    for side in SIDES:
        joints = np.asarray(state[f"{side}_arm_joint_state"], dtype=np.float64).reshape(-1)
        if joints.shape != (ARM_JOINTS,):
            raise ValueError(f"{side}_arm_joint_state must hold {ARM_JOINTS} joint angles, got {joints.shape}")
        opening = np.asarray(state[f"{side}_ee_joint_state"], dtype=np.float64).reshape(-1)[:1]
        values.extend((joints, opening))
    values = np.concatenate(values)
    if not np.isfinite(values).all():
        raise ValueError(f"the observed joint state must be finite, got {values}")
    rows, _mask = dual_arm_rows(values[None], ARX_X5_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSED)
    return rows[0]


def flange_transform(row: np.ndarray, side: str) -> np.ndarray:
    """Return the transform [4, 4] of one arm's flange in its base frame from a robot row [80], in the axes of the simulator's ``link6``."""
    transform = np.eye(4)
    rotation = rot6d_to_rotation(row[ROBOT_FIELDS[f"{side}_eef_rotation"]])
    transform[:3, :3] = rotation @ FLANGE_AXES.T
    transform[:3, 3] = row[ROBOT_FIELDS[f"{side}_eef_position"]]
    return transform


def gripper_opening(row: np.ndarray, side: str) -> np.ndarray:
    """Return the gripper opening [1] the environment takes from the closedness of a robot row [80]."""
    closedness = np.clip(float(row[ROBOT_FIELDS[f"{side}_gripper"]][0]), 0.0, 1.0)
    return np.array([GRIPPER_OPEN + closedness * (GRIPPER_CLOSED - GRIPPER_OPEN)], dtype=np.float32)


def actions_from_rows(rows: np.ndarray, action_type: str, world_from_base: dict) -> list:
    """Return one action dict per robot row.

    Args:
        rows: The rows the robot runs, [n, 80] in the dataset's units.
        action_type: ``joint`` for ``*_arm_joint_state`` targets, ``ee`` for ``*_ee_pose`` targets
            [x, y, z, qw, qx, qy, qz] of the flange in the environment's frame.
        world_from_base: The transform of each arm's base from ``root_transforms``; read for ``ee`` only.

    Returns:
        The actions: per arm the joint or pose target and ``*_ee_joint_state``, the gripper opening.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(rows).all():
        raise ValueError("the generated actions must be finite")
    actions = []
    for row in rows:
        action = {}
        for side in SIDES:
            if action_type == ACTION_TYPE_JOINT:
                joints = row[ROBOT_FIELDS[f"{side}_joints"]][:ARM_JOINTS]
                action[f"{side}_arm_joint_state"] = joints.astype(np.float32)
            else:
                pose = transform_to_pose(world_from_base[side] @ flange_transform(row, side))
                if not np.isfinite(pose).all():
                    raise ValueError(f"the {side} end-effector pose must be finite: its rotation is degenerate")
                action[f"{side}_ee_pose"] = pose.astype(np.float32)
            action[f"{side}_ee_joint_state"] = gripper_opening(row, side)
        actions.append(action)
    return actions


def flange_pose_errors(state: dict, row: np.ndarray, world_from_base: dict) -> dict:
    """Return per arm the (position error in metres, rotation error in degrees) between the ``*_ee_pose`` an observation reports and the pose the robot row [80] of the same observation holds; an arm without a reported pose is left out."""
    errors = {}
    for side in SIDES:
        observed = state.get(f"{side}_ee_pose")
        if observed is None:
            continue
        observed = pose_to_transform(observed)
        expected = world_from_base[side] @ flange_transform(row, side)
        cosine = (np.trace(observed[:3, :3].T @ expected[:3, :3]) - 1.0) / 2.0
        errors[side] = (
            float(np.linalg.norm(observed[:3, 3] - expected[:3, 3])),
            float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
        )
    return errors
