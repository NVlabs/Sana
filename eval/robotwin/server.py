# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from multiprocessing.connection import Listener

import numpy as np
import pyrallis
from scipy.spatial.transform import Rotation

from eval.policy import EpisodePolicy
from sana_robot.data.format.robot80 import ROBOT_FIELDS, rot6d_to_rotation
from sana_robot.data.utils.convert_robotwin import ALOHA_AGILEX_JOINTS, GRIPPER_CLOSED, GRIPPER_OPEN
from sana_robot.data.utils.kinematics import FLANGE_AXES, dual_arm_rows
from sana_robot.deploy.runtime.deployment import DeployConfig, Deployment

SIDES = ("left", "right")


@dataclass
class RobotwinServerConfig(DeployConfig):
    """The policy server of the RoboTwin 2.0 evaluation; the other fields are a deployment's.

    Args:
        port: The local port the server listens on.
    """

    port: int = 8000


def state_row(joints) -> np.ndarray:
    """Return the robot state row [80] of the simulator's joint readings [14]: per arm six joint angles and the gripper opening in [0, 1], the left arm first."""
    readings = np.asarray(joints, dtype=np.float64)[None]
    return dual_arm_rows(readings, ALOHA_AGILEX_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSED)[0][0]


def end_effector_commands(actions: np.ndarray, world_from_base: np.ndarray, flange_from_tool: np.ndarray) -> np.ndarray:
    """Return the simulator's end-effector commands [T, 16] of action rows [T, 80]: per arm the position and the quaternion (w, x, y, z) of its tool frame in the world and the gripper opening in [0, 1], the left arm first. ``world_from_base`` and ``flange_from_tool`` hold the [4, 4] pose of each arm's base in the world and of its tool frame in its last link."""
    actions = np.asarray(actions, dtype=np.float64)
    commands = np.empty((len(actions), 16), dtype=np.float32)
    for index, side in enumerate(SIDES):
        flange = np.tile(np.eye(4), (len(actions), 1, 1))
        flange[:, :3, 3] = actions[:, ROBOT_FIELDS[f"{side}_eef_position"]]
        flange[:, :3, :3] = rot6d_to_rotation(actions[:, ROBOT_FIELDS[f"{side}_eef_rotation"]]) @ FLANGE_AXES.T
        tool = world_from_base[index] @ flange @ flange_from_tool[index]
        quaternion = Rotation.from_matrix(tool[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        closedness = np.clip(actions[:, ROBOT_FIELDS[f"{side}_gripper"]].reshape(-1), 0.0, 1.0)
        commands[:, 8 * index : 8 * index + 8] = np.column_stack((tool[:, :3, 3], quaternion, 1.0 - closedness))
    return commands


def receive(connection) -> tuple[dict, list]:
    """Return the next message of the simulator: its fields, and the camera frames uint8 [H, W, 3] that follow it."""
    message = json.loads(connection.recv_bytes())
    shapes = message.get("image_shapes", [])
    return message, [np.frombuffer(connection.recv_bytes(), dtype=np.uint8).reshape(shape).copy() for shape in shapes]


def serve(connection, policy: EpisodePolicy) -> None:
    """Answer one simulator process until it closes the connection.

    The simulator sends ``reset`` before an episode, ``plan`` with the frames, the joint readings, the instruction
    and the frames of its arms when it needs a chunk, and ``observe`` with the frames every ``video_frame_stride``
    control steps in between. A ``plan`` is answered with the commands of the next ``n_action_steps`` control
    steps; the noise of the k-th chunk of an episode has seed k.
    """
    (opened, requests) = (False, 0)
    while True:
        try:
            (message, images) = receive(connection)
        except EOFError:
            return
        reply = {}
        if message["type"] == "reset":
            (opened, requests) = (False, 0)
        elif message["type"] == "observe":
            policy.observe(images)
        else:
            if opened:
                policy.observe(images)
            else:
                policy.start(images, message["instruction"])
            actions = policy.plan(state_row(message["joints"]), message["instruction"], seed=requests)
            (opened, requests) = (True, requests + 1)
            frames = (np.asarray(message["world_from_base"]), np.asarray(message["flange_from_tool"]))
            reply = {"commands": end_effector_commands(actions, *frames).tolist(), "frame_stride": policy.frame_stride}
        connection.send_bytes(json.dumps(reply).encode())


@pyrallis.wrap()
def main(config: RobotwinServerConfig) -> None:
    """Serve SANA-Robot-AR to the RoboTwin 2.0 evaluation: ``python -m eval.robotwin.server --config_path eval/robotwin/deploy.yaml``."""
    policy = EpisodePolicy(Deployment(config))
    with Listener(("localhost", config.port)) as listener:
        print(f"SANA-Robot-AR serves RoboTwin on port {config.port}", flush=True)
        while True:
            with listener.accept() as connection:
                serve(connection, policy)


if __name__ == "__main__":
    main()
