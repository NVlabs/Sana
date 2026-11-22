# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

import json
from multiprocessing.connection import Client

import numpy as np

CAMERAS = ("head_camera", "left_camera", "right_camera")
SIDES = ("left", "right")
BASE_LINKS = {"left": "fl_base_link", "right": "fr_base_link"}
TOOL_OFFSET = 0.12


class PolicyClient:
    """The RoboTwin side of the SANA-Robot-AR policy server (``eval/robotwin/server.py``).

    Args:
        port: The local port of the server.
    """

    def __init__(self, port: int) -> None:
        self.connection = Client(("localhost", port))

    def call(self, message: dict, images=()) -> dict:
        """Send a message, followed by camera frames uint8 [H, W, 3], and return the server's answer."""
        frames = [np.ascontiguousarray(image, dtype=np.uint8) for image in images]
        message = dict(message, image_shapes=[frame.shape for frame in frames])
        self.connection.send_bytes(json.dumps(message).encode())
        for frame in frames:
            self.connection.send_bytes(frame.tobytes())
        return json.loads(self.connection.recv_bytes())


def camera_images(observation) -> list:
    """Return the frames of the head camera and of the two wrist cameras of a RoboTwin observation."""
    return [observation["observation"][camera]["rgb"] for camera in CAMERAS]


def joint_readings(observation) -> list:
    """Return the joint readings [14] of a RoboTwin observation: per arm six joint angles and the gripper opening, the left arm first."""
    joints = observation["joint_action"]
    readings = [*joints["left_arm"], joints["left_gripper"], *joints["right_arm"], joints["right_gripper"]]
    return [float(value) for value in readings]


def arm_frames(robot) -> tuple[list, list]:
    """Return, per arm of the running robot, the [4, 4] pose of its base in the world and of the tool frame of RoboTwin's end-effector commands in its last link."""
    (world_from_base, flange_from_tool) = ([], [])
    for side in SIDES:
        base = getattr(robot, f"{side}_entity").find_link_by_name(BASE_LINKS[side])
        tool = np.eye(4)
        tool[:3, :3] = np.asarray(getattr(robot, f"{side}_global_trans_matrix")) @ np.asarray(
            getattr(robot, f"{side}_delta_matrix")
        )
        tool[:3, 3] = tool[:3, :3] @ np.array([getattr(robot, f"{side}_gripper_bias") - TOOL_OFFSET, 0.0, 0.0])
        joint = getattr(robot, f"{side}_ee").pose_in_child.to_transformation_matrix()
        world_from_base.append(np.asarray(base.get_pose().to_transformation_matrix()).tolist())
        flange_from_tool.append((np.asarray(joint) @ tool).tolist())
    return world_from_base, flange_from_tool


def get_model(arguments) -> PolicyClient:
    """RoboTwin hook: connect to the policy server on the ``port`` of the evaluation's arguments."""
    return PolicyClient(int(arguments["port"]))


def reset_model(model: PolicyClient) -> None:
    """RoboTwin hook: start a new episode."""
    model.call({"type": "reset"})


def eval(task_env, model: PolicyClient, observation) -> None:
    """RoboTwin hook: ask the server for the next chunk and run it, one end-effector command per control step; the server gets the camera frames again every few steps."""
    (world_from_base, flange_from_tool) = arm_frames(task_env.robot)
    request = {
        "type": "plan",
        "instruction": task_env.get_instruction(),
        "joints": joint_readings(observation),
        "world_from_base": world_from_base,
        "flange_from_tool": flange_from_tool,
    }
    answer = model.call(request, camera_images(observation))
    commands = np.asarray(answer["commands"], dtype=np.float32)
    for step, command in enumerate(commands, start=1):
        task_env.take_action(command, action_type="ee")
        if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
            return
        if step % answer["frame_stride"] == 0 and step < len(commands):
            model.call({"type": "observe"}, camera_images(task_env.get_obs()))
