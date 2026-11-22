# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from eval.policy import EpisodePolicy
from sana_robot.data.format.robot80 import ROBOT_FIELDS, rot6d_to_rotation
from sana_robot.data.utils.convert_libero import FRANKA_FLANGE, FRANKA_FLANGE_AXES, FRANKA_JOINTS, state_rows
from sana_robot.data.utils.kinematics import flange_poses

CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
SETTLE_COMMAND = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def camera_images(observation) -> list:
    """Return the frame of the external and of the wrist camera of a simulator observation, uint8 [H, W, 3], turned by 180 degrees as the training videos are."""
    return [np.ascontiguousarray(np.asarray(observation[camera])[::-1, ::-1]) for camera in CAMERAS]


def state_row(observation) -> np.ndarray:
    """Return the robot state row [80] of a simulator observation, from its joint angles and finger positions."""
    joints = np.asarray(observation["robot0_joint_pos"], dtype=np.float64)[None]
    fingers = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64)[None]
    return state_rows(joints, fingers)[0]


def robots_of(env) -> list:
    """Return the robots of a LIBERO environment, wrapped or not."""
    robots = getattr(env, "robots", None)
    return env.env.robots if robots is None else robots


def use_absolute_targets(env, absolute: bool) -> None:
    """Make the controller of the arm read its commands as absolute targets, or as changes of the current pose."""
    for robot in robots_of(env):
        robot.controller.use_delta = not absolute


def settle(env, observation, steps: int):
    """Let the scene come to rest for ``steps`` control steps with the gripper open, then switch the controller to absolute targets; returns the last observation."""
    use_absolute_targets(env, False)
    for _ in range(steps):
        (observation, _, _, _) = env.step(SETTLE_COMMAND)
    use_absolute_targets(env, True)
    return observation


def controller_frames(env) -> tuple[np.ndarray, np.ndarray]:
    """Return the pose [4, 4] of the arm's base in the world and the pose [4, 4] of the controller's grip site in the flange frame of the policy, both read from the running simulation."""
    robot = robots_of(env)[0]
    simulation = env.env.sim
    simulation.forward()
    (base, site) = (np.eye(4), np.eye(4))
    body = simulation.model.body_name2id(robot.robot_model.naming_prefix + "link0")
    base[:3, 3] = simulation.data.body_xpos[body]
    base[:3, :3] = simulation.data.body_xmat[body].reshape(3, 3)
    site[:3, 3] = simulation.data.site_xpos[robot.eef_site_id]
    site[:3, :3] = simulation.data.site_xmat[robot.eef_site_id].reshape(3, 3)
    joints = np.asarray(robot._joint_positions, dtype=np.float64)[None]
    flange = base @ flange_poses(FRANKA_JOINTS, joints, FRANKA_FLANGE, FRANKA_FLANGE_AXES)[0]
    return base, np.linalg.inv(flange) @ site


def controller_commands(actions: np.ndarray, world_from_base: np.ndarray, flange_from_site: np.ndarray) -> np.ndarray:
    """Return the controller commands [T, 7] of action rows [T, 80]: the absolute position and rotation vector of the grip site in the world, and -1 to open the gripper or 1 to close it."""
    actions = np.asarray(actions, dtype=np.float64)
    goals = np.tile(np.eye(4), (len(actions), 1, 1))
    goals[:, :3, 3] = actions[:, ROBOT_FIELDS["right_eef_position"]]
    goals[:, :3, :3] = rot6d_to_rotation(actions[:, ROBOT_FIELDS["right_eef_rotation"]])
    goals = world_from_base @ goals @ flange_from_site
    closedness = np.clip(actions[:, ROBOT_FIELDS["right_gripper"]].reshape(-1), 0.0, 1.0)
    gripper = np.where(2.0 * closedness - 1.0 < 0, -1.0, 1.0)
    commands = np.column_stack((goals[:, :3, 3], Rotation.from_matrix(goals[:, :3, :3]).as_rotvec(), gripper))
    return commands.astype(np.float32)


def run_episode(
    env, observation, policy: EpisodePolicy, instruction, max_steps: int, seed=None, replay=None, on_query=None
) -> dict:
    """Run one episode in a settled LIBERO environment whose controller reads absolute targets.

    Args:
        env: The environment.
        observation: Its current observation.
        policy: The policy.
        instruction: The task text, or a function that returns the text at the current step.
        max_steps: The control steps after which the episode counts as failed.
        seed: A function of the control step that returns the seed of the noise of the request made there.
        replay: A function of the control step that returns the commands [n, 7] to run instead of the policy's at
            that step, or None to ask the policy; while it returns commands the policy only observes.
        on_query: Called with (commands, instruction, replayed) at every step the policy is asked.

    Returns:
        Dict with ``success`` and ``policy_steps``, the control steps the episode took.
    """
    text = instruction() if callable(instruction) else instruction
    policy.start(camera_images(observation), text)
    (world_from_base, flange_from_site) = controller_frames(env)
    (success, step) = (False, 0)
    while step < max_steps and not success:
        text = instruction() if callable(instruction) else instruction
        commands = None if replay is None else replay(step)
        replayed = commands is not None
        actions = policy.plan(state_row(observation), text, not replayed, None if seed is None else seed(step))
        if not replayed:
            commands = controller_commands(actions, world_from_base, flange_from_site)
        commands = np.asarray(commands, dtype=np.float32)
        if on_query is not None:
            on_query(commands, text, replayed)
        for command in commands[: min(policy.n_action_steps, max_steps - step)]:
            (observation, _, done, _) = env.step(command.tolist())
            step += 1
            if step % policy.frame_stride == 0:
                policy.observe(camera_images(observation))
            if done:
                success = True
                break
    return {"success": success, "policy_steps": step}
