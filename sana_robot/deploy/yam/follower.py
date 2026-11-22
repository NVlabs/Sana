# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import threading
import time

import lcm
import numpy as np
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType

from sana_robot.deploy.yam.client import COMMAND_CHANNEL, START_SECONDS, STATE_CHANNEL

ARM_JOINTS = 6
CONTROL_HZ = 300.0
STATE_HZ = 100.0
GRIPPER_KP = 1.0
GRIPPER_KD = 0.5
MAX_TIP_FORCE = 8.0
TORQUE_PER_TIP_FORCE = 0.096 / 6.57
FRICTION_TORQUE = 0.12
ERROR_DEADBAND = 0.02


def arm_state(arms) -> np.ndarray:
    """Return the state [14] of the two arms: the joints of the left arm, the joints of the right arm, the opening of the left and of the right gripper."""
    readings = [np.asarray(arm.get_joint_pos(), dtype=np.float64) for arm in arms]
    joints = [reading[:ARM_JOINTS] for reading in readings]
    return np.concatenate((*joints, *(reading[ARM_JOINTS:] for reading in readings)))


def arm_command(command: np.ndarray, side: int) -> tuple[np.ndarray, float]:
    """Return the joint targets [6] and the gripper opening of one arm from a command [14]."""
    joints = command[ARM_JOINTS * side : ARM_JOINTS * (side + 1)]
    return (joints, float(np.clip(command[2 * ARM_JOINTS + side], 0.0, 1.0)))


def move_slowly(arms, command: np.ndarray, seconds: float, publisher: lcm.LCM) -> None:
    """Bring both arms from where they are to a command in ``seconds``, by position control, publishing their state on the way."""
    starts = [np.asarray(arm.get_joint_pos(), dtype=np.float64) for arm in arms]
    goals = [np.append(*arm_command(command, side)) for side in range(len(arms))]
    steps = max(1, int(seconds * STATE_HZ))
    for step in range(1, steps + 1):
        for arm, start, goal in zip(arms, starts, goals):
            arm.command_joint_pos(start + (goal - start) * step / steps)
        publisher.publish(STATE_CHANNEL, arm_state(arms).tobytes())
        time.sleep(1.0 / STATE_HZ)


def command_arm(arm, joints: np.ndarray, opening: float) -> None:
    """Command one arm: its joints by the SDK's position control, its gripper by a torque.

    The gripper motor's own position control is switched off. Its torque is a proportional-derivative term on
    the gripper's motor angle, limited to what ``MAX_TIP_FORCE`` at the fingertips needs, plus
    ``FRICTION_TORQUE`` towards the target while the gripper is farther than ``ERROR_DEADBAND`` from it, which
    overcomes the static friction of the drive. The limit bounds the force on a held object.
    """
    gripper = arm._gripper_index
    with arm._state_lock:
        position = float(arm.remapper.to_robot_joint_pos_space(arm._joint_state.pos.copy())[gripper])
        velocity = float(arm.remapper.to_robot_joint_vel_space(arm._joint_state.vel.copy())[gripper])
    (closed, opened) = (float(arm._gripper_limits[0]), float(arm._gripper_limits[1]))
    error = closed + opening * (opened - closed) - position
    limit = MAX_TIP_FORCE * TORQUE_PER_TIP_FORCE
    torque = float(np.clip(GRIPPER_KP * error - GRIPPER_KD * velocity, -limit, limit))
    if abs(error) > ERROR_DEADBAND:
        torque += float(np.sign(error)) * FRICTION_TORQUE
    motors = len(arm.motor_chain)
    (target, torques, kp, kd) = (np.zeros(motors), np.zeros(motors), arm._kp.copy(), arm._kd.copy())
    target[:gripper] = np.clip(joints, arm._joint_limits[:, 0], arm._joint_limits[:, 1])
    target[gripper] = position
    torques[gripper] = torque
    (kp[gripper], kd[gripper]) = (0.0, 0.0)
    with arm._command_lock:
        arm._commands.torques = torques
        arm._commands.pos = target
        arm._commands.vel = np.zeros(motors)
        arm._commands.kp = kp
        arm._commands.kd = kd


def receive(connection: lcm.LCM) -> None:
    """Handle the messages of an LCM connection as they arrive."""
    while True:
        connection.handle_timeout(100)


def main() -> None:
    """Drive two YAM arms from the commands of the control loop and publish their state: ``python -m sana_robot.deploy.yam.follower --left_can can_left --right_can can_right``. The arms first move slowly to the first command they receive; ``--sim`` runs the SDK's simulated arms, whose grippers follow by position."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_can", default="can_left")
    parser.add_argument("--right_can", default="can_right")
    parser.add_argument("--arm", default="yam_ultra")
    parser.add_argument("--gripper", default="linear_4310")
    parser.add_argument("--sim", action="store_true")
    arguments = parser.parse_args()
    (arm_type, gripper_type) = (
        ArmType.from_string_name(arguments.arm),
        GripperType.from_string_name(arguments.gripper),
    )
    arms = [
        get_yam_robot(channel=channel, arm_type=arm_type, gripper_type=gripper_type, sim=arguments.sim)
        for channel in (arguments.left_can, arguments.right_can)
    ]
    latest = {}
    (listener, publisher) = (lcm.LCM(), lcm.LCM())
    listener.subscribe(COMMAND_CHANNEL, lambda channel, data: latest.update(command=np.frombuffer(data, np.float64)))
    threading.Thread(target=receive, args=(listener,), daemon=True).start()
    try:
        while "command" not in latest:
            publisher.publish(STATE_CHANNEL, arm_state(arms).tobytes())
            time.sleep(1.0 / STATE_HZ)
        move_slowly(arms, latest["command"], START_SECONDS, publisher)
        published = 0.0
        while True:
            started = time.perf_counter()
            command = latest["command"]
            for side, arm in enumerate(arms):
                (joints, opening) = arm_command(command, side)
                if arguments.sim:
                    arm.command_joint_pos(np.append(joints, opening))
                else:
                    command_arm(arm, joints, opening)
            if started - published >= 1.0 / STATE_HZ:
                publisher.publish(STATE_CHANNEL, arm_state(arms).tobytes())
                published = started
            time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - started)))
    finally:
        for arm in arms:
            arm.close()


if __name__ == "__main__":
    main()
