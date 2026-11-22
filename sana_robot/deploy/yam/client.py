# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import threading
import time
from multiprocessing import resource_tracker, shared_memory

import lcm
import numpy as np

from sana_robot.deploy.client import PolicyClient, Robot, episode_summary, run_episode

STATE_CHANNEL = "SANA_ROBOT_YAM_STATE"
COMMAND_CHANNEL = "SANA_ROBOT_YAM_COMMAND"
FRAME_SIZE = (480, 640)
PUBLISH_HZ = 100.0
START_SECONDS = 2.0


def block_name(camera: str) -> str:
    """Return the name of the shared-memory block that holds the newest frame of a camera."""
    return f"sana_robot_yam_{camera}"


class YamRobot(Robot):
    """Two YAM arms and their cameras, behind the follower and the camera bridge of this directory.

    A state and a command are 14 values: the six joints of the left arm, the six of the right arm, then the
    opening of the left and of the right gripper, 1 for open. The frames come from the shared memory the camera
    bridge writes. Commands go to the follower at ``PUBLISH_HZ``: the command of the running control step moves
    linearly to the following one over the step, and is held when the plan ends with it or the next step is late.
    Until the first command the arms are held where the follower reports them, for at least the
    ``START_SECONDS`` the follower takes to reach its first target.

    Args:
        cameras: The names of the policy's cameras.
        fps: The control frequency.
    """

    def __init__(self, cameras, fps: float) -> None:
        self.period = 1.0 / float(fps)
        self.blocks = [shared_memory.SharedMemory(name=block_name(camera)) for camera in cameras]
        for block in self.blocks:
            resource_tracker.unregister(block._name, "shared_memory")
        self.frames = {
            camera: np.ndarray((*FRAME_SIZE, 3), dtype=np.uint8, buffer=block.buf)
            for (camera, block) in zip(cameras, self.blocks)
        }
        self.lock = threading.Lock()
        self.state = None
        (self.listener, self.publisher) = (lcm.LCM(), lcm.LCM())
        self.listener.subscribe(STATE_CHANNEL, self.keep_state)
        threading.Thread(target=self.listen, daemon=True).start()
        while self.state is None:
            time.sleep(0.01)
        (self.command, self.following, self.sent) = (self.state, None, time.perf_counter())
        threading.Thread(target=self.publish, daemon=True).start()
        time.sleep(START_SECONDS + 1.0)

    def keep_state(self, channel: str, data: bytes) -> None:
        """Keep the newest state the follower published."""
        with self.lock:
            self.state = np.frombuffer(data, dtype=np.float64).copy()

    def listen(self) -> None:
        """Receive the follower's states."""
        while True:
            self.listener.handle_timeout(100)

    def publish(self) -> None:
        """Send the follower its target at ``PUBLISH_HZ``."""
        while True:
            with self.lock:
                (command, following, sent) = (self.command, self.following, self.sent)
            target = command
            if following is not None:
                target = command + (following - command) * min((time.perf_counter() - sent) / self.period, 1.0)
            self.publisher.publish(COMMAND_CHANNEL, target.tobytes())
            time.sleep(1.0 / PUBLISH_HZ)

    def read(self) -> tuple[dict, np.ndarray]:
        """Return the newest frame of every camera and the newest state of the arms."""
        with self.lock:
            state = self.state
        return ({camera: frame.copy() for (camera, frame) in self.frames.items()}, state)

    def send(self, command: np.ndarray, following: np.ndarray | None) -> None:
        """Make ``command`` the target of the running control step and ``following`` the one it moves to."""
        command = np.asarray(command, dtype=np.float64)
        following = None if following is None else np.asarray(following, dtype=np.float64)
        with self.lock:
            (self.command, self.following, self.sent) = (command, following, time.perf_counter())


def main() -> None:
    """Run the control loop of two YAM arms against a policy server: ``python -m sana_robot.deploy.yam.client --host <server> --port 8000 --instruction "stack the three cups"``, with the camera bridge and the follower running."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--control_steps", type=int, default=256)
    arguments = parser.parse_args()
    policy = PolicyClient(arguments.host, arguments.port)
    robot = YamRobot(list(policy.description["cameras"]), policy.description["fps"])
    result = run_episode(policy, robot, arguments.instruction, arguments.control_steps)
    policy.close()
    print(episode_summary(result, arguments.control_steps), flush=True)


if __name__ == "__main__":
    main()
