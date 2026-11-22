# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import socket
import struct
import time
from dataclasses import dataclass

import cv2
import numpy as np

MAGIC = b"SANAROBT"
PREFIX = struct.Struct("!8sIQ")


def send_message(connection: socket.socket, header: dict, arrays: dict | None = None) -> None:
    """Send a header and named NumPy arrays as one message."""
    arrays = {name: np.ascontiguousarray(array) for (name, array) in (arrays or {}).items()}
    layout = [{"name": name, "dtype": array.dtype.str, "shape": list(array.shape)} for (name, array) in arrays.items()]
    head = json.dumps({**header, "arrays": layout}).encode()
    connection.sendall(PREFIX.pack(MAGIC, len(head), sum(array.nbytes for array in arrays.values())) + head)
    for array in arrays.values():
        connection.sendall(array.data)


def receive_bytes(connection: socket.socket, size: int) -> bytearray:
    """Return the next ``size`` bytes of a connection."""
    data = bytearray(size)
    view = memoryview(data)
    received = 0
    while received < size:
        count = connection.recv_into(view[received:])
        if count == 0:
            raise ConnectionError("the connection was closed")
        received += count
    return data


def receive_message(connection: socket.socket) -> tuple[dict, dict]:
    """Return the header and the named NumPy arrays of the next message of a connection."""
    (magic, head_size, arrays_size) = PREFIX.unpack(receive_bytes(connection, PREFIX.size))
    if magic != MAGIC:
        raise ConnectionError("the peer does not speak the policy server's protocol")
    header = json.loads(receive_bytes(connection, head_size))
    data = receive_bytes(connection, arrays_size)
    (arrays, offset) = ({}, 0)
    for entry in header.pop("arrays"):
        dtype = np.dtype(entry["dtype"])
        count = int(np.prod(entry["shape"]))
        arrays[entry["name"]] = np.frombuffer(data, dtype, count, offset).reshape(entry["shape"])
        offset += count * dtype.itemsize
    return (header, arrays)


def resize_frame(frame: np.ndarray, size, resize: str) -> np.ndarray:
    """Return a uint8 frame [H, W, 3] at ``size`` = (h, w) as the policy's training frames were resized: by bilinear interpolation of float values, stretched to the size (``stretch``), or scaled until it covers the size and cut at the centre (``crop``)."""
    (height, width) = frame.shape[:2]
    (target_height, target_width) = (int(side) for side in size)
    if (height, width) == (target_height, target_width):
        return frame
    if resize == "stretch":
        (scaled_height, scaled_width) = (target_height, target_width)
    elif target_height / height > target_width / width:
        (scaled_height, scaled_width) = (target_height, round(width * (target_height / height)))
    else:
        (scaled_height, scaled_width) = (round(height * (target_width / width)), target_width)
    scaled = cv2.resize(frame.astype(np.float32), (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
    first_row = int(round(scaled_height - target_height) / 2.0)
    first_column = int(round(scaled_width - target_width) / 2.0)
    cut = scaled[first_row : first_row + target_height, first_column : first_column + target_width]
    return np.rint(cut).astype(np.uint8)


@dataclass
class Plan:
    """What the robot runs of one generated chunk: the commands [n, K], one per control step, the chunk's index in the episode and the milliseconds the server took to generate it."""

    commands: np.ndarray
    chunk: int
    generate_ms: float


class PolicyClient:
    """The methods of a policy server, called over one TCP connection.

    ``start`` opens an episode and returns the plan of its first chunk. From then on the control loop hands over
    a frame every ``video_frame_stride`` control steps with ``push_frame`` and takes the next plan with
    ``next_plan`` when the running one is used up. The server generates the chunks on its own schedule, so the
    loop is the same for synchronous and asynchronous execution.
    """

    def __init__(self, host: str, port: int, timeout: float = 600.0) -> None:
        """Connect to a policy server.

        Args:
            host: The server's address.
            port: The server's port.
            timeout: The seconds to wait for a reply before the call fails; the first episode of a task the
                server has not prepared takes the longest.
        """
        self.connection = socket.create_connection((host, port), timeout=timeout)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.description = self.describe()

    def call(self, method: str, arguments: dict | None = None, arrays: dict | None = None) -> tuple[dict, dict]:
        """Call a method of the server and return the result and the arrays of its reply."""
        send_message(self.connection, {"method": method, "arguments": arguments or {}}, arrays)
        (reply, reply_arrays) = receive_message(self.connection)
        if "error" in reply:
            raise RuntimeError(f"the policy server failed in {method}:\n{reply['error']}")
        return (reply["result"], reply_arrays)

    def describe(self) -> dict:
        """Return what the policy expects of the robot.

        Returns:
            Dict with ``fps``, the control frequency; ``video_frame_stride``, the control steps between two frames
            the policy reads; ``cameras``, the (height, width) the policy reads every camera at, by camera name;
            ``resize``, how a frame is brought to that size (``stretch`` or ``crop``); ``async_delay_steps``, the
            control steps the robot runs while the next chunk is generated, 0 for synchronous execution;
            ``state_fields`` and ``command_fields``, the values of a state vector and of a command in order, as
            [field, number of values] pairs.
        """
        return self.call("describe")[0]

    def observation(self, images: dict, state) -> dict:
        """Return the arrays of an observation as a request carries them: the frame of every camera at the size the policy reads it at, under ``image/<camera>``, and the state under ``state``."""
        (sizes, resize) = (self.description["cameras"], self.description["resize"])
        arrays = {
            f"image/{camera}": resize_frame(np.asarray(image, dtype=np.uint8), sizes[camera], resize)
            for (camera, image) in images.items()
        }
        arrays["state"] = np.asarray(state, dtype=np.float64)
        return arrays

    def start(self, instruction: str, images: dict, state) -> Plan:
        """Open an episode and return the plan of its first chunk.

        Args:
            instruction: The task.
            images: The first frame of every camera, ``{camera: uint8 [H, W, 3]}`` in RGB, at any resolution.
            state: The robot's state vector read with them.

        Returns:
            The plan of chunk 0.
        """
        (result, arrays) = self.call("start", {"instruction": instruction}, self.observation(images, state))
        return Plan(arrays["commands"], result["chunk"], result["generate_ms"])

    def push_frame(self, images: dict, state) -> None:
        """Hand over the next frame of every camera, taken ``video_frame_stride`` control steps after the one before, and the state vector read with it."""
        self.call("push_frame", arrays=self.observation(images, state))

    def next_plan(self) -> Plan:
        """Return the plan that takes over now, waiting for it when it is late."""
        (result, arrays) = self.call("next_plan")
        return Plan(arrays["commands"], result["chunk"], result["generate_ms"])

    def close(self) -> None:
        """Close the connection."""
        self.connection.close()


class Robot:
    """The robot of the control loop: fill in the two methods with your cameras, your state reading and your controller.

    ``PolicyClient.description`` tells what the policy expects: the names of the cameras, and what a state vector
    and a command hold. The frames may have any resolution: the client sends them at the policy's.
    """

    def read(self) -> tuple[dict, np.ndarray]:
        """Return the newest frame of every camera, ``{camera: uint8 [H, W, 3]}`` in RGB, and the robot's state vector read with them."""
        raise NotImplementedError("read your cameras and your robot's state here")

    def send(self, command: np.ndarray, following: np.ndarray | None) -> None:
        """Send the command of one control step to the robot. ``following`` is the command of the next step, for a controller that interpolates between the two; None when the running plan ends with this command."""
        raise NotImplementedError("send the command to your robot here")


def run_episode(policy: PolicyClient, robot: Robot, instruction: str, control_steps: int, paced: bool = True) -> dict:
    """Run one episode as the robot's control loop.

    Every control step sends one command. An observation is read every ``video_frame_stride`` steps and handed to
    the server after the command of its step has gone out, so that resizing and sending it do not delay the
    command; in synchronous execution the next plan is generated from that observation, which then goes first.

    Args:
        policy: The client of the policy server.
        robot: The robot.
        instruction: The task.
        control_steps: The control steps to run.
        paced: Run the steps at the control frequency; otherwise as fast as the plans come.

    Returns:
        Dict with ``generate_ms``, the time the server took for every chunk; ``late_ms``, how long the robot waited
        for every plan after chunk 0; and ``seconds``, the time of the loop.
    """
    (fps, stride) = (float(policy.description["fps"]), int(policy.description["video_frame_stride"]))
    synchronous = int(policy.description["async_delay_steps"]) == 0
    plan = policy.start(instruction, *robot.read())
    queue = list(plan.commands)
    (generate, late, observation) = ([plan.generate_ms], [], None)
    clock = started = time.perf_counter()
    for step in range(int(control_steps)):
        if not queue:
            if synchronous and observation is not None:
                policy.push_frame(*observation)
                observation = None
            waited = time.perf_counter()
            plan = policy.next_plan()
            stall = time.perf_counter() - waited
            late.append(stall * 1000)
            generate.append(plan.generate_ms)
            clock += stall
            queue = list(plan.commands)
        command = queue.pop(0)
        robot.send(command, queue[0] if queue else None)
        if observation is not None:
            policy.push_frame(*observation)
            observation = None
        if paced:
            time.sleep(max(0.0, clock + (step + 1) / fps - time.perf_counter()))
        if (step + 1) % stride == 0 and step + 1 < int(control_steps):
            observation = robot.read()
    return {"generate_ms": generate, "late_ms": late, "seconds": time.perf_counter() - started}


def episode_summary(result: dict, control_steps: int) -> str:
    """Return one line on an episode of ``run_episode``: its length, the chunks and how long the robot waited for them."""
    (generate, late) = (result["generate_ms"], result["late_ms"])
    return (
        f"{control_steps} control steps in {result['seconds']:.1f} s, {len(generate)} chunks, "
        f"{float(np.median(generate)):.1f} ms per chunk (median); "
        f"waited for a chunk {sum(wait > 1.0 for wait in late)} times, {max(late, default=0.0):.1f} ms at most"
    )


def main() -> None:
    """Run the control loop of ``Robot`` against a policy server: ``python client.py --host <server> --port 8000 --instruction "put the red block into the box"``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--control_steps", type=int, default=256)
    arguments = parser.parse_args()
    policy = PolicyClient(arguments.host, arguments.port)
    result = run_episode(policy, Robot(), arguments.instruction, arguments.control_steps)
    policy.close()
    print(episode_summary(result, arguments.control_steps), flush=True)


if __name__ == "__main__":
    main()
