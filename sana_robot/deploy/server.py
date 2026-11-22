# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

import gc
import socket
import traceback
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pyrallis
import torch

from sana_robot.data.format.robot80 import GRIPPER_FIELDS, ROBOT_DIM, ROBOT_FIELDS, field_slots, gripper_closedness
from sana_robot.data.storage.lerobot import LeRobotStorage
from sana_robot.deploy.async_exec.runner import ChunkPlan, ChunkRunner, run_episode
from sana_robot.deploy.client import receive_message, send_message
from sana_robot.deploy.runtime.deployment import DeployConfig, Deployment


@dataclass
class ServerConfig(DeployConfig):
    """A policy server: a deployment that the control loop of a robot drives over TCP.

    Args:
        host: The address to listen on; ``0.0.0.0`` accepts a client on another machine.
        port: The port to listen on.
        instructions: The tasks to prepare before the first client connects; any other task is prepared when its
            first episode starts.
        warmup_steps: The control steps of the episode of black frames a task is prepared with, in which the CUDA
            graphs are captured.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    instructions: List[str] = field(default_factory=list)
    warmup_steps: int = 128


def field_layout(mask: np.ndarray) -> list:
    """Return, for a slot mask [80], the layout of a vector that holds the robot fields in use one after the other: per field its name, its slots in the robot row, its positions in the vector and no gripper range."""
    (layout, position) = ([], 0)
    for name, slots in ROBOT_FIELDS.items():
        used = (np.flatnonzero(mask[slots]) + slots.start).tolist()
        if used:
            layout.append((name, used, list(range(position, position + len(used))), None))
            position += len(used)
    return layout


def column_layout(fields: dict) -> list:
    """Return the layout of a vector that is one column of a LeRobot dataset, from the column's field mapping: per field its name, its slots in the robot row, its positions in the column and, for a gripper stored as a reading, the readings of the open and the closed gripper."""
    layout = []
    for name, source in fields.items():
        positions = [int(index) for index in source["index"]]
        gripper = (source["open"], source["closed"]) if name in GRIPPER_FIELDS and "open" in source else None
        layout.append((name, field_slots(name, len(positions)), positions, gripper))
    return sorted(layout, key=lambda entry: entry[2][0])


class RobotAdapter:
    """What the server converts for the robot of its client: the state vector into the robot row the policy reads, and the action rows the policy writes into commands.

    The policy works on robot rows of 80 slots, filled as its training dataset fills them
    (``sana_robot.data.format.robot80``). For a LeRobot dataset whose mapping reads the state from one column and
    the action from one column, the client speaks those columns: a state vector is laid out as a row of the state
    column and a command as a row of the action column, gripper readings included. For any other dataset the
    client speaks the dataset's own values: a state vector holds the fields of ``state_fields`` one after the
    other, and a command those of ``command_fields``. Rewrite the two methods for a robot that reports or takes
    something else, e.g. joint angles alone for a dataset whose rows also hold the end-effector pose
    (``sana_robot.data.utils.kinematics.dual_arm_rows``), or joint targets for a policy that writes end-effector
    poses.

    Args:
        storage: The reader of the policy's training dataset.
    """

    def __init__(self, storage) -> None:
        info = storage.info
        (self.state_layout, self.command_layout) = (field_layout(info.state_mask), field_layout(info.action_mask))
        if isinstance(storage, LeRobotStorage):
            (state, action) = (storage.mapping["state"], storage.mapping["action"])
            if all(len({source["column"] for source in fields.values()}) == 1 for fields in (state, action)):
                (self.state_layout, self.command_layout) = (column_layout(state), column_layout(action))
        self.state_fields = [[name, len(positions)] for (name, _slots, positions, _gripper) in self.state_layout]
        self.command_fields = [[name, len(positions)] for (name, _slots, positions, _gripper) in self.command_layout]
        self.command_size = 1 + max(position for entry in self.command_layout for position in entry[2])

    def state_row(self, state: np.ndarray) -> np.ndarray:
        """Return the robot row [80] of a state vector of the client, in the dataset's units."""
        state = np.asarray(state, dtype=np.float64)
        row = np.zeros(ROBOT_DIM, dtype=np.float32)
        for _name, slots, positions, gripper in self.state_layout:
            values = state[positions]
            row[slots] = values if gripper is None else gripper_closedness(values, *gripper)
        return row

    def commands(self, rows: np.ndarray) -> np.ndarray:
        """Return the commands [n, K] of action rows [n, 80], one per control step."""
        commands = np.zeros((len(rows), self.command_size), dtype=np.float32)
        for _name, slots, positions, gripper in self.command_layout:
            values = rows[:, slots]
            if gripper is not None:
                (open_value, closed_value) = gripper
                values = open_value + np.clip(values, 0.0, 1.0) * (closed_value - open_value)
            commands[:, positions] = values
        return commands


class PolicyServer:
    """The methods a policy server answers: ``describe``, ``start``, ``push_frame`` and ``next_plan``, the last three those of ``ChunkRunner``.

    Every task keeps its own runner, so an episode of a task that ran before starts without capturing CUDA graphs.

    Args:
        deployment: The deployment to serve.
        warmup_steps: The control steps of the episode of black frames a new task is prepared with.
    """

    def __init__(self, deployment: Deployment, warmup_steps: int) -> None:
        self.deployment = deployment
        self.warmup_steps = int(warmup_steps)
        self.adapter = RobotAdapter(deployment.storage)
        self.runners = {}
        self.runner = None

    def description(self) -> dict:
        """Return what the client has to know of the policy: the control frequency, the control steps between two frames, the size the policy reads every camera at and how a frame is brought to it, the delay of asynchronous execution, and the fields of a state vector and of a command."""
        (config, info) = (self.deployment.config, self.deployment.storage.info)
        return {
            "fps": float(info.fps),
            "video_frame_stride": int(config.video_frame_stride),
            "cameras": {view.name: list(size) for (view, size) in zip(info.views, config.view_sizes)},
            "resize": info.resize,
            "async_delay_steps": int(config.async_delay_steps),
            "state_fields": self.adapter.state_fields,
            "command_fields": self.adapter.command_fields,
        }

    def runner_of(self, instruction: str) -> ChunkRunner:
        """Return the runner of a task. A new task first runs an episode of black frames; the objects it leaves are then kept out of the garbage collector, whose full collections would stall the control loop."""
        if instruction not in self.runners:
            config = self.deployment.config
            runner = self.deployment.runner(instruction)
            images = [np.zeros((*size, 3), dtype=np.uint8) for size in config.view_sizes]
            state_row = np.zeros(ROBOT_DIM, dtype=np.float32)
            fps = self.deployment.storage.info.fps
            run_episode(runner, self.warmup_steps, lambda step: images, lambda step: state_row, fps, paced=False)
            runner.wait()
            self.runners[instruction] = runner
            gc.collect()
            gc.freeze()
        return self.runners[instruction]

    def observation(self, arrays: dict) -> tuple:
        """Return the frame of every camera, in the dataset's camera order, and the robot row [80] of a request's arrays."""
        images = [arrays[f"image/{view.name}"] for view in self.deployment.storage.info.views]
        return (images, self.adapter.state_row(arrays["state"]))

    def plan_reply(self, plan: ChunkPlan) -> tuple:
        """Return the result and the arrays of the reply that carries a plan: the chunk's index, the milliseconds its generation took and its commands."""
        step = plan.index * self.runner.controller.n_action_steps
        timings = self.runner.timings
        generate = [timing["ms"] for timing in timings if timing["event"] == "generate" and timing["step"] == step]
        return ({"chunk": plan.index, "generate_ms": generate[-1]}, {"commands": self.adapter.commands(plan.targets)})

    def handle(self, method: str, arguments: dict, arrays: dict) -> tuple:
        """Answer one request.

        Args:
            method: ``describe``, ``start``, ``push_frame`` or ``next_plan``.
            arguments: The request's arguments: ``instruction`` for ``start``.
            arrays: The request's arrays: the camera frames ``image/<camera>``, at the size the policy reads
                them at, and ``state`` for ``start`` and ``push_frame``.

        Returns:
            The result and the arrays of the reply.
        """
        if method == "describe":
            return (self.description(), {})
        if method == "push_frame":
            self.runner.push_frame(*self.observation(arrays))
            return ({}, {})
        if method == "next_plan":
            return self.plan_reply(self.runner.next_plan())
        if self.runner is not None:
            self.runner.wait()
        self.runner = self.runner_of(arguments["instruction"])
        self.runner.timings.clear()
        return self.plan_reply(self.runner.start(*self.observation(arrays)))


def serve(connection: socket.socket, server: PolicyServer) -> None:
    """Answer the requests of one client until it closes the connection; a request that fails is answered with its error."""
    while True:
        try:
            (request, arrays) = receive_message(connection)
            try:
                (result, reply_arrays) = server.handle(request["method"], request["arguments"], arrays)
                reply = {"result": result}
            except Exception:
                (reply, reply_arrays) = ({"error": traceback.format_exc()}, {})
                print(reply["error"], flush=True)
            send_message(connection, reply, reply_arrays)
        except ConnectionError:
            return


@pyrallis.wrap()
def main(config: ServerConfig) -> None:
    """Serve a deployment to the control loop of a robot, one client at a time: ``python -m sana_robot.deploy.server --config_path <deploy config> --host 0.0.0.0 --port 8000``, where the config file holds the fields of ``DeployConfig``. The server's work on the CPU, the resizing of the camera frames, runs on one thread: a pool of threads stalls it when the machine is busy."""
    server = PolicyServer(Deployment(config), config.warmup_steps)
    torch.set_num_threads(1)
    for instruction in config.instructions:
        server.runner_of(instruction)
    with socket.create_server((config.host, config.port)) as listener:
        print(f"SANA-Robot policy server on {config.host}:{config.port}", flush=True)
        while True:
            (connection, address) = listener.accept()
            with connection:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"client {address[0]} connected", flush=True)
                serve(connection, server)
                print(f"client {address[0]} left", flush=True)


if __name__ == "__main__":
    main()
