# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from sana_robot.deploy.io.robot import RobotInterface


@dataclass
class ChunkPlan:
    """One generated chunk: its index in the episode, its normalized actions [1, A, 80] and the action rows the robot runs of it, [n, 80] in the dataset's units."""

    index: int
    actions: torch.Tensor
    targets: np.ndarray


def executed_rows(index: int, async_delay_steps: int, n_action_steps: int) -> slice:
    """Return the rows of chunk ``index`` the robot runs: the first P + d of chunk 0, rows d to d + P of every later chunk."""
    if index == 0:
        return slice(0, n_action_steps + async_delay_steps)
    return slice(async_delay_steps, async_delay_steps + n_action_steps)


class ChunkRunner:
    """Drives a policy from a robot's control loop, the policy's work on one worker thread.

    A chunk is generated every P control steps. Chunk 0 is generated at step 0 while the robot waits. Chunk k is
    generated at step k * P from the frames and the state observed there, while the robot runs the next d rows
    (the delay) of the chunk before; it takes over at step k * P + d, so its first d rows are dropped. With d = 0
    the robot waits for every chunk: P = C is synchronous execution, P < C replans inside a chunk.

    The control loop calls ``start`` with the first frame and state, runs the rows of the plan it returns, hands
    over every ``frame_stride``-th frame with ``push_frame``, and takes the next plan with ``next_plan`` when the
    running one is used up. One runner serves episode after episode.
    """

    def __init__(self, controller, robot: RobotInterface) -> None:
        """Build a runner.

        Args:
            controller: ``RTCController`` (SANA-Robot-base) or ``MemoryRTCController`` (SANA-Robot-AR).
            robot: The robot's interface.
        """
        self.controller = controller
        self.robot = robot
        self.frames_per_chunk = controller.chunk_size // controller.frame_stride
        self.window = deque(maxlen=self.frames_per_chunk)
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.plans = {}
        self.pending = []
        self.step = 0
        self.generated = 0
        self.running = 0
        self.timings = []

    def start(self, images, state_row) -> ChunkPlan:
        """Open an episode and generate chunk 0, waiting for it.

        Args:
            images: The first frame of every camera, uint8 [H, W, 3].
            state_row: The robot row [80] read with it, in the dataset's units.

        Returns:
            The plan of chunk 0: its first P + d rows.
        """
        frames = self.robot.frames(images)
        self.window.clear()
        self.window.extend([frames] * self.frames_per_chunk)
        (self.step, self.generated, self.running, self.pending) = (0, 0, 0, [])
        self.plans = {0: self.worker.submit(self._generate, 0, 0, self._clips(), state_row, None)}
        return self.plans[0].result()

    def push_frame(self, images, state_row) -> None:
        """Hand over the next video frame of every camera, taken ``frame_stride`` control steps after the one before, with the robot row read with it. At a multiple of P the next chunk is generated from them."""
        self.window.append(self.robot.frames(images))
        self.step += self.controller.frame_stride
        step = self.step
        if step % self.controller.window_steps:
            return
        clips = self._clips()
        if step % self.controller.n_action_steps:
            self.pending.append(self.worker.submit(self._timed, "observe", step, self.controller.observe, step, clips))
            return
        self.generated += 1
        before = self.plans[self.generated - 1]
        self.plans[self.generated] = self.worker.submit(self._generate, self.generated, step, clips, state_row, before)

    def next_plan(self) -> ChunkPlan:
        """Return the plan that takes over now, waiting for it when it is late."""
        waited = time.perf_counter()
        self.running += 1
        plan = self.plans[self.running].result()
        self.timings.append(
            {"event": "next_plan", "chunk": self.running, "late_ms": (time.perf_counter() - waited) * 1000}
        )
        self.plans = {index: plan for (index, plan) in self.plans.items() if index >= self.running}
        for work in [work for work in self.pending if work.done()]:
            work.result()
            self.pending.remove(work)
        return plan

    def wait(self) -> None:
        """Wait for the queued work."""
        self.worker.submit(lambda: None).result()

    def finish(self) -> None:
        """Wait for the queued work and stop the worker."""
        self.worker.shutdown(wait=True)

    def _clips(self) -> list:
        """Return the last ``frames_per_chunk`` frames of every camera as one clip [3, T, h, w] per camera."""
        return [torch.cat(view, dim=1) for view in zip(*self.window)]

    def _timed(self, event: str, step: int, work, *arguments):
        started = time.perf_counter()
        result = work(*arguments)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timings.append({"event": event, "step": step, "ms": (time.perf_counter() - started) * 1000})
        return result

    def _generate(self, index: int, step: int, clips, state_row, before) -> ChunkPlan:
        controller = self.controller
        (state, state_mask) = self.robot.state(state_row)
        running = None if before is None else before.result().actions
        if index == 0:
            actions = self._timed("generate", step, controller.open, clips, state, state_mask, self.robot.action_mask)
        else:
            actions = self._timed(
                "generate", step, controller.generate, step, clips, state, state_mask, self.robot.action_mask, running
            )
        rows = executed_rows(index, controller.async_delay_steps, controller.n_action_steps)
        return ChunkPlan(index, actions, self.robot.actions(actions)[rows])


def run_episode(runner: ChunkRunner, control_steps: int, images, state_row, fps: float, paced: bool = True) -> dict:
    """Run one episode as the robot's control loop.

    Args:
        runner: The runner, between episodes.
        control_steps: The control steps to run.
        images: ``images(step)``, the frame of every camera at a control step.
        state_row: ``state_row(step)``, the robot row [80] read at a control step.
        fps: The control frequency.
        paced: Run the steps at the control frequency; otherwise as fast as the plans come.

    Returns:
        Dict with ``targets`` [control_steps, 80], the rows the robot ran; ``plans``, the chunk that ran every step;
        ``late_ms``, how long the robot waited for every plan after chunk 0; and ``seconds``, the time of the loop.
    """
    stride = runner.controller.frame_stride
    plan = runner.start(images(0), state_row(0))
    queue = list(plan.targets)
    (targets, plans, late) = ([], [], [])
    clock = started = time.perf_counter()
    for step in range(int(control_steps)):
        if not queue:
            waited = time.perf_counter()
            plan = runner.next_plan()
            stall = time.perf_counter() - waited
            late.append(stall * 1000)
            clock += stall
            queue = list(plan.targets)
        targets.append(queue.pop(0))
        plans.append(plan.index)
        if paced:
            time.sleep(max(0.0, clock + (step + 1) / float(fps) - time.perf_counter()))
        if (step + 1) % stride == 0 and step + 1 < int(control_steps):
            runner.push_frame(images(step + 1), state_row(step + 1))
    return {"targets": np.stack(targets), "plans": plans, "late_ms": late, "seconds": time.perf_counter() - started}
