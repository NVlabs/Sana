# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import dataclass

import numpy as np
import pyrallis
import torch

from sana_robot.deploy.runtime.deployment import POLICY_AR, DeployConfig, Deployment


@dataclass
class BenchmarkConfig(DeployConfig):
    """A latency measurement of a deployment in synchronous execution.

    Args:
        episode: The recorded episode whose frames and states are fed.
        requests: The timed requests.
        warmup: The requests before them that are not timed.
    """

    episode: int = 0
    requests: int = 30
    warmup: int = 6


def timed(work) -> tuple:
    """Run ``work`` and return its result and its time in milliseconds, the GPU's queue empty before and after."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = work()
    torch.cuda.synchronize()
    return (result, (time.perf_counter() - started) * 1000)


def measure(deployment: Deployment, config: BenchmarkConfig) -> dict:
    """Feed a recorded episode chunk by chunk and time every stage of a request: video encoding, the memory commit (SANA-Robot-AR) and the generation."""
    (storage, robot, stream) = (deployment.storage, deployment.robot, deployment.stream)
    record = storage.episodes[config.episode]
    (state, _action) = storage.read_rows(record)
    stride = config.video_frame_stride
    chunk_frames = config.chunk_size // stride
    chunks = (record.rows - 1) // config.chunk_size
    frame_rows = np.arange(0, chunks * config.chunk_size + 1, stride)
    frames = robot.frames([storage.read_frames(record, view.name, frame_rows) for view in storage.info.views])
    session = deployment.session(record.instructions[0])
    memory = config.policy_type == POLICY_AR
    times = {"encode": [], "commit": [], "generate": []}
    for request in range(config.warmup + config.requests):
        chunk = request % chunks
        first = chunk * chunk_frames
        (state_row, state_mask) = robot.state(state[chunk * config.chunk_size])
        stages = {}
        if chunk == 0:
            observation = stream.start([clip[:, :1] for clip in frames])
            if memory:
                session.start(observation)
        if memory and chunk > 0:
            window = [clip[:, first - chunk_frames + 1 : first + 1] for clip in frames]
            (latents, stages["encode"]) = timed(
                lambda: (stream.encode_chunk(window), stream.encode_observation([clip[:, -1:] for clip in window]))
            )
            (_, stages["commit"]) = timed(lambda: session.commit(latents[0]))
            observation = latents[1]
        elif chunk > 0:
            (observation, stages["encode"]) = timed(
                lambda: stream.encode_observation([clip[:, first : first + 1] for clip in frames])
            )
        (_, stages["generate"]) = timed(lambda: session.generate(observation, state_row, state_mask, robot.action_mask))
        if request >= config.warmup and chunk > 0:
            for stage, milliseconds in stages.items():
                times[stage].append(milliseconds)
    return {stage: float(np.median(values)) for (stage, values) in times.items() if values}


@pyrallis.wrap()
def main(config: BenchmarkConfig) -> None:
    """Time the requests of a deployment: ``python -m sana_robot.utils.benchmark --config_path <deploy config>``, where the config file holds the fields of ``DeployConfig``; add ``--optimized false`` for the eager path."""
    stages = measure(Deployment(config), config)
    parts = ", ".join(f"{stage} {milliseconds:.1f} ms" for (stage, milliseconds) in stages.items())
    print(
        f"{config.policy_type}, {config.steps} steps, optimized {config.optimized}: "
        f"{sum(stages.values()):.1f} ms per request ({parts}; medians of {config.requests} requests)",
        flush=True,
    )


if __name__ == "__main__":
    main()
