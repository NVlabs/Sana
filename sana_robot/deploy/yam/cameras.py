# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import time
from multiprocessing import shared_memory

import numpy as np
import pyrealsense2 as rs

from sana_robot.deploy.yam.client import FRAME_SIZE, block_name


def open_block(name: str, size: int) -> shared_memory.SharedMemory:
    """Return the shared-memory block of a name, created when it does not exist yet."""
    try:
        return shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        return shared_memory.SharedMemory(name=name)


def main() -> None:
    """Write the newest RGB frame of every RealSense camera to shared memory: ``python -m sana_robot.deploy.yam.cameras --cameras top=<serial> left_wrist=<serial> right_wrist=<serial>``, the names being the policy's cameras."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", nargs="+", required=True)
    parser.add_argument("--fps", type=int, default=30)
    arguments = parser.parse_args()
    (height, width) = FRAME_SIZE
    (blocks, streams) = ([], [])
    for camera in arguments.cameras:
        (name, serial) = camera.split("=")
        block = open_block(block_name(name), height * width * 3)
        settings = rs.config()
        settings.enable_device(serial)
        settings.enable_stream(rs.stream.color, width, height, rs.format.rgb8, arguments.fps)
        pipeline = rs.pipeline()
        pipeline.start(settings)
        blocks.append(block)
        streams.append((pipeline, np.ndarray((height, width, 3), dtype=np.uint8, buffer=block.buf)))
    try:
        while True:
            for pipeline, frame in streams:
                color = pipeline.poll_for_frames().get_color_frame()
                if color:
                    frame[:] = np.asanyarray(color.get_data())
            time.sleep(0.001)
    finally:
        for (pipeline, _frame), block in zip(streams, blocks):
            pipeline.stop()
            block.close()
            block.unlink()


if __name__ == "__main__":
    main()
