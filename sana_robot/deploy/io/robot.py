# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

from sana_robot.data.base.dataset import view_layout
from sana_robot.data.format.episode import DatasetInfo
from sana_robot.data.format.normalization import denormalize_rows, normalize_rows
from sana_robot.data.format.prompt import build_prompts
from sana_robot.data.format.video import frames_to_model_range, model_frames


class RobotInterface:
    """What a deployed policy reads from one robot and writes to it, in the conventions of its training dataset.

    The state and the actions are robot rows of 80 slots (``sana_robot.data.format.robot80``), in the units and
    frames the dataset stores; the interface normalizes the state, restores the actions, resizes the camera
    frames and words the prompts as training did.
    """

    def __init__(
        self,
        info: DatasetInfo,
        normalization: dict,
        view_sizes,
        chunk_size: int = 32,
        video_frame_stride: int = 4,
        device="cuda",
    ) -> None:
        """Build the interface of a robot.

        Args:
            info: The description of the training dataset: control frequency, embodiment, action mode, cameras
                and the robot slots in use.
            normalization: The normalization of the robot rows, as ``load_normalization`` returns it.
            view_sizes: The pixel (height, width) of every camera, in the dataset's camera order.
            chunk_size: Control steps per action chunk.
            video_frame_stride: Control steps between two video frames.
            device: The device of the policy.
        """
        self.info = info
        self.normalization = normalization
        self.view_sizes = [tuple(int(side) for side in size) for size in view_sizes]
        self.chunk_size = int(chunk_size)
        self.video_frame_stride = int(video_frame_stride)
        self.device = torch.device(device)
        self.state_mask = torch.from_numpy(info.state_mask)[None].to(self.device)
        self.action_mask = torch.from_numpy(info.action_mask)[None, None].expand(1, self.chunk_size, -1).to(self.device)

    def prompts(self, instruction: str) -> tuple:
        """Return the prompts of a task, one per camera and one for the robot rows."""
        descriptions = [view.description for view in self.info.views]
        return build_prompts(self.info.embodiment, self.info.action_mode, descriptions, instruction)

    def data_info(self) -> dict:
        """Return what the policy reads about the cameras and the clock: ``view_count``, ``view_latent_shape``, ``view_slot_ids``, ``model_fps`` and ``video_frame_stride``."""
        layout = {
            name: value[None].to(self.device) for (name, value) in view_layout(self.info, self.view_sizes).items()
        }
        fps = torch.tensor([float(self.info.fps)], dtype=torch.float64, device=self.device)
        return dict(layout, view_count=len(self.view_sizes), model_fps=fps, video_frame_stride=self.video_frame_stride)

    def frames(self, images) -> tuple:
        """Return camera frames in the policy's input form.

        Args:
            images: Per camera, uint8 frames [T, H, W, 3], or one frame [H, W, 3].

        Returns:
            Per camera, a float tensor [3, T, h_v, w_v] in [-1, 1] on the device, at the camera's size under the
            dataset's resize rule.
        """
        clips = []
        for image, size in zip(images, self.view_sizes):
            image = np.asarray(image)
            clip = model_frames(image[None] if image.ndim == 3 else image, size, self.info.resize)
            clips.append(frames_to_model_range(clip).transpose(0, 1).to(self.device))
        return tuple(clips)

    def state(self, row) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the normalized state [1, 80] of a robot row [80] in the dataset's units, and its slot mask [1, 80]."""
        state = torch.as_tensor(np.asarray(row, dtype=np.float32), device=self.device).reshape(1, -1)
        return (normalize_rows(state, self.state_mask, self.normalization), self.state_mask)

    def actions(self, normalized: torch.Tensor) -> np.ndarray:
        """Return the actions the policy generated, [1, A, 80] normalized, as robot rows [A, 80] in the dataset's units."""
        restored = denormalize_rows(normalized.float(), self.action_mask[:, : normalized.shape[1]], self.normalization)
        return restored[0].cpu().numpy()
