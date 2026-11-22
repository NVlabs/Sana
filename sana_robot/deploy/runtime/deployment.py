# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from sana_robot.data.format.normalization import load_normalization
from sana_robot.data.storage import open_storage
from sana_robot.deploy.async_exec.memory_rtc import MemoryRTCController
from sana_robot.deploy.async_exec.rtc import RTCController
from sana_robot.deploy.async_exec.runner import ChunkRunner
from sana_robot.deploy.io.robot import RobotInterface
from sana_robot.deploy.memory.view_stream import ViewStream
from sana_robot.deploy.session.ar_session import ARSession
from sana_robot.deploy.session.base_session import BaseSession
from sana_robot.kernels.inference.presets import install_kernels
from sana_robot.policy.sana_robot_ar import SanaRobotAR
from sana_robot.policy.sana_robot_base import SanaRobotBase
from sana_robot.utils.text_encoder import load_text_encoder
from sana_robot.utils.vae import load_video_encoder

POLICY_AR = "ar"
POLICY_BASE = "base"
WEIGHT_KEYS = ("state_dict_ema", "state_dict")


@dataclass
class DeployConfig:
    """A deployment of SANA-Robot-AR or SANA-Robot-base.

    Args:
        policy_type: ``ar`` or ``base``.
        checkpoint: The policy weights.
        vae: The directory of the LTX-2 video VAE.
        text_encoder: The directory of the Gemma 2 2B weights.
        policy: Keyword arguments of ``SanaRobotAR`` or ``SanaRobotBase``.
        dataset: The directory of the training dataset; its description gives the control frequency, the cameras,
            the prompt fields and the robot slots.
        lerobot_mapping: The mapping file of a LeRobot dataset; unused for a zip archive dataset.
        normalization: The normalization file of the robot rows.
        view_sizes: The pixel (height, width) of every camera.
        chunk_size: Control steps per action chunk.
        video_frame_stride: Control steps between two video frames.
        steps: Denoising steps per chunk.
        video_flow_shift: Flow shift of the video latents.
        action_flow_shift: Flow shift of the actions.
        video_cfg_scale: Classifier-free guidance scale of the video velocity of SANA-Robot-AR, against the prompt
            without the instruction; 1 turns it off. The actions are not guided.
        async_delay_steps: The control steps the robot runs from the running chunk while the next one is generated; 0
            makes the robot wait for every chunk.
        n_action_steps: The control steps between two generated chunks; ``chunk_size - async_delay_steps`` when unset.
        optimized: Run the policy with the inference kernels of the GPU and as CUDA graphs, and compile the video
            encoder; turn it off for the plain eager path.
        precision: The precision of the large matrix products of an optimized policy: ``bf16``, ``fp8`` or
            ``nvfp4`` (see ``install_kernels``).
        device: The device of the models.
    """

    policy_type: str = POLICY_AR
    checkpoint: str = ""
    vae: str = ""
    text_encoder: str = ""
    policy: Dict[str, Any] = field(default_factory=dict)
    dataset: str = ""
    lerobot_mapping: Optional[str] = None
    normalization: str = ""
    view_sizes: List[List[int]] = field(default_factory=lambda: [[256, 320], [256, 320], [256, 320]])
    chunk_size: int = 32
    video_frame_stride: int = 4
    steps: int = 4
    video_flow_shift: float = 5.0
    action_flow_shift: float = 1.0
    video_cfg_scale: float = 1.0
    async_delay_steps: int = 0
    n_action_steps: Optional[int] = None
    optimized: bool = True
    precision: str = "bf16"
    device: str = "cuda"


def load_policy(policy_type: str, arguments: dict, checkpoint: str, device, dtype=torch.bfloat16) -> torch.nn.Module:
    """Build a policy and load its weights from a file that holds a state dict, bare or under ``state_dict_ema`` or ``state_dict``; returned in eval mode."""
    policy = SanaRobotAR(**arguments) if policy_type == POLICY_AR else SanaRobotBase(**arguments)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    for key in WEIGHT_KEYS:
        state = state.get(key, state)
    policy.load_state_dict(state, strict=True)
    return policy.to(device=device, dtype=dtype).eval().requires_grad_(False)


class Deployment:
    """The models of a deployment, loaded once, and the runner of a task.

    Args:
        config: The deployment.
    """

    def __init__(self, config: DeployConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.storage = open_storage(config.dataset, config.lerobot_mapping)
        self.robot = RobotInterface(
            self.storage.info,
            load_normalization(config.normalization),
            config.view_sizes,
            config.chunk_size,
            config.video_frame_stride,
            self.device,
        )
        self.policy = load_policy(config.policy_type, config.policy, config.checkpoint, self.device)
        if config.optimized:
            install_kernels(self.policy, config.precision)
        self.encode_text = load_text_encoder(config.text_encoder, self.device)
        encode = load_video_encoder(config.vae, self.device, compiled=config.optimized)
        self.stream = ViewStream(encode, config.optimized)

    def session(self, instruction: str):
        """Return a session for a task: ``ARSession`` or ``BaseSession``."""
        config = self.config
        (text_embeddings, text_mask) = self.encode_text([self.robot.prompts(instruction)])
        arguments = (
            self.policy,
            text_embeddings,
            text_mask,
            self.robot.data_info(),
            config.steps,
            config.video_flow_shift,
            config.action_flow_shift,
            config.optimized,
        )
        if config.policy_type != POLICY_AR:
            return BaseSession(*arguments)
        unconditional = (None, None)
        if config.video_cfg_scale != 1.0:
            unconditional = self.encode_text([self.robot.prompts(None)])
        return ARSession(*arguments, config.video_cfg_scale, *unconditional)

    def runner(self, instruction: str) -> ChunkRunner:
        """Return the runner of a task, which a control loop drives; see ``ChunkRunner``."""
        config = self.config
        session = self.session(instruction)
        if config.policy_type == POLICY_AR:
            controller = MemoryRTCController(session, self.stream, config.async_delay_steps, config.n_action_steps)
        else:
            controller = RTCController(
                session, self.stream, config.chunk_size, config.async_delay_steps, config.n_action_steps
            )
        return ChunkRunner(controller, self.robot)
