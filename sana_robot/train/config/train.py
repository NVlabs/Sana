# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from sana_robot.train.config.data import ChunkDataConfig, WindowDataConfig
from sana_robot.train.config.optimizer import OptimizerConfig


@dataclass
class ARTrainConfig:
    """A training run of SANA-Robot-AR.

    Args:
        work_dir: The directory of the run: its config, its checkpoints and the place it resumes from.
        checkpoint: The policy weights a new run starts from.
        vae: The directory of the LTX-2 video VAE; unused with a latent store.
        text_encoder: The directory of the Gemma 2 2B weights.
        policy: Keyword arguments of ``SanaRobotAR``.
        data: The dataset and its loader.
        optimizer: The optimizer, the schedule and the clipping.
        max_steps: The training steps of the run, also where the learning rate reaches its floor.
        save_steps: The steps between two checkpoints.
        log_steps: The steps between two log rows.
        seed: The seed of the step plan and of the noise.
        action_loss_weight: The weight of the action loss next to the video loss.
        timestep_weighting: Weigh the loss of a chunk by its timestep, more in the middle of the noise range; every
            chunk weighs 1 when off.
        instruction_dropout: The probability that a window is trained without its instruction.
        wandb_project: The Weights & Biases project.
        run_name: The Weights & Biases run name and id; the name of ``work_dir`` when unset.
    """

    work_dir: str = "output/sana_robot_ar"
    checkpoint: Optional[str] = None
    vae: str = ""
    text_encoder: str = ""
    policy: Dict[str, Any] = field(default_factory=dict)
    data: WindowDataConfig = field(default_factory=WindowDataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    max_steps: int = 50000
    save_steps: int = 500
    log_steps: int = 10
    seed: int = 1
    action_loss_weight: float = 1.0
    timestep_weighting: bool = True
    instruction_dropout: float = 0.1
    wandb_project: str = "sana-robot"
    run_name: Optional[str] = None


@dataclass
class BaseTrainConfig:
    """A training run of SANA-Robot-base.

    Args:
        work_dir: The directory of the run: its config, its checkpoints and the place it resumes from.
        checkpoint: The policy weights a new run starts from.
        vae: The directory of the LTX-2 video VAE; unused with a latent store.
        text_encoder: The directory of the Gemma 2 2B weights.
        policy: Keyword arguments of ``SanaRobotBase``.
        data: The dataset and its loader.
        optimizer: The optimizer, the schedule and the clipping.
        max_steps: The training steps of the run, also where the learning rate reaches its floor.
        save_steps: The steps between two checkpoints.
        log_steps: The steps between two log rows.
        seed: The seed of the step plan and of the noise.
        action_loss_weight: The weight of the action loss next to the video loss.
        instruction_dropout: The probability that a window is trained without its instruction.
        wandb_project: The Weights & Biases project.
        run_name: The Weights & Biases run name and id; the name of ``work_dir`` when unset.
    """

    work_dir: str = "output/sana_robot_base"
    checkpoint: Optional[str] = None
    vae: str = ""
    text_encoder: str = ""
    policy: Dict[str, Any] = field(default_factory=dict)
    data: ChunkDataConfig = field(default_factory=ChunkDataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    max_steps: int = 36180
    save_steps: int = 500
    log_steps: int = 10
    seed: int = 1
    action_loss_weight: float = 1.0
    instruction_dropout: float = 0.1
    wandb_project: str = "sana-robot"
    run_name: Optional[str] = None
