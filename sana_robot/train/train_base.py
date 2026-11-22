# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyrallis

from sana_robot.data.base.dataset import ChunkDataset, chunk_dataloader
from sana_robot.data.format.normalization import load_normalization
from sana_robot.data.storage import open_storage
from sana_robot.data.storage.latent_store import LatentStore
from sana_robot.policy.sana_robot_base import SanaRobotBase
from sana_robot.train.config.train import BaseTrainConfig
from sana_robot.train.runtime.accelerator import build_accelerator
from sana_robot.train.runtime.loop import train_policy
from sana_robot.train.runtime.tracking import start_tracking
from sana_robot.train.step.base import chunk_training_loss, draw_chunk_noise, encode_chunk_latents
from sana_robot.train.step.batch import training_prompts
from sana_robot.utils.text_encoder import load_text_encoder
from sana_robot.utils.vae import load_video_encoder


def train_base(config: BaseTrainConfig) -> None:
    """Train SANA-Robot-base on one-chunk windows.

    A step holds ``data.batch_size`` windows per GPU, noises every window at its own timestep and takes the mean
    video and action flow loss over the windows. A run resumes from the newest checkpoint in ``work_dir``. From the
    repository root: ``torchrun --standalone --nproc_per_node 8 -m sana_robot.train.train_base --config_path
    <recipe>``, the command of ``scripts/train/train_base.sh``; the recipe is a file of the fields of
    ``BaseTrainConfig``.

    Args:
        config: The run.
    """
    policy = SanaRobotBase(**config.policy)
    accelerator = build_accelerator(type(policy.blocks[0]).__name__)
    device, world_size = accelerator.device, accelerator.num_processes
    start_tracking(accelerator, config, config.run_name or Path(config.work_dir).name)

    data = config.data
    latent_store = LatentStore(data.latent_store) if data.latent_store else None
    dataset = ChunkDataset(
        open_storage(data.path, data.lerobot_mapping),
        load_normalization(data.normalization),
        data.view_sizes,
        data.chunk_size,
        data.video_frame_stride,
        latent_store,
    )
    loader = chunk_dataloader(
        dataset, data.batch_size, world_size, accelerator.process_index, config.seed, data.num_workers
    )
    plan = {"windows": len(dataset), "batch_size": data.batch_size, "world_size": world_size, "seed": config.seed}
    encode_video = None if latent_store else load_video_encoder(config.vae, device)
    encode_text = load_text_encoder(config.text_encoder, device)

    def step_loss(model, batch: dict) -> dict:
        text_embeddings, text_mask = encode_text(training_prompts(batch, config.instruction_dropout))
        latents = encode_chunk_latents(batch, encode_video, device)
        noise = draw_chunk_noise(latents, batch["action"].shape[1], batch["action"].shape[2])
        return chunk_training_loss(model, batch, latents, text_embeddings, text_mask, noise, config.action_loss_weight)

    train_policy(config, accelerator, policy, loader, plan, step_loss)


if __name__ == "__main__":
    train_base(pyrallis.parse(config_class=BaseTrainConfig))
