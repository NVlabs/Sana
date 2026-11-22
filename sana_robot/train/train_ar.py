# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from pathlib import Path

import pyrallis

from sana_robot.data.ar.dataset import EpisodeWindowDataset, window_dataloader
from sana_robot.data.format.normalization import load_normalization
from sana_robot.data.storage import open_storage
from sana_robot.data.storage.latent_store import LatentStore
from sana_robot.policy.sana_robot_ar import SanaRobotAR
from sana_robot.train.config.train import ARTrainConfig
from sana_robot.train.runtime.accelerator import build_accelerator
from sana_robot.train.runtime.loop import train_policy
from sana_robot.train.runtime.tracking import start_tracking
from sana_robot.train.step.ar import draw_window_noise, encode_window_latents, window_training_loss
from sana_robot.train.step.batch import training_prompts
from sana_robot.train.utils.augmentation import augment_views
from sana_robot.utils.text_encoder import load_text_encoder
from sana_robot.utils.vae import load_video_encoder


def train_ar(config: ARTrainConfig) -> None:
    """Train SANA-Robot-AR on whole-episode windows.

    A step packs windows into one row of at most ``data.chunk_budget`` chunks per GPU, noises every chunk at its
    own timestep and takes the mean video and action flow loss over the trained chunks of all GPUs. A run resumes
    from the newest checkpoint in ``work_dir``. From the repository root: ``torchrun --standalone --nproc_per_node 8
    -m sana_robot.train.train_ar --config_path configs/robodojo_ar.yaml``, the command of
    ``scripts/train/train_ar.sh``.

    Args:
        config: The run.
    """
    policy = SanaRobotAR(**config.policy)
    accelerator = build_accelerator(type(policy.blocks[0]).__name__)
    device, world_size = accelerator.device, accelerator.num_processes
    start_tracking(accelerator, config, config.run_name or Path(config.work_dir).name)

    data = config.data
    latent_store = LatentStore(data.latent_store) if data.latent_store else None
    dataset = EpisodeWindowDataset(
        open_storage(data.path, data.lerobot_mapping),
        load_normalization(data.normalization),
        data.view_sizes,
        data.chunk_size,
        data.video_frame_stride,
        latent_store,
        partial(augment_views, settings=data.augmentation) if data.augmentation else None,
    )
    loader = window_dataloader(
        dataset, data.chunk_budget, world_size, accelerator.process_index, config.seed, data.num_workers
    )
    plan = {"windows": len(dataset), "chunk_budget": data.chunk_budget, "world_size": world_size, "seed": config.seed}
    encode_video = None if latent_store else load_video_encoder(config.vae, device)
    encode_text = load_text_encoder(config.text_encoder, device)

    def step_loss(model, batch: dict) -> dict:
        text_embeddings, text_mask = encode_text(training_prompts(batch, config.instruction_dropout))
        window_latents, observation_latents = encode_window_latents(batch, encode_video, device)
        noise = draw_window_noise(
            window_latents, batch["teacher_forcing_num_chunks"], data.chunk_size, batch["action"].shape[-1]
        )
        return window_training_loss(
            model,
            batch,
            window_latents,
            observation_latents,
            text_embeddings,
            text_mask,
            noise,
            config.action_loss_weight,
            world_size,
            config.timestep_weighting,
        )

    train_policy(config, accelerator, policy, loader, plan, step_loss)


if __name__ == "__main__":
    train_ar(pyrallis.parse(config_class=ARTrainConfig))
