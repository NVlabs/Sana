# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import torch
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from sana_robot.train.checkpoint.weights import AVERAGE_KEY
from sana_robot.train.optim.ema import averaged_weights

CHECKPOINT_FOLDER = "checkpoints"
LATEST_LINK = "latest"
STATE_FOLDER = "model"
PROGRESS_FILE = "metadata.pth"
AVERAGE_FILE = "model_ema.pth"


def checkpoint_directory(work_dir, step: int) -> Path:
    """Return the directory of the checkpoint of a training step."""
    return Path(work_dir) / CHECKPOINT_FOLDER / f"step_{int(step)}"


def latest_checkpoint(work_dir):
    """Return the newest complete checkpoint of a run, or None when the run has none."""
    link = Path(work_dir) / CHECKPOINT_FOLDER / LATEST_LINK
    return link.resolve() if link.exists() else None


def full_state_dict(model: torch.nn.Module, world_size: int) -> dict:
    """Return the state dict of the whole model; under FSDP every rank calls this and rank 0 receives the tensors. Without FSDP the tensors are the model's own, so they follow its weights."""
    if not isinstance(model, FSDP):
        return model.state_dict()
    gathered = world_size > 1
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=gathered, rank0_only=gathered)
    ):
        return model.state_dict()


def save_checkpoint(accelerator, model: torch.nn.Module, average: dict, work_dir, progress: dict) -> Path:
    """Save a checkpoint that a run resumes from and that holds the averaged weights for inference.

    The directory ``checkpoints/step_<step>`` receives ``model/`` (the state Accelerate saves: the training weights,
    the optimizer, the learning rate schedule and the random states), ``model_ema.pth`` (the averaged weights under
    ``state_dict_ema``) and ``metadata.pth`` (``progress``). ``checkpoints/latest`` points at it once it is complete.

    Args:
        accelerator: The ``Accelerator`` of the run; every rank calls this function.
        model: The prepared policy.
        average: The averaged weights from ``start_average``.
        work_dir: The directory of the run.
        progress: The position of the run: ``step``, ``epoch`` and ``epoch_step``, the steps done in the epoch,
            and ``plan``, what the step plan of an epoch depends on.

    Returns:
        The checkpoint directory.
    """
    directory = checkpoint_directory(work_dir, progress["step"])
    accelerator.wait_for_everyone()
    accelerator.save_state(str(directory / STATE_FOLDER))
    with averaged_weights(average, model):
        averaged_state = full_state_dict(model, accelerator.num_processes)
        if accelerator.is_main_process:
            torch.save({AVERAGE_KEY: averaged_state}, directory / AVERAGE_FILE)
    if accelerator.is_main_process:
        torch.save(progress, directory / PROGRESS_FILE)
        link = directory.parent / LATEST_LINK
        fresh = directory.parent / f"{LATEST_LINK}.tmp"
        fresh.unlink(missing_ok=True)
        fresh.symlink_to(directory.name)
        fresh.replace(link)
    accelerator.wait_for_everyone()
    return directory


def load_progress(directory) -> dict:
    """Return the position a checkpoint was saved at."""
    return torch.load(Path(directory) / PROGRESS_FILE, map_location="cpu", weights_only=True)
