# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import math

from diffusers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from diffusion.utils.logger import get_root_logger


def build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio):
    if not config.get("lr_schedule_args", None):
        config.lr_schedule_args = dict()
    if config.get("lr_warmup_steps", None):
        config["num_warmup_steps"] = config.get("lr_warmup_steps")  # for compatibility with old version

    logger = get_root_logger()
    logger.info(
        f"Lr schedule: {config.lr_schedule}, "
        + ",".join([f"{key}:{value}" for key, value in config.lr_schedule_args.items()])
        + "."
    )
    if config.lr_schedule == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            **config.lr_schedule_args,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
        )
    elif config.lr_schedule == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            **config.lr_schedule_args,
        )
    elif config.lr_schedule == "cosine_decay_to_constant":
        assert lr_scale_ratio >= 1
        lr_scheduler = get_cosine_decay_to_constant_with_warmup(
            optimizer=optimizer,
            **config.lr_schedule_args,
            final_lr=1 / lr_scale_ratio,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
        )
    else:
        raise RuntimeError(f"Unrecognized lr schedule {config.lr_schedule}.")
    return lr_scheduler


def get_cosine_decay_to_constant_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    final_lr: float = 0.0,
    num_decay: float = 0.667,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a cosine annealing lr followed by a constant lr.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The number of total training steps.
        final_lr (`int`):
            The final constant lr after cosine decay.
        num_decay (`int`):
            The
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        num_decay_steps = int(num_training_steps * num_decay)
        if current_step > num_decay_steps:
            return final_lr

        progress = float(current_step - num_warmup_steps) / float(max(1, num_decay_steps - num_warmup_steps))
        return (
            max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * (1 - final_lr) + final_lr
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
