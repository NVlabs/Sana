# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from sana_robot.train.checkpoint.state import (
    AVERAGE_FILE,
    STATE_FOLDER,
    latest_checkpoint,
    load_progress,
    save_checkpoint,
)
from sana_robot.train.checkpoint.weights import load_weights
from sana_robot.train.optim.clipping import gradient_groups
from sana_robot.train.optim.ema import start_average
from sana_robot.train.optim.optimizer import build_optimizer
from sana_robot.train.optim.schedule import cosine_schedule
from sana_robot.train.runtime.accelerator import shard_process_group
from sana_robot.train.runtime.tracking import step_logs
from sana_robot.train.step.batch import batch_to_device
from sana_robot.train.step.update import update_parameters


def train_policy(config, accelerator: Accelerator, policy: torch.nn.Module, loader, plan: dict, step_loss) -> None:
    """Run the training loop shared by both policies: optimizer, schedule, averaged weights, checkpoints and logs.

    Args:
        config: The run, an ``ARTrainConfig`` or a ``BaseTrainConfig``.
        accelerator: The ``Accelerator`` of the run.
        policy: The policy as built; its weights are loaded here, from the run's newest checkpoint or from
            ``config.checkpoint``.
        loader: The training loader; its ``batch_sampler.set_epoch(epoch, first_step)`` selects where it starts.
        plan: What the step plan of an epoch depends on; a checkpoint of another plan resumes at the next epoch.
        step_loss: ``step_loss(model, batch)``, which returns the losses of a batch on the device, among them
            ``scaled_loss`` to backpropagate.
    """
    device = accelerator.device
    resume = latest_checkpoint(config.work_dir)
    if resume:
        load_weights(policy, resume / AVERAGE_FILE)
    elif config.checkpoint:
        load_weights(policy, config.checkpoint)
    policy.grad_checkpointing = True
    settings = config.optimizer
    optimizer = build_optimizer(
        policy,
        settings.learning_rate,
        settings.betas,
        settings.eps,
        settings.weight_decay,
        settings.decay_biases_and_norms,
    )
    schedule = cosine_schedule(optimizer, settings.warmup_steps, config.max_steps, settings.min_learning_rate_ratio)
    model = accelerator.prepare(policy)
    optimizer = accelerator.prepare(optimizer)
    accelerator.register_for_checkpointing(schedule)
    groups = gradient_groups(model, settings.gradient_clip_groups, device)
    average = start_average(model)
    process_group = shard_process_group(model)

    progress = {"step": 0, "epoch": 1, "epoch_step": 0, "plan": plan}
    if resume:
        accelerator.load_state(str(resume / STATE_FOLDER))
        progress = load_progress(resume)
        if progress["plan"] != plan:
            progress = {"step": progress["step"], "epoch": progress["epoch"] + 1, "epoch_step": 0, "plan": plan}
    step, epoch, epoch_step = progress["step"], progress["epoch"], progress["epoch_step"]
    set_seed(config.seed + (step + 1) // config.save_steps + accelerator.process_index)

    model.train()
    while step < config.max_steps:
        loader.batch_sampler.set_epoch(epoch, epoch_step)
        for batch in loader:
            optimizer.zero_grad()
            losses = step_loss(model, batch_to_device(batch, device))
            accelerator.backward(losses["scaled_loss"])
            learning_rate = schedule.get_last_lr()[0]
            norms = update_parameters(
                model,
                optimizer,
                schedule,
                groups,
                average,
                accelerator.clip_grad_norm_,
                settings.gradient_clip,
                process_group,
            )
            step, epoch_step = step + 1, epoch_step + 1
            if step % config.log_steps == 0:
                logs = step_logs(accelerator, losses, norms, groups, learning_rate)
                logs["epoch"] = epoch
                accelerator.log(logs, step=step)
            if step % config.save_steps == 0 or step == config.max_steps:
                position = {"step": step, "epoch": epoch, "epoch_step": epoch_step, "plan": plan}
                save_checkpoint(accelerator, model, average, config.work_dir, position)
            if step == config.max_steps:
                break
        else:
            epoch, epoch_step = epoch + 1, 0
    accelerator.end_training()
