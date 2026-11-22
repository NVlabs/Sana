# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from pathlib import Path

import pyrallis
import torch
from accelerate import Accelerator


def start_tracking(accelerator: Accelerator, config, name: str) -> None:
    """Write the config into the run directory and open the Weights & Biases run."""
    if accelerator.is_main_process:
        Path(config.work_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(config.work_dir) / "config.yaml", "w", encoding="utf-8") as stream:
            pyrallis.dump(config, stream)
    accelerator.init_trackers(
        config.wandb_project,
        config=dataclasses.asdict(config),
        init_kwargs={"wandb": {"name": name, "id": name, "resume": "allow"}},
    )


def step_logs(accelerator: Accelerator, losses: dict, norms: dict, groups, learning_rate: float) -> dict:
    """Return the log row of a step: the losses averaged over the ranks, the learning rate and the gradient norms."""
    names = ("loss", "video_loss", "action_loss")
    values = accelerator.gather(torch.stack([losses[name].detach().float() for name in names])[None]).mean(dim=0)
    logs = dict(zip(names, values.tolist()))
    logs["learning_rate"] = learning_rate
    logs["gradient_norm"] = float(norms["gradient_norm"])
    logs["gradient_norm_after_group_clip"] = float(norms["clipped_norm"])
    for name, value in zip(groups.names, norms["group_norms"].tolist()):
        logs[f"gradient_norm/{name}"] = value
    for name, value in zip(groups.clip_names, norms["clip_scales"].tolist()):
        logs[f"clip_scale/{name}"] = value
    return logs
