# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

"""What the adapter needs from a checkpoint directory to build a SANA-Robot deployment.

A checkpoint directory is a released checkpoint or the run directory of ``train.sh``::

    <checkpoint>/
    ├── config.yaml                          the training config: the policy's arguments and the data settings
    ├── meta.json                            the description of the training dataset
    ├── normalization.json                   the normalization of the robot rows
    └── checkpoints/latest/model_ema.pth     the weights (or one ``*.pth`` file in the directory)

Every piece can be given by a ``deploy.yml`` key instead; see the README.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

import sana_robot
from sana_robot.deploy.runtime.deployment import DeployConfig, Deployment

SANA_ROOT_ENV = "SANA_ROBOT_ROOT"
VAE_ENV = "SANA_ROBOT_VAE"
TEXT_ENCODER_ENV = "SANA_ROBOT_TEXT_ENCODER"

TRAIN_CONFIG_FILE = "config.yaml"
META_FILE = "meta.json"
NORMALIZATION_FILE = "normalization.json"
WEIGHTS_FILE = "model_ema.pth"
CHECKPOINT_FOLDER = "checkpoints"
LATEST_LINK = "latest"
RECIPE = "configs/robodojo_ar.yaml"
DEPLOY_FIELDS = (
    "steps",
    "video_flow_shift",
    "action_flow_shift",
    "video_cfg_scale",
    "async_delay_steps",
    "n_action_steps",
    "optimized",
    "precision",
    "device",
)


@dataclass(frozen=True)
class RunFiles:
    """The files of a checkpoint directory and the settings its policy was trained with."""

    root: Path
    weights: Path
    normalization: Path
    meta: Path
    vae: Path
    text_encoder: Path
    policy: dict
    view_sizes: Optional[list]
    chunk_size: Optional[int]
    video_frame_stride: Optional[int]


def sana_root(model_cfg: Mapping[str, Any]) -> Path:
    """Return the checkout of the SANA repository: the ``sana_robot_root`` key, ``$SANA_ROBOT_ROOT``, or the checkout the ``sana_robot`` package is installed from."""
    root = model_cfg.get("sana_robot_root") or os.environ.get(SANA_ROOT_ENV) or Path(sana_robot.__file__).parents[1]
    return Path(os.path.expanduser(str(root))).resolve()


def existing(candidates, what: str) -> Path:
    """Return the first existing path of ``candidates``, which may hold None."""
    paths = [Path(os.path.expanduser(str(path))) for path in candidates if path]
    for path in paths:
        if path.exists():
            return path.resolve()
    checked = "\n  ".join(str(path) for path in paths) or "<nothing to check>"
    raise FileNotFoundError(f"SANA_Robot found no {what}. Checked:\n  {checked}")


def under(base: Path, path) -> Optional[Path]:
    """Return ``path`` under ``base`` when it is relative; None for an empty path."""
    if not path:
        return None
    path = Path(os.path.expanduser(str(path)))
    return path if path.is_absolute() else base / path


def find_weights(root: Path, model_cfg: Mapping[str, Any]) -> Path:
    """Return the weights file of a checkpoint directory: ``checkpoint_file``, the step ``checkpoint_step`` of a run, its latest step, or the directory's ``*.pth`` file."""
    if root.is_file():
        return root
    steps = root / CHECKPOINT_FOLDER
    if model_cfg.get("checkpoint_file"):
        return existing([under(root, model_cfg["checkpoint_file"])], "weights file (checkpoint_file)")
    if model_cfg.get("checkpoint_step") is not None:
        step = steps / f"step_{int(model_cfg['checkpoint_step'])}" / WEIGHTS_FILE
        return existing([step], "weights file (checkpoint_step)")
    return existing([steps / LATEST_LINK / WEIGHTS_FILE, *sorted(root.glob("*.pth"))], f"weights file in {root}")


def resolve_run(model_cfg: Mapping[str, Any], checkpoint: Path, repository: Path) -> RunFiles:
    """Return the files and the training settings of a checkpoint.

    A ``deploy.yml`` key comes first, then what the checkpoint directory holds, then what its training config
    names; paths of the training config are relative to the repository, where training runs. A SANA-Robot-AR
    directory without a training config reads the RoboDojo recipe of the repository.

    Args:
        model_cfg: The deploy config.
        checkpoint: The checkpoint directory, or a weights file whose directory holds the other files.
        repository: The checkout of the SANA repository.

    Returns:
        The files and the settings.
    """
    checkpoint = Path(checkpoint)
    root = checkpoint.parent if checkpoint.is_file() else checkpoint
    config_file = root / TRAIN_CONFIG_FILE
    if not config_file.exists() and (model_cfg.get("policy_type") or "ar") == "ar":
        config_file = repository / RECIPE
    train_config = yaml.safe_load(config_file.read_text()) if config_file.exists() else {}
    data = train_config.get("data") or {}
    dataset = under(repository, data.get("path"))
    return RunFiles(
        root=root,
        weights=find_weights(checkpoint, model_cfg),
        normalization=existing(
            [
                model_cfg.get("normalization_path"),
                root / NORMALIZATION_FILE,
                under(repository, data.get("normalization")),
            ],
            "normalization file (normalization_path)",
        ),
        meta=existing(
            [model_cfg.get("dataset_meta_path"), root / META_FILE, dataset / META_FILE if dataset else None],
            "dataset description meta.json (dataset_meta_path)",
        ),
        vae=existing(
            [model_cfg.get("vae_path"), os.environ.get(VAE_ENV), under(repository, train_config.get("vae"))],
            "LTX-2 video VAE directory (vae_path)",
        ),
        text_encoder=existing(
            [
                model_cfg.get("text_encoder_path"),
                os.environ.get(TEXT_ENCODER_ENV),
                under(repository, train_config.get("text_encoder")),
            ],
            "Gemma 2 2B directory (text_encoder_path)",
        ),
        policy=dict(model_cfg.get("policy") or train_config["policy"]),
        view_sizes=model_cfg.get("view_sizes") or data.get("view_sizes"),
        chunk_size=model_cfg.get("chunk_size") or data.get("chunk_size"),
        video_frame_stride=model_cfg.get("video_frame_stride") or data.get("video_frame_stride"),
    )


def description_dataset(meta: Path) -> tempfile.TemporaryDirectory:
    """Return a dataset directory that holds the description of the training dataset and no episode, which is what a deployment reads of its dataset. The caller keeps the returned object for as long as the directory is needed."""
    directory = tempfile.TemporaryDirectory(prefix="sana_robot_dataset_")
    shutil.copyfile(meta, Path(directory.name) / META_FILE)
    return directory


def build_deployment(model_cfg: Mapping[str, Any], run: RunFiles, dataset: str) -> Deployment:
    """Load the models of a checkpoint as a ``sana_robot`` ``Deployment``.

    Args:
        model_cfg: The deploy config; its keys of ``DEPLOY_FIELDS`` that are set replace the defaults of
            ``DeployConfig``.
        run: The checkpoint's files and settings.
        dataset: The dataset directory of ``description_dataset``.

    Returns:
        The deployment.
    """
    settings = {
        "policy_type": model_cfg.get("policy_type") or "ar",
        "checkpoint": str(run.weights),
        "vae": str(run.vae),
        "text_encoder": str(run.text_encoder),
        "policy": run.policy,
        "dataset": str(dataset),
        "normalization": str(run.normalization),
        "view_sizes": run.view_sizes,
        "chunk_size": run.chunk_size,
        "video_frame_stride": run.video_frame_stride,
    }
    settings.update({name: model_cfg.get(name) for name in DEPLOY_FIELDS})
    return Deployment(DeployConfig(**{name: value for name, value in settings.items() if value is not None}))
