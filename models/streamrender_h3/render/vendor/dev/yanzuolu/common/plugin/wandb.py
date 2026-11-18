"""wandb tracker as a Logger plugin.

Attach via the logging plugins list::

    logging:
      plugins:
        - module: common.plugin.wandb
          class_name: WandbPlugin
          mode: offline        # optional (default: wandb default / WANDB_MODE env)

Optional config keys: ``project``, ``name``, ``id`` (default to persistence
proj_name/exp_name), ``mode``, ``entity``, ``tags``.

wandb is an optional dependency: this module is imported only when the plugin is
configured.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
import wandb


class WandbPlugin:
    """wandb tracker as a Logger plugin (rank0-only; other ranks no-op).

    Design decisions:
    - id-pinned run + ``resume="allow"``: restarts (including jobs rescheduled
      onto a different node) continue the same wandb run by id. ``"auto"`` is
      not used because it only resumes from machine-local wandb state.
    - the run id is sanitized: wandb ids only allow ``[A-Za-z0-9_-]`` (max 128
      chars) while ``exp_name`` comes from a config filename stem.
    - ``close(exit_code=...)``: ``Logger.close`` runs in launch's ``finally``,
      so the real exit code must be forwarded -- otherwise a crashed run would
      be marked finished in wandb.
    - define_metric("*", step_metric="train/step") + explicit ``train/step`` in
      every payload decouples charts from wandb's monotonic internal step
      (safe when a restart re-logs an already-seen step range).
    - media logging/publishing capabilities are injected directly onto Logger.
    """

    def __init__(self, plugin_config: Any):
        self.plugin_config = plugin_config
        self.run = None

    def before_configuration(self, logger: Any, **kwargs: Any) -> None:
        logger.inject_methods(
            self,
            {
                "log_video": self.log_video,
                "log_image": self.log_image,
            },
        )

    def after_configuration(self, config: Any = None, **kwargs: Any) -> None:
        if int(os.environ.get("RANK", "0")) != 0:
            return

        from ..persistence import to_plain_dict

        p = config.persistence
        run_id = re.sub(r"[^A-Za-z0-9_-]", "-", str(self.plugin_config.get("id", p.exp_name)))[:128]
        self.run = wandb.init(
            project=self.plugin_config.get("project", p.proj_name),
            name=self.plugin_config.get("name", p.exp_name),
            id=run_id,
            resume=self.plugin_config.get("resume", "allow"),
            dir=p.run_dir,
            mode=self.plugin_config.get("mode", None),
            entity=self.plugin_config.get("entity", None),
            tags=self.plugin_config.get("tags", None),
            config=to_plain_dict(config),
        )
        self.run.define_metric("*", step_metric="train/step")

    def after_log_metrics(self, metrics: dict[str, Any] = None, step: int | None = None, **kwargs: Any) -> None:
        if self.run is None:
            return
        payload = {key: self._to_scalar(value) for key, value in metrics.items()}
        if step is not None:
            payload["train/step"] = step
        self.run.log(payload)

    def log_video(
        self,
        videos: Mapping[str, str | Path],
        step: int | None = None,
        collection: str | None = None,
        captions: Mapping[str, str] | None = None,
        fps: int | None = None,
    ) -> None:
        if self.run is None:
            return

        captions = captions or {}
        payload: dict[str, Any] = {}
        if collection is not None:
            payload[collection] = [
                wandb.Video(
                    str(path),
                    format=Path(path).suffix.lower().lstrip("."),
                    caption=captions.get(name),
                    fps=fps,
                )
                for name, path in videos.items()
            ]
        else:
            for name, path in videos.items():
                ext = Path(path).suffix.lower()
                key = str(PurePosixPath(name).with_suffix(""))
                payload[key] = wandb.Video(
                    str(path), format=ext.lstrip("."), caption=captions.get(name), fps=fps
                )
        self._log_media_payload(payload, "video", videos, collection, step)

    def log_image(
        self,
        images: Mapping[str, str | Path],
        step: int | None = None,
        collection: str | None = None,
        captions: Mapping[str, str] | None = None,
    ) -> None:
        if self.run is None:
            return

        captions = captions or {}
        if collection is not None:
            payload: dict[str, Any] = {
                collection: [wandb.Image(str(path), caption=captions.get(name)) for name, path in images.items()]
            }
        else:
            payload = {
                str(PurePosixPath(name).with_suffix("")): wandb.Image(str(path), caption=captions.get(name))
                for name, path in images.items()
            }
        self._log_media_payload(payload, "image", images, collection, step)

    def flush(self) -> None:
        pass

    def close(self, exit_code: int = 0, **kwargs: Any) -> None:
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
            self.run = None

    def _log_media_payload(
        self,
        payload: dict[str, Any],
        kind: str,
        items: Mapping[str, str | Path],
        collection: str | None,
        step: int | None,
    ) -> None:
        if step is not None:
            payload["train/step"] = step
        self.run.log(payload)
        keys = [collection] if collection is not None else [str(PurePosixPath(name).with_suffix("")) for name in items]
        label = kind if len(items) == 1 else f"{kind}s"
        logging.getLogger().info("WandB logged %d %s at step %s: %s", len(items), label, step, ", ".join(keys))

    @staticmethod
    def _to_scalar(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.item() if value.numel() == 1 else value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value
