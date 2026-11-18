"""Run persistence and checkpoint saving."""

from __future__ import annotations

import copy
import importlib
import os
import pickle
import random
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import yaml
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from .logging import get_logger

logger = get_logger()


def resolve_run_dirs(config: Any, config_path: str | Path) -> Any:
    """Pure path derivation, no I/O.

    Split out of ``configure_persistence`` so it can run first in the launch
    sequence: ``configure_logging`` needs ``run_dir`` before text logging is up,
    and everything after (distributed init, persistence proper) can then log.
    """
    persistence = config.persistence
    config_path = Path(config_path)

    if persistence.get("exp_name", None) is None:
        persistence.exp_name = config_path.stem

    # proj_name is required, not derived. It used to be inferred from
    # ``entry.module`` by taking the segment after a literal ``projects`` part,
    # which silently tied the on-disk run layout to where the engine module
    # happened to live -- moving an engine would have changed run_dir, or
    # raised ValueError once the path no longer contained ``projects`` at all.
    assert persistence.get("proj_name", None), (
        "persistence.proj_name must be set explicitly in the trial config; "
        "it names the run directory and is no longer derived from entry.module"
    )

    run_dir = (Path(persistence.output_dir).expanduser() / persistence.proj_name / persistence.exp_name).resolve()
    persistence.run_dir = str(run_dir)
    persistence.configs_dir = str(run_dir / "configs")
    persistence.checkpoints_dir = str(run_dir / "checkpoints")
    return config


def configure_persistence(config: Any, config_path: str | Path, overrides: list[str]) -> Any:
    persistence = config.persistence
    run_dir = Path(persistence.run_dir)

    plugins = []
    for plugin_config in config.persistence.get("plugins", []):
        module = importlib.import_module(plugin_config.module)
        plugin_cls = getattr(module, plugin_config.class_name)
        plugins.append(plugin_cls(plugin_config))
    config.persistence_plugins = PersistencePlugins(plugins)
    config.persistence_plugins.before_run_dir_create(config=config, run_dir=run_dir)
    for path in [run_dir, Path(persistence.configs_dir), Path(persistence.checkpoints_dir)]:
        path.mkdir(parents=True, exist_ok=True)
    config.persistence_plugins.after_run_dir_ready(config=config, run_dir=run_dir)

    save_config_snapshot(config, Path(config_path), overrides)
    config.checkpointer = Checkpointer(config)
    logger.info("Persistence ready: run_dir=%s async_checkpoint=%s", run_dir, config.checkpointer.async_mode)
    return config


def save_config_snapshot(config: Any, config_path: Path, overrides: list[str]) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    configs_dir = Path(config.persistence.configs_dir)
    config.persistence_plugins.before_configs_saved(config=config, configs_dir=configs_dir)

    files = {
        "original": configs_dir / "original.yaml",
        "overrides": configs_dir / "overrides.txt",
        "git_commit": configs_dir / "git_commit.txt",
        "git_status": configs_dir / "git_status.txt",
        "git_tracked_patch": configs_dir / "git_tracked_changes.patch",
        "git_untracked_list": configs_dir / "git_untracked_files.txt",
        "git_untracked_archive": configs_dir / "git_untracked_files.tar",
        "resolved": configs_dir / "resolved.yaml",
    }

    files["original"].write_text(config_path.read_text())
    files["overrides"].write_text("\n".join(overrides) + ("\n" if overrides else ""))
    files["git_commit"].write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True))
    files["git_status"].write_text(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    files["git_tracked_patch"].write_text(subprocess.check_output(["git", "diff", "HEAD", "--binary"], text=True))

    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"])
    untracked_files = [path.decode() for path in untracked.split(b"\0") if path]
    files["git_untracked_list"].write_text("\n".join(untracked_files) + ("\n" if untracked_files else ""))
    with tarfile.open(files["git_untracked_archive"], "w") as tar:
        for path in untracked_files:
            tar.add(path, arcname=path)

    with files["resolved"].open("w") as f:
        yaml.safe_dump(to_plain_dict(config), f, sort_keys=False)

    config.persistence_plugins.after_configs_saved(config=config, configs_dir=configs_dir, files=files)


class PersistencePlugins:
    def __init__(self, plugins: list[Any]):
        self.plugins = plugins

    def before_run_dir_create(self, **payload: Any) -> None:
        self.emit("before_run_dir_create", **payload)

    def after_run_dir_ready(self, **payload: Any) -> None:
        self.emit("after_run_dir_ready", **payload)

    def before_configs_saved(self, **payload: Any) -> None:
        self.emit("before_configs_saved", **payload)

    def after_configs_saved(self, **payload: Any) -> None:
        self.emit("after_configs_saved", **payload)

    def before_checkpoint_saved(self, **payload: Any) -> None:
        self.emit("before_checkpoint_saved", **payload)

    def after_checkpoint_saved(self, **payload: Any) -> None:
        self.emit("after_checkpoint_saved", **payload)

    def before_checkpoint_loaded(self, **payload: Any) -> None:
        self.emit("before_checkpoint_loaded", **payload)

    def after_checkpoint_loaded(self, **payload: Any) -> None:
        self.emit("after_checkpoint_loaded", **payload)

    def emit(self, hook_name: str, **payload: Any) -> None:
        for plugin in self.plugins:
            hook = getattr(plugin, hook_name, None)
            if hook is not None:
                hook(**payload)


# ---------------------------------------------------------------------------
# Handlers
#
# Every handler shares the constructor signature (path, leaf, tree, args) and
# implements the DCP top-level ``Stateful`` protocol (``state_dict`` /
# ``load_state_dict``). That protocol is ``runtime_checkable`` duck typing, so
# handlers do not need to subclass anything -- DCP will call ``state_dict()`` on
# each top-level value to obtain the structure to save/load and call
# ``load_state_dict()`` back after loading.
# ---------------------------------------------------------------------------


class ModelHandler:
    """Handles ``torch.nn.Module`` leaves via the DCP model state-dict helpers."""

    def __init__(self, path: str, leaf: Any, tree: dict, args: dict):
        self.path = path
        self.leaf = leaf
        self.tree = tree
        self.args = args

    def state_dict(self) -> Any:
        return get_model_state_dict(
            model=self.leaf,
            options=StateDictOptions(full_state_dict=self.args.get("full_state_dict", False), cpu_offload=True),
        )

    def load_state_dict(self, sd: Any) -> None:
        full_state_dict = self.args.get("full_state_dict", False)
        set_model_state_dict(
            model=self.leaf,
            model_state_dict=sd,
            options=StateDictOptions(full_state_dict=full_state_dict, broadcast_from_rank0=full_state_dict),
        )


class OptimizerHandler:
    """Handles ``torch.optim.Optimizer`` leaves and needs the paired model.

    Checkpoints restore per-parameter optimizer state, including moments and
    steps. Every param-group field other than ``params`` always comes from the
    newly constructed optimizer, so the current YAML and optimizer defaults
    have global priority over checkpoint hyperparameters. The training loop's
    LR scheduler may continue to overwrite ``lr`` after resume.

    The model is resolved from ``args["model"]`` (a dotted path into the state
    tree). When absent, the convention swaps the first path segment
    ``optimizers`` for ``models`` (``optimizers.backbone`` -> ``models.backbone``)
    and then strips trailing segments until a node is found, so a nested
    multi-optimizer FQN (``optimizers.backbone.matrices``) still pairs with its
    model (``models.backbone``).
    """

    def __init__(self, path: str, leaf: Any, tree: dict, args: dict):
        self.path = path
        self.leaf = leaf
        self.tree = tree
        self.args = args
        self.model = self._resolve_model()

    def _resolve_model(self) -> Any:
        explicit = self.args.get("model")
        if explicit is not None:
            node = self.tree
            try:
                for segment in explicit.split("."):
                    node = node[segment]
            except (KeyError, TypeError) as exc:
                raise KeyError(
                    f"OptimizerHandler at '{self.path}' could not resolve its model at '{explicit}'. "
                    f"Specify it explicitly via persistence.checkpoint.handlers.{self.path}.args.model"
                ) from exc
            return node

        segments = self.path.split(".")
        segments[0] = "models"
        # Strip trailing segments until a node resolves: optimizers.backbone.matrices
        # -> models.backbone.matrices (miss) -> models.backbone (hit). Never strip
        # below models.<name> (2 segments), or a miss would return the whole
        # models container instead of failing.
        while len(segments) >= 2:
            node = self.tree
            for segment in segments:
                node = node.get(segment) if isinstance(node, dict) else None
                if node is None:
                    break
            else:
                return node
            segments.pop()
        raise KeyError(
            f"OptimizerHandler at '{self.path}' could not resolve its model by convention. "
            f"Specify it explicitly via persistence.checkpoint.handlers.{self.path}.args.model"
        )

    def state_dict(self) -> Any:
        return get_optimizer_state_dict(
            model=self.model,
            optimizers=self.leaf,
            options=StateDictOptions(full_state_dict=self.args.get("full_state_dict", False), cpu_offload=True),
        )

    def load_state_dict(self, sd: Any) -> None:
        current_group_fields = [
            copy.deepcopy({key: value for key, value in group.items() if key != "params"})
            for group in self.leaf.param_groups
        ]
        current_group_count = len(current_group_fields)
        checkpoint_group_count = (
            len(sd["param_groups"])
            if isinstance(sd, dict) and isinstance(sd.get("param_groups"), list)
            else None
        )

        full_state_dict = self.args.get("full_state_dict", False)
        try:
            set_optimizer_state_dict(
                model=self.model,
                optimizers=self.leaf,
                optim_state_dict=sd,
                options=StateDictOptions(full_state_dict=full_state_dict, broadcast_from_rank0=full_state_dict),
            )
        except ValueError as exc:
            if checkpoint_group_count is not None and checkpoint_group_count != current_group_count:
                raise ValueError(
                    f"OptimizerHandler at '{self.path}' param-group count mismatch. "
                    f"Loaded {checkpoint_group_count}, current {current_group_count}"
                ) from exc
            raise

        loaded_group_count = len(self.leaf.param_groups)
        if loaded_group_count != current_group_count:
            raise ValueError(
                f"OptimizerHandler at '{self.path}' param-group count mismatch. "
                f"Loaded {loaded_group_count}, current {current_group_count}"
            )

        for loaded_group, current_fields in zip(self.leaf.param_groups, current_group_fields, strict=True):
            loaded_params = loaded_group["params"]
            loaded_group.clear()
            loaded_group["params"] = loaded_params
            loaded_group.update(current_fields)

        logger.info("OptimizerHandler loaded path=%s groups=%d", self.path, current_group_count)


class StatefulHandler:
    """Passthrough for leaves that already implement ``state_dict``/``load_state_dict``."""

    def __init__(self, path: str, leaf: Any, tree: dict, args: dict):
        self.path = path
        self.leaf = leaf
        self.tree = tree
        self.args = args

    def state_dict(self) -> Any:
        return self.leaf.state_dict()

    def load_state_dict(self, sd: Any) -> None:
        self.leaf.load_state_dict(sd)


class ValueHandler:
    """Handles scalar leaves (int/float/str/bool/None) as plain checkpoint values."""

    def __init__(self, path: str, leaf: Any, tree: dict, args: dict):
        self.path = path
        self.leaf = leaf
        self.tree = tree
        self.args = args
        self.value = leaf

    def state_dict(self) -> Any:
        return {"value": self.leaf}

    def load_state_dict(self, sd: Any) -> None:
        self.value = sd["value"]

    @property
    def restored_value(self) -> Any:
        return self.value


class RngStateHandler:
    """Captures/restores python, numpy and torch RNG state for the local rank.

    The whole payload is deliberately ``pickle.dumps``-ed into a single ``bytes``
    object so it travels DCP's BYTE_IO path: this avoids the in-place tensor
    shape constraints that DCP enforces when loading tensor values. The result is
    keyed per rank (``rank_{RANK}``) because DCP deduplicates identically named
    top-level keys across ranks -- without the per-rank key each rank's RNG state
    would collapse into one.

    Injected automatically by ``Checkpointer`` (no leaf), so its constructor
    takes no arguments.
    """

    def __init__(self):
        pass

    def state_dict(self) -> Any:
        rank = int(os.environ.get("RANK", "0"))
        payload = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["torch_cuda"] = torch.cuda.get_rng_state_all()
        return {f"rank_{rank}": pickle.dumps(payload)}

    def load_state_dict(self, sd: Any) -> None:
        rank = int(os.environ.get("RANK", "0"))
        blob = sd[f"rank_{rank}"]
        payload = pickle.loads(blob)
        random.setstate(payload["python"])
        np.random.set_state(payload["numpy"])
        torch.set_rng_state(payload["torch_cpu"])
        if "torch_cuda" in payload and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["torch_cuda"])


def walk_state_tree(tree: dict, overrides: dict) -> dict[str, Any]:
    """Flatten a nested state tree into ``{dotted_path: handler}``.

    Recurses into plain dicts, accumulating a dotted path; anything that is not a
    dict is a leaf and is dispatched to a handler. The returned flat dict is fed
    straight to ``dcp.save``/``dcp.load`` (DCP calls ``state_dict()`` on each
    top-level value automatically).
    """
    entries: dict[str, Any] = {}

    def recurse(node: dict, prefix: str) -> None:
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                recurse(value, path)
            elif path in overrides:
                override = overrides[path]
                module = importlib.import_module(override["module"])
                handler_cls = getattr(module, override["class_name"])
                entries[path] = handler_cls(path, value, tree, override.get("args", {}))
            elif isinstance(value, torch.nn.Module):
                entries[path] = ModelHandler(path, value, tree, {})
            elif isinstance(value, torch.optim.Optimizer):
                entries[path] = OptimizerHandler(path, value, tree, {})
            elif hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                entries[path] = StatefulHandler(path, value, tree, {})
            elif isinstance(value, (int, float, str, bool)) or value is None:
                entries[path] = ValueHandler(path, value, tree, {})
            else:
                raise TypeError(f"No handler for leaf at '{path}' of type {type(value).__name__}")

    recurse(tree, "")
    return entries


class Checkpointer:
    """Single DCP-based checkpointer.

    Walks a nested state tree into per-leaf handlers, dispatched by type or by
    config override, then performs a single ``dcp.save``/``dcp.async_save`` (and
    the mirror ``dcp.load``). Design mirrors torchtitan's ``CheckpointManager``.
    """

    def __init__(self, config: Any):
        self.config = config
        ckpt_cfg = config.persistence.get("checkpoint", {})
        self.async_mode = ckpt_cfg.get("async", False)
        self.allow_partial_load = ckpt_cfg.get("allow_partial_load", False)
        self.keep_latest_k = ckpt_cfg.get("keep_latest_k", -1)
        if self.keep_latest_k == 1:
            raise ValueError(
                "keep_latest_k=1 is unsafe: the newest checkpoint may still be writing "
                "asynchronously and would be deleted by cleanup. Use keep_latest_k>=2 or -1."
            )
        self.handler_overrides = ckpt_cfg.get("handlers", {})
        self._pending: tuple[Any, int, Path, Any] | None = None
        self._async_pg: Any = None
        if self.async_mode and dist.is_available() and dist.is_initialized():
            self._async_pg = self._create_async_pg()

    def _create_async_pg(self) -> Any:
        """Dedicated gloo group for async saves (torchtitan pattern).

        The async writer coordinates its plan/metadata exchange from a background
        thread; that must never share the training (NCCL) process group. Created
        eagerly at construction time -- before any training collective is in
        flight -- and warmed up with one barrier so gloo's lazy TCP connect does
        not happen inside the first save.
        """
        pg = dist.new_group(backend="gloo")
        dist.barrier(group=pg)
        return pg

    def save(self, state: dict[str, Any], step: int, metadata: dict[str, Any] | None = None) -> Path:
        self._wait_pending()

        entries = walk_state_tree(state, self.handler_overrides)
        assert "rng" not in entries, "'rng' is reserved for the checkpointer-injected RNG state"
        entries["rng"] = RngStateHandler()

        checkpoint_dir = Path(self.config.persistence.checkpoints_dir) / f"{step:07d}"
        self.config.persistence_plugins.before_checkpoint_saved(
            config=self.config,
            checkpoint_dir=checkpoint_dir,
            step=step,
            metadata=metadata,
            checkpointer=self,
        )

        if int(os.environ.get("RANK", "0")) == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving checkpoint step=%s to %s (async=%s)", step, checkpoint_dir, self.async_mode)
        if self.async_mode:
            if dist.is_available() and dist.is_initialized() and self._async_pg is None:
                # dist was not yet initialized when this Checkpointer was built,
                # so the group is created mid-training: drain in-flight NCCL work
                # and align all ranks on the default pg first (new_group "Safe
                # concurrent usage": pending async ops on one pg must complete
                # before another pg is used, and all ranks must create groups in
                # the same order).
                torch.cuda.synchronize()
                dist.barrier()
                self._async_pg = self._create_async_pg()
            future = dcp.async_save(entries, checkpoint_id=str(checkpoint_dir), process_group=self._async_pg)
            self._pending = (future, step, checkpoint_dir, metadata)
        else:
            dcp.save(entries, checkpoint_id=str(checkpoint_dir))
            logger.info("Checkpoint step=%s complete", step)
            self.config.persistence_plugins.after_checkpoint_saved(
                config=self.config,
                checkpoint_dir=checkpoint_dir,
                step=step,
                metadata=metadata,
                checkpointer=self,
            )
            self._cleanup()

        return checkpoint_dir

    def load(self, state: dict[str, Any], step: int | None = None, metadata: dict[str, Any] | None = None, checkpoints_dir: str | Path | None = None) -> dict[str, Any]:
        self._wait_pending()

        if checkpoints_dir is None:
            checkpoints_dir = self.config.persistence.checkpoints_dir
        if step is None:
            step = self.find_latest_step(checkpoints_dir)
        if step is None:
            raise RuntimeError(f"no complete checkpoint found in {checkpoints_dir}")

        checkpoint_dir = Path(checkpoints_dir) / f"{step:07d}"
        if not (checkpoint_dir / ".metadata").exists():
            raise RuntimeError(
                f"checkpoint at {checkpoint_dir} is incomplete (missing .metadata, the DCP completion marker)"
            )

        self.config.persistence_plugins.before_checkpoint_loaded(
            config=self.config,
            checkpoint_dir=checkpoint_dir,
            step=step,
            metadata=metadata,
            checkpointer=self,
        )

        logger.info("Loading checkpoint step=%s from %s", step, checkpoint_dir)
        entries = walk_state_tree(state, self.handler_overrides)
        assert "rng" not in entries, "'rng' is reserved for the checkpointer-injected RNG state"
        entries["rng"] = RngStateHandler()
        planner = DefaultLoadPlanner(allow_partial_load=self.allow_partial_load)
        dcp.load(entries, checkpoint_id=str(checkpoint_dir), planner=planner)

        self.config.persistence_plugins.after_checkpoint_loaded(
            config=self.config,
            checkpoint_dir=checkpoint_dir,
            step=step,
            metadata=metadata,
            checkpointer=self,
        )

        result: dict[str, Any] = {}
        for path, handler in entries.items():
            if isinstance(handler, ValueHandler):
                keys = path.split(".")
                node = result
                for key in keys[:-1]:
                    node = node.setdefault(key, {})
                node[keys[-1]] = handler.restored_value
        result["step"] = step
        return result

    def find_latest_step(self, checkpoints_dir: str | Path | None = None) -> int | None:
        if checkpoints_dir is None:
            checkpoints_dir = self.config.persistence.checkpoints_dir
        checkpoints_dir = Path(checkpoints_dir)
        if not checkpoints_dir.exists():
            return None
        steps = [
            int(path.name)
            for path in checkpoints_dir.iterdir()
            if path.is_dir() and path.name.isdigit() and (path / ".metadata").exists()
        ]
        return max(steps) if steps else None

    def finalize(self) -> None:
        """Flush any in-flight async write (call before the engine exits)."""
        self._wait_pending()

    def _wait_pending(self) -> None:
        if self._pending is None:
            return
        future, step, checkpoint_dir, metadata = self._pending
        future.result()
        self._pending = None
        logger.info("Async checkpoint step=%s complete", step)
        self.config.persistence_plugins.after_checkpoint_saved(
            config=self.config,
            checkpoint_dir=checkpoint_dir,
            step=step,
            metadata=metadata,
            checkpointer=self,
        )
        self._cleanup()

    def _cleanup(self) -> None:
        if self.keep_latest_k < 0:
            return
        if int(os.environ.get("RANK", "0")) != 0:
            return
        checkpoints_dir = Path(self.config.persistence.checkpoints_dir)
        dirs = sorted(
            (path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
        )
        for path in dirs[: max(0, len(dirs) - self.keep_latest_k)]:
            shutil.rmtree(path)

def to_plain_dict(value: Any) -> Any:
    if isinstance(value, dict):
        runtime_keys = {"persistence_plugins", "checkpointer", "logger"}
        return {key: to_plain_dict(item) for key, item in value.items() if key not in runtime_keys}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return value
