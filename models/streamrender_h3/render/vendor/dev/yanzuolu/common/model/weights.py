"""Initial weight loading with explicit concrete/diffusers/transformers routes."""

from __future__ import annotations

import gc
import importlib
import json
from pathlib import Path
from typing import Any

import torch

from ..config import CfgNode
from ..logging import get_logger

logger = get_logger()


def load_state_dict_file(path: Path, mmap: bool = True) -> dict[str, Any]:
    """Load a state dict from a single file, dispatched by suffix.

    ``.safetensors``/``.safetensor`` go through safetensors; ``.pkl`` and other
    torch-serialized files go through ``torch.load`` (the latter memory-mapped by
    default).
    """
    if path.suffix in [".safetensor", ".safetensors"]:
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    if path.suffix == ".pkl":
        return torch.load(path, map_location="cpu")
    return torch.load(path, map_location="cpu", mmap=mmap)


def _snapshot_weight_path(config: Any) -> Path:
    root = Path(config.pretrained_model_name_or_path)
    if not root.is_dir():
        # Not a local directory: resolve the hub repo through the HF cache. With
        # HF_HUB_OFFLINE=1 this remains a pure cache lookup on clusters.
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(config.pretrained_model_name_or_path))
    if "subfolder" in config:
        root = root / config.subfolder
    files = (
        sorted(root.glob("*.safetensors.index.json"))
        or sorted(root.glob("*.safetensors"))
        or sorted([*root.glob("*.bin"), *root.glob("*.pt"), *root.glob("*.pth")])
    )
    assert len(files) == 1, (
        f"expected exactly one weight file in snapshot {root}, found {files}; "
        "set weight.path explicitly"
    )
    return files[0]


def load_initial_weight(state: dict[str, Any]) -> dict[str, Any]:
    """Load configured initial weights.

    The singular ``weight`` stage selects exactly one loader class. The default
    loader handles a concrete path or no source; snapshot model configs must
    explicitly select their diffusers or transformers loader class.

    A configured EMA is loaded identically: the same loader class and config are
    applied once more to the independently instantiated EMA shadow.
    """
    config = state["config"]
    stage_config = config.get("weight", {})
    module = importlib.import_module(stage_config.get("module", __name__))
    loader_cls = getattr(module, stage_config.get("class_name", "DefaultWeightLoader"))
    state = loader_cls(stage_config)(state)
    if "ema_model" in state:
        ema_state = {"name": state["name"] + "_ema", "config": config, "model": state["ema_model"]}
        state["ema_model"] = loader_cls(stage_config)(ema_state)["model"]

    return state


class DefaultWeightLoader:
    """Load a concrete ``path``, or preserve constructor init with no source.

    With no ``path``, a snapshot-bearing model config fails loudly and names the
    two explicit snapshot loaders; a model with no source is returned unchanged.

    ``assign=True`` (the default) makes this the materialization entry point for
    meta-initialized shells: the loaded tensors replace the meta placeholders in
    place rather than being copied into pre-allocated storage.

    An optional ``dtype`` casts every floating-point tensor in the state dict
    before load (non-float tensors, e.g. integer buffers, are left untouched).
    This privatizes the cast tensors' memory: they are no longer the mmap'd file
    pages shared across processes on a node, but a per-process copy. For a large
    frozen model the cheaper path is to convert the file to the target dtype once
    offline (``common.plugin.export``) and mmap-load it normally.

    Tied-weight guard blind spot: the guard below inspects the built model's
    parameter identities. ``meta_init`` (accelerate ``init_empty_weights``)
    rewraps parameter objects at construction, so ties are already broken by the
    time we look -- the guard cannot see them. When the checkpoint stores tied
    tensors deduplicated, the tied keys stay meta and the placement stage catches
    them; when they are stored duplicated, the load silently unties and the user
    must know to set ``retie: true`` themselves.
    """

    def __init__(self, config: Any):
        self.config = config

    def _resolved_config(self, state: dict[str, Any]) -> Any:
        if "path" in self.config:
            logger.info("[%s] weight source: explicit weight.path %s", state["name"], self.config.path)
        return self.config

    def _remap_keys(
        self,
        state: dict[str, Any],
        values: dict[str, Any],
        *,
        log_decision: bool,
    ) -> dict[str, Any]:
        """Return a key-compatible mapping; concrete and diffusers loads are no-op."""
        return values

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        name = state["name"]
        config = self._resolved_config(state)
        if "path" not in config:
            pretrained = state["config"].get("pretrained_model_name_or_path", None)
            assert pretrained is None, (
                "DefaultWeightLoader does not handle pretrained_model_name_or_path; "
                "select DiffusersWeightLoader or TransformersWeightLoader explicitly"
            )
            logger.info("[%s] weight source: none, keeping constructor init", name)
            return state
        path = Path(config.path)

        if "assign" not in config and not config.get("retie", False):
            # assign=True replaces parameter objects key by key, silently untying
            # weight-tied parameters (multiple names sharing one tensor). When the
            # config does not handle ties explicitly, detect them and fail loud.
            # Ways out: retie: true (keeps assign=True and re-ties after loading,
            # the memory-safe path), assign: false (copy semantics preserve ties,
            # but cannot materialize meta shells), or assign: true (accept untying,
            # e.g. for frozen encoders).
            groups: dict[int, list[str]] = {}
            for param_name, param in state["model"].named_parameters(remove_duplicate=False):
                groups.setdefault(id(param), []).append(param_name)
            tied = [names for names in groups.values() if len(names) > 1]
            assert not tied, (
                f"[{name}] tied parameters {tied[:2]} would be silently untied by the "
                "default assign=True; set weight.retie or weight.assign explicitly"
            )

        dtype = getattr(torch, config.dtype) if "dtype" in config else None
        cast_count = 0

        if str(path).endswith(".safetensors.index.json"):
            # Sharded load is inherently strict=False (each shard covers only its
            # own keys) and safetensors is inherently mmap-backed, so neither key
            # has any effect here -- reject them instead of silently ignoring.
            assert "strict" not in config, "weight.strict has no effect on a sharded index load (each shard is strict=False); remove it"
            assert "mmap" not in config, "weight.mmap has no effect on a sharded safetensors load (always mmap-backed); remove it"
            from safetensors.torch import load_file

            with path.open("r", encoding="utf-8") as f:
                index = json.load(f)

            weight_map = self._remap_keys(state, index["weight_map"], log_decision=True)
            loaded_keys = weight_map.keys()
            shard_files = list(dict.fromkeys(weight_map.values()))
            for shard_file in shard_files:
                state_dict = load_file(str(path.parent / shard_file), device="cpu")
                state_dict = self._remap_keys(state, state_dict, log_decision=False)
                if dtype is not None:
                    cast_count += sum(1 for v in state_dict.values() if v.is_floating_point() and v.dtype != dtype)
                    state_dict = {k: v.to(dtype) if v.is_floating_point() else v for k, v in state_dict.items()}
                state["model"].load_state_dict(
                    state_dict,
                    strict=False,
                    assign=config.get("assign", True),
                )
                del state_dict
                gc.collect()

            model_keys = state["model"].state_dict().keys()
            result = torch.nn.modules.module._IncompatibleKeys(
                [key for key in model_keys if key not in loaded_keys],
                [key for key in loaded_keys if key not in model_keys],
            )
        else:
            state_dict = load_state_dict_file(path, mmap=config.get("mmap", True))
            state_dict = self._remap_keys(state, state_dict, log_decision=True)
            if dtype is not None:
                cast_count = sum(1 for v in state_dict.values() if v.is_floating_point() and v.dtype != dtype)
                state_dict = {k: v.to(dtype) if v.is_floating_point() else v for k, v in state_dict.items()}
            result = state["model"].load_state_dict(
                state_dict,
                strict=config.get("strict", False),
                assign=config.get("assign", True),
            )
        if dtype is not None:
            logger.info("[%s] cast %d state dict tensors to %s before load", name, cast_count, config.dtype)
        if result.missing_keys or result.unexpected_keys:
            logger.info(
                "[%s] weights loaded from %s: %d missing %s, %d unexpected %s",
                name, path,
                len(result.missing_keys), result.missing_keys[:3],
                len(result.unexpected_keys), result.unexpected_keys[:3],
            )
        else:
            logger.info("[%s] weights loaded from %s: clean load", name, path)

        if config.get("retie", False):
            # Re-establish weight tying after assign=True replaced parameter
            # objects. Checkpoints usually store tied tensors deduplicated, so
            # this also fills the keys that stayed meta because they were absent
            # from the file. The model owns its tying convention via the standard
            # tie_weights() method.
            state["model"].tie_weights()
            logger.info("[%s] weight tying re-established via tie_weights()", name)
        return state


class DiffusersWeightLoader(DefaultWeightLoader):
    """Resolve a diffusers snapshot weight file and load its 1:1 keys."""

    def _resolved_config(self, state: dict[str, Any]) -> CfgNode:
        path = _snapshot_weight_path(state["config"])
        config = CfgNode({
            key: value for key, value in self.config.items()
            if key not in {"module", "class_name"}
        })
        config.path = str(path)
        logger.info("[%s] weight source: auto-detected from snapshot %s", state["name"], path)
        return config


class TransformersWeightLoader(DefaultWeightLoader):
    """Resolve a transformers snapshot and deterministically normalize wrapper keys."""

    def _resolved_config(self, state: dict[str, Any]) -> CfgNode:
        path = _snapshot_weight_path(state["config"])
        config = CfgNode({
            key: value for key, value in self.config.items()
            if key not in {"module", "class_name"}
        })
        config.path = str(path)
        logger.info("[%s] weight source: auto-detected from snapshot %s", state["name"], path)
        return config

    def _remap_keys(
        self,
        state: dict[str, Any],
        values: dict[str, Any],
        *,
        log_decision: bool,
    ) -> dict[str, Any]:
        prefix = type(state["model"]).base_model_prefix
        if not prefix:
            return values

        marker = prefix + "."
        prefixed = [key.startswith(marker) for key in values]
        if prefixed and all(prefixed):
            if log_decision:
                logger.info("[%s] stripping transformers checkpoint key prefix %s", state["name"], marker)
            # Rekey only: tensor objects remain the mmap-backed values.
            return {key[len(marker):]: value for key, value in values.items()}
        assert not any(prefixed), (
            f"[{state['name']}] transformers checkpoint mixes {marker!r}-prefixed and bare keys"
        )
        return values
