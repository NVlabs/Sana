# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""Loading the HyperFlow LoRA onto an official MiniMax-H3 transformer.

The weights file holds LoRA ``A``/``B`` matrices only, under diffusers-style keys::

    transformer.transformer_blocks.{i}.attn.to_q.lora_A.weight        (to_k, to_v, to_out.0, ff.net.0.proj, ff.net.2)
    transformer.token_refiner.refiner_blocks.{j}.attn.to_q.lora_A.weight
    transformer.time_embedder.linear_1.lora_A.weight                   (linear_2)
    transformer.endpoint_time_embedder.linear_1.lora_A.weight          (linear_2)

The set of target modules is whatever the file contains: the loader maps every key onto a module of the transformer,
injects a PEFT adapter on exactly those modules, and loads the tensors. Anything that does not line up — a key with no
module, a module with no tensor, a shape mismatch — is an error, never a warning.

``endpoint_time_embedder`` is not a module of the official transformer. It is the copy of ``time_embedder`` that
[`TwoTimeEmbedder`] creates, so the loader installs the wrapper first and then treats ``time_embedder.*`` as
``time_embedder.base.*`` and ``endpoint_time_embedder.*`` as ``time_embedder.endpoint.*``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from .embedder import TwoTimeEmbedder

logger = logging.getLogger(__name__)

TRANSFORMER_CLASS_NAME = "MiniMaxH3Transformer3DModel"
_ADAPTER_NAME = "hyperflow"
# Sidecar of a weights repo or directory: `{"default": <file>, "weights": {<file>: {"hyperflow_version", "sha256"}}}`
# where `default` is the recommended file, what a bare repo id loads. Published file names are never reused.
MANIFEST_NAME = "hyperflow.json"

_KEY_RE = re.compile(r"^transformer\.(?P<module>.+)\.lora_(?P<matrix>[AB])\.weight$")
_REQUIRED_METADATA = ("hyperflow", "hyperflow_version", "hyperflow_gate", "lora_alpha", "base_model")


@dataclass(frozen=True)
class HyperFlowMetadata:
    """The safetensors header of a HyperFlow weights file, parsed and validated."""

    version: str
    gate: float
    lora_alpha: float
    base_model: str
    base_model_revision: str | None = None
    lora_rank: int | None = None
    sigmas: tuple[float, ...] | None = None
    video_shift: float | None = None
    audio_shift: float | None = None
    trained_on_subfolder: str | None = None
    compatible_subfolders: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    raw: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, metadata: dict[str, str] | None) -> HyperFlowMetadata:
        metadata = dict(metadata or {})
        missing = [key for key in _REQUIRED_METADATA if key not in metadata]
        if missing or metadata.get("hyperflow", "").lower() != "true":
            raise ValueError(
                "Not a HyperFlow weights file: the safetensors header must carry "
                f'{list(_REQUIRED_METADATA)} with `hyperflow == "true"`; missing {missing or ["hyperflow=true"]}.'
            )
        return cls(
            version=metadata["hyperflow_version"],
            gate=float(metadata["hyperflow_gate"]),
            lora_alpha=float(metadata["lora_alpha"]),
            base_model=metadata["base_model"],
            base_model_revision=metadata.get("base_model_revision"),
            lora_rank=int(metadata["lora_rank"]) if "lora_rank" in metadata else None,
            sigmas=_json_tuple(metadata.get("hyperflow_sigmas"), float),
            video_shift=float(metadata["hyperflow_video_shift"]) if "hyperflow_video_shift" in metadata else None,
            audio_shift=float(metadata["hyperflow_audio_shift"]) if "hyperflow_audio_shift" in metadata else None,
            trained_on_subfolder=metadata.get("trained_on_subfolder"),
            compatible_subfolders=_json_tuple(metadata.get("compatible_subfolders"), str) or (),
            tasks=_json_tuple(metadata.get("tasks"), str) or (),
            raw=metadata,
        )


def _json_tuple(value: str | None, cast) -> tuple | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list in the safetensors header, got {value!r}.")
    return tuple(cast(item) for item in parsed)


def read_metadata(path: str | Path) -> HyperFlowMetadata:
    """Read and validate the HyperFlow header of a safetensors file without loading any tensor."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return HyperFlowMetadata.from_dict(handle.metadata())


def resolve_weights(
    path_or_repo: str | Path,
    *,
    filename: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Turn a local file, a local directory or a Hub repo id into the path of one ``.safetensors`` file.

    A directory or repo names its recommended file in [`MANIFEST_NAME`]; without ``filename`` that file is taken. On
    the Hub this is two direct downloads (manifest, then weights), both cached, so a repo id keeps working under
    ``HF_HUB_OFFLINE=1`` once fetched; the repo is only listed when it has no manifest, and then a single
    ``.safetensors`` is accepted. ``filename`` pins one file of the directory or repo regardless of the manifest.
    """
    local = Path(path_or_repo)
    if local.is_file():
        return local
    if local.is_dir():
        if filename is None and (local / MANIFEST_NAME).is_file():
            filename = _manifest_default(local / MANIFEST_NAME)
        candidates = sorted(p.name for p in local.glob("*.safetensors"))
        return local / _pick_one(candidates, str(local), filename)

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    repo_id = str(path_or_repo)
    if filename is None:
        try:
            manifest = hf_hub_download(repo_id, MANIFEST_NAME, revision=revision, token=token)
        except LocalEntryNotFoundError:
            raise  # offline or unreachable with nothing cached: listing the repo could not succeed either
        except EntryNotFoundError:  # no manifest: accept a repo that holds a single weights file
            files = HfApi(token=token).list_repo_files(repo_id, revision=revision)
            filename = _pick_one([f for f in files if f.endswith(".safetensors")], repo_id, None)
        else:
            filename = _manifest_default(Path(manifest))
    return Path(hf_hub_download(repo_id, filename, revision=revision, token=token))


def _manifest_default(path: Path) -> str:
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    default = manifest.get("default") if isinstance(manifest, dict) else None
    if not isinstance(default, str) or not default:
        raise ValueError(f"{path} has no `default` entry naming the recommended weights file.")
    return default


def _pick_one(candidates: Iterable[str], where: str, filename: str | None) -> str:
    candidates = list(candidates)
    if filename is not None:
        if filename not in candidates:
            raise FileNotFoundError(f"{filename!r} not found in {where}; available: {candidates}")
        return filename
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected a {MANIFEST_NAME} naming the default or exactly one .safetensors file in {where}, found "
            f"{candidates}. Pass `filename=` to choose."
        )
    return candidates[0]


def _public_to_module_path(public_module: str) -> str:
    """Map a public key's module path onto the transformer with the [`TwoTimeEmbedder`] installed."""
    if public_module.startswith("endpoint_time_embedder."):
        return "time_embedder.endpoint." + public_module[len("endpoint_time_embedder.") :]
    if public_module.startswith("time_embedder."):
        return "time_embedder.base." + public_module[len("time_embedder.") :]
    return public_module


def plan_from_keys(keys: Iterable[str]) -> dict[str, dict[str, str]]:
    """Group the file's keys by target module: ``{module_path: {"A": key, "B": key}}``."""
    plan: dict[str, dict[str, str]] = {}
    for key in keys:
        match = _KEY_RE.match(key)
        if match is None:
            raise ValueError(
                f"Unexpected key {key!r} in a HyperFlow weights file. Only "
                "`transformer.<module>.lora_A.weight` / `lora_B.weight` keys are allowed."
            )
        module_path = _public_to_module_path(match["module"])
        plan.setdefault(module_path, {})[match["matrix"]] = key
    incomplete = [module for module, matrices in plan.items() if set(matrices) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"Modules with only one of lora_A / lora_B: {incomplete[:8]}")
    return plan


def is_h3_transformer(module: Any) -> bool:
    return isinstance(module, nn.Module) and any(
        klass.__name__ == TRANSFORMER_CLASS_NAME for klass in type(module).__mro__
    )


def find_transformers(target: Any) -> dict[str, nn.Module]:
    """The MiniMax-H3 transformers of a pipeline (``transformer`` and/or ``transformer_ref``), or a bare model."""
    if isinstance(target, nn.Module):
        if not is_h3_transformer(target):
            raise TypeError(f"Expected a {TRANSFORMER_CLASS_NAME}, got {type(target).__name__}.")
        return {"transformer": target}
    found: dict[str, nn.Module] = {}
    components = getattr(target, "components", None)
    if isinstance(components, dict):
        for name, component in components.items():
            if is_h3_transformer(component):
                found[name] = component
    else:
        for name in ("transformer", "transformer_ref"):
            component = getattr(target, name, None)
            if is_h3_transformer(component):
                found[name] = component
    if not found:
        raise ValueError(
            f"No {TRANSFORMER_CLASS_NAME} found on {type(target).__name__}. Call `pipe.load_components(...)` first, "
            "or pass the transformer module directly."
        )
    return found


def load_hyperflow_lora(
    target: Any,
    path_or_repo: str | Path,
    *,
    filename: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    gate: float | None = None,
    configure_blocks: bool = True,
) -> HyperFlowMetadata:
    """Load the HyperFlow LoRA onto every MiniMax-H3 transformer of ``target``.

    Call it after ``pipe.load_components(...)`` and before any ``enable_group_offload(...)``: PEFT replaces the target
    ``nn.Linear`` modules with wrappers, and module-level offload hooks must be attached to the final module tree.
    ``ComponentsManager.enable_auto_cpu_offload`` moves whole components and is fine in either order.

    Args:
        target: A ``ModularPipeline`` (its ``transformer`` and/or ``transformer_ref`` are patched — the same file fits
            both partitions) or a ``MiniMaxH3Transformer3DModel``.
        path_or_repo: Local ``.safetensors`` file, or a local directory / Hub repo id whose ``hyperflow.json`` names
            the recommended file (a single ``.safetensors`` is accepted without one).
        filename: Pin one file of the directory or repo instead of the manifest's default.
        revision: Hub revision.
        token: Hub token.
        gate: Override the blend factor stored in the file (for ablations only).
        configure_blocks: When ``target`` is a pipeline, point its [`HyperFlowSetTimestepsStep`] at the sigma grid
            stored in the file. Raises if the pipeline still runs the official blocks — build them with
            [`hyperflow_blocks`].

    Returns:
        The parsed file metadata.
    """
    path = resolve_weights(path_or_repo, filename=filename, revision=revision, token=token)
    metadata = read_metadata(path)
    transformers = find_transformers(target)
    effective_gate = metadata.gate if gate is None else float(gate)

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        plan = plan_from_keys(handle.keys())
        for name, transformer in transformers.items():
            _load_into(transformer, handle, plan, metadata, gate=effective_gate)
            logger.info(
                "HyperFlow %s (%s) loaded onto %s (%d target modules, rank %s, gate %.4g).",
                metadata.version,
                path.name,
                name,
                len(plan),
                metadata.lora_rank,
                effective_gate,
            )

    if configure_blocks and not isinstance(target, nn.Module):
        blocks = getattr(target, "blocks", None)
        if blocks is not None:
            from .blocks import configure_hyperflow_blocks

            configure_hyperflow_blocks(blocks, metadata)
    return metadata


def _load_into(
    transformer: nn.Module,
    handle: Any,
    plan: dict[str, dict[str, str]],
    metadata: HyperFlowMetadata,
    *,
    gate: float,
) -> None:
    try:
        from peft import LoraConfig
    except ImportError as error:  # pragma: no cover - import guard
        raise RuntimeError("Loading the HyperFlow LoRA requires `peft` (`pip install peft`).") from error

    existing_adapters = sorted((getattr(transformer, "peft_config", None) or {}).keys())
    if existing_adapters:
        raise ValueError(
            f"This transformer already has PEFT adapter(s) {existing_adapters}. HyperFlow owns one fixed adapter and "
            "cannot be loaded alongside or on top of another adapter; use a fresh base model."
        )

    embedder = TwoTimeEmbedder.wrap(transformer, gate)
    embedder.gate = gate

    modules = dict(transformer.named_modules())
    missing_modules = [path for path in plan if not isinstance(modules.get(path), nn.Linear)]
    if missing_modules:
        raise KeyError(
            f"{len(missing_modules)} LoRA target(s) in the file have no `nn.Linear` on this transformer, e.g. "
            f"{missing_modules[:5]}. The file does not match this model architecture."
        )

    ranks = {int(handle.get_slice(matrices["A"]).get_shape()[0]) for matrices in plan.values()}
    if len(ranks) != 1:
        raise ValueError(f"All LoRA targets must share one rank, found {sorted(ranks)}.")
    rank = ranks.pop()
    if metadata.lora_rank is not None and metadata.lora_rank != rank:
        raise ValueError(f"Header says lora_rank={metadata.lora_rank} but the tensors have rank {rank}.")

    config = LoraConfig(
        r=rank,
        lora_alpha=metadata.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=sorted(plan),
        init_lora_weights=True,
    )
    transformer.add_adapter(config, adapter_name=_ADAPTER_NAME)

    parameters = dict(transformer.named_parameters())
    expected = {
        name for name in parameters if f".lora_A.{_ADAPTER_NAME}." in name or f".lora_B.{_ADAPTER_NAME}." in name
    }
    state: dict[str, torch.Tensor] = {}
    for module_path, matrices in plan.items():
        for matrix, key in matrices.items():
            state[f"{module_path}.lora_{matrix}.{_ADAPTER_NAME}.weight"] = handle.get_tensor(key)

    missing = sorted(expected - state.keys())
    unexpected = sorted(state.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            "The injected adapter and the file disagree: "
            f"{len(missing)} parameter(s) without a tensor {missing[:4]}, "
            f"{len(unexpected)} tensor(s) without a parameter {unexpected[:4]}."
        )
    for name, tensor in state.items():
        if tuple(parameters[name].shape) != tuple(tensor.shape):
            raise ValueError(
                f"Shape mismatch for {name}: model {tuple(parameters[name].shape)}, file {tuple(tensor.shape)}."
            )

    with torch.no_grad():
        for name, tensor in state.items():
            parameter = parameters[name]
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            parameter.requires_grad_(False)


def _set_hyperflow(target: Any, enabled: bool) -> None:
    for transformer in find_transformers(target).values():
        embedder = transformer.time_embedder
        if not isinstance(embedder, TwoTimeEmbedder):
            raise RuntimeError("This transformer has no HyperFlow LoRA loaded; call `load_hyperflow_lora` first.")
        if enabled:
            transformer.enable_adapters()
        else:
            transformer.disable_adapters()
        embedder.passthrough = not enabled


def disable_hyperflow(target: Any) -> None:
    """Turn a patched transformer back into the exact base model: LoRA layers off, endpoint ignored.

    Both switches move together — LoRA on with the gate off (or the reverse) is a state the adapter was never trained
    in. When ``target`` is a pipeline built with [`hyperflow_blocks`], those blocks detect passthrough mode and
    delegate to the Diffusers scheduler and denoiser, so one loaded model can produce same-seed base / HyperFlow pairs.
    """
    _set_hyperflow(target, enabled=False)


def enable_hyperflow(target: Any) -> None:
    """Undo [`disable_hyperflow`]."""
    _set_hyperflow(target, enabled=True)


__all__ = [
    "MANIFEST_NAME",
    "HyperFlowMetadata",
    "disable_hyperflow",
    "enable_hyperflow",
    "find_transformers",
    "is_h3_transformer",
    "load_hyperflow_lora",
    "plan_from_keys",
    "read_metadata",
    "resolve_weights",
]
