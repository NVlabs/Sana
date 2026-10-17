"""Load initial weights from a DCP checkpoint, after the model is sharded.

This is a placement plugin rather than a weights-stage loader, and the ordering
is the whole point. The build runs instantiate -> weights -> adapters -> runtime
-> placement, so at weights time every rank still holds the *whole* model: a
loader there can only hand each rank a full state dict and let ``fully_shard``
slice it afterwards. For a file-backed source that means 64 ranks scatter-read
the same bytes through mmap page faults, which on a parallel filesystem is the
slow path -- measured at 753 s for a 124 GB safetensors where a plain sequential
read of the same file takes 31 s.

DCP inverts that only if the load happens *after* sharding, when the targets are
DTensors: the planner then knows each rank's exact chunk and every rank streams
its own slice. The same checkpoint that takes 753 s through the weights stage
loads in 119-179 s this way (371 GB, 64 ranks, measured on the same cluster).

So the checkpoint load runs in ``after_placement``. A PEFT model whose base is
still meta needs one additional step before sharding: PEFT creates LoRA A/B from
the base layer's shape, then moves them to the base layer's meta device, losing
the temporary initialization values. ``before_placement`` first inspects the DCP
metadata. If every LoRA A/B leaf is present, it stays meta and the DCP fills its
shard directly; if every leaf is absent, the hook materializes only those meta
vanilla-LoRA leaves on CPU and calls PEFT's own initializer. A partial adapter in
the archive is rejected. The hook never touches the base layer,
``modules_to_save`` or project-specific parameters such as a discriminator;
those retain their own initialization plugins.

The model config therefore carries no ``weight`` node at all -- leaving one would
pay the very cost this avoids, loading bytes that are then overwritten::

    models:
      backbone:
        module: ...
        class_name: ...
        meta_init: true
        placement:
          fsdp: { wrap_modules: [TransformerBlock] }
          plugins:
            - module: common.plugin.dcp_weights
              class_name: ShardedDCPWeights
              path: /path/to/exported.dcp

``key`` addresses a subtree when the archive is a training checkpoint rather than
a bare module export, so another run's checkpoint can seed this one directly with
no export step in between::

              path: runs/<proj>/<trial>/checkpoints/0001200
              key: models.backbone

Only the model is read; optimizer state and iteration in that archive are
ignored, which is what makes this an initialization rather than a resume.

PEFT changes live parameter names (for example by adding ``base_model.model`` and
``base_layer``) while a released archive may keep the original model namespace.
``key_map`` is an ordered chain of regex substitutions from the model-local LIVE
FQN to the archive FQN; ``key`` is prepended after the substitutions::

              key_map:
                - {src: '^base_model\\.model\\.', tgt: ''}
                - {src: '\\.base_layer(?=\\.(weight|bias)$)', tgt: ''}
              allow_missing:
                - '\\.lora_[AB]\\.'
                - 'discriminator\\.'

``allow_missing`` is the safe alternative to broad ``strict: false``: every
missing live FQN must match one of its regexes and must already have real storage
from a before-placement initializer. Legacy ``strict: false`` remains supported
for existing configs, but deliberately retains its old broad semantics.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Pattern

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from ..distributed import ops
from ..logging import get_logger
from ..model.placement import PlacementPlugin
from ..seed import local_seed

logger = get_logger()


def _compile_key_map(config: Any) -> list[tuple[Pattern[str], str]]:
    """Compile ordered ``{src, tgt}`` substitutions from plugin config."""
    rules: list[tuple[Pattern[str], str]] = []
    for index, rule in enumerate(config.get("key_map", [])):
        assert isinstance(rule, dict) and set(rule) == {"src", "tgt"}, (
            f"key_map[{index}] must contain exactly src and tgt, got {rule}"
        )
        try:
            pattern = re.compile(str(rule["src"]))
        except re.error as exc:
            raise AssertionError(f"key_map[{index}].src is not a valid regex: {exc}") from exc
        rules.append((pattern, str(rule["tgt"])))
    return rules


def _build_fqn_map(live_fqns: Any, config: Any) -> dict[str, str]:
    """Return ``live FQN -> archive FQN`` with collision/coverage checks."""
    rules = _compile_key_map(config)
    key = config.get("key", None)
    prefix = f"{key}." if key else ""
    mapped: dict[str, str] = {}
    reverse: dict[str, list[str]] = defaultdict(list)
    changed = 0
    for live_fqn in live_fqns:
        archive_local = live_fqn
        for pattern, replacement in rules:
            archive_local = pattern.sub(replacement, archive_local)
        changed += int(archive_local != live_fqn)
        archive_fqn = f"{prefix}{archive_local}"
        mapped[live_fqn] = archive_fqn
        reverse[archive_fqn].append(live_fqn)

    collisions = {archive: lives for archive, lives in reverse.items() if len(lives) > 1}
    assert not collisions, (
        "key_map maps multiple live FQNs onto one archive key; first collisions: "
        f"{list(collisions.items())[:3]}"
    )
    assert not rules or changed > 0, "key_map did not change any live FQN; check its regexes"
    return mapped


def _initialize_meta_lora(state: dict[str, Any], config: Any) -> int:
    """Initialize only meta vanilla-LoRA A/B leaves absent from this DCP.

    PEFT's base layer and all non-LoRA parameters stay untouched. A checkpoint
    that contains every A/B leaf fills the adapter after sharding without ever
    materializing a full copy per rank. A checkpoint that contains none gets the
    normal fresh-PEFT fallback. Partial adapter state is never a valid seed.
    """
    adapter_config = state["config"].get("adapter", None)
    model = state["model"]
    if adapter_config is None or not isinstance(model, nn.Module):
        return 0

    from peft.tuners.lora.layer import LoraLayer

    layers = [module for module in model.modules() if isinstance(module, LoraLayer)]
    if not layers:
        return 0

    path = Path(config.path)
    assert (path / ".metadata").exists(), (
        f"{path} is not a complete DCP checkpoint (no .metadata, the DCP completion marker)"
    )
    stored = dcp.FileSystemReader(str(path)).read_metadata().state_dict_metadata
    named_parameters: dict[int, list[str]] = defaultdict(list)
    all_live_fqns: list[str] = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        all_live_fqns.append(name)
        named_parameters[id(parameter)].append(name)
    fqn_map = _build_fqn_map(all_live_fqns, config)

    peft_configs = getattr(model, "peft_config", None)
    assert peft_configs is not None, "model contains LoraLayer modules but exposes no peft_config"
    adapter_weight = adapter_config.get("weight", None)
    if adapter_weight is not None:
        meta_adapter_fqns = [
            name
            for name, parameter in model.named_parameters()
            if parameter.is_meta and any(
                token in name
                for token in (".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B.")
            )
        ]
        assert not meta_adapter_fqns, (
            f"adapter.weight={adapter_weight} was configured but {len(meta_adapter_fqns)} "
            f"adapter tensors are still meta; first: {meta_adapter_fqns[:3]}"
        )
    initialized = 0

    # One deterministic stream across the stable module traversal, matching
    # PEFT's use of the global CPU RNG while restoring every ambient RNG on exit.
    with local_seed(int(adapter_config.get("seed", 1019))):
        for layer in layers:
            embedding_params = [
                parameter
                for container_name in ("lora_embedding_A", "lora_embedding_B")
                for parameter in getattr(layer, container_name, {}).values()
            ]
            if embedding_params:
                embedding_fqns: list[str] = []
                for parameter in embedding_params:
                    names = named_parameters.get(id(parameter), [])
                    assert len(names) == 1, f"LoRA embedding parameter must have one live FQN, got {names}"
                    embedding_fqns.append(names[0])
                embedding_present = [fqn_map[fqn] in stored for fqn in embedding_fqns]
                assert all(embedding_present) or not any(embedding_present), (
                    f"DCP {path} contains only part of a LoRA embedding adapter; "
                    f"present: {[f for f, yes in zip(embedding_fqns, embedding_present) if yes]}, "
                    f"missing: {[f for f, yes in zip(embedding_fqns, embedding_present) if not yes]}"
                )
                assert all(embedding_present) or not any(parameter.is_meta for parameter in embedding_params), (
                    "ShardedDCPWeights only initializes linear LoRA A/B; DCP-missing meta LoRA "
                    "embedding parameters require their own initializer"
                )

            a_names = set(layer.lora_A.keys())
            b_names = set(layer.lora_B.keys())
            assert a_names == b_names, (
                f"LoRA layer has mismatched A/B adapters: A={sorted(a_names)} B={sorted(b_names)}"
            )
            for adapter_name in sorted(a_names):
                lora_a = layer.lora_A[adapter_name]
                lora_b = layer.lora_B[adapter_name]
                parameters = list(lora_a.parameters()) + list(lora_b.parameters())
                live_fqns: list[str] = []
                for parameter in parameters:
                    names = named_parameters.get(id(parameter), [])
                    assert len(names) == 1, (
                        f"LoRA adapter {adapter_name!r} parameter must have one live FQN, got {names}"
                    )
                    live_fqns.append(names[0])
                present = [fqn_map[fqn] in stored for fqn in live_fqns]
                assert all(present) or not any(present), (
                    f"DCP {path} contains only part of LoRA adapter {adapter_name!r}; "
                    f"present live FQNs: {[f for f, yes in zip(live_fqns, present) if yes]}, "
                    f"missing: {[f for f, yes in zip(live_fqns, present) if not yes]}"
                )
                if all(present):
                    continue

                meta = [parameter.is_meta for parameter in parameters]
                if not any(meta):
                    continue
                assert all(meta), (
                    f"LoRA adapter {adapter_name!r} is partially meta; refusing to guess which "
                    "values should be preserved"
                )
                peft_config = peft_configs[adapter_name]
                init = peft_config.init_lora_weights
                assert init is True or (isinstance(init, str) and init.lower() == "gaussian"), (
                    "meta LoRA initialization supports PEFT's vanilla True/gaussian schemes only; "
                    f"adapter {adapter_name!r} requested {init!r}"
                )
                assert not bool(getattr(peft_config, "use_dora", False)), (
                    f"DoRA adapter {adapter_name!r} requires base-weight-aware initialization"
                )
                assert adapter_name not in getattr(layer, "lora_variant", {}), (
                    f"LoRA variant for adapter {adapter_name!r} requires a dedicated initializer"
                )
                assert not getattr(layer, "merged_adapters", []), (
                    "cannot initialize a merged LoRA layer"
                )

                lora_a.to_empty(device=torch.device("cpu"))
                lora_b.to_empty(device=torch.device("cpu"))
                layer.reset_lora_parameters(adapter_name, init)
                initialized += sum(1 for _ in lora_a.parameters())
                initialized += sum(1 for _ in lora_b.parameters())

    if initialized:
        logger.info(
            "[%s] initialized %d DCP-missing meta LoRA parameter tensors with PEFT "
            "defaults (seed=%d)",
            state["name"],
            initialized,
            int(adapter_config.get("seed", 1019)),
        )
    return initialized


def _materialize_shards(model: nn.Module, param_device: torch.device, buffer_device: torch.device) -> None:
    """``to_empty`` the meta parameter shards while leaving buffers on the compute device.

    Two devices rather than one, because CPU offload does not treat them alike.
    ``fully_shard`` moves a wrap unit's parameters AND buffers to the mesh device
    and then returns only the resulting parameter *shard* to host memory --
    ``CPUOffloadPolicy`` offloads parameters, gradients and optimizer states, and
    buffers are none of those. A single ``to_empty(cpu)`` drags the buffers down
    with the parameters, and the failure surfaces far away: the first rotary
    embedding multiplies a cpu ``inv_freq`` against cuda positions and reports a
    device mismatch inside transformers.

    Buffers were left real by ``init_empty_weights`` and hold constructor values
    (rotary tables and friends) that ``to_empty`` would replace with empty
    storage, so they are saved first and restored onto ``buffer_device``.
    """
    saved = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers(remove_duplicate=False)
        if not buffer.is_meta
    }
    # Filter by is_meta instead of wiping parameters initialized by a
    # before-placement plugin (for example LoRA or a discriminator).
    model._apply(
        lambda tensor: torch.empty_like(tensor, device=param_device) if tensor.is_meta else tensor
    )
    for module_path, module in model.named_modules():
        for name, buffer in list(module._buffers.items()):
            if buffer is None:
                continue
            full_name = f"{module_path}.{name}" if module_path else name
            module._buffers[name] = saved.get(full_name, buffer).to(buffer_device)


class ShardedDCPWeights(PlacementPlugin):
    """Initialize meta LoRA leaves, then fill placed model shards from DCP."""

    def before_placement(self, state: dict[str, Any]) -> dict[str, Any]:
        _initialize_meta_lora(state, self.config)
        return state

    def after_placement(self, state: dict[str, Any]) -> dict[str, Any]:
        model = state["model"]
        if not isinstance(model, nn.Module):
            return state

        path = Path(self.config.path)
        assert (path / ".metadata").exists(), (
            f"{path} is not a complete DCP checkpoint (no .metadata, the DCP completion marker)"
        )

        placement = state["config"].get("placement", None)
        fsdp = placement.get("fsdp", None) if placement else None
        offloaded = bool(fsdp.get("cpu_offload", False)) if fsdp else False
        compute = torch.device("cuda", ops.get_local_rank())
        device = torch.device("cpu") if offloaded else compute

        # Inspect FQNs and meta state before materialization. Selectively allowed
        # missing keys must already have been initialized by a before hook; this
        # catches a forgotten LoRA/discriminator initializer before to_empty could
        # disguise it as a real but uninitialized tensor.
        pre_sharded = get_model_state_dict(
            model=model,
            options=StateDictOptions(full_state_dict=False),
        )
        fqn_map = _build_fqn_map(pre_sharded.keys(), self.config)
        stored = dcp.FileSystemReader(str(path)).read_metadata().state_dict_metadata
        missing_live = sorted(
            live_fqn for live_fqn, archive_fqn in fqn_map.items() if archive_fqn not in stored
        )

        allow_missing = [re.compile(str(pattern)) for pattern in self.config.get("allow_missing", [])]
        strict = bool(self.config.get("strict", True))
        assert not (allow_missing and not strict), (
            "allow_missing is the selective replacement for strict:false; do not configure both"
        )
        if allow_missing:
            unexpected_missing = [
                fqn for fqn in missing_live
                if not any(pattern.search(fqn) for pattern in allow_missing)
            ]
            assert not unexpected_missing, (
                f"[{state['name']}] {len(unexpected_missing)} live FQNs are missing from {path} "
                f"and match no allow_missing regex; first: {unexpected_missing[:3]}"
            )
            uninitialized_missing = [fqn for fqn in missing_live if pre_sharded[fqn].is_meta]
            assert not uninitialized_missing, (
                f"[{state['name']}] {len(uninitialized_missing)} allowed-missing tensors are still "
                f"meta; initialize them before placement; first: {uninitialized_missing[:3]}"
            )
            if missing_live:
                logger.info(
                    "[%s] %d live FQNs are absent from %s and allowed to retain initialized "
                    "values (first live/archive: %s -> %s)",
                    state["name"],
                    len(missing_live),
                    path.name,
                    missing_live[0],
                    fqn_map[missing_live[0]],
                )
        elif strict:
            assert not missing_live, (
                f"[{state['name']}] {len(missing_live)} live FQNs are missing from {path}; "
                f"first: {missing_live[:3]}"
            )
        elif missing_live:
            # Backward-compatible broad mode. New configs should use
            # allow_missing so a broken key map cannot silently skip the trunk.
            logger.info(
                "[%s] %d parameters are absent from %s under legacy strict:false "
                "and are left untouched (first live/archive: %s -> %s)",
                state["name"],
                len(missing_live),
                path.name,
                missing_live[0],
                fqn_map[missing_live[0]],
            )

        if any(parameter.is_meta for parameter in model.parameters()):
            _materialize_shards(model, param_device=device, buffer_device=compute)

        # Re-read after materialization: dcp.load must receive the model's current
        # live tensors, not references captured before to_empty replaced storage.
        sharded = get_model_state_dict(
            model=model,
            options=StateDictOptions(full_state_dict=False),
        )
        assert set(sharded) == set(fqn_map), "model state keys changed during shard materialization"
        archive = {
            fqn_map[live_fqn]: tensor
            for live_fqn, tensor in sharded.items()
            if fqn_map[live_fqn] in stored
        }

        # DCP validates shape but copies across dtype. Frozen conversions are
        # intentional; trainable conversions would destroy optimizer masters.
        converted = [
            (live_fqn, fqn_map[live_fqn], stored[fqn_map[live_fqn]].properties.dtype, tensor.dtype)
            for live_fqn, tensor in sharded.items()
            if isinstance(stored.get(fqn_map[live_fqn]), TensorStorageMetadata)
            and stored[fqn_map[live_fqn]].properties.dtype != tensor.dtype
        ]
        trainable = {fqn for fqn, parameter in model.named_parameters() if parameter.requires_grad}
        assert trainable <= set(sharded), (
            "trainable parameter FQNs and model state-dict FQNs disagree; dtype protection "
            f"would be incomplete: {sorted(trainable - set(sharded))[:3]}"
        )
        harmful = [entry for entry in converted if entry[0] in trainable]
        assert not harmful, (
            f"[{state['name']}] {len(harmful)} TRAINABLE tensors would be silently converted by "
            "dcp.load, destroying their optimizer master; first three "
            f"(live, archive, stored dtype, model dtype): {harmful[:3]}"
        )
        if converted:
            logger.info(
                "[%s] %d frozen tensors converted on load (e.g. %s -> %s %s->%s); "
                "trainable are exact",
                state["name"],
                len(converted),
                converted[0][0],
                converted[0][1],
                converted[0][2],
                converted[0][3],
            )

        if archive:
            dcp.load(archive, checkpoint_id=str(path))

        remaining_meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
        assert not remaining_meta, (
            f"[{state['name']}] DCP initialization left {len(remaining_meta)} parameters meta; "
            f"first: {remaining_meta[:3]}"
        )

        dtypes = Counter(str(tensor.dtype).replace("torch.", "") for tensor in archive.values())
        key = self.config.get("key", None)
        logger.info(
            "[%s] weights loaded from DCP %s%s: %d/%d tensors onto %s, dtypes %s",
            state["name"],
            path,
            f" (key={key})" if key else "",
            len(archive),
            len(sharded),
            device,
            dict(dtypes),
        )
        return state


__all__ = ["ShardedDCPWeights"]
