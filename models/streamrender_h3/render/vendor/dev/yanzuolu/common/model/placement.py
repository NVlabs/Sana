"""Model placement stage: how the model lands on hardware.

A single default implementation covers both answers to that question: plain
``.to(device)`` when no ``fsdp`` node is configured (``device`` defaults to
``cuda``), ``fully_shard`` across the default device mesh when it is. The EMA
model, when present, is placed identically by the entry (placement classes face
a single model, like every other stage).

Placement always runs: an omitted ``placement`` node is treated exactly like
``placement: {}``, so a built model is never silently left on CPU/meta -- it
lands on cuda by default. Opt out of the device move only by configuring it
(``placement.device: cpu``).

Like runtime, placement supports before/after plugins (subclasses of
``PlacementPlugin``), routed on ``module`` / ``class_name`` under a ``plugins``
list and run on the main model and the EMA. ``before_placement`` runs before the
core class, ``after_placement`` after it. This is where from-scratch
materialization lives (there is no meta guard: a model that reaches placement on
meta without a plugin to materialize it will fail loud on torch's own error --
the user's risk to take). A meta model is materialized + initialized by an init
plugin at the point dictated by the path: non-fsdp before ``.to()`` (``.to``
cannot move a meta tensor), fsdp after ``fully_shard`` (which shards meta tensors
natively, then ``to_empty`` materializes each rank's shard).

::

    # (no placement node)            # same as placement: {} -> .to(cuda)

    placement: {}                     # .to(cuda)

    placement:
      device: cpu                     # .to(cpu)

    placement:
      weight_dtype: float32           # optional, storage dtype of trained params
      fsdp:                           # node-local HSDP by default; shard within
                                      # LOCAL_WORLD_SIZE and replicate across nodes
        wrap_modules: [TransformerBlock]   # submodule class names to shard
        param_dtype: bfloat16         # default: bf16 all-gather/compute
        reduce_dtype: float32         # default: fp32 gradient reduction
        cpu_offload: true             # optional, offload sharded params/grads
                                      # to CPU (CPUOffloadPolicy)
        shard_size: 8                 # optional, hybrid shard: shard within
                                      # groups of this many ranks, replicate
                                      # across the world_size/shard_size groups
                                      # (typically = GPUs per node)
                                      # explicit null on either dtype = follow
                                      # the parameters' own dtype (torch raw)

``device`` and ``fsdp`` are mutually exclusive: the fsdp path derives the
device from ``LOCAL_RANK``, so a configured ``device`` would be silently
ignored.

FSDP always reshards every wrapped unit, including the root, after forward.
Placement shards one unit at a time and leaves managed parameter/buffer device
movement to ``fully_shard`` for both regular and CPU-offloaded FSDP. PyTorch may
stage each full wrap unit on the target device internally before creating its
local shard; CPU offload then moves that shard back to CPU.

``weight_dtype`` is the storage dtype of the parameters being trained (the
master weights). The cast runs on each parameter's current device before its
wrap unit is sharded. With the standard mmap+assign load path this is a per-unit
CPU cast, so only the current unit receives private target-dtype CPU storage;
the full model is never cast at once. Only trainable
floating-point parameters are cast; frozen parameters keep their file dtype
(convert those via ``weight.dtype`` or offline if needed). The EMA's mirror
parameters (those matching the main model's trainable names) are cast the same
way, so the engine's EMA update sees matching dtypes on both sides and never
rounds. This is orthogonal to ``fsdp.param_dtype``: ``weight_dtype`` governs the
sharded storage, ``param_dtype`` the transient all-gather/compute -- the
standard recipe is ``weight_dtype: float32`` with the default bf16 ``mp_policy``.
"""

from __future__ import annotations

import copy
import importlib
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from ..config import CfgNode
from ..distributed import ops
from ..logging import get_logger

logger = get_logger()

# Device meshes are backed by real process groups, so identical shapes must be
# created once and shared across every model (and EMA) placed in this process.
_MESHES: dict[tuple[int, int], Any] = {}


def _get_hsdp_mesh(shape: tuple[int, int]) -> DeviceMesh:
    """Create or reuse a 2D CUDA mesh without splitting the composite default PG."""
    if shape in _MESHES:
        return _MESHES[shape]

    replicate_size, shard_size = shape
    rank_mesh = torch.arange(replicate_size * shard_size).reshape(shape)
    rank = dist.get_rank()
    shard_group = None
    replicate_group = None

    # Every rank must create every group in exactly the same order. Creating the
    # groups directly avoids DeviceMesh splitting the mixed-backend default
    # process group and passing NCCL options to Gloo on PyTorch 2.9.1. The
    # groups carry both backends because DTensor collectives dispatch by tensor
    # device: CPU-offloaded grads make clip_grad_norm_'s partial-norm reduction
    # run on CPU, which needs Gloo on this mesh (same fix as torchtune#2108).
    for row in rank_mesh:
        ranks = row.tolist()
        group = dist.new_group(ranks=ranks, backend="cpu:gloo,cuda:nccl")
        if rank in ranks:
            shard_group = group
    for column in rank_mesh.T:
        ranks = column.tolist()
        group = dist.new_group(ranks=ranks, backend="cpu:gloo,cuda:nccl")
        if rank in ranks:
            replicate_group = group

    mesh = DeviceMesh.from_group(
        [replicate_group, shard_group],
        device_type="cuda",
        mesh=rank_mesh,
        mesh_dim_names=("replicate", "shard"),
    )
    _MESHES[shape] = mesh
    return mesh


class PlacementPlugin:
    """Base class for placement plugins.

    Override ``before_placement`` and/or ``after_placement`` -- they run,
    respectively, before and after the core placement class for the single model
    the plugin faces. Both default to a no-op and must return the (possibly
    mutated) state dict. ``self.config`` is the plugin's config node (its own
    fields plus ``module`` / ``class_name``; on the EMA pass the ema overrides are
    already merged in).
    """

    def __init__(self, config: Any):
        self.config = config

    def before_placement(self, state: dict[str, Any]) -> dict[str, Any]:
        return state

    def after_placement(self, state: dict[str, Any]) -> dict[str, Any]:
        return state


def place_model(state: dict[str, Any]) -> dict[str, Any]:
    # Generic stage orchestrator: route to the placement class and run each
    # plugin's before/after hooks around it, for the main model and the EMA
    # shadow (placement classes face a single model, like every stage). Nothing
    # framework-specific lives here -- the nn.Module / CUDA / meta assumptions and
    # the trainable scan all belong to the placement class, so an alternative
    # implementation can target a different runtime. A model with no placement
    # node still gets placed: absent is treated exactly like ``placement: {}``.
    config = state["config"].get("placement", CfgNode({}))
    module = importlib.import_module(config.get("module", __name__))
    placement_cls = getattr(module, config.get("class_name", "DefaultPlacement"))
    plugin_configs = config.get("plugins", [])

    # Main model: each plugin with its own config (ema node stripped); before
    # hooks, core placement, after hooks. Deep-copied so a plugin can never mutate
    # the config tree in place.
    main_plugins = [
        (pc.class_name, getattr(importlib.import_module(pc.module), pc.class_name)(
            CfgNode(copy.deepcopy({k: v for k, v in pc.items() if k != "ema"}))))
        for pc in plugin_configs
    ]
    for _, plugin in main_plugins:
        state = plugin.before_placement(state)
    state = placement_cls(config)(state)
    for class_name, plugin in main_plugins:
        state = plugin.after_placement(state)
        logger.info("[%s] placement plugin applied: %s", state["name"], class_name)

    # EMA shadow: use one effective placement config for both its plugins and
    # core placement. The copy is taken AFTER the main pass so DefaultPlacement's
    # derived ``trainable`` set carries over, while EMA overrides can never mutate
    # the live model's config. Placement plugins still receive their own ``ema``
    # overrides independently.
    if "ema_model" in state:
        ema_placement = state["config"].get("ema", CfgNode({})).get("placement", None)
        ema_config = CfgNode(copy.deepcopy(dict(config)))
        if ema_placement:
            CfgNode._merge_dict(ema_config, copy.deepcopy(dict(ema_placement)))
        ema_model_config = CfgNode(copy.deepcopy(dict(state["config"])))
        ema_model_config["placement"] = ema_config

        ema_plugins = []
        for pc in plugin_configs:
            ema_node = pc.get("ema", {})
            if not ema_node.get("enabled", True):
                continue
            effective = CfgNode(copy.deepcopy({k: v for k, v in pc.items() if k != "ema"}))
            CfgNode._merge_dict(effective, {k: v for k, v in ema_node.items() if k != "enabled"})
            ema_plugins.append((pc.class_name, getattr(importlib.import_module(pc.module), pc.class_name)(effective)))
        ema_state = {
            "name": state["name"] + "_ema",
            "config": ema_model_config,
            "model": state["ema_model"],
        }
        for _, plugin in ema_plugins:
            ema_state = plugin.before_placement(ema_state)
        ema_state = placement_cls(ema_config)(ema_state)
        for class_name, plugin in ema_plugins:
            ema_state = plugin.after_placement(ema_state)
            logger.info("[%s] placement plugin applied: %s", ema_state["name"], class_name)
        state["ema_model"] = ema_state["model"]
    return state


class DefaultPlacement:
    """Move to a device, or shard with ``fully_shard`` when ``fsdp`` is set.

    Without an ``fsdp`` node the model is moved wholesale to ``device``
    (default ``cuda``, resolving to the current device set at distributed
    init). With one, every submodule whose class name is listed in
    ``fsdp.wrap_modules`` (the unit of parameter grouping / communication) is
    sharded, then the root module.

    Weight loading is entirely the weights stage's job; placement only moves
    and shards already-materialized tensors. For models too large to fit one
    device, keep the default ``meta_init`` and load through the weights stage's
    mmap+``assign`` path (the meta shells are materialized in place from the
    checkpoint, so node CPU stays at ~1x the shared page cache).

    The fsdp path processes one wrap unit at a time and lets ``fully_shard``
    move that unit's managed parameters and buffers to the mesh device before
    sharding. Transient GPU peak is therefore about the largest single wrap unit
    plus the accumulated shards, not the whole model. CPU offload moves each
    resulting local shard back to CPU; root wrapping never recursively moves
    already-sharded children.
    """

    def __init__(self, config: Any):
        self.config = config

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        model = state["model"]

        # Placement speaks Module/CUDA APIs; a non-Module build product has nothing to place.
        if not isinstance(model, torch.nn.Module):
            return state

        assert not ("device" in self.config and "fsdp" in self.config), (
            "placement.device and placement.fsdp are mutually exclusive: the fsdp path derives the "
            "device from LOCAL_RANK, so a configured device would be silently ignored"
        )

        # No meta guard: an unmaterialized meta model fails loud on torch's own error; init is an opt-in plugin's job.

        # Trainable set: from requires_grad on the main pass, reused by the frozen EMA via the shared config.
        trainable = self.config.get("trainable")
        if trainable is None:
            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            self.config["trainable"] = trainable

        weight_dtype = getattr(torch, self.config.weight_dtype) if "weight_dtype" in self.config else None

        # fsdp shards the model; otherwise move wholesale to device, casting trainable params.
        if "fsdp" in self.config:
            state["model"] = self._shard(state["model"], self.config.fsdp, state["name"], trainable, weight_dtype)
        else:
            device = torch.device(self.config.get("device", "cuda"))
            model = state["model"].to(device)
            cast_count = 0
            if weight_dtype is not None:
                for param_name, param in model.named_parameters():
                    if param_name in trainable and param.is_floating_point() and param.dtype != weight_dtype:
                        param.data = param.data.to(weight_dtype)
                        cast_count += 1
            state["model"] = model
            cast_note = f", cast {cast_count} params to {self.config.weight_dtype}" if weight_dtype is not None else ""
            logger.info("[%s] placed on %s%s", state["name"], device, cast_note)

        # Informational GPU memory report; skipped when CUDA is absent.
        if torch.cuda.is_available():
            logger.info(
                "[%s] GPU memory after placement: allocated %.1f GiB, reserved %.1f GiB",
                state["name"], torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30,
            )
        return state

    def _shard(self, model: Any, config: Any, model_name: str, trainable: set, weight_dtype: Any) -> Any:
        # Mixed-precision policy: bf16 all-gather/compute with fp32 gradient
        # reduction by default; an explicit null follows the parameters' own dtype.
        param_dtype = config.get("param_dtype", "bfloat16")
        reduce_dtype = config.get("reduce_dtype", "float32")
        cpu_offload = bool(config.get("cpu_offload", False))
        kwargs: dict[str, Any] = {
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=getattr(torch, param_dtype) if param_dtype else None,
                reduce_dtype=getattr(torch, reduce_dtype) if reduce_dtype else None,
                # False, which is also torch's own default. True downcasts every
                # floating-point forward input to param_dtype, and that is only
                # safe for activations. Inputs that are coordinates or scales
                # rather than activations -- timesteps, guidance values, rotary
                # positions -- lose distinctions instead of precision: bf16's
                # spacing is 8 at magnitude 1000 and 0.0625 at magnitude 8, so
                # neighbouring values silently collapse onto each other and the
                # run degrades without erroring. Every trial wrapping such a
                # module already sets False by hand; defaulting to it means a
                # trial that omits the line is safe rather than quietly wrong.
                cast_forward_inputs=config.get("cast_forward_inputs", False),
            ),
            "reshard_after_forward": True,
        }
        if cpu_offload:
            kwargs["offload_policy"] = CPUOffloadPolicy()

        # shard_size defaults to the node-local GPU count, clamped to the world;
        # torchrun always exports LOCAL_WORLD_SIZE when the world spans ranks.
        world_size = ops.get_world_size()
        assert world_size == 1 or "LOCAL_WORLD_SIZE" in os.environ, (
            "LOCAL_WORLD_SIZE must be set when world size is greater than 1"
        )
        shard_size = config.get("shard_size")
        if shard_size is None:
            shard_size = min(ops.get_local_world_size(), world_size)
        assert world_size % shard_size == 0, f"shard_size {shard_size} must divide world size {world_size}"

        if shard_size < world_size:
            # Hybrid shard: shard within groups of shard_size ranks, replicate across
            # groups on a 2D mesh; DCP deduplicates the replicas at checkpoint time.
            shape = (world_size // shard_size, shard_size)
            kwargs["mesh"] = _get_hsdp_mesh(shape)

        # Cast and shard one wrap unit at a time. fully_shard owns managed-state
        # device movement; CPU offload returns the resulting local shard to CPU.
        wrap_classes = set(config.get("wrap_modules", []))
        hits = {class_name: 0 for class_name in wrap_classes}
        wrap_units = 0
        cast_count = 0
        for prefix, module in model.named_modules():
            class_name = type(module).__name__
            if class_name in wrap_classes:
                hits[class_name] += 1
                wrap_units += 1
                # Meta units shard directly (an init plugin materializes them
                # after), and the cast applies to them too: ``.to(dtype)`` on a
                # meta tensor is legal and free, and it is what decides the dtype
                # the local shard is later created at.
                #
                # Gating this on "no meta parameters" is what silently demoted a
                # 33 B fp32 master to bf16 once weight loading moved after
                # placement. The DiT's shell is bf16 for the trunk, the weights
                # stage that used to arrive first replaced those parameters
                # outright (``load_state_dict(assign=True)`` takes the FILE's
                # dtype), and a post-placement loader instead fills them in place
                # -- where ``Tensor.copy_`` converts dtype without complaint.
                # Nothing downstream catches it: FSDP all-gathers compute weights
                # in bf16 either way, so every forward is bit-identical and only
                # the optimizer master changes. At lr 2e-6 against bf16's 3e-5 ULP
                # that master rounds nearly every update to zero.
                #
                # DTensors are skipped because a before_placement plugin may have
                # sharded some modules already under a policy of their own -- the
                # same exclusion the residual loop below applies.
                if weight_dtype is not None:
                    # Full name = unit prefix + unit-relative param name.
                    for local_name, param in module.named_parameters():
                        if (
                            f"{prefix}.{local_name}" in trainable
                            and not isinstance(param.data, DTensor)
                            and param.is_floating_point()
                            and param.dtype != weight_dtype
                        ):
                            param.data = param.data.to(weight_dtype)
                            cast_count += 1
                fully_shard(module, **kwargs)

        # A wrap_modules entry matching no submodule is almost always a typo.
        missed = sorted(class_name for class_name, count in hits.items() if count == 0)
        assert not missed, f"wrap_modules {missed} matched no submodule of {model_name}; check the class names"

        # Cast residual parameters outside every wrap unit (embeddings, norms,
        # ...); fully_shard moves their managed parameters and buffers.
        if weight_dtype is not None:
            # Params sharded above are DTensors, already cast in their unit -- skip.
            # Meta parameters are cast like any other; see the wrap-unit loop.
            for param_name, param in model.named_parameters():
                if (
                    param_name in trainable
                    and not isinstance(param.data, DTensor)
                    and param.is_floating_point()
                    and param.dtype != weight_dtype
                ):
                    param.data = param.data.to(weight_dtype)
                    cast_count += 1
        fully_shard(model, **kwargs)

        # Shard summary, plus cast and cpu_offload notes when they apply.
        if shard_size < world_size:
            mesh_note = f"mesh=({world_size // shard_size},{shard_size}) hybrid"
        else:
            mesh_note = f"mesh=default({world_size})"
        logger.info("[%s] fsdp sharded: %d wrap units, %s", model_name, wrap_units, mesh_note)
        if weight_dtype is not None:
            logger.info("[%s] cast %d trainable params to %s during shard", model_name, cast_count, weight_dtype)
        if cpu_offload:
            logger.info("[%s] fsdp cpu_offload enabled (CPUOffloadPolicy)", model_name)
        return model
