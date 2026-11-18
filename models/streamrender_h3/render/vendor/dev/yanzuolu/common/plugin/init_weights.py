"""From-scratch weight-init placement plugins.

Two general-purpose initializers for a meta-shell model that has no checkpoint to
materialize it (training from scratch). Both are opt-in placement plugins (not
enabled by default) and cover the common weight-bearing layer types
(Linear/Conv/norm/Embedding) with sensible standard init, seeded for
reproducibility -- broad enough to use directly on a stock model; a model with a
bespoke scheme or semantic buffers should supply its own init plugin instead.

The two differ by where in placement they run and how they drive the RNG, both
dictated by how the core placement op treats meta tensors:

* ``InitWeights`` -- non-fsdp, ``before_placement``. ``.to(device)`` cannot move a
  meta tensor, so the model is materialized + initialized *before* the core moves
  it: on CPU, with an explicit CPU ``torch.Generator`` (plain tensors, isolated
  from global RNG). The core ``.to(device)`` then moves the real, initialized model.
* ``ShardedInitWeights`` -- fsdp, ``after_placement``. ``fully_shard`` shards meta
  tensors natively, so the core runs first and leaves meta DTensor shards; this
  then materializes each rank's shard on cuda and initializes. DTensor's RNG must
  be driven by the default generator seeded identically on every rank (an explicit
  generator triggers DTensor's deprecated rank0 RNG-sync), so the init runs inside
  ``local_seed`` -- which seeds the global RNGs and restores every previous state
  (python/numpy/torch-CPU/torch-CUDA) on exit, leaving the build-time global RNG
  undisturbed for any later design that relies on it. FSDP-path correctness is
  verified on the cluster, not locally (DTensor RNG is CUDA-only).

Both preserve buffers: instantiation uses ``init_empty_weights``, which leaves
buffers materialized (only parameters go to meta), so the constructor's buffer
values (RoPE tables, etc.) are already correct. ``to_empty`` would wipe them, so
they are saved and restored around it; only the meta parameters are truly
(re)initialized. A model already materialized (no meta parameters) is skipped
untouched -- these never clobber loaded or constructor-initialized weights.

Example::

    placement:
      fsdp: { wrap_modules: [TransformerBlock] }
      plugins:
        - module: common.plugin.init_weights
          class_name: ShardedInitWeights
          seed: 1019
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from ..distributed import ops
from ..model.placement import PlacementPlugin
from ..seed import local_seed


def _init_general(model: nn.Module, generator: torch.Generator | None) -> None:
    """Standard init for common weight-bearing layers.

    ``generator`` is used for the random draws (an explicit CPU generator on the
    non-fsdp path; ``None`` -- i.e. the caller-seeded default RNG -- on the fsdp
    path, so DTensor's RNG dispatch manages per-shard offsets). Parameters only;
    buffers are handled by the caller. Unknown/custom module types are left as-is.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, generator=generator)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                                 nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=generator)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm,
                                 nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # affine weight/bias are parameters (running stats are buffers, handled
            # by the caller's save/restore); guard for affine=False / no bias.
            if getattr(module, "weight", None) is not None:
                nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=generator)


def _materialize(model: nn.Module, device: torch.device) -> None:
    """``to_empty(device)`` the model's meta tensors while preserving the buffers
    that are already real.

    ``to_empty`` materializes parameters (including sharded DTensor shards) but
    also re-creates buffers as empty; instantiation (init_empty_weights) left
    buffers with real constructor values, so they are cloned first and copied back
    afterward. Parameters are left uninitialized here -- the caller inits them.
    """
    saved_buffers = {name: buf.detach().clone() for name, buf in model.named_buffers() if not buf.is_meta}
    model.to_empty(device=device)
    for name, buf in model.named_buffers():
        if name in saved_buffers:
            buf.copy_(saved_buffers[name].to(buf.device))


class InitWeights(PlacementPlugin):
    """Non-fsdp from-scratch init: materialize + init on CPU before the core
    ``.to(device)`` moves the model (``.to`` cannot move a meta tensor)."""

    def before_placement(self, state: dict[str, Any]) -> dict[str, Any]:
        model = state["model"]
        if not isinstance(model, nn.Module) or not any(p.is_meta for p in model.parameters()):
            return state  # nothing to place, or already materialized -- do not clobber
        _materialize(model, torch.device("cpu"))
        generator = torch.Generator()  # CPU, isolated from global RNG
        generator.manual_seed(self.config.get("seed", 0))
        _init_general(model, generator)
        return state


class ShardedInitWeights(PlacementPlugin):
    """FSDP from-scratch init: after ``fully_shard`` has produced meta DTensor
    shards, materialize each rank's shard on cuda and init via DTensor's RNG."""

    def after_placement(self, state: dict[str, Any]) -> dict[str, Any]:
        model = state["model"]
        if not isinstance(model, nn.Module) or not any(p.is_meta for p in model.parameters()):
            return state
        device = torch.device("cuda", ops.get_local_rank())
        _materialize(model, device)
        # DTensor RNG: seed the default generator identically on every rank so
        # DTensor's dispatch draws per-shard-unique, reproducible values (an
        # explicit generator triggers DTensor's deprecated rank0 RNG-sync).
        # local_seed does the seeding and restores every previous global RNG state
        # on exit, so the build-time global RNG is left undisturbed.
        with local_seed(self.config.get("seed", 0)):
            _init_general(model, generator=None)
        return state
