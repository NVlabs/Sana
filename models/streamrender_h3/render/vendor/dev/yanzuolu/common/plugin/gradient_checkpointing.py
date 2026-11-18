"""Gradient checkpointing runtime plugins.

Ecosystems expose gradient checkpointing through different entry-point APIs --
transformers' ``gradient_checkpointing_enable(gradient_checkpointing_kwargs=...)``,
diffusers' no-argument ``enable_gradient_checkpointing()``, and the
``set_gradient_checkpointing(**kwargs)`` that vendored model code in this repo
implements natively -- so each gets its own plugin class rather than one
branching on library.

Like every runtime plugin this also runs on the EMA by default; checkpointing
is inert on the eval-mode EMA, so that is harmless (``ema: {enabled: false}``
opts out).

Example::

    runtime:
      plugins:
        - module: common.plugin.gradient_checkpointing
          class_name: TransformersGradientCheckpointing
          use_reentrant: false
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from torch import nn
from torch.distributed.fsdp._fully_shard import _fsdp_state

from ..model.runtime import RuntimePlugin


def _backport_fsdp_checkpoint_recompute_unshard() -> None:
    """Apply PyTorch PR 171779 when the installed FSDP2 predates it."""
    current = _fsdp_state.FSDPState._pre_forward
    if getattr(current, "_checkpoint_recompute_unshards", False):
        return
    original = inspect.unwrap(current)
    if "is_unsharded" in original.__code__.co_names:
        return

    @functools.wraps(original)
    def _pre_forward(self, module, args, kwargs):
        if self._training_state == _fsdp_state.TrainingState.PRE_BACKWARD:
            group = self._fsdp_param_group
            if group is not None and not group.is_unsharded:
                group.unshard()
                group.wait_for_unshard()
        return original(self, module, args, kwargs)

    patched = _fsdp_state.disable_if_config_true(_pre_forward)
    patched._checkpoint_recompute_unshards = True
    _fsdp_state.FSDPState._pre_forward = patched


class TransformersGradientCheckpointing(RuntimePlugin):
    """Enable transformers-style gradient checkpointing (after the core runtime)."""

    def after_runtime(self, state: dict[str, Any]) -> dict[str, Any]:
        _backport_fsdp_checkpoint_recompute_unshard()
        kwargs = {k: v for k, v in self.config.items() if k not in ("module", "class_name")}
        state["model"].gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs or None)
        return state


class DiffusersGradientCheckpointing(RuntimePlugin):
    """Enable diffusers-style gradient checkpointing (after the core runtime)."""

    def after_runtime(self, state: dict[str, Any]) -> dict[str, Any]:
        _backport_fsdp_checkpoint_recompute_unshard()
        state["model"].enable_gradient_checkpointing()
        return state


class NativeGradientCheckpointing(RuntimePlugin):
    """Call a model's own ``set_gradient_checkpointing(**kwargs)``.

    The third entry-point API, and the one the vendored model code in this repo
    implements itself, so it takes whatever keyword arguments that model chose
    to expose (``enable``, ``gc_start_idx``, ``gc_step``, ...) straight from the
    plugin config. Nothing here is specific to a model family -- the only
    requirement is that the method exists.
    """

    def after_runtime(self, state: dict[str, Any]) -> dict[str, Any]:
        _backport_fsdp_checkpoint_recompute_unshard()
        model = state["model"]
        if not isinstance(model, nn.Module):
            raise TypeError(
                "NativeGradientCheckpointing requires state['model'] to be a torch.nn.Module, "
                f"got {type(model).__name__}"
            )

        set_gradient_checkpointing = getattr(model, "set_gradient_checkpointing", None)
        if not callable(set_gradient_checkpointing):
            raise TypeError(
                "NativeGradientCheckpointing requires the model to implement callable "
                "set_gradient_checkpointing(...)"
            )

        kwargs = {k: v for k, v in self.config.items() if k not in ("module", "class_name")}
        set_gradient_checkpointing(**kwargs)
        return state
