"""Base class for multigpu transformer builders that delegate to an inner builder.
Shared boilerplate for SP and TDP builders — both wrap a
:class:`SingleGPUModelBuilder` and forward the ``ModelBuilderProtocol`` surface.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from ltx_core.loader.fuse_loras import FuseRule
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder

if TYPE_CHECKING:
    from typing_extensions import Self

InnerModelT = TypeVar("InnerModelT", bound=torch.nn.Module)


class DelegatingBuilder(Generic[InnerModelT]):
    """Thin wrapper that delegates all ``ModelBuilderProtocol`` accessors to *inner*.
    ``InnerModelT`` is the type produced by the inner builder.  Subclasses only
    need to implement ``__init__`` and ``build`` (whose return type may differ).
    """

    _inner: Builder[InnerModelT]

    # -- delegated properties / with_* methods --------------------------------

    @property
    def checkpoint(self) -> str | tuple[str, ...]:
        return self._inner.checkpoint

    @property
    def model_sd_ops(self) -> SDOps | None:
        return self._inner.model_sd_ops

    def with_sd_ops(self, sd_ops: SDOps | None) -> Self:
        clone = copy.copy(self)
        clone._inner = self._inner.with_sd_ops(sd_ops)
        return clone

    @property
    def module_ops(self) -> tuple[ModuleOps, ...]:
        return self._inner.module_ops

    def with_module_ops(self, module_ops: tuple[ModuleOps, ...]) -> Self:
        clone = copy.copy(self)
        clone._inner = self._inner.with_module_ops(module_ops)
        return clone

    @property
    def loras(self) -> tuple[LoraPathStrengthAndSDOps, ...]:
        return self._inner.loras

    def with_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> Self:
        clone = copy.copy(self)
        clone._inner = self._inner.with_loras(loras)
        return clone

    @property
    def registry(self) -> Registry:
        return self._inner.registry

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        return self._inner.keeps_gpu_resident_weights

    def with_registry(self, registry: Registry) -> Self:
        clone = copy.copy(self)
        clone._inner = self._inner.with_registry(registry)
        return clone

    def with_lora_load_device(self, device: torch.device) -> Self:
        clone = copy.copy(self)
        clone._inner = self._inner.with_lora_load_device(device)
        return clone

    def with_fuse_rule(self, fuse_rule: FuseRule) -> Self:
        clone = copy.copy(self)
        clone._inner = self._inner.with_fuse_rule(fuse_rule)
        return clone

    def model_config(self) -> dict:
        return self._inner.model_config()
