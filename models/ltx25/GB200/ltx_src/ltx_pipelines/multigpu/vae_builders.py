"""Multi-GPU VAE decoder builder.
Wrapping builder that produces a :class:`DistributedVideoDecoder`.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.multiprocessing import Queue

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import BuilderProtocol
from ltx_core.loader.registry import Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.multigpu.vae.distributed_decoder import DistributedVideoDecoder
from ltx_core.tiling import TileCountConfig

if TYPE_CHECKING:
    from typing_extensions import Self


class DistributedDecoderBuilder(BuilderProtocol):
    """Builder that wraps a base decoder builder with distributed logic.
    Forwards ``module_ops`` / ``with_module_ops`` to the inner builder so
    pipeline flags like ``diffvae_optimization`` (``build_diffvae_mode_op``)
    still apply to the base ``DiffusionVideoDecoder`` before wrapping.
    """

    def __init__(
        self,
        inner: BuilderProtocol,
        queue: Queue,  # type: ignore[type-arg]
        vae_group: dist.ProcessGroup,
        vae_tiling: TileCountConfig,
        driver_rank: int,
        registry: Registry,
    ) -> None:
        self._inner = inner.with_registry(registry)
        self._queue = queue
        self._vae_group = vae_group
        self._vae_tiling = vae_tiling
        self._driver_rank = driver_rank

    @property
    def registry(self) -> Registry:
        return self._inner.registry

    def with_registry(self, registry: Registry) -> Self:
        clone = copy.copy(self)
        clone._inner = self._inner.with_registry(registry)
        return clone

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

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> DistributedVideoDecoder:
        base_decoder = self._inner.build(device=device, dtype=dtype, **kwargs)
        return DistributedVideoDecoder(
            base_decoder,
            queue=self._queue,
            vae_group=self._vae_group,
            vae_tiling=self._vae_tiling,
            driver_rank=dist.get_group_rank(self._vae_group, self._driver_rank),
        )
