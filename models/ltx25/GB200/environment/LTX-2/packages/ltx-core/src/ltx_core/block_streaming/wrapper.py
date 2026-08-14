"""Block streaming wrapper: streams transformer blocks through a WeightsProvider."""

from __future__ import annotations

import itertools
from typing import Any

import torch
from torch import nn

from ltx_core.block_streaming.provider import WeightsProvider
from ltx_core.block_streaming.utils import assign_tensor_to_module
from ltx_core.model.disposable import Disposable


class BlockStreamingWrapper(nn.Module, Disposable):
    """Streams sequential model blocks through GPU buffer caches.
    The wrapper delegates all weight management to a :class:`WeightsProvider`
    which handles CPU-to-GPU copies, caching, LoRA fusion, and stream
    synchronization internally.
    Use :class:`StreamingModelBuilder` to construct this wrapper -- it
    handles checkpoint parsing, source selection, and provider creation.
    Args:
        model: The wrapped model (non-block params already on GPU).
        blocks: Sequential blocks to stream (``nn.ModuleList``).
        provider: Provides GPU-ready weights on demand.
        target_device: GPU device for compute.
    """

    def __init__(
        self,
        model: nn.Module,
        blocks: nn.ModuleList,
        provider: WeightsProvider,
        target_device: torch.device,
    ) -> None:
        super().__init__()
        self._model = model
        self._blocks = blocks
        self._target_device = target_device
        self._provider = provider

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    @property
    def num_blocks(self) -> int:
        return self._model.num_blocks

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _pre_hook(self, block_idx: int) -> None:
        """Load GPU weights for a block and inject them into its parameters."""
        gpu_weights = self._provider.get(block_idx)

        block = self._blocks[block_idx]
        for name, _param in itertools.chain(block.named_parameters(), block.named_buffers()):
            assign_tensor_to_module(block, name, gpu_weights[name])

    def _post_hook(self, block_idx: int) -> None:
        """Release the block weights once its forward pass has been enqueued.
        The provider guards the buffer against reuse until this block's compute
        completes."""
        self._provider.mark_block_done(block_idx)

    def _register_hooks(self) -> None:
        for idx, block in enumerate(self._blocks):
            pre = block.register_forward_pre_hook(
                lambda _mod, _args, *, idx=idx: self._pre_hook(idx),
            )
            post = block.register_forward_hook(
                lambda _mod, _args, _out, *, idx=idx: self._post_hook(idx),
            )
            self._hooks.extend([pre, post])

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        """Remove hooks and release all resources."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._provider.cleanup()

    # ------------------------------------------------------------------
    # Forward and attribute delegation
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Proxy attribute access to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)
