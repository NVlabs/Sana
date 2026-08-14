"""Batch-parallel Gemma text encoder builder.
Each rank materialises a full :class:`LTXGemmaTextEncoder` on its own local
CUDA device via the standard :class:`SingleGPUModelBuilder` pipeline (the
same code path used by the non-MGPU pipelines). No Accelerate, no
``device_map``, no per-layer dispatch hooks.
Every ``build()`` reconstructs the encoder through
:class:`SingleGPUModelBuilder`: a fresh meta module is created, the
family-specific model-ops chain from :func:`get_gemma_ops` re-runs
(recomputing the rotary / position buffers that live outside the
safetensors file), and the trained weights
are bound from the provided :class:`Registry`. The registry caches the
loaded state dict so subsequent calls skip disk I/O while still rebuilding
the module tree -- mirroring the rebuild logic of
:class:`AccelerateGemmaBuilder` on this branch. Encoder-instance caching is
intentionally left out; it will arrive later as a global builder refactor.
The result is wrapped in :class:`BatchParallelGemmaWrapper`, which
partitions prompt lists across ranks in ``encode`` and routes
non-deterministic sampling (``enhance_t2v`` / ``enhance_i2v``) through a
single ``src_rank``.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from ltx_core.loader.primitives import BuilderProtocol
from ltx_core.loader.registry import Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.multigpu.gemma.batch_parallel_wrapper import BatchParallelGemmaWrapper
from ltx_core.text_encoders.gemma import (
    GemmaTextEncoderConfigurator,
    get_gemma_ops,
    resolve_gemma_weight_paths,
)

logger = logging.getLogger(__name__)


class BatchParallelGemmaBuilder(BuilderProtocol):
    """Per-rank Gemma replica builder for the batch-parallel encode path.
    Mirrors the inline Gemma builder construction inside
    :class:`PromptEncoder` (single-GPU path) and adds the MGPU wiring --
    broadcast group + source rank for non-deterministic methods. Each
    ``build()`` reconstructs the encoder via :class:`SingleGPUModelBuilder`;
    the registry caches the state dict so only disk I/O is skipped across
    calls.
    """

    def __init__(
        self,
        gemma_root_path: str,
        broadcast_group: dist.ProcessGroup | None,
        registry: Registry,
        *,
        src_rank: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        weight_paths = resolve_gemma_weight_paths(gemma_root_path)
        gemma_sd_ops, gemma_module_ops = get_gemma_ops(gemma_root_path)
        self._inner = Builder(
            model_path=weight_paths,
            model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(gemma_root_path),
            model_sd_ops=gemma_sd_ops,
            module_ops=gemma_module_ops,
            registry=registry,
        )
        self._broadcast_group = broadcast_group
        self._src_rank = src_rank
        self._dtype = dtype

    def model_config(self) -> dict:
        return {}

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BatchParallelGemmaWrapper:
        dtype = dtype or self._dtype
        encoder = self._inner.build(device=device, dtype=dtype).eval()
        return BatchParallelGemmaWrapper(
            encoder=encoder,
            broadcast_group=self._broadcast_group,
            src_rank=dist.get_group_rank(self._broadcast_group, self._src_rank),
            dtype=dtype,
            device=device,
        )
