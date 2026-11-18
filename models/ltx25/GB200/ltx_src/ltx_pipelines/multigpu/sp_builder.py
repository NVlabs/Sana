"""Sequence parallel transformer builder.
Wrapping builder that produces a transformer model with sequence parallelism applied.
Requires ``ltx-kernels`` to be installed.
"""

from __future__ import annotations

from typing import Generic

import torch

from ltx_core.loader.primitives import ModelBuilderProtocol
from ltx_core.loader.registry import Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.model_protocol import LTXModelProtocol
from ltx_core.multigpu.transformer.attention import AttentionManager
from ltx_core.multigpu.transformer.sequence_parallel import (
    SequenceParallelModelWrapper,
    create_video_self_attention_module_ops,
)
from ltx_pipelines.multigpu.delegating_builder import DelegatingBuilder, InnerModelT
from ltx_pipelines.multigpu.weight_tracker import TransformerWeightTracker


class SequenceParallelBuilder(DelegatingBuilder[InnerModelT], Generic[InnerModelT]):
    """Builder that injects SP module ops and wraps with :class:`SequenceParallelModelWrapper`."""

    def __init__(
        self,
        inner: ModelBuilderProtocol[LTXModelProtocol],
        attn_mgr: AttentionManager,
        registry: Registry,
        tracker: TransformerWeightTracker,
    ) -> None:
        if not isinstance(inner, Builder):
            raise TypeError(f"SequenceParallelBuilder wraps a SingleGPUModelBuilder, got {type(inner).__name__}")
        cuda_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        inner = inner.with_registry(registry).with_lora_load_device(cuda_device)
        sp_ops = create_video_self_attention_module_ops(attn_mgr)
        self._inner = inner.with_module_ops((*inner.module_ops, sp_ops))
        self._tracker = tracker
        self._attn_mgr = attn_mgr

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        # Weight tracker keeps registry tensors GPU-resident and rebinds them across builds.
        return True

    @property
    def all2all_timeout_seconds(self) -> float:
        """The SP all2all barrier timeout (seconds); forwards to the AttentionManager that owns the refs."""
        return self._attn_mgr.all2all_timeout_seconds

    @all2all_timeout_seconds.setter
    def all2all_timeout_seconds(self, seconds: float) -> None:
        self._attn_mgr.all2all_timeout_seconds = seconds

    def build(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs: object
    ) -> SequenceParallelModelWrapper:
        model = self._tracker.build(self._inner, device=device, dtype=dtype, **kwargs)
        return SequenceParallelModelWrapper(model, self._attn_mgr)
