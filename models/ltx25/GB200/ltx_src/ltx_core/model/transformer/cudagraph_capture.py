"""Self-managed CUDA-graph capture of the transformer block loop.
Captures ``_process_transformer_blocks`` as one CUDA graph per (shape, perturbation signature)
and replays it from persistent static buffers. RoPE prepare / output projection stay eager.
Not thread-safe: :class:`CudaGraphRunner` (including the shared graph mempool) assumes a single
caller; concurrent capture/replay is unsupported.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING

import torch
import torch.utils._pytree as pytree

from ltx_core.guidance.perturbations import BatchedPerturbationConfig

if TYPE_CHECKING:
    # torch imports this under TYPE_CHECKING too (a runtime top-level import cycles).
    from torch.cuda import _POOL_HANDLE

    from ltx_core.model.transformer.transformer_args import BlockPerturbationsProcessor, TransformerArgs

_BlockLoopFn = Callable[
    ["TransformerArgs | None", "TransformerArgs | None", "BatchedPerturbationConfig"],
    "tuple[TransformerArgs | None, TransformerArgs | None]",
]
_ShapeKey = tuple[tuple[int, ...], ...]
# One capture identity: input shapes plus the block's perturbation graph signature (what the block
# specialises on), so a perturbed pass keys to -- and recaptures -- its own graph.
_CaptureKey = tuple[_ShapeKey, Hashable]


@dataclasses.dataclass
class _Capture:
    graph: torch.cuda.CUDAGraph
    # Persistent static buffers the graph reads/writes. Inputs are copied in, outputs cloned out.
    static_video: TransformerArgs | None
    static_audio: TransformerArgs | None
    static_perturbations: BatchedPerturbationConfig  # wraps the static block_masks buffer read in-graph
    static_video_out: TransformerArgs | None
    static_audio_out: TransformerArgs | None


def _shape_key(video: TransformerArgs | None, audio: TransformerArgs | None, block_masks: torch.Tensor) -> _ShapeKey:
    tensors = [*pytree.tree_leaves(video), *pytree.tree_leaves(audio), block_masks]
    return tuple(tuple(t.shape) for t in tensors if isinstance(t, torch.Tensor))


def _clone_value(value: object) -> object:
    """Clone a ``TransformerArgs`` field. Tensors and the rope ``(cos, sin)`` tuples are copied;
    bools and ``None`` are immutable and shared as-is."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(t.clone() for t in value)
    return value


def _copy_value(dst: object, src: object) -> None:
    """Copy ``src`` into ``dst`` in place. Non-tensor fields (bools/None) are baked into the captured
    graph and need no copy; tensors and the rope ``(cos, sin)`` tuples are copied element-wise."""
    if isinstance(dst, torch.Tensor):
        dst.copy_(src, non_blocking=True)
    elif isinstance(dst, tuple):
        for d, s in zip(dst, src, strict=True):
            d.copy_(s, non_blocking=True)


def _clone_args(args: TransformerArgs | None) -> TransformerArgs | None:
    """Clone args into fresh persistent buffers -- the one-time static-INPUT allocation in ``_capture``."""
    if args is None:
        return None
    return dataclasses.replace(args, **{f.name: _clone_value(getattr(args, f.name)) for f in dataclasses.fields(args)})


def _copy_args(dst: TransformerArgs | None, src: TransformerArgs | None) -> None:
    """Copy ``src``'s tensors into ``dst``'s persistent static buffers, field by field, in place."""
    if dst is None:
        return
    for f in dataclasses.fields(dst):
        _copy_value(getattr(dst, f.name), getattr(src, f.name))


class CudaGraphRunner:
    """Wrap ``_process_transformer_blocks(video, audio, perturbations) -> (video_out, audio_out)``
    with a per-(shape, perturbation) captured CUDA graph. Holds the video/audio ``TransformerArgs``
    (and the perturbation ``block_masks``) as persistent static buffers: each call's tensors are
    copied into the static inputs, the graph replayed, and the static OUTPUT buffers returned
    directly (standard CUDA-graph idiom -- the caller reads them, via the eager output projection,
    before the next replay overwrites them).
    Not thread-safe. Capture dict mutation, static-buffer copy-in/out, and the class-level shared
    mempool all assume a single-threaded caller; do not share a runner (or concurrent first-time
    captures across runners) across threads.
    """

    # Shared mempool across runners so captures reuse intermediate memory (lazy init; not locked).
    _shared_pool: _POOL_HANDLE | None = None

    def __init__(
        self, block_loop_fn: _BlockLoopFn, block_input_processor: BlockPerturbationsProcessor, warmup_iters: int = 3
    ) -> None:
        self._block_loop = block_loop_fn
        # Keys the capture on the block's perturbation graph signature -- the SAME processor the
        # wrapped block-loop uses, so a pass recaptures exactly when the block recompiles and the
        # signature can never diverge from the actual guards.
        self._block_input_processor = block_input_processor
        self._warmup_iters = warmup_iters
        self._captures: dict[_CaptureKey, _Capture] = {}

    def __call__(
        self, video: TransformerArgs | None, audio: TransformerArgs | None, perturbations: BatchedPerturbationConfig
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        key = self._cache_key(video, audio, perturbations)
        cap = self._captures.get(key)
        if cap is None:
            cap = self._capture(video, audio, perturbations)  # static inputs already hold this call's values
            self._captures[key] = cap
        else:
            # Copy this call's inputs into the static buffers the graph reads.
            _copy_args(cap.static_video, video)
            _copy_args(cap.static_audio, audio)
            cap.static_perturbations.block_masks.copy_(perturbations.block_masks, non_blocking=True)
        # Always replay (even right after capture); never return the capture-run outputs.
        cap.graph.replay()
        return cap.static_video_out, cap.static_audio_out

    def _cache_key(
        self, video: TransformerArgs | None, audio: TransformerArgs | None, perturbations: BatchedPerturbationConfig
    ) -> _CaptureKey:
        """Capture identity: input shapes + the block's perturbation graph signature. Keying on the
        signature makes a pass recapture exactly when it would recompile (and no more) -- the perturbed
        pass no longer silently replays the clean same-shape graph."""
        shape = _shape_key(video, audio, perturbations.block_masks)
        return shape, self._block_input_processor.graph_signature(perturbations)

    def _capture(
        self, video: TransformerArgs | None, audio: TransformerArgs | None, perturbations: BatchedPerturbationConfig
    ) -> _Capture:
        static_video = _clone_args(video)
        static_audio = _clone_args(audio)
        # Device masks are in-graph inputs; host mirror feeds eager any/all during warmup (may be None).
        static_perturbations = BatchedPerturbationConfig.from_masks(
            perturbations.block_masks.clone(), perturbations.block_masks_cpu
        )
        # Warmup on a side stream so any (first-shape) Dynamo/Inductor compile, autotune, cuBLAS
        # workspace, and IPC-kernel residency land OUTSIDE the capture. All ranks run this lockstep.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self._warmup_iters):
                self._block_loop(static_video, static_audio, static_perturbations)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()  # avoid double-counting cuBLAS workspace into the pool
        # Restore this call's inputs (warmup may have written into the static buffers) so the capture
        # records with correct values and the post-capture replay yields this call's output.
        _copy_args(static_video, video)
        _copy_args(static_audio, audio)
        static_perturbations.block_masks.copy_(perturbations.block_masks, non_blocking=True)
        if CudaGraphRunner._shared_pool is None:
            CudaGraphRunner._shared_pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=CudaGraphRunner._shared_pool, capture_error_mode="global"):
            video_out, audio_out = self._block_loop(static_video, static_audio, static_perturbations)
        return _Capture(graph, static_video, static_audio, static_perturbations, video_out, audio_out)
