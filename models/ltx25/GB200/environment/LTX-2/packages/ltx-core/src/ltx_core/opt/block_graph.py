"""Per-BLOCK CUDA graphs, so the step cache can still choose what runs.

The first attempt captured LTXModel.forward whole, which baked whatever the
capture step decided about skipping into the graph and made graph and cache
mutually exclusive. That was a consequence of the capture granularity, not a
real conflict: within one denoising step the only tensors that change from
block to block are video.x and audio.x -- timesteps, positions, context and the
perturbation masks are constant across the 48 blocks. So capture each block
against one persistent (SV, SA) pair and the skip decision stays on the host,
where the cache already makes it. Skipping block k is then just "do not replay
graph k".

Cost of the granularity: 48 replays instead of 1 (~7 us each, 0.15% of a 220 ms
step) plus one copy of the hidden state per block to write back into the
persistent buffer (~0.6 ms/step, 0.3%). Both are far below what the cache buys.
"""

from __future__ import annotations

import dataclasses
import os

import torch

STATS = {"captured": 0, "replays": 0, "replays_bf16": 0, "bypass": 0,
         "fail": None}
_MIN_TOKENS = int(os.environ.get("LTX_BGRAPH_MIN_TOKENS", "1024"))


class _BlockGraphs:
    """One entry per (model, input signature). Holds the persistent hidden-state
    buffers, the per-step constant statics, and one graph per block."""

    __slots__ = ("sv", "sa", "vmod", "amod", "pert", "graphs", "graphs_bf16",
                 "keep", "sig")

    def __init__(self):
        self.sv = self.sa = None
        self.vmod = self.amod = self.pert = None
        self.graphs = []
        self.graphs_bf16 = []
        self.keep = []
        self.sig = None


def _sig(video, audio, pert):
    def one(m):
        if m is None:
            return None
        return tuple((f.name, tuple(v.shape), v.dtype) if isinstance(v := getattr(m, f.name), torch.Tensor)
                     else (f.name, v if isinstance(v, bool) else v is None)
                     for f in dataclasses.fields(m))
    pm = getattr(pert, "_block_masks", None)
    return (one(video), one(audio),
            None if pm is None else (tuple(pm.shape), pm.dtype))


def _static(m):
    if m is None:
        return None
    upd = {f.name: getattr(m, f.name).clone()
           for f in dataclasses.fields(m)
           if isinstance(getattr(m, f.name), torch.Tensor)}
    return dataclasses.replace(m, **upd)


def _static_pert(pert):
    """copy.copy + clone the mask tensor: it is not a dataclass, so _static's
    dataclasses.fields() walk does not reach it."""
    if pert is None:
        return None
    import copy as _copy
    new = _copy.copy(pert)
    m = getattr(pert, "_block_masks", None)
    if isinstance(m, torch.Tensor):
        new._block_masks = m.clone()
    return new


def _fill_pert(static, live):
    if static is None or live is None:
        return
    m, lm = getattr(static, "_block_masks", None), getattr(live, "_block_masks", None)
    if isinstance(m, torch.Tensor) and isinstance(lm, torch.Tensor):
        m.copy_(lm, non_blocking=True)


def _fill(static, live):
    if static is None or live is None:
        return
    for f in dataclasses.fields(static):
        v = getattr(static, f.name)
        if isinstance(v, torch.Tensor):
            v.copy_(getattr(live, f.name), non_blocking=True)


def build(self, video, audio, perturbations, warmup=2):
    """Capture one graph per transformer block. Raises on failure -- the caller
    decides whether to fall back, and must say so rather than silently running
    eager while reporting as graphed."""
    from ltx_core.opt.ours import install_graph_prereqs
    from ltx_core.guidance.perturbations import PerturbationType

    install_graph_prereqs(self)
    e = _BlockGraphs()
    e.sig = _sig(video, audio, perturbations)
    e.vmod, e.amod = _static(video), _static(audio)
    e.pert = _static_pert(perturbations)
    e.sv = video.x.clone() if video is not None else None
    e.sa = audio.x.clone() if audio is not None else None

    def one_block(block_idx, block):
        v = dataclasses.replace(e.vmod, x=e.sv) if e.vmod is not None else None
        a = dataclasses.replace(e.amod, x=e.sa) if e.amod is not None else None
        if v is not None:
            v = self.block_input_processor(
                v, e.pert, block_idx,
                self_attn_type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                cross_attn_type=PerturbationType.SKIP_A2V_CROSS_ATTN)
        if a is not None:
            a = self.block_input_processor(
                a, e.pert, block_idx,
                self_attn_type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                cross_attn_type=PerturbationType.SKIP_V2A_CROSS_ATTN)
        v, a = block(video=v, audio=a)
        if v is not None:
            e.sv.copy_(v.x)
        if a is not None:
            e.sa.copy_(a.x)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            for i, b in enumerate(self.transformer_blocks):
                one_block(i, b)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    from ltx_core.opt import nvfp4 as _nv

    pool = None
    # The bf16 set only gets replayed when the step guard is on; capturing it
    # otherwise doubles capture time for graphs nothing will ever select.
    _sets = [("fp4", e.graphs)]
    if _nv.GUARD["first"] or _nv.GUARD["last"]:
        _sets.append(("bf16", e.graphs_bf16))
    for mode, dest in _sets:
        _nv.FORCE["mode"] = mode
        # re-warm per mode: the two precisions run different kernels and the
        # first call of each would otherwise JIT inside the capture
        side2 = torch.cuda.Stream()
        side2.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side2):
            for i, b in enumerate(self.transformer_blocks):
                one_block(i, b)
        torch.cuda.current_stream().wait_stream(side2)
        torch.cuda.synchronize()
        for i, b in enumerate(self.transformer_blocks):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                one_block(i, b)
            pool = g.pool()
            dest.append(g)
        torch.cuda.synchronize()
    _nv.FORCE["mode"] = None
    # fp4 weights are derived state that NVFP4CastLinear drops in _apply; a live
    # graph still points at them, and replaying after they are freed segfaults.
    e.keep = [m._nvfp4 for m in self.modules() if getattr(m, "_nvfp4", None) is not None]
    STATS["captured"] += 1
    return e


def load_step(e, video, audio, perturbations=None):
    """Per-step: refresh the constants and seed the hidden state."""
    _fill(e.vmod, video)
    _fill(e.amod, audio)
    _fill_pert(e.pert, perturbations)
    if e.sv is not None:
        e.sv.copy_(video.x, non_blocking=True)
    if e.sa is not None:
        e.sa.copy_(audio.x, non_blocking=True)


def replay(e, idx, dense=False):
    """dense=True replays the bf16 set. Falls back to the fp4 set only if the
    bf16 set is missing, and says so rather than silently running fp4."""
    if dense:
        if not e.graphs_bf16:
            if STATS["fail"] is None:
                STATS["fail"] = "bf16 graph set missing"
                print("[bgraph] bf16 set missing, guarded step ran FP4", flush=True)
        else:
            e.graphs_bf16[idx].replay()
            STATS["replays"] += 1
            STATS["replays_bf16"] += 1
            return
    e.graphs[idx].replay()
    STATS["replays"] += 1


def current(e, video, audio):
    """Wrap the persistent buffers back into Modalities for the caller."""
    v = dataclasses.replace(video, x=e.sv) if video is not None else None
    a = dataclasses.replace(audio, x=e.sa) if audio is not None else None
    return v, a
