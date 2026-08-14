"""O7 as a pipeline feature, not just a benchmark arm.

Measured on the stage-1 DiT (M=6144, single GPU):

    eager                       297.7 ms
    + kernels                   255.8
    + nvfp4                     269.0     <- NOT the GPU cost
    graph                       276.8
    graph + nvfp4               226.2
    graph + kernels + nvfp4     190.6

Every eager NVFP4 configuration lands on ~269 ms no matter what else is turned
on -- including one with 96 fewer quantizer launches per forward. That is a CPU
dispatch floor: NVFP4 trades a big cuBLAS call for a Triton quantize plus a
_scaled_mm plus Python, so it *adds* launches while removing GPU work, and once
the GPU finishes ahead of the CPU nothing downstream can show up. Under a graph
the same stack composes exactly as the per-linear numbers predict.

So the graph is what makes low precision mergeable at all, and it has to live
in the pipeline rather than in dit_bench.

Requirements it inherits: no CPU->GPU copies during capture (install_graph_
prereqs) and no data-dependent control flow, which is why the step cache cannot
be inside the captured region.
"""

from __future__ import annotations

import copy
import dataclasses
import os

import torch

_WARMUP = int(os.environ.get("LTX_GRAPH_WARMUP", "3"))
_MIN_TOKENS = int(os.environ.get("LTX_GRAPH_MIN_TOKENS", "1024"))
STATS = {"capture": 0, "replay": 0, "bypass": 0, "keys": []}


def _tensor_fields(obj):
    if dataclasses.is_dataclass(obj):
        names = [f.name for f in dataclasses.fields(obj)]
    else:
        names = [n for n in vars(obj)]
    return [n for n in names if isinstance(getattr(obj, n, None), torch.Tensor)]


def _sig(obj):
    if obj is None:
        return None
    parts = []
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            parts.append((f.name, tuple(v.shape), v.dtype) if isinstance(v, torch.Tensor)
                         else (f.name, v if isinstance(v, bool) else v is None))
    else:
        for n in sorted(vars(obj)):
            v = getattr(obj, n)
            if isinstance(v, torch.Tensor):
                parts.append((n, tuple(v.shape), v.dtype))
            elif v is None:
                parts.append((n, None))
    return tuple(parts)


def _static_copy(obj):
    """A private, address-stable clone the graph can be captured against."""
    if obj is None:
        return None
    if dataclasses.is_dataclass(obj):
        upd = {n: getattr(obj, n).clone() for n in _tensor_fields(obj)}
        return dataclasses.replace(obj, **upd)
    new = copy.copy(obj)
    for n in _tensor_fields(obj):
        setattr(new, n, getattr(obj, n).clone())
    return new


def _fill(static, live):
    if static is None or live is None:
        return
    for n in _tensor_fields(static):
        getattr(static, n).copy_(getattr(live, n), non_blocking=True)


class _Entry:
    __slots__ = ("g", "v", "a", "p", "out", "keep")

    def __init__(self, g, v, a, p, out, keep=()):
        self.g, self.v, self.a, self.p, self.out = g, v, a, p, out
        # Derived tensors the captured kernels point at. NVFP4CastLinear drops its
        # fp4 copy in _apply (every .to()/dispose goes through it), which would
        # free memory a live graph still references -- replaying that is a
        # segfault, not an error. Pin them to the graph's lifetime.
        self.keep = keep


def _clone_out(out):
    # Outputs live in the graph's private pool and are overwritten by the next
    # replay, so callers must never see them directly.
    if isinstance(out, tuple):
        return tuple(o.clone() if isinstance(o, torch.Tensor) else o for o in out)
    return out.clone() if isinstance(out, torch.Tensor) else out


def _capture(orig, self, video, audio, perturbations):
    # Capture is illegal while anything still copies CPU->GPU (the resident
    # modulation tables and the rope index vector); idempotent, so it is safe
    # to run again per capture.
    from ltx_core.opt.ours import install_graph_prereqs
    install_graph_prereqs(self)
    sv, sa, sp = (_static_copy(video), _static_copy(audio), _static_copy(perturbations))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(_WARMUP):
            orig(self, sv, sa, sp)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = orig(self, sv, sa, sp)
    torch.cuda.synchronize()
    STATS["capture"] += 1
    keep = [m._nvfp4 for m in self.modules() if getattr(m, "_nvfp4", None) is not None]
    return _Entry(g, sv, sa, sp, out, tuple(keep))


def install(model_mod) -> int:
    """Patch LTXModel.forward to capture-once / replay-per-step, keyed on the
    input signature (stage 1 and stage 2 get their own graph)."""
    cls = model_mod.LTXModel
    orig = cls.forward
    if getattr(orig, "_ltx_graphed", False):
        return 0

    def forward(self, video, audio, perturbations):
        ref = video if video is not None else audio
        if ref is None or ref.latent.shape[1] < _MIN_TOKENS:
            STATS["bypass"] += 1
            return orig(self, video, audio, perturbations)
        store = self.__dict__.setdefault("_ltx_graphs", {})
        key = (_sig(video), _sig(audio), _sig(perturbations))
        ent = store.get(key)
        if ent is None:
            try:
                ent = _capture(orig, self, video, audio, perturbations)
            except Exception as e:
                # A failed capture must not silently become a slow path that
                # still reports as "graphed" -- say so, once, and bypass.
                print(f"[graph] capture FAILED, running eager: "
                      f"{type(e).__name__}: {str(e)[:200]}", flush=True)
                if os.environ.get("LTX_GRAPH_TRACE") == "1":
                    _trace_host_copies(orig, self, video, audio, perturbations)
                store[key] = False
                return orig(self, video, audio, perturbations)
            store[key] = ent
            STATS["keys"].append(str(ref.latent.shape))
        if ent is False:
            STATS["bypass"] += 1
            return orig(self, video, audio, perturbations)
        _fill(ent.v, video)
        _fill(ent.a, audio)
        _fill(ent.p, perturbations)
        ent.g.replay()
        STATS["replay"] += 1
        return _clone_out(ent.out)

    forward._ltx_graphed = True
    cls.forward = forward
    return 1


def _trace_host_copies(orig, self, video, audio, perturbations):
    """Capture dies on the FIRST host->device copy and names neither the tensor
    nor the call site. Re-run the forward eagerly with .to() instrumented so the
    remaining offender is identified instead of guessed at."""
    import traceback
    orig_to = torch.Tensor.to
    hits = []

    def traced(self_t, *a, **k):
        dev = k.get("device") or (a[0] if a and not isinstance(a[0], torch.dtype) else None)
        if dev is not None and not self_t.is_cuda and "cuda" in str(dev) and len(hits) < 4:
            hits.append(1)
            print(f"[graph] host->device copy {tuple(self_t.shape)} {self_t.dtype}",
                  flush=True)
            for ln in traceback.format_stack()[-6:-1]:
                print("[graph]    " + ln.strip().replace("\n", " | "), flush=True)
        return orig_to(self_t, *a, **k)

    torch.Tensor.to = traced
    try:
        orig(self, video, audio, perturbations)
    finally:
        torch.Tensor.to = orig_to
    if not hits:
        print("[graph] no host->device copy seen; capture failed for another reason",
              flush=True)
