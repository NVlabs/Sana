"""Stage-level timing for the whole pipeline, not just the denoising loops.

LTX_TIME only covers euler_denoising_loop. The VAE decode is a DIFFUSION decoder
(NADiffusionDecoder) and was never measured at all, so every "end-to-end" number so
far has been denoise-only. Hot inference time is the metric, so warmup/loading must
stay outside these spans -- each span here is wall time of the call itself, and the
model build happens before the first span.
"""
import os
import time

import torch

_T = {}
_FILE = os.environ.get("STAGE_TIME_FILE")


def _rec(name, dt):
    _T[name] = _T.get(name, 0.0) + dt
    line = f"[STAGE] {name}={dt:.3f}s (cum {_T[name]:.3f}s) pid={os.getpid()}"
    print(line, flush=True)
    if _FILE:
        with open(_FILE, "a") as f:
            f.write(line + "\n")


def install():
    import ltx_pipelines.utils.blocks as B

    for cls_name in ("VideoDecoder", "AudioDecoder", "PromptEncoder", "VideoUpsampler"):
        cls = getattr(B, cls_name, None)
        if cls is None or getattr(cls.__call__, "_timed", False):
            continue
        orig = cls.__call__

        def make(orig, nm):
            def timed(self, *a, **kw):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = orig(self, *a, **kw)
                # PromptEncoder's first __call__ also builds/loads gemma (24 GB off
                # Lustre). Time a second identical call to separate load from compute:
                # first = load + compute, second = compute.
                # Every one of these builds its model lazily on first __call__, so the
                # first span is load+compute. Hot inference time is the metric, so run a
                # second identical call and report that as the compute-only number.
                if os.environ.get("STAGE_SPLIT_LOAD") == "1" \
                        and not getattr(self, "_split_done", False):
                    self._split_done = True
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    _o2 = orig(self, *a, **kw)
                    if nm == "VideoDecoder" and hasattr(_o2, "__iter__") and not torch.is_tensor(_o2):
                        list(_o2)
                    torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    _rec(nm + "_HOT", time.perf_counter() - t1)
                # decoders return generators/iterators -- force them so the time lands here
                if nm == "VideoDecoder":
                    out = list(out) if hasattr(out, "__iter__") and not torch.is_tensor(out) else out
                torch.cuda.synchronize()
                _rec(nm, time.perf_counter() - t0)
                return out
            timed._timed = True
            return timed

        cls.__call__ = make(orig, cls_name)
    print("[STAGE] instrumented VideoDecoder/AudioDecoder/PromptEncoder/VideoUpsampler", flush=True)
