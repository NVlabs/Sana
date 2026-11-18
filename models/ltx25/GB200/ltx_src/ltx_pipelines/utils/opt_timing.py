"""Timing instrumentation. In-tree, and deliberately free of control flow.

An earlier external version materialised the video decoder's iterator to time it.
That changed when work happened, and every mgpu run afterwards deadlocked in
teardown and produced no file. Nothing here consumes an iterator, re-calls a stage,
or branches on anything: it reads a clock around calls that were going to happen
anyway.

Every per-step duration is written out, not just a total. torch.compile pays
compilation in the first steps and the step cache makes step cost deliberately
uneven, so no single mean is comparable across arms -- the steady-state window has
to be cut from the raw sequence afterwards.

Enabled by setting LTX_TIME_FILE. Results are flushed when the last stage reports,
not only at exit: the launcher stops a run once its compute is done, and a killed
process runs no atexit handler.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from collections.abc import Callable, Iterator
from typing import Any

_fwd: list[float] = []
_step: list[float] = []
_tail: list = []          # (stage name, seconds)
_shape: list[tuple] = []
_flushed: list[int] = []

# Request-scoped timing is deliberately separate from the legacy forward/step arrays.
# A resident server handles many requests in one worker process; the old timer flushed
# once at the first AudioDecoder and therefore recorded only request 1.  These fields
# are reset by begin_request(), which is called at the actual runner boundary.
_request_id = 0
_request_active = False
_request_started = 0.0
_request_stage_calls = 0
_request_stage: list[dict[str, Any]] = []
_request_tail: list[dict[str, Any]] = []
_request_video_iter = 0.0
_request_output = 0.0
_request_stage1_steps = 0


def _path() -> str | None:
    p = os.environ.get("LTX_TIME_FILE")
    return f"{p}.{os.getpid()}" if p else None


def _request_path() -> str | None:
    p = _path()
    return f"{p}.requests.jsonl" if p else None


def begin_request(stage1_steps: int) -> None:
    """Start one resident-server request in this worker process."""
    global _request_id, _request_active, _request_started, _request_stage_calls
    global _request_stage, _request_tail, _request_video_iter, _request_output
    global _request_stage1_steps

    if _path() is None:
        return
    if _request_active:
        raise RuntimeError("timing request started before the previous request ended")
    _request_id += 1
    _request_active = True
    _request_started = time.perf_counter()
    _request_stage_calls = 0
    _request_stage = []
    _request_tail = []
    _request_video_iter = 0.0
    _request_output = 0.0
    _request_stage1_steps = int(stage1_steps)


def _rank() -> int | None:
    try:
        import torch.distributed as dist

        return dist.get_rank() if dist.is_available() and dist.is_initialized() else None
    except Exception:
        return None


def end_request() -> None:
    """Write one JSONL record after the worker has completed the real request path."""
    global _request_active
    if not _request_active:
        return

    import torch

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - _request_started
    cache: dict[str, Any] | None = None
    try:
        from ltx_core.opt.step_cache import CTRL

        if CTRL.armed:
            cache = {
                "policy": CTRL.policy,
                "threshold": CTRL.threshold,
                "calls": CTRL.n_calls,
                "skipped": CTRL.n_skipped,
                "local_wanted": CTRL.n_local_skip,
            }
    except Exception:
        cache = None

    row = {
        "pid": os.getpid(),
        "rank": _rank(),
        "request": _request_id,
        "stage1_steps": _request_stage1_steps,
        "worker_total": elapsed,
        "stages": _request_stage,
        "tail": _request_tail,
        "video_iterator": _request_video_iter,
        "output_encode_total": _request_output,
        "cache": cache,
    }
    p = _request_path()
    if p:
        with open(p, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    stage_text = " ".join(f"{x['name']}={x['total']:.3f}s" for x in _request_stage)
    tail_text = " ".join(f"{x['name']}={x['total']:.3f}s" for x in _request_tail)
    cache_text = ""
    if cache is not None:
        cache_text = f" cache={cache['skipped']}/{cache['calls']}"
    print(
        f"[opt] request pid={os.getpid()} rank={_rank()} n={_request_id} "
        f"worker={elapsed:.3f}s {stage_text} {tail_text} "
        f"video_iter={_request_video_iter:.3f}s output={_request_output:.3f}s{cache_text}",
        flush=True,
    )
    _request_active = False


def time_output_call(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Time the existing output call without changing when its iterator is consumed."""
    global _request_output
    if not _request_active:
        return fn(*args, **kwargs)

    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    _request_output += time.perf_counter() - t0
    return out


def _time_video_iterator(it: Iterator[Any]) -> Iterator[Any]:
    """Lazily time ``next`` calls; never pre-consume or materialise the iterator."""
    global _request_video_iter
    import torch

    try:
        while True:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                item = next(it)
            except StopIteration:
                torch.cuda.synchronize()
                _request_video_iter += time.perf_counter() - t0
                return
            torch.cuda.synchronize()
            _request_video_iter += time.perf_counter() - t0
            yield item
    finally:
        close = getattr(it, "close", None)
        if close is not None:
            close()


def flush() -> None:
    if _flushed or not _fwd:
        return
    _flushed.append(1)
    if _step:
        # Denoiser-call time, one entry per denoising step. The forward timer brackets
        # LTXModel.forward only, and on the compiled arms it accounted for barely 40% of
        # the loop's measured wall -- so a ratio taken from it compared two different
        # fractions of the work. This one sits at the step boundary, outside everything
        # torch.compile adds around the forward.
        st = sum(_step)
        print(f"[opt] step pid={os.getpid()} steps={len(_step)} total={st:.3f}s "
              f"first={_step[0]:.3f}s excl_first={st - _step[0]:.3f}s", flush=True)
    total = sum(_fwd)
    line = (f"[opt] forward pid={os.getpid()} calls={len(_fwd)} total={total:.3f}s "
            f"first={_fwd[0]:.3f}s excl_first={total - _fwd[0]:.3f}s "
            f"shape={_shape[0] if _shape else '?'}")
    print(line, flush=True)
    # Skip counts, printed here rather than left to atexit. The launcher SIGKILLs the
    # process group once the last stage reports, and a killed process runs no atexit
    # handler -- so the one number that says what the cache actually did was riding on
    # a hook that does not fire. local_wanted vs skipped is the diagnostic that
    # separates "this rank rarely wanted to skip" from "the ranks disagreed".
    try:
        from ltx_core.opt.step_cache import CTRL

        if CTRL.armed:
            print(f"[opt] cache stats pid={os.getpid()} {CTRL.stats()}", flush=True)
    except Exception as e:
        print(f"[opt] cache stats unavailable: {type(e).__name__}: {e}", flush=True)
    p = _path()
    if p:
        with open(p, "w") as f:
            f.write(f"pid={os.getpid()}\ncalls={len(_fwd)}\ntotal={total:.3f}\n"
                    f"first={_fwd[0]:.3f}\nshape={_shape[0] if _shape else '?'}\n")
            f.write("all=" + ",".join(f"{d:.4f}" for d in _fwd) + "\n")
            f.write("step_all=" + ",".join(f"{d:.4f}" for d in _step) + "\n")
            f.write("tail=" + ",".join(f"{n}:{d:.4f}" for n, d in _tail) + "\n")


def install(blocks_module) -> None:
    if not os.environ.get("LTX_TIME_FILE"):
        return
    import torch

    import ltx_core.model.transformer.model as model_mod

    original_forward = model_mod.LTXModel.forward
    time_forwards = os.environ.get("LTX_TIME_FORWARD", "1") != "0"
    if time_forwards and not getattr(original_forward, "_ltx_timed", False):

        def timed(self, *args, **kwargs):
            if not _shape:
                # The shape actually being denoised. --height/--width are the FINAL
                # dimensions and the pipeline halves them for stage 1, so a run can
                # denoise a quarter of the intended pixels with every flag looking
                # right. Record the tensor; the flags are not evidence.
                for obj in list(args) + list(kwargs.values()):
                    t = obj if hasattr(obj, "shape") else None
                    if t is None and hasattr(obj, "__dict__"):
                        t = next((v for v in vars(obj).values() if hasattr(v, "shape")), None)
                    if t is not None:
                        _shape.append(tuple(t.shape))
                        print(f"[opt] stage-1 forward input {tuple(t.shape)}", flush=True)
                        break
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = original_forward(self, *args, **kwargs)
            torch.cuda.synchronize()
            _fwd.append(time.perf_counter() - t0)
            return out

        timed._ltx_timed = True
        model_mod.LTXModel.forward = timed

    # The last stage before teardown. A pure marker: no re-call, no iterator consumed.
    audio_decoder = getattr(blocks_module, "AudioDecoder", None)
    if audio_decoder is not None and not getattr(audio_decoder.__call__, "_ltx_marked", False):

        def make_marker(inner):
            # Bound through a factory, not closed over a local. A later loop in this same
            # function reused the name and this wrapper silently picked up the loop's last
            # value, so the marker ended up calling SimpleDenoiser.__call__ with
            # AudioDecoder's arguments. Nothing fails at install time; it fails at the end
            # of a generation, after every timing has already been recorded.
            def marked(self, *args, **kwargs):
                torch.cuda.synchronize()
                _t0 = time.perf_counter()
                out = inner(self, *args, **kwargs)
                torch.cuda.synchronize()
                _dt = time.perf_counter() - _t0
                _tail.append(("AudioDecoder", _dt))
                if _request_active:
                    _request_tail.append(
                        {"name": "AudioDecoder", "total": _dt, "load": 0.0, "compute": _dt}
                    )
                print(f"[opt] tail AudioDecoder={_dt:.3f}s", flush=True)
                flush()
                print(f"[opt] AudioDecoder done pid={os.getpid()}", flush=True)
                return out

            marked._ltx_marked = True
            return marked

        audio_decoder.__call__ = make_marker(audio_decoder.__call__)

    # The untimed tail. Each is wrapped through a factory so no wrapper closes over a
    # loop variable -- doing that once made the AudioDecoder marker call a denoiser.
    for stage in ("PromptEncoder", "VideoUpsampler", "VideoDecoder"):
        cls = getattr(blocks_module, stage, None)
        if cls is None or getattr(cls.__call__, "_ltx_tailtimed", False):
            continue

        def make_tail(inner, label):
            def tail_timed(self, *a, **kw):
                # Charge lazy construction to LOAD so the remainder is compute. The
                # builder is whichever attribute of this stage ends in "_builder".
                load = [0.0]
                for attr in [x for x in vars(self) if x.endswith("_builder")]:
                    b = getattr(self, attr)
                    bo = getattr(b, "build", None)
                    if bo is None or getattr(bo, "_ltx_loadtimed", False):
                        continue

                    def make_build(bo, sink):
                        def build_timed(*ba, **bkw):
                            torch.cuda.synchronize()
                            t = time.perf_counter()
                            m = bo(*ba, **bkw)
                            torch.cuda.synchronize()
                            sink[0] += time.perf_counter() - t
                            return m

                        build_timed._ltx_loadtimed = True
                        return build_timed

                    b.build = make_build(bo, load)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = inner(self, *a, **kw)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                _tail.append((label, dt - load[0]))
                if _request_active:
                    _request_tail.append(
                        {"name": label, "total": dt, "load": load[0], "compute": dt - load[0]}
                    )
                print(f"[opt] tail {label} total={dt:.3f}s load={load[0]:.3f}s "
                      f"compute={dt - load[0]:.3f}s pid={os.getpid()}", flush=True)
                if label == "VideoDecoder" and hasattr(out, "__next__"):
                    out = _time_video_iterator(out)
                return out

            tail_timed._ltx_tailtimed = True
            return tail_timed

        cls.__call__ = make_tail(cls.__call__, stage)

    # Time each full diffusion stage, including its hot per-request builder/context
    # overhead, with only one synchronization pair per stage.  The legacy step timer
    # is optional below because synchronizing every denoising step can perturb the
    # end-to-end latency we are trying to report.
    diffusion_stage = getattr(blocks_module, "DiffusionStage", None)
    if diffusion_stage is not None and not getattr(diffusion_stage.__call__, "_ltx_stagetimed", False):

        def make_stage(inner):
            def stage_timed(self, *a, **kw):
                global _request_stage_calls
                if not _request_active:
                    return inner(self, *a, **kw)
                _request_stage_calls += 1
                label = f"Stage{_request_stage_calls}"
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = inner(self, *a, **kw)
                torch.cuda.synchronize()
                _request_stage.append({"name": label, "total": time.perf_counter() - t0})
                return out

            stage_timed._ltx_stagetimed = True
            return stage_timed

        diffusion_stage.__call__ = make_stage(diffusion_stage.__call__)

    # One entry per denoising step, at the step boundary.
    from ltx_pipelines.utils import denoisers

    time_steps = os.environ.get("LTX_TIME_STEPS", "1") != "0"
    for name in ("FactoryGuidedDenoiser", "GuidedDenoiser", "SimpleDenoiser") if time_steps else ():
        cls = getattr(denoisers, name, None)
        if cls is None or getattr(cls.__call__, "_ltx_steptimed", False):
            continue
        def make_step(inner):
            def step_timed(self, *a, **kw):
                # *a/**kw rather than the concrete signature: the three denoiser classes
                # agree today, but a wrapper that hard-codes an arity breaks the moment
                # one of them does not.
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = inner(self, *a, **kw)
                torch.cuda.synchronize()
                _step.append(time.perf_counter() - t0)
                return out

            step_timed._ltx_steptimed = True
            return step_timed

        cls.__call__ = make_step(cls.__call__)

    atexit.register(flush)
