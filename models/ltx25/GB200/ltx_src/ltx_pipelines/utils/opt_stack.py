"""Optimizations that are not vendor flags. Installed at import time from blocks.py.

Selected by environment so one build serves every arm of a comparison:

  LTX_S1_PARALLEL=cfg        stage-1 CFG parallelism. Wired at its two call sites
                             (ti2vid_two_stages_mgpu.py and blocks.py), not here --
                             it replaces a builder and lifts a batch limit, and both
                             belong where those decisions are made.
  LTX_STACK_CACHE=<policy>   fbcache | teacache | easycache, stage 1 only.

The cache does NOT compose with the vendor's whole-loop CUDA graph
(``--compile capture=true``): its skip decision reads a GPU scalar back to the host,
which is illegal mid-capture, and a captured graph would in any case bake in whatever
the capture step happened to decide. Per-block ``torch.compile`` (``capture=false``)
composes fine -- that is the pairing that measured 2.02x.
"""

from __future__ import annotations

import os


def begin_request() -> None:
    """Reset request-local optimization state inside each fleet worker.

    The warm-server driver lives in a different process from the workers, so resetting
    ``CTRL`` in the driver only resets an unused controller.  The runner calls this at
    the real worker request boundary, before prompt encoding and denoising begin.
    """
    try:
        from ltx_core.opt.step_cache import CTRL

        if CTRL.armed:
            CTRL.reset()
    except ImportError:
        # The optimization package is optional when the cache is disabled.
        return


def install_cache() -> None:
    policy = os.environ.get("LTX_STACK_CACHE", "off")
    if policy in ("", "off", "none"):
        return
    if policy not in ("fbcache", "teacache", "easycache"):
        raise ValueError(
            f"unknown LTX_STACK_CACHE={policy!r}; step_cache implements only "
            "fbcache / teacache / easycache, and anything else would arm the "
            "controller with a signal it does not compute"
        )

    import ltx_core.model.transformer.model as model_mod
    from ltx_core.opt.step_cache import CTRL, install

    install(model_mod)
    CTRL.configure(
        policy=policy,
        threshold=float(os.environ.get("LTX_CACHE_THRESH", "0.06")),
        warmup=int(os.environ.get("LTX_CACHE_WARMUP", "1")),
        max_consecutive=int(os.environ.get("LTX_CACHE_MAXCONSEC", "3")),
    )
    CTRL.reset()
    CTRL.armed = True
    _install_step_driver(CTRL)

    # Per-rank skip counts at exit. The pre-release tree added these for a reason worth
    # keeping: single-GPU fbcache at 0.06 skipped 30% and gained 1.40x, but the 4-GPU
    # stack gained only 1.18x -- consistent with a ~15% effective rate. A skip needs
    # every rank to agree (_unified takes the cross-rank MIN), so either the per-rank
    # rate fell or the ranks disagreed, and those two have very different fixes. Without
    # the counts, which one it is stays a guess.
    import atexit

    atexit.register(
        lambda: print(f"[opt] cache stats pid={os.getpid()} {CTRL.stats()}", flush=True)
    )
    print(
        f"[opt] step cache: policy={policy} "
        f"thresh={os.environ.get('LTX_CACHE_THRESH', '0.06')} "
        f"maxconsec={os.environ.get('LTX_CACHE_MAXCONSEC', '3')}",
        flush=True,
    )


def _install_step_driver(ctrl) -> None:
    """Advance the cache's step counter once per denoising step.

    NOT once per transformer forward. ``begin_step`` resets the pass counter, so when
    the guidance batch is split into four forwards the reset fired four times per step:
    all four guidance passes shared one cache slot, the signal compared cond against
    uncond instead of comparing adjacent timesteps, the skip therefore always fired,
    and the reused residual reproduced the computed pass exactly -- making
    ``cond - uncond`` identically zero. That run denoised with no guidance at all and
    reported a 5x speedup for it.

    ``euler_denoising_loop`` passes ``step_index`` to the denoiser explicitly, so this
    is the one place where a step is defined by the code instead of inferred from how
    many times something downstream happened to be called.
    """
    from ltx_pipelines.utils import denoisers

    hooked = []
    for name in ("FactoryGuidedDenoiser", "GuidedDenoiser", "SimpleDenoiser"):
        cls = getattr(denoisers, name, None)
        if cls is None or getattr(cls.__call__, "_ltx_step_driver", False):
            continue
        original = cls.__call__

        # In the public two-stage TI2Vid pipeline, guided denoisers are Stage 1
        # and SimpleDenoiser is the unchanged Stage 2 refiner.
        stage = 2 if name == "SimpleDenoiser" else 1

        def make(original, stage):
            def stepped(self, transformer, video_state, audio_state, sigmas, step_index):
                ctrl.begin_step(step_index, stage=stage)
                return original(self, transformer, video_state, audio_state, sigmas, step_index)

            stepped._ltx_step_driver = True
            return stepped

        cls.__call__ = make(original, stage)
        hooked.append(name)

    if not hooked:
        raise RuntimeError(
            "the step cache is armed but no denoiser class was found to drive its step "
            "counter. Without one the counter never advances, nothing is ever skipped, "
            "and the arm reports as cached while doing the full work"
        )
    print(f"[opt] cache step driver on {hooked}", flush=True)
