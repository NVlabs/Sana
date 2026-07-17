# Copyright 2025 SGLang authors
#
# StepCache -- whole-step denoiser-output cache (the runtime form of the SCSP
# stage-1 cache-core: on scheduled steps, skip recomputing the denoiser output
# and reuse / delta-extrapolate the previous step's output).
#
# Phase ON_STEP: wraps the per-step compute. writes STEP_OUTPUT (exclusive --
# only one thing may own the step output). enabled schedule selects the SKIP
# steps (e.g. SCSP preset 8of15_last_29calls skips a fixed set of late steps);
# step 0 always computes to seed the buffer. OFF (no skip step active) ==
# byte-identical baseline.

from __future__ import annotations

from techniques.registry import register_technique
from techniques.schedule import as_schedule, at_steps
from techniques.technique import (
    Phase,
    Seam,
    Technique,
    TechniqueContext,
)


@register_technique("step_cache")
class StepCache(Technique):
    """Skip-and-reuse the whole denoiser-output on scheduled steps.

    Parameters
    ----------
    skip : Schedule[bool] | str | bool -- True on steps whose compute is
        skipped and replaced by the cached (optionally delta-extrapolated)
        previous output. A *string* like ``"16-28"`` / ``"1-2,5,7-9"`` is
        parsed as a step set (for example, a late-cluster skip policy);
        ``""`` / ``False`` / ``None`` disables; a bare ``True`` skips every
        step. Pass a pre-built Schedule for stage/policy-aware skips (see
        ``efficiency.presets.ltx_full_opt`` for the stage-gated form).
    delta_scale : float -- 0.0 reuses the last output verbatim; >0 linearly
        extrapolates using the last computed delta (output_t - output_{t-1}).
    """

    name = "step_cache"
    phase = Phase.ON_STEP
    reads = frozenset({Seam.STEP_OUTPUT})
    writes = frozenset({Seam.STEP_OUTPUT})

    def __init__(self, skip="", delta_scale: float = 0.0, enabled=True):
        # A string-skip spec ("16-28", "1-2,5") must be PARSED into a step set
        # via at_steps -- otherwise as_schedule wraps the literal string in a
        # const() schedule whose .at(step)="16-28" (truthy) and the technique
        # is active on every step. Bug history: a 6.4x "speedup" on Cosmos3
        # with skip='16-28' that was actually skipping all 35 steps.
        #
        # An EMPTY string means "no steps to skip" -> the technique is OFF
        # (compose() still accepts it; it just never fires). When ``skip`` is
        # a pre-built Schedule we trust it as-is.
        if isinstance(skip, str):
            sched = at_steps(skip, True, False) if skip else False
        else:
            sched = as_schedule(skip) if skip is not None else enabled
        super().__init__(enabled=sched)
        self.delta_scale = float(delta_scale)

    def on_step(self, ctx: TechniqueContext, run_step):
        key = ("step_cache", ctx.cache_key)
        prev = ctx.scratch.get(key)  # (last_output, last_delta)
        if not self.is_active(ctx) or prev is None:
            out = run_step()  # full compute (and always on the seed step)
            last_out = prev[0] if prev is not None else None
            delta = (out - last_out) if last_out is not None else None
            ctx.scratch[key] = (out.detach() if hasattr(out, "detach") else out, delta)
            return out
        # SKIP this step: reuse / delta-extrapolate the cached output
        last_out, last_delta = prev
        if self.delta_scale and last_delta is not None:
            return last_out + self.delta_scale * last_delta
        return last_out
