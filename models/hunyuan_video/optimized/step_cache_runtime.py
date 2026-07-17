#!/usr/bin/env python3
"""TeaCache seam for HunyuanVideo (diffusers) — the model-specific glue.

This is NOT a new optimization. The algorithm is the public TeaCache controller
already in ``efficiency/techniques/teacache.py`` (rescaled cumulative relative-L1
of a per-step signal -> reuse the cached step output while it stays under a
threshold). This module is only the SEAM that:

  1. feeds HunyuanVideo's ``time_text_embed`` output (``temb``, the combined
     timestep + pooled-text (+guidance) embedding) into that controller as the
     per-step signal, and
  2. wraps ``transformer.forward`` so the expensive dual/single-stream block stack
     is skipped (its output reused) on steps the controller marks reusable.

OFF == baseline: if no ``SGLANG_HQ_TEACACHE_*`` knob is set, ``maybe_enable()`` is
a no-op and the pipeline stays byte-identical to the vanilla baseline.

Knobs (published by search/plan_eval.py ``_RUNTIME_TECHNIQUE_ENV["teacache"]``):
  SGLANG_HQ_TEACACHE_THRESHOLD   accumulate rescaled rel-L1; reuse while < this.
                                 Presence (>0) ENABLES the seam.
  SGLANG_HQ_TEACACHE_START_STEP  always compute before this step (warm-up); default 6.
  SGLANG_HQ_TEACACHE_MAX_HITS    cap consecutive reuses (1 = never two in a row); default 1.

STATUS: first cut = WHOLE-STEP output cache (efficiency ``TeaCache``, Phase.ON_STEP):
wraps the whole transformer.forward and reuses its returned output. The
block-residual variant (``TeaCacheResidual``, replays block residuals — more
accurate) is the planned refinement once this is GPU-validated against the
baseline. No speed/quality claim until then.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the orchestration repo's `efficiency` package importable (the generic
# TeaCache controller lives there). repo root = runtime/<name>/../..
_REPO = Path(os.environ.get("AUTOVIDEO_REPO_ROOT", str(Path(__file__).resolve().parents[3])))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _env_float(name: str):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else None


def _env_int(name: str):
    v = os.environ.get(name)
    return int(float(v)) if v not in (None, "") else None


def maybe_enable(pipe) -> dict | None:
    """Install the TeaCache seam on ``pipe.transformer`` iff a knob is set.

    Returns a diagnostics dict (knobs + a live ``stats`` counter the caller can
    read after generation) when enabled, else ``None`` (baseline untouched).
    """
    threshold = _env_float("SGLANG_HQ_TEACACHE_THRESHOLD")
    if threshold is None or threshold <= 0:
        return None  # disabled -> vanilla baseline, no wrapping

    from techniques.registry import build_technique
    from techniques.technique import TechniqueContext
    from techniques.methods.teacache import teacache_relative_l1

    kwargs = {"threshold": threshold}
    start_step = _env_int("SGLANG_HQ_TEACACHE_START_STEP")
    max_hits = _env_int("SGLANG_HQ_TEACACHE_MAX_HITS")
    if start_step is not None:
        kwargs["start_step"] = start_step
    if max_hits is not None:
        kwargs["max_continuous_hits"] = max_hits
    teacache = build_technique("teacache", **kwargs)  # generic controller, reused

    transformer = pipe.transformer
    ctx = TechniqueContext(step=0, stage="denoise", cache_key="main", scratch={})
    orig_forward = transformer.forward
    stats = {"calls": 0, "reuse": 0, "compute": 0}
    trace = []                       # per-step calibration log (logging only)
    _log = {"prev": None}

    def wrapped(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                pooled_projections, guidance=None, attention_kwargs=None, return_dict=True):
        # Per-step signal = HunyuanVideo's combined timestep+text(+guidance) embedding.
        # Cheap MLP relative to the block stack; recomputed inside run_step on a
        # compute step, which is negligible.
        temb, _ = transformer.time_text_embed(timestep, pooled_projections, guidance)
        signal = temb.detach()
        ctx.scratch[("teacache_signal", ctx.cache_key)] = signal
        # logging-only rel-L1 (same formula the controller uses) for threshold calibration
        rel_l1 = None
        if _log["prev"] is not None:
            try:
                rel_l1 = round(float(teacache_relative_l1(signal, _log["prev"])), 6)
            except Exception:
                rel_l1 = None
        _log["prev"] = signal

        ran = {"computed": False}

        def run_step():
            ran["computed"] = True
            return orig_forward(
                hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                pooled_projections, guidance, attention_kwargs, return_dict,
            )

        out = teacache.on_step(ctx, run_step)  # reuse cached output, or compute+cache
        cstate = ctx.scratch.get(("teacache", ctx.cache_key)) or {}
        trace.append({"step": ctx.step, "rel_l1": rel_l1,
                      "acc": round(float(cstate.get("acc") or 0.0), 6),
                      "reused": not ran["computed"]})
        stats["calls"] += 1
        stats["compute" if ran["computed"] else "reuse"] += 1
        ctx.step += 1
        return out

    transformer.forward = wrapped
    return {
        "technique": "teacache",
        "variant": "whole_step_output_cache",
        "threshold": threshold,
        "start_step": kwargs.get("start_step"),
        "max_continuous_hits": kwargs.get("max_continuous_hits"),
        "signal": "time_text_embed",
        "stats": stats,  # live; read after generation for reuse/compute counts
        "trace": trace,  # per-step rel_l1 / acc / reused, for threshold calibration
    }
