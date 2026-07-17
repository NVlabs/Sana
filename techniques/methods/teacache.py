# Copyright 2025 SGLang authors
#
# TeaCache -- timestep-embedding-similarity step cache (the framework form of
# the LTX2/Cosmos TeaCache: skip recomputing the denoiser when the accumulated,
# polynomial-rescaled L1 distance of the timestep-conditioned "modulated input"
# stays below a threshold, and reuse the cached output instead).
#
# Like token-prune, it has a GENERIC core (the distance accumulation + rescale +
# skip decision, the dedup target for the existing ltx2_teacache.py /
# TeaCacheMixin) and a MODEL-SPECIFIC seam: the "modulated input" signal each
# step. The model stashes that signal into ctx.scratch[("teacache_signal",
# cache_key)] before the step compute; the polynomial `coefficients` are a
# model-calibrated rescale (identity by default).
#
# Phase ON_STEP, writes STEP_OUTPUT (exclusive -> conflicts with StepCache, which
# is correct for this whole-step baseline: you don't run two whole-step caches at
# once). Public Cosmos TeaCache uses the same controller with a model-specific
# block-residual adapter; this class intentionally keeps only the generic
# thresholded signal-reuse controller and whole-step fallback boundary.

from __future__ import annotations

from techniques.registry import register_technique
from techniques.technique import (
    Capability,
    Phase,
    Seam,
    Technique,
    TechniqueContext,
)


def teacache_poly_rescale(coefficients, x: float) -> float:
    """Horner evaluation of the TeaCache indicator polynomial.

    Coefficients are highest-degree first, matching ``numpy.poly1d`` in the
    public TeaCache scripts.
    """

    y = 0.0
    for coeff in coefficients:
        y = y * x + float(coeff)
    return y


def teacache_relative_l1(current, previous) -> float:
    """Mean relative L1 distance used by the public TeaCache controller."""

    num = float((current - previous).abs().mean())
    den = max(float(previous.abs().mean()), 1e-8)
    return num / den


def teacache_indicator(current, previous, coefficients) -> float:
    """Public TeaCache rel-L1 distance after polynomial rescaling."""

    return teacache_poly_rescale(coefficients, teacache_relative_l1(current, previous))


@register_technique("teacache")
class TeaCache(Technique):
    """Reuse the denoiser output while the rescaled cumulative timestep-embedding
    distance stays under ``threshold``.

    Parameters
    ----------
    threshold : accumulate rescaled rel-L1 distance; reuse while < threshold.
    start_step : always compute before this step (warmup / seed).
    coefficients : polynomial (highest-degree first) rescaling rel-L1 -> the
        TeaCache "indicator"; model-calibrated. Default [1, 0] (identity).
    max_continuous_hits : cap consecutive reuses (1 = never skip two in a row,
        0 or less = no cap, matching the public TeaCache controller profile).
    periodic_recompute : force a recompute every N steps (0 = off).
    """

    name = "teacache"
    phase = Phase.ON_STEP
    reads = frozenset({Seam.STEP_OUTPUT})
    writes = frozenset({Seam.STEP_OUTPUT})

    def __init__(
        self,
        threshold: float = 0.04,
        start_step: int = 6,
        coefficients=None,
        max_continuous_hits: int = 1,
        periodic_recompute: int = 0,
        enabled="" or True,
    ):
        super().__init__(enabled=enabled)
        self.threshold = float(threshold)
        self.start_step = int(start_step)
        self.coefficients = list(coefficients) if coefficients else [1.0, 0.0]
        self.max_continuous_hits = int(max_continuous_hits)
        self.periodic_recompute = int(periodic_recompute)

    def on_step(self, ctx: TechniqueContext, run_step):
        key = ("teacache", ctx.cache_key)
        st = ctx.scratch.get(key) or {"prev": None, "acc": 0.0, "hits": 0, "out": None, "since": 0}
        modulated = ctx.scratch.get(("teacache_signal", ctx.cache_key))

        force = (
            not self.is_active(ctx)
            or modulated is None
            or ctx.step < self.start_step
            or st["prev"] is None
            or (self.periodic_recompute and st["since"] >= self.periodic_recompute)
        )
        reuse = False
        if not force:
            st["acc"] += teacache_indicator(
                modulated, st["prev"], self.coefficients
            )
            hit_cap_ok = (
                self.max_continuous_hits <= 0
                or st["hits"] < self.max_continuous_hits
            )
            reuse = st["acc"] < self.threshold and hit_cap_ok

        if modulated is not None:
            st["prev"] = modulated.detach() if hasattr(modulated, "detach") else modulated

        if reuse and st["out"] is not None:
            st["hits"] += 1
            st["since"] += 1
            ctx.scratch[key] = st
            return st["out"]

        out = run_step()  # full compute
        st["out"] = out.detach() if hasattr(out, "detach") else out
        st["acc"] = 0.0
        st["hits"] = 0
        st["since"] = 0
        ctx.scratch[key] = st
        return out


@register_technique("teacache_residual")
class TeaCacheResidual(Technique):
    """TeaCache controller for model adapters that can replay block residuals.

    The pure part remains the same public TeaCache controller: accumulate a
    polynomial-rescaled relative L1 signal distance and reuse while it stays
    below ``threshold``. The model adapter supplies the signal and consumes a
    ``("reuse",)`` carry by skipping its transformer-block loop after this hook
    has added the cached residual to the block input.
    """

    name = "teacache_residual"
    phase = Phase.PRE_BLOCKS
    reads = frozenset({Seam.HIDDEN_STATES, Seam.RESIDUAL_CACHE})
    writes = frozenset({Seam.HIDDEN_STATES, Seam.RESIDUAL_CACHE})
    required_capabilities = frozenset(
        {Capability.BLOCKS, Capability.SUPPORTS_STEP_CACHE}
    )

    def __init__(
        self,
        threshold: float = 0.3,
        start_step: int = 0,
        coefficients=None,
        max_continuous_hits: int = 0,
        periodic_recompute: int = 0,
        enabled="" or True,
    ):
        super().__init__(enabled=enabled)
        self.threshold = float(threshold)
        self.start_step = int(start_step)
        self.coefficients = list(coefficients) if coefficients else [1.0, 0.0]
        self.max_continuous_hits = int(max_continuous_hits)
        self.periodic_recompute = int(periodic_recompute)

    @staticmethod
    def _detach(value):
        return value.detach() if hasattr(value, "detach") else value

    def before_blocks(self, ctx: TechniqueContext, hidden):
        key = ("teacache_residual", ctx.cache_key)
        st = ctx.scratch.get(key) or {
            "prev": None,
            "acc": 0.0,
            "hits": 0,
            "since": 0,
            "residual": None,
        }
        signal = ctx.scratch.get(("teacache_signal", ctx.cache_key))
        final_force = bool(ctx.scratch.get(("teacache_force_compute", ctx.cache_key), False))
        force = (
            not self.is_active(ctx)
            or signal is None
            or ctx.step < self.start_step
            or st["prev"] is None
            or final_force
            or (self.periodic_recompute and st["since"] >= self.periodic_recompute)
        )
        reuse = False
        indicator = None
        if not force:
            indicator = teacache_indicator(signal, st["prev"], self.coefficients)
            st["acc"] += indicator
            hit_cap_ok = (
                self.max_continuous_hits <= 0
                or st["hits"] < self.max_continuous_hits
            )
            reuse = (
                st["acc"] < self.threshold
                and hit_cap_ok
                and st["residual"] is not None
            )

        if signal is not None:
            st["prev"] = self._detach(signal)

        stats = ctx.scratch.setdefault(
            ("teacache_residual_stats",),
            {
                "calls": 0,
                "compute": 0,
                "reuse": 0,
                "forced": 0,
                "last_indicator": None,
                "max_indicator": None,
                "last_acc": None,
                "max_acc": None,
                "by_key": {},
            },
        )
        stats["calls"] += 1
        key_stats = stats["by_key"].setdefault(
            str(ctx.cache_key),
            {
                "calls": 0,
                "compute": 0,
                "reuse": 0,
                "forced": 0,
                "last_indicator": None,
                "max_indicator": None,
                "last_acc": None,
                "max_acc": None,
            },
        )
        key_stats["calls"] += 1
        if indicator is not None:
            stats["last_indicator"] = indicator
            stats["last_acc"] = st["acc"]
            key_stats["last_indicator"] = indicator
            key_stats["last_acc"] = st["acc"]
            stats["max_indicator"] = (
                indicator
                if stats["max_indicator"] is None
                else max(stats["max_indicator"], indicator)
            )
            stats["max_acc"] = (
                st["acc"] if stats["max_acc"] is None else max(stats["max_acc"], st["acc"])
            )
            key_stats["max_indicator"] = (
                indicator
                if key_stats["max_indicator"] is None
                else max(key_stats["max_indicator"], indicator)
            )
            key_stats["max_acc"] = (
                st["acc"]
                if key_stats["max_acc"] is None
                else max(key_stats["max_acc"], st["acc"])
            )

        if reuse:
            stats["reuse"] += 1
            key_stats["reuse"] += 1
            st["hits"] += 1
            st["since"] += 1
            ctx.scratch[key] = st
            return hidden + st["residual"], ("reuse", key)

        stats["compute"] += 1
        key_stats["compute"] += 1
        if final_force:
            stats["forced"] += 1
            key_stats["forced"] += 1
        st["pending_input"] = self._detach(hidden).clone()
        ctx.scratch[key] = st
        return hidden, ("compute", key)

    def after_blocks(self, ctx: TechniqueContext, hidden, carry):
        if not carry or carry[0] != "compute":
            return hidden
        key = carry[1]
        st = ctx.scratch.get(key)
        if not st:
            return hidden
        original = st.pop("pending_input", None)
        if original is not None and getattr(original, "shape", None) == getattr(hidden, "shape", None):
            st["residual"] = self._detach(hidden - original)
            st["acc"] = 0.0
            st["hits"] = 0
            st["since"] = 0
            ctx.scratch[key] = st
        return hidden
