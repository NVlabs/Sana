# Search Space: Caching

Goal: reduce repeated denoiser, block, or attention work while preserving a
clean OFF path and auditable quality gates.

This file defines method families and search axes only. It intentionally does
not provide thresholds, step windows, layer ranges, or operating points. Those
choices are model-specific and must be discovered from Cosmos3 inference code,
traces, and artifacts.

## Method Families

- TeaCache-style signal reuse: use changes in timestep-conditioned, modulated,
  residual, hidden, or other model-specific signals to decide whether reuse is
  safe.
- Whole-step denoiser output reuse: reuse, extrapolate, blend, or otherwise
  predict the denoiser output across nearby steps.
- Block or residual reuse: cache intermediate block outputs, residual streams,
  norm/modulation products, or FFN outputs for selected layers and steps.
- Attention feature reuse: reuse K/V tensors, attention outputs, score/routing
  metadata, or post-attention projections when the model-specific signal says
  the attention state is stable.
- PAB-style broadcast windows: broadcast selected computation across step,
  layer, spatial, temporal, or attention-type axes.
- Step-output reuse with correction: reuse a previous output and apply a
  learned, analytic, or measured correction from local deltas.

## Search Axes

- Signal source: timestep embedding, hidden state, residual, block output,
  attention output, K/V cache, modulation input, latent delta, or another
  code-discovered feature.
- Scope: per step, per layer, per block, per attention type, per token region,
  or combinations of those axes.
- Decision rule: threshold, accumulated change, periodic recompute,
  error-predictor, confidence model, deterministic schedule, or hybrid rule.
- Reuse payload: full denoiser output, block output, residual, attention output,
  K/V tensors, routing metadata, or delta/correction term.
- Safety policy: warmup, forced recompute, max consecutive reuse, dense fallback,
  disabled regions, or artifact-triggered rollback.
- Quality risk: flicker, blur, ghosting, patch discontinuity, temporal popping,
  snow/static, and prompt/object identity drift.

## Required Exploration

- Inspect the live Cosmos3 inference path before choosing any parameter.
- Record every tested signal and why it was accepted or rejected.
- Discover all layer, step, signal, and threshold choices from model behavior;
  do not predefine them from this document.
- Prove OFF identity before claiming any speed or memory gain.
