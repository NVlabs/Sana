# Search Space: Caching

Goal: reduce repeated denoiser, block, or attention work while preserving a
clean OFF path and auditable quality gates.

This file defines method families and search axes only. It intentionally does
not provide thresholds, step windows, layer ranges, or operating points. Those
choices are model-specific and must be discovered from target-model inference code,
traces, and artifacts.

## Method Families

- Whole-step denoiser output reuse: reuse, extrapolate, blend, or otherwise
  predict the complete denoiser output across nearby steps. Start with simple
  deterministic schedules only as baselines; useful transfeat should discover
  when the output is stable enough to reuse.
- TeaCache-style timestep-aware reuse: estimate output change from cheap input
  signals such as timestep embedding, timestep-modulated noisy input, hidden
  deltas, or accumulated relative distance; refresh when the accumulated or
  rescaled signal exceeds a threshold.
- EasyCache-style runtime-adaptive transform-vector reuse: cache reusable
  transformation vectors or residual updates and refresh online from runtime
  error estimates. This family should avoid offline profiling and should adapt
  to the current prompt/video dynamics.
- PAB-style attention broadcast: cache attention outputs and broadcast them
  across stable denoising intervals. Explore separate policies for spatial,
  temporal, cross, joint, or model-specific attention paths, with longer
  broadcast windows for more stable paths.
- Block/layer feature caching: cache transformer, U-Net, residual, attention,
  FFN/MLP, norm/modulation, or stage outputs for selected layers/blocks. Discover
  per-block schedules from observed feature deltas rather than using one global
  cadence.
- FORA-style transformer layer caching: cache and reuse attention and MLP
  intermediate outputs across denoising steps, with explicit layer/stage guards
  and refresh intervals.
- Token-wise feature caching: cache only low-risk token features instead of
  caching every token uniformly. Transfeat signals can include temporal
  redundancy, token-feature distance, spatial coverage, cross-attention
  importance, accumulated cache error, and expected error propagation.
- CFG-aware feature caching: exploit redundancy between conditional and
  unconditional branches, or between paired guidance computations, while keeping
  guidance scale, prompt alignment, and branch-specific artifacts guarded.
- Content- or motion-adaptive schedules: allocate more computation to prompts,
  regions, or clips with higher motion or semantic change, and cache more
  aggressively in stable regions or samples.
- Predictive, delta, or forecast caching: use first-order deltas, second-order
  deltas, Taylor-style forecasting, correction terms, or lightweight error
  models to predict future features instead of blindly reusing stale ones.
- Architecture-aware feature reuse: for U-Net-like models, reuse high-level or
  low-resolution features while cheaply refreshing low-level/detail features; for
  DiT-like models, prefer block/token/attention/MLP payloads discovered from the
  live transformer path.

## Search Axes

- Signal source: timestep embedding, timestep-modulated input, hidden state,
  residual, block output, attention output, K/V cache, modulation input, latent
  delta, transform vector, feature derivative, token score, branch similarity,
  motion proxy, or another code-discovered feature.
- Scope: per step, per layer, per block, per attention type, per token group, per
  modality, per spatial/temporal region, per guidance branch, per stage, or
  combinations of those axes.
- Decision rule: fixed schedule, U-shaped stable interval, threshold,
  accumulated change, runtime error estimate, polynomial/rescaled indicator,
  periodic recompute, content-adaptive schedule, motion-adaptive schedule,
  confidence model, derivative forecast error, or hybrid rule.
- Reuse payload: full denoiser output, transformer block output, residual,
  attention output, K/V tensors, score/routing metadata, FFN/MLP output,
  norm/modulation products, token features, transform vector, guidance branch
  output, delta, correction term, or forecasted feature.
- Refresh policy: warmup, tail protection, forced recompute, max consecutive
  reuse, accumulated-error cap, dense fallback, sensitive-token fallback,
  sensitive-layer fallback, motion-triggered refresh, prompt-change refresh, or
  artifact-triggered rollback.
- Forecast policy: zero-order reuse, linear delta, higher-order delta,
  Taylor-style extrapolation, learned-free correction, measured local
  correction, or branch-specific correction.
- Compatibility policy: decide whether a cache can coexist with token pruning,
  quantization, sparse attention, kernel fusion, CFG changes, sequence
  parallelism, compile caching, or distributed attention broadcast.
- Quality risk: flicker, blur, ghosting, patch discontinuity, temporal popping,
  snow/static, prompt/object identity drift, guidance drift, motion smoothing,
  fine-detail loss, background drift, token layout corruption, and error
  accumulation.
- Measurements to log: payload shape and dtype, refresh/skip pattern, cache hit
  rate, max consecutive reuse, accumulated signal/error, per-layer latency,
  per-attention latency, branch reuse rate, memory overhead, warm/cold compile
  state, OFF identity, aligned LPIPS, and aligned pairwise visual verdict.

## Required Exploration

- Inspect the live target-model inference path before choosing any parameter.
- Map the denoising loop, block boundaries, attention paths, MLP/FFN paths,
  guidance branches, token layout, and any existing cache or backend hooks.
- Record every tested signal, cache payload, refresh rule, and why it was
  accepted or rejected.
- Compare at least five distinct caching families before declaring the dimension
  exhausted. Include timestep-aware reuse, runtime-adaptive transform reuse,
  attention broadcast, block/layer feature caching, and one of token-wise,
  CFG-aware, content-adaptive, or predictive/delta caching when applicable.
- Discover all layer, step, signal, threshold, refresh, and fallback choices from
  target-model behavior; do not predefine them from this document.
- Prove OFF identity before claiming any speed or memory gain.
- When a transfeat fails, record whether the root cause is stale-feature error,
  schedule too aggressive, wrong payload boundary, guidance drift, token/layout
  mismatch, memory overhead, compile/cold-start distortion, or no real compute
  saved.
