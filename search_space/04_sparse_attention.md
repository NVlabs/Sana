# Search Space: Sparse Attention

Goal: reduce attention cost with sparse, routed, approximate, or cached
attention while preserving visual quality and temporal coherence.

This file defines method families and search axes only. It intentionally does
not provide density values, block sizes, layer ranges, step windows, or env-var
settings. Those choices are model-specific and must be discovered from Cosmos3
inference code, traces, and artifacts.

## Method Families

- Piecewise sparse attention: route query/key/value blocks through exact or
  approximate block selection.
- Local/window attention: restrict attention to local spatial, temporal, or
  token-neighborhood windows where the model supports it.
- Routed global attention: keep selected global blocks dense based on proxy
  scores, layout, or salience.
- Cross-attention-specific sparsity: use a distinct approximation for
  conditioning/prompt attention when self-attention and cross-attention have
  different sensitivity.
- Attention output reuse: cache and reuse attention outputs, K/V tensors, or
  routing decisions across layers or steps when signals are stable.
- Dense fallback schedules: keep sensitive steps, layers, token regions, or
  attention types dense.

## Search Axes

- Attention path: self-attention, cross-attention, joint/GEN attention, temporal
  attention, spatial attention, or model-specific variants.
- Routing signal: centroid score, local window, attention statistics, feature
  similarity, token layout, layer role, timestep, or code-discovered signal.
- Scope: per layer, per step, per head, per attention type, per token region, or
  combinations of those axes.
- Approximation payload: exact selected blocks, centroid approximation, zero
  remainder, cached output, cached K/V, or hybrid correction.
- Dense guard policy: warmup, tail protection, sensitive layer protection,
  attention-type fallback, artifact-triggered fallback, or periodic dense pass.
- Quality risk: flicker, blur, ghosting, snow/static, patch discontinuity,
  temporal popping, and prompt/object identity drift.

## Required Exploration

- Inspect Cosmos3 attention implementations before choosing any backend or
  routing policy.
- Treat self-attention, cross-attention, and joint/GEN attention as separate
  candidates unless evidence says they can share a policy.
- Discover density, block size, layer, step, and fallback choices from model
  behavior; do not predefine them from this document.
- Prove OFF identity and dense fallback behavior before claiming any speed or
  memory gain.
