# Search Space: Token Pruning

Goal: reduce token computation while preserving positional alignment, prompt
conditioning, and temporal coherence.

This file defines method families and search axes only. It intentionally does
not provide keep ratios, layer windows, step windows, or scoring constants.
Those choices are model-specific and must be discovered from Cosmos3 inference
code, traces, and artifacts.

## Method Families

- Token pruning: remove or skip low-importance tokens for selected layers/steps,
  then restore the full token layout before downstream code requires it.
- Token merging: merge similar tokens into representatives and unmerge or
  broadcast outputs later.
- Token masking: keep tensor shapes stable but mask selected tokens from costly
  work.
- Region-aware pruning: treat spatial, temporal, prompt, conditioning, or latent
  token regions differently based on code-discovered layout.
- Attention-guided pruning: use attention statistics, routing scores, feature
  norms, velocity, residual change, or other model-specific signals.

## Search Axes

- Token layout: generated video tokens, prompt/text tokens, conditioning tokens,
  spatial/temporal ordering, and sequence-parallel partitioning.
- Salience signal: feature norm, feature delta, attention score, residual
  magnitude, motion/velocity proxy, uncertainty, or code-discovered signal.
- Scope: per step, per layer, per block, per modality, per region, or
  combinations of those axes.
- Restoration policy: scatter, broadcast, merge reversal, zero/previous-state
  compensation, residual correction, or dense fallback.
- Alignment safety: RoPE/position tensors, masks, K/V layout, cross-attention
  inputs, batch/sequence parallel state, and output ordering.
- Quality risk: local detail loss, identity drift, motion popping, temporal
  inconsistency, patch boundaries, and prompt-conditioning degradation.

## Required Exploration

- Inspect the live Cosmos3 token layout before choosing any pruning site.
- Prove that gathered/masked/merged tokens restore correctly before quality runs.
- Discover all keep policies, layer windows, and step windows from model
  behavior; do not predefine them from this document.
- Prove OFF identity before claiming any speed or memory gain.
