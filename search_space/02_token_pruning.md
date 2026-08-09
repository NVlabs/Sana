# Search Space: Token Pruning

Goal: reduce token computation while preserving positional alignment, prompt
conditioning, and temporal coherence.

This file defines method families and search axes only. It intentionally does
not provide keep ratios, layer windows, step windows, or scoring constants.
Those choices are model-specific and must be discovered from target-model inference
code, traces, and artifacts.

## Method Families

- Token pruning: remove, skip, or bypass low-importance latent/video tokens for
  selected layers and steps, then restore the full token layout before downstream
  code observes the output.
- ToMe-style token merging: merge similar visual tokens into representatives,
  run expensive blocks on the reduced sequence, then unmerge, broadcast, or
  residual-correct outputs later.
- Importance-preserving token merging: protect semantic, high-motion,
  high-attention, prompt-aligned, or high-error tokens while merging only
  redundant low-risk tokens.
- Token masking / compute masking: keep tensor shapes stable but mask selected
  tokens from attention, MLP, FFN, routing, or update paths. This is useful when
  gather/scatter would break kernels, sequence parallelism, or positional state.
- Region-aware token reduction: treat spatial, temporal, prompt, conditioning,
  latent, background, foreground, or ROI token regions differently based on
  code-discovered layout and quality sensitivity.
- Attention-guided token reduction: use attention statistics, routing scores,
  query-key redundancy, prompt-token contribution, cross-attention salience, or
  mediator-token compression to reduce redundant interactions.
- Dynamics-aware token pruning: preserve high-dynamic tokens and skip/prune
  low-dynamic tokens using feature delta, noise relative magnitude, residual
  change, temporal velocity, or accumulated error signals.
- Cluster-aware token pruning: cluster spatial or latent tokens and update only
  representative or high-change cluster members, with cluster-level broadcast or
  interpolation for the rest.
- Dynamic token-density control: vary token count by timestep, layer, block,
  modality, resolution, and sample difficulty instead of applying one global
  keep ratio.
- Video token carving: reduce tokens per operation in video DiTs by exploiting
  spatial and temporal redundancy, while preserving enough tokens for motion and
  temporal coherence.
- Context/reference token pruning: for in-context, reference, image-to-video, or
  edit-style generation, prune non-essential reference/context tokens while
  preserving semantic anchors and periodically refreshing selection.
- Token-wise feature caching: cache or reuse selected low-risk token features
  rather than pruning them outright. This overlaps with caching, but belongs here
  when token selection is the main mechanism.
- Conservative/aggressive dual token policies: alternate between aggressive
  token reduction for speed and conservative token updates to repair accumulated
  quality error.

## Search Axes

- Token layout: generated video/image latent tokens, prompt/text tokens,
  reference/context tokens, conditioning tokens, control tokens, spatial/temporal
  ordering, packed sequences, sequence-parallel partitions, and any rotary or
  frequency layout.
- Reduction primitive: drop, skip update, gather shorter sequence, merge,
  cluster, representative pooling, mediator tokens, mask attention, mask MLP,
  cache token features, route to cheaper block, or route to dense fallback.
- Salience signal: feature norm, feature delta, residual magnitude, attention
  score, cross-attention score, prompt contribution, QK redundancy, token
  similarity, noise relative magnitude, temporal velocity, motion proxy,
  uncertainty, accumulated error, region label, cluster membership, or
  code-discovered signal.
- Scope: per step, per layer, per block, per attention type, per MLP/FFN path,
  per modality, per token region, per temporal segment, per cluster, per guidance
  branch, or combinations of those axes.
- Schedule: fixed keep ratio, timestep-dependent keep ratio, U-shaped schedule,
  layer-dependent keep ratio, sample-adaptive density, region-adaptive density,
  periodic selection refresh, error-triggered refresh, conservative/aggressive
  alternation, or budgeted controller.
- Restoration policy: scatter, broadcast, merge reversal, representative
  interpolation, cluster broadcast, zero/previous-state compensation, cached
  feature fill, residual correction, dense recompute, or branch-specific repair.
- Alignment safety: RoPE/position tensors, timestep embeddings, attention masks,
  K/V layout, cross-attention inputs, guidance branches, batch/sequence parallel
  state, packed sequence metadata, output ordering, and downstream shape
  contracts.
- Kernel compatibility: whether the reduced-token path actually speeds up
  attention/MLP kernels, whether dynamic shapes trigger recompilation, whether
  masking wastes compute, whether FlashAttention/xFormers/SGLang backends accept
  the layout, and whether gather/scatter overhead erases savings.
- Quality risk: local detail loss, identity drift, object disappearance, prompt
  misalignment, motion popping, temporal inconsistency, spatial patch
  boundaries, texture smearing, background drift, face/hand/text degradation,
  guidance drift, and restoration artifacts.
- Measurements to log: original/reduced token counts, per-region keep ratios,
  selection refresh pattern, gather/scatter time, attention/MLP time, compile
  cache state, memory overhead, alignment proof, OFF identity, aligned LPIPS,
  aligned pairwise visual verdict, and per-failure root cause.

## Required Exploration

- Inspect the live target-model token layout before choosing any pruning site.
- Prove that gathered/masked/merged tokens restore correctly before quality runs.
- Map prompt/context/video/control token boundaries, position encodings, masks,
  K/V layout, guidance branches, sequence parallelism, and downstream shape
  assumptions before launching active pruning.
- Compare at least five distinct token-reduction families before declaring early
  stop or structured negative. Include pruning, merging, masking, region-aware or
  dynamics-aware selection, and one of mediator tokens, cluster-aware pruning,
  context-token pruning, token-wise caching, or dynamic token-density control
  when applicable.
- Discover all keep policies, layer windows, step windows, refresh intervals,
  salience signals, restoration policies, and dense fallback rules from
  target-model behavior; do not predefine them from this document.
- Prove OFF identity before claiming any speed or memory gain.
- When a transfeat fails, record whether the root cause is positional/layout
  mismatch, scatter/merge restoration error, prompt-conditioning loss,
  cross-attention sensitivity, insufficient kernel speedup, dynamic-shape compile
  overhead, quality cliff from too few tokens, or no actual token compute saved.
