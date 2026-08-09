# Search Space: Training-Free Sparse Attention

Goal: reduce attention cost with training-free sparse, routed, approximate, or
cached attention while preserving visual quality and temporal coherence.

This file defines method families and search axes only. It is not a fixed
hyperparameter grid. Density values, block sizes, layer ranges, step windows,
head policies, and env vars must be discovered from target-model inference code,
attention traces, and reproduction artifacts.

Training-free means no weight update, no finetune, and no distillation. Offline
profiling and online mask search are allowed when they are config artifacts
and can be reproduced.

---

## Method Families

### 1. Existing Piecewise / PISA-Style Block Sparse Attention

Use the current piecewise attention backend or a direct target-runtime equivalent.
This is the existing wired path for `SparseAttention`: choose a component,
block size, sparsity, route mode, dense fallback, dense stage/layer guards, and
whether only video self-attention is approximated.

Useful axes:

- attention component: stage, transformer, self-attention, cross-attention;
- route mode: score, local-exact, exact-only, or target-discovered variant;
- approximate remainder policy;
- dense fallback backend;
- per-layer and per-step dense guards;
- whether prompt/cross attention stays dense.

### 2. Sparse VideoGen-Style Spatial/Temporal Head Routing

Sparse VideoGen observes that video diffusion attention heads can behave like
spatial heads or temporal heads, then uses online profiling and hardware-aware
layout changes to route each head. Abstract this as:

- profile a small number of early/warmup attention calls;
- classify heads or layers as spatial, temporal, mixed, or dense;
- use spatial windows for spatial heads and temporal stripes/windows for
  temporal heads;
- keep ambiguous or quality-sensitive heads dense;
- record whether a layout transform or custom kernel is needed for real speed.

### 3. Sparse VideoGen2 Semantic-Aware Permutation

SVG2 clusters or permutes tokens so semantically related tokens become contiguous
in memory, making dynamic sparse attention more hardware-efficient. Abstract this
as:

- derive token grouping from feature similarity, latent layout, or attention
  statistics;
- permute/gather tokens into hardware-friendly contiguous groups;
- run block sparse attention inside grouped neighborhoods;
- scatter back exactly and prove positional/layout restoration;
- compare permutation overhead against attention savings.

### 4. AdaSpa-Style Online Precise Search With Mask Reuse

AdaSpa uses blockified sparse patterns, online precise mask search, and cached
search signals across denoising steps. Abstract this as:

- warmup dense attention for selected steps;
- run online block-mask search on search steps;
- reuse masks across later denoising steps when query/key or LSE signals are
  stable;
- make sparsity head-adaptive rather than one global threshold;
- refresh masks periodically or when drift exceeds a threshold.

### 5. SpargeAttn-Style Universal Mask Prediction

SpargeAttn predicts sparse masks by compressing blocks to representative tokens
and adds online softmax-aware filtering. Abstract this as:

- compress each query/key block to a proxy token only when the block is internally
  coherent;
- predict config sparse blocks from proxy attention;
- use softmax-aware filtering to skip low-contribution products;
- keep dense fallback when proxy prediction is uncertain;
- optionally combine with lower-precision attention kernels only as a separate
  interaction config.

### 6. LVSA-Style Rotating Anchors And Long-Video Windows

LVSA combines structured windows with rotating global anchors to avoid fixed-grid
bias in long videos. Abstract this as:

- local spatial/temporal window attention;
- rotating or staggered global anchors;
- horizon-aware anchor spacing;
- dense anchors for scene or prompt identity tokens;
- explicit checks for frozen, repetitive, or loopy video failure modes.

### 7. SVOO-Style Layer-Wise Profiling And QK Co-Clustering

SVOO treats sparsity as a relatively input-stable layer property and combines
offline layer-wise sensitivity profiling with online QK co-clustering. Abstract
this as:

- profile per-layer sparsity tolerance offline on reproducible prompts;
- assign per-layer pruning budgets;
- cluster Q and K blocks jointly at runtime;
- preserve coupled query-key regions rather than pruning by Q or K alone;
- compare quality-speed frontier against purely online mask search.

### 8. HASTE-Style Head-Wise Adaptive Budgets

HASTE targets two practical costs: repeated mask prediction and shared thresholds
despite head heterogeneity. Abstract this as:

- temporal mask reuse when query-key drift is small;
- per-head sparse budgets rather than one threshold;
- error-guided calibration under a global sparsity budget;
- dense fallback for unstable heads;
- record mask-prediction overhead separately from attention-kernel speedup.

### 9. MInference-Inspired Dynamic Patterns

MInference is designed for long-context LLM prefill, not video diffusion, but its
patterns are useful as probes when target attention maps show similar structure:

- A-shape: initial/global tokens plus local windows;
- vertical/slash stripes: selected global columns or periodic diagonals;
- block-sparse: dynamic clusters with local aggregation;
- per-head pattern assignment;
- online sparse-index construction only if overhead is small at the target length.

---

## Search Axes

- Attention path: video self-attention, temporal attention, spatial attention,
  cross-attention, joint/GEN attention, text/prompt attention, or model-specific
  variants.
- Pattern family: piecewise, spatial/temporal head routing, semantic permutation,
  online precise block search, proxy-mask prediction, rotating anchors, QK
  co-clustering, head-wise budgets, or MInference-style patterns.
- Routing signal: attention scores, QK proxy scores, LSE/cache signal,
  query-key drift, feature similarity, semantic cluster id, token layout, layer
  role, timestep, head id, or code-discovered signal.
- Scope: per layer, per step, per head, per attention type, per token region,
  per frame, per spatial tile, or combinations.
- Approximation payload: exact selected blocks, local windows, temporal stripes,
  global anchors, centroid/proxy blocks, zero remainder, cached masks, cached
  K/V, cached attention output, or hybrid correction.
- Dense guard policy: warmup, tail protection, sensitive layer protection,
  attention-type fallback, head fallback, artifact-triggered fallback, periodic
  dense pass, or drift-triggered mask refresh.
- Kernel/runtime path: existing backend, target-runtime direct patch, FlashInfer
  block sparse, Triton prototype, layout permutation, gather/scatter wrapper, or
  metadata-only diagnostic.
- Quality risk: flicker, blur, ghosting, snow/static, patch discontinuity,
  temporal popping, object identity drift, prompt neglect, frozen/loopy motion,
  and long-range temporal repetition.

---

## Required Exploration

- Inspect target-model attention implementations before choosing any backend or
  routing policy.
- Treat self-attention, cross-attention, and joint/GEN attention as separate
  config unless evidence says they can share a policy.
- Run an attention preflight: identify sequence length, frame/tile layout,
  head count, attention call timing, backend selection path, and whether a real
  sparse kernel or only a masking wrapper is used.
- Compare at least five training-free sparse-attention families before declaring
  structured negative.
- Discover density, block size, layer, step, head, and fallback choices from
  model behavior; do not predefine them from this document.
- Prove OFF identity and dense fallback behavior before claiming any speed or
  memory gain.
- Measure mask-search/permutation overhead separately from sparse attention
  kernel time.

---

## Frontier Retention

Use the same frontier rule as step cache and token pruning:

- Retain a config when quality improves or speed/memory improves.
- Discard a config only when neither quality nor speed/memory improves.
- Reject hard-invalid config such as broken OFF identity, missing artifacts,
  metadata-only env with claimed runtime behavior, or non-reproducible masks.
- Continue until max_iters, real blocker, or orchestrator release. A
  structured-negative proposal is logged as evidence but does not stop the
  default fixed-budget loop by itself. Do not stop merely because one sparse
  pattern fails a tier gate.

For retained config, record:

- method family and pattern;
- attention path and component;
- layer/head/step scope;
- sparsity, block/window size, and dense guards;
- mask source, refresh policy, and search overhead;
- backend/kernel path and layout transform;
- speedup, peak memory, LPIPS, pairwise Gemini, and video artifacts;
- whether the config improved quality, speed/memory, or both.

---

## Structured Negative Requirements

Do not declare structured negative until the summary covers:

- attention preflight and runtime/backend feasibility;
- at least five training-free method families from this file;
- best speed/memory point and its quality failure;
- best quality point and its speed/memory failure;
- mask-search/permutation overhead;
- common failure signatures across rejected config;
- why remaining pattern families are redundant, unsupported, or lower value.

---

## Source Pointers

- Sparse VideoGen: spatial/temporal head routing with online profiling.
- Sparse VideoGen2: semantic-aware permutation and dynamic sparse kernels.
- AdaSpa: dynamic blockified pattern with online precise search and mask reuse.
- SpargeAttn: universal proxy-mask prediction and softmax-aware filtering.
- LVSA: structured windows with rotating global anchors for long video.
- SVOO: offline layer-wise sparsity profiling plus online QK co-clustering.
- HASTE: temporal mask reuse and head-wise adaptive sparse budgets.
- MInference: A-shape, vertical-slash, and block-sparse dynamic patterns.
