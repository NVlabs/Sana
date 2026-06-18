# Open-Ended Exploration: Sparse Attention

Start from `search_space/04_sparse_attention.md`, then inspect the target-model
inference code directly. If `search_space/` is missing, stop and ask the main
agent to repair the search-space contract.

Explore training-free sparse attention as a model-specific backend, mask-search,
and routing problem:

- Inspect target-model self-attention, cross-attention, and any joint/GEN attention
  paths separately.
- Identify which components route through `supported_attention_backends`,
  `USPAttention`, or related SGLang attention selectors.
- Run attention preflight before GPU search: call timing, token/frame/tile
  layout, sequence length, dominant attention path, available sparse kernels,
  dense fallback, and OFF identity.
- Test dense fallback behavior before any approximate path.
- Compare at least five training-free sparse-attention families before declaring
  structured negative: piecewise/PISA, Sparse VideoGen-style spatial/temporal
  head routing, SVG2-style semantic permutation, AdaSpa-style online precise
  search and mask reuse, SpargeAttn-style proxy masks, LVSA-style rotating
  anchors, SVOO-style QK co-clustering, HASTE-style head-wise budgets, or
  MInference-style patterns.
- Derive sparsity, block size, window/anchor policy, mask refresh, dense warmup,
  layer/head selection, route mode, and component mapping from traces/code, not
  from predefined constants.
- Consider whether cross-attention needs a distinct approximation from video
  self-attention.
- Measure mask-search/permutation overhead separately from sparse kernel time.

Required output:

- Candidate manifest or structured-negative proposal for orchestrator review; do
  not use the proposal to stop the fixed-budget loop.
- Evidence for which attention path dominates runtime.
- Evidence for method family, routing signal, mask source, dense fallback,
  backend/kernel path, and candidate-specific overhead.
- OFF identity and structured frontier-retention / speed-quality evidence status.
