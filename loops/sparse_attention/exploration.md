# Open-Ended Exploration: Sparse Attention

Start from `search_space/04_sparse_attention.md`, then inspect the Cosmos3
inference code directly. If `search_space/` is missing, stop and ask the main
agent to repair the search-space contract.

Explore sparse attention as a model-specific backend and routing problem:

- Inspect Cosmos3 self-attention, cross-attention, and any joint/GEN attention
  paths separately.
- Identify which components route through `supported_attention_backends`,
  `USPAttention`, or related SGLang attention selectors.
- Test dense fallback behavior before any approximate path.
- Derive sparsity, block size, dense warmup, layer selection, route mode, and
  component mapping from traces/code, not from predefined constants.
- Consider whether cross-attention needs a distinct approximation from video
  self-attention.

Required output:

- Candidate manifest or structured negative result.
- Evidence for which attention path dominates runtime.
- OFF identity and structured quality-gate status.
