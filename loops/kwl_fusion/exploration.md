# Open-Ended Exploration: Kernel And Operator Fusion

Start from `search_space/05_kernel_fusion.md`, then inspect the Cosmos3
inference code directly. If `search_space/` is missing, stop and ask the main
agent to repair the search-space contract.

Explore fusion as a model-specific graph and kernel selection problem:

- Inspect the Cosmos3 hot path before selecting fusions.
- Derive candidate fused ops from measured repeated patterns rather than copying
  any fixed flag bundle.
- Separate implementation-level exact fusions from lossy approximations.
- Record compile-cache state and warm/cold timing context.

Required output:

- Candidate manifest or structured negative result.
- Evidence for the hot operation being fused.
- OFF identity and structured quality-gate status.
