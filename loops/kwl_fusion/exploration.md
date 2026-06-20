# Open-Ended Exploration: Kernel And Operator Fusion

Start from `search_space/05_kernel_fusion.md`, then inspect the target-model
inference code directly. If `search_space/` is missing, stop and ask the main
agent to repair the search-space contract.

Explore fusion as a model-specific graph and kernel selection problem:

- Inspect the target-model hot path before selecting fusions.
- Derive candidate fused ops from measured repeated patterns rather than copying
  any fixed flag bundle.
- Separate KWL-safe kernel/backend approximations from algorithm changes that
  belong to cache, pruning, sparse attention, quantization, or scheduler
  dimensions.
- Record compile-cache state and warm/cold timing context.
- Compare at least six KWL method families before declaring a structured
  negative, including exact-preferred and quality-gated approximate variants
  where relevant: GEMM epilogue, norm/modulation/residual fusion,
  attention-adjacent dense fusion, compile/CUDA graph capture, layout/copy
  elimination, launch batching, stream overlap, decode/postprocess fusion, or
  backend selection.
- Treat cache reuse, token reduction, sparse attention, changed precision, or
  scheduler changes as out of scope for KWL.
- For every speed candidate, prove OFF identity and record ON aligned quality
  evidence before retaining it in the frontier; ON bit-exactness is not
  required.

Required output:

- Candidate manifest or structured-negative proposal for orchestrator review; do
  not use the proposal to stop the fixed-budget loop.
- Evidence for the hot operation being fused.
- Launch count, memory traffic, backend/fallback, and cold/warm timing context.
- OFF identity and structured speed/quality evidence status.
- Expected numeric tolerance: bit-exact, dtype-rounding-only, reduction-order
  drift, FMA/epilogue drift, fast-math drift, or approximate-kernel drift.
