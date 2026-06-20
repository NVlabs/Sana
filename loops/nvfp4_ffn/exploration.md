# Open-Ended Exploration: Low-Precision Hot Linear

Start from `search_space/03_quantization.md`, then inspect the target-model
inference code directly. If `search_space/` is missing, stop and ask the main
agent to repair the search-space contract.

Explore low-precision hot-linear quantization as a model-specific module and
calibration problem:

- Run NVFP4 hardware/runtime preflight first: GPU architecture, TransformerEngine
  import/version, `NVFP4BlockScaling`, minimal TE/loader smoke, FP4 GEMM backend,
  OFF identity, and env-consumption proof.
- Inspect which target-model linear-heavy modules dominate runtime and support
  safe replacement or load-time quantization, including FFN/MLP, attention
  projections, and output projections.
- Derive precision scope, excluded layers, dense layer/step guards, TE recipe
  flags, backend/padding policy, and fallback policy from traces/code.
- Separate already-wired runtime env from metadata-only env that still needs
  candidate-side loader wiring.
- Treat attention quantization and FFN quantization as separate candidates unless
  evidence says they should be combined.
- Record hardware/library prerequisites explicitly.

Required output:

- Candidate manifest or structured-negative proposal for orchestrator review; do
  not use the proposal to stop the fixed-budget loop.
- Evidence for module scope, recipe flags, backend, dense guards, and fallback
  behavior.
- OFF identity and structured frontier-retention / speed-quality evidence status.
