# Open-Ended Exploration: Low-Precision FFN

Start from `search_space/03_quantization.md`, then inspect the Cosmos3
inference code directly. If `search_space/` is missing, stop and ask the main
agent to repair the search-space contract.

Explore low-precision FFN as a model-specific module and calibration problem:

- Inspect which Cosmos3 FFN/MLP modules dominate runtime and support safe
  replacement or load-time quantization.
- Derive precision scope, excluded layers, warmup layers, and fallback policy from
  traces/code.
- Treat attention quantization and FFN quantization as separate candidates unless
  evidence says they should be combined.
- Record hardware/library prerequisites explicitly.

Required output:

- Candidate manifest or structured negative result.
- Evidence for exact module scope and fallback behavior.
- OFF identity and structured quality-gate status.
