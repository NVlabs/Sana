# Open-Ended Exploration: Kernel And Operator Fusion

Start from `search_space/05_kernel_fusion.md`, then inspect the target-model
inference code directly. If `search_space/` is missing, stop and ask the main
agent to repair the search-space contract.

Explore fusion as a model-specific graph and kernel selection problem:

- Inspect the target-model hot path before selecting fusions.
- Derive candidate fused ops from measured repeated patterns rather than copying
  any fixed flag bundle.
- Do not implement, resume, or rerun backend-selection, SDPA-backend, framework
  dispatch, or env-flag-only probes. They are not KWL startup candidates.
- If prior local status or journal files contain backend-selection work, record
  it as stale/cancelled and start a new module/DiT microbench candidate.
- Prefer lossless implementation optimizations first: fuse already-adjacent
  operator chains, remove redundant layout/copy/allocation work, batch or pack
  equivalent launches. Apply compile or graph capture only after local
  module/kernel candidates are exhausted.
- Write a module-level or DiT-block-level warm paired microbenchmark before any
  full denoising/video generation run. The microbench must compare OFF baseline
  and ON candidate tensors in the same process/allocation/GPU, after explicit
  warmup, with the same tensors and dtype. It must report median/p25/p75/min/max
  latency, max/mean numerical difference, launch/profile evidence, expected
  full contribution, and exact reproduction commands.
- Promote to a full denoising run only after the microbench shows positive
  speed or memory movement and acceptable tensor drift.
- State the exact semantic equivalence for every candidate: same tensor inputs,
  parameters, masks, shape contract, dtype contract, dependency ordering, and
  output placement unless the declared tolerance class explicitly allows a
  kernel-level floating-point-order difference.
- Separate KWL-safe kernel/operator approximations from algorithm changes that
  belong to cache, pruning, sparse attention, quantization, or scheduler
  dimensions.
- Record compile-cache state and warm/cold timing context. Do not use a single
  full run against a historical canonical baseline as the primary speed evidence
  for small KWL changes.
- Compare at least seven KWL method families before declaring a structured
  negative, including exact-preferred and quality-gated approximate variants
  where relevant: GEMM epilogue, norm/modulation/residual fusion,
  attention-adjacent dense fusion, layout/copy elimination, launch batching,
  stream overlap, decode/postprocess fusion, and compile/CUDA graph capture only
  after local module candidates are exhausted.
- Treat cache reuse, token reduction, sparse attention, changed precision, or
  scheduler changes as out of scope for KWL.
- If final denoising shows visual artifacts after a passing microbench, first
  suspect a kernel/module-boundary bug: aliasing, layout, mask, split/concat,
  dtype, stream ordering, or stale workspace. Do not attribute artifacts to
  harmless numeric drift without module-level evidence.

## Hunyuan Diffusers Examples

When the model is `hunyuan_diffusers`, use these concrete DiT fusion candidates
as references:

- attention Q/K projection output + QK RMSNorm + RoPE application;
- packed latent QKV and packed text added-QKV projections;
- single-stream `cat(attn_output, mlp_output) -> proj_out -> gate -> residual`;
- dual-stream `hidden + gate * attn` and `context + c_gate * context_attn`;
- dual-stream `LayerNorm -> scale/shift` before FFN on latent/text branches;
- FFN output epilogue `x + gate * ff(x)` on latent/text branches;
- attention output split plus latent/text output projections;
- final `norm_out -> proj_out -> reshape/permute/flatten` output layout;
- static attention mask and RoPE/layout descriptor construction for the official
  benchmark shape.

Required output:

- Candidate manifest or structured-negative proposal for orchestrator review; do
  not use the proposal to stop the fixed-budget loop.
- Evidence for the hot operation being fused.
- Microbench JSON with paired OFF/ON warm latency statistics, tensor diff,
  shape/dtype, expected full contribution, and exact command.
- Launch count, memory traffic, kernel/fallback, and cold/warm timing context.
- OFF identity and structured microbench speed/numeric evidence status.
- Expected numeric tolerance: bit-exact, dtype-rounding-only, reduction-order
  drift, FMA/epilogue drift, fast-math drift, or approximate-kernel drift.
