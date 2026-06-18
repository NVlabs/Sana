# Open-Ended Exploration: Caching

Start from `search_space/01_cache.md`, then inspect the target-model inference code
directly. If `search_space/` is missing, stop and ask the main agent to repair
the search-space contract.

Explore caching as a family of model-specific mechanisms, not a fixed grid:

- TeaCache-style reuse from timestep/modulated-input similarity.
- Whole-step denoiser output reuse or delta extrapolation.
- Block feature or residual reuse inside selected DiT/GEN layers.
- Attention feature, attention output, or K/V reuse.
- PAB-style broadcast windows across step, layer, spatial, temporal, or attention
  type axes.

Required exploration behavior:

- Inspect target-model traces/code to choose signals, layers, step windows, thresholds,
  and schedules.
- Do not use predefined thresholds or schedules. Discover them from target-model
  code, traces, and artifacts.
- Record rejected mechanisms and why they failed.
- Prove OFF identity before claiming speedup.
- Produce a runnable candidate manifest or a structured-negative proposal for
  orchestrator review; do not use the proposal to stop the fixed-budget loop.
