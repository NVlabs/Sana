# Search Space

This directory is the canonical search-space contract for native Codex
implementation goals. It names method families and axes to investigate; it is
not a recipe archive and must not be treated as a fixed hyperparameter grid.

## Method Families

- `01_cache.md`: denoiser-step caching, including TeaCache and EasyCache
  directions.
- `02_token_pruning.md`: token pruning, token merging, token masking, and
  region-aware token-routing directions.
- `03_quantization.md`: NVFP4 linear quantization with hardware/runtime
  preflight, module profiling, TE recipe variants, backend/padding policy, and
  dense guards by layer and denoising step.
- `04_sparse_attention.md`: training-free sparse attention, including
  piecewise/PISA, Sparse VideoGen-style spatial/temporal routing, SVG2 semantic
  permutation, AdaSpa online search and mask reuse, SpargeAttn proxy masks, LVSA
  rotating anchors, SVOO QK co-clustering, HASTE head-wise budgets, and
  MInference-style dynamic patterns.
- `05_kernel_fusion.md`: quality-gated kernel and operator optimization,
  including GEMM epilogues, norm/modulation/residual fusion, attention-adjacent
  dense fusion, compile or CUDA graph capture, layout/copy elimination, launch
  batching, stream overlap, decode/postprocess fusion, backend selection,
  approximate kernel/backend paths, and fallback policy.

Goal agents should turn these directions into target-model experiments by
reading the live inference code directly. Layer, step, signal, threshold,
routing, and fallback choices are discovered by each subagent from code, traces,
and local reproduction artifacts.
