# Search Space

This directory is the canonical search-space contract for native Codex
implementation goals. It names method families and axes to investigate; it is
not a recipe archive and must not be treated as a fixed hyperparameter grid.

- Original source: https://github.com/Efficient-Large-Model/Sol-LTX-Infer/tree/cosmos_exp/search_space_docs
- Imported source: git@github.com:Efficient-Large-Model/Sol-LTX-Infer.git
- Branch: cosmos_exp
- Commit: 4049c2a0588d39c2939eef9a4700ce24eadba5b1
- Source path: search_space_docs

See `SOURCE.json` for machine-readable provenance.

## Method Families

- `01_cache.md`: denoiser-step caching, including TeaCache and EasyCache
  directions.
- `02_token_pruning.md`: token pruning, token merging, token masking, and
  region-aware token-routing directions.
- `03_quantization.md`: NVFP4 linear quantization with module profiling and
  dense guards by layer and denoising step.
- `04_sparse_attention.md`: PISA sparse attention with density, block size,
  dense guards, routing mode, and remainder approximation.
- `05_kernel_fusion.md`: kernel fusion and lossless operator optimization,
  including FFN residual epilogues, QK norm/RoPE fusion, attention bias fusion,
  AdaLN reuse, QKV batching, compile-based fusion, and CUDA stream overlap.

Goal agents should turn these directions into model-specific experiments by
reading Cosmos3 inference code directly. Layer, step, signal, threshold, routing,
and fallback choices are discovered by each subagent from code, traces, and
local reproduction artifacts.
