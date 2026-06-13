<!-- ported from Sol-LTX-Infer docs/ltx23_sglang_hq_variants.md @ 29d0d9e -->

# LTX-2.3 Sparse Attention Report Excerpt

The LTX-2.3 HQ runner supported optimized lossy variants on top of KWL:

```bash
bash scripts/run_ltx23_sglang_hq_1080p10s.sh kwl_sparse
bash scripts/run_ltx23_sglang_hq_1080p10s.sh kwl_sparse_cache
```

Relevant sparse-attention settings from the report:

- `kwl_sparse`: KWL plus `piecewise_attn` sparse video self-attention.
- backend: `piecewise_attn` for `transformer` and `transformer_2`
- block size: `64`
- only video self-attention is approximated; other attention falls back dense
- stage 1 schedule: first `3` steps dense, then sparsity ramps from `0.8` to
  `0.9`
- final sparsity: `0.9` (`density=0.1`)
- layer selective guard: layer `0` remains dense by default via
  `piecewise_dense_layers=0`

Latest validated 1080p 10s matrix, prompt `antique brass clockwork train`,
`241` frames, seed `42`, warmup excluded from request runtime:

| Variant | Total s | Denoise s | Total speedup vs KWL | Denoise speedup vs KWL | Notes |
|---|---:|---:|---:|---:|---|
| `kwl` | 69.120 | 63.646 | 1.000x | 1.000x | lossless KWL baseline for this matrix |
| `kwl_cache` | 60.598 | 55.075 | 1.141x | 1.156x | PAB start=6, stage2 PAB off |
| `kwl_sparse` | 61.405 | 56.008 | 1.126x | 1.136x | sparse only |
| `kwl_sparse_cache` | 53.778 | 48.004 | 1.285x | 1.326x | best current combined setting |

Artifacts named in the source report:

```bash
outputs/ltx23-sglang-hq-kwl-sparse-cache-matrix-pab6-stage2off-1080p10s/benchmark_summary.json
outputs/ltx23-sglang-hq-kwl-sparse-cache-matrix-pab6-stage2off-1080p10s/kwl-vs-kwl-sparse-cache-side-by-side.mp4
```
