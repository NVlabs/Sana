# HunyuanVideo SOL v3 — upstream PISA alignment + Morton ablation (2026-07-19)

Reference: hp-l33/Sparse-VideoGen @ pisa-bidirectional
`pisa_kernels/kernels/piecewise_sparse_attn_hyvideo.py`, vendored unmodified
(import renames only) under `techniques/sparse_backends/pisa_hyvideo/`.
Module: `sol_attn_hunyuan_v3.py` (`HUNYUAN_SOL_V3=1`,
`HUNYUAN_SOLV3_MORTON=1` for the Morton variant). Semantics: per-query-block
top-k routing, non-selected KV blocks contribute via block centroids (mass
conserved), text suffix forced exact sink, padding cropped outside the kernel.

## Correctness
density→1.0 = true identity vs dense SDPA (rel_l2≈0.003 bf16, 5 structures,
raster AND morton) — achievable here because top-k covers all blocks, unlike
the colmask route (saturates ~0.53). v3 plumbing fully verified.

## Results (canonical config, seed 42, density 0.15, no compile/cache)

| path | generate | speedup | SSIM | PSNR | LPIPS(alex) |
|---|---|---|---|---|---|
| baseline dense | 863.1s | 1.00x | — | — | — |
| **v2 colmask hard-drop + Morton** | 391.9s | 2.20x | **0.7793** | **15.73dB** | **0.4050** |
| v3 upstream-aligned (raster) | **365.2s** | **2.36x** | 0.6722 | 12.53dB | 0.5396 |
| v3 + Morton | 386.1s | 2.24x | 0.7261 | 13.46dB | 0.4972 |

Runs: v3 raster `20260719-031049` (job 5472663), v3 morton `20260719-064725`
(job 5475290). Per-frame metrics via `scripts/_video_quality_metrics.py`.

## Conclusions
1. Morton reorder helps the upstream kernel (LPIPS 0.540→0.497, SSIM
   0.672→0.726) — confirms raster 64-token strips weaken routing/centroids —
   but does not close the gap.
2. At equal density 0.15, our v2 (colmask per-column threshold + Morton)
   remains the best trajectory fidelity; the query-block-centroid top-k
   routing appears to be a weaker selection signal than the calibrated
   per-column mask, and centroid mass from heterogeneous far blocks adds noise.
3. None of the three pass the delivery gate (LPIPS 0.03–0.37). Open levers on
   every variant: higher density, `*_DENSE_STEPS` (exact early denoising),
   dense first/last layers.
