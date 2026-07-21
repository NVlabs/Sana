# HunyuanVideo SOL — split-screen artifact root cause + fix (2026-07-19)

Symptom: greenhouse (Penguin #298, a left|right two-region composition prompt)
rendered as TWO STACKED SCREENS (top: greenhouse interior / bottom: unrelated
open field) under both sparse variants (v2 colmask AND v3 PISA), never under
baseline. Looked like a wrong attention mask.

## Audit (independent subagent, full report in session log)
- Vendored PISA kernel is byte-identical to upstream (only 3 import renames);
  `piecewise_sparse_attn_0th.py` diff-empty. Integration matches upstream
  `svg/models/hyvideo/attention.py::pisa_attention` line-for-line (query
  routing, crop, text sink, scale/dtype, ragged 118800%64=16 block handling).
  **Mask/kernel misalignment ruled out.**
- One real divergence found: our Morton bit-lane order (f fastest) vs upstream
  (w fastest). Ours produced pathological 64-token blocks (up to 33x45x32
  extent). FIXED: `sol_attn_hunyuan_v3._morton3d_perm_v3` now ports upstream's
  `_morton3d_perm` exactly; block extents now mean (3.95,4.17,4.43), max
  (6,13,20).
- Second real bug found while testing: the SOL (step,layer) clock is advanced
  by a transformer forward pre-hook, so the WARMUP pass exhausted
  `*_DENSE_STEPS` before the timed pass (first dense-steps run was
  pixel-identical to fully-sparse). FIXED in `gpu_infer.py`: ctx clocks
  (v1/v2/v3) reset after warmup, before the timed pass.

## Isolation experiments (greenhouse, d=0.15, seed 42, vs same baseline)

| run | config | generate | speedup | SSIM | PSNR | LPIPS | split? |
|---|---|---|---|---|---|---|---|
| gh-fixm | fixed-lane Morton, fully sparse | 388s | 2.22x | 0.177 | 8.6dB | 0.654 | YES |
| gh-raster | raster (upstream default), fully sparse | ~390s | 2.21x | 0.180 | 9.0dB | 0.646 | YES |
| **gh-fixm-ds10b** | fixed Morton + **dense first 10 steps** | 490s | **1.76x** | **0.701** | **21.1dB** | **0.154** | **NO** |

## Conclusions
1. Split-screen = **early-denoising layout decoherence** under fully-sparse
   d=0.15 attention, triggered by two-region composition prompts. Not a mask
   bug, not the Morton curve (both orders split identically).
2. `HUNYUAN_SOLV3_DENSE_STEPS=10` (upstream's `pisa_dense_first_steps`
   equivalent) eliminates it AND takes trajectory fidelity from LPIPS 0.65 to
   **0.154** — comfortably inside the historical delivery-gate range
   (0.03-0.37) — at 1.76x vs 2.2x fully sparse.
3. Next: sweep dense_steps {5,10} x density {0.15,0.25} across the 4 official
   prompts (incl. v2 + dense_steps, likely faster at similar quality) to pick
   the delivery point.
