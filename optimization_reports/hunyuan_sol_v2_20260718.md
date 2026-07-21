# HunyuanVideo SOL Attention v2 — independent dense-text / sparse-video path (2026-07-18)

## What was built
`techniques/sparse_backends/sol_attn_hunyuan_v2.py` — independent implementation
of the joint [video, text] split: text↔anything exact dense, ONLY video×video
sparse (PISA2 SM100 colmask kernel, consumed unmodified via
`integrations.wan.run/calibrate_tau`), exact disjoint-key LSE merge (fp32,
query-chunked). Own custom op `sol2::attn_hunyuan_v2` + dispatch hook; v1
(`sol_attn_backend.py`) untouched. Runtime gate: `HUNYUAN_SOL_V2=1`
(`HUNYUAN_SOLV2_DENSITY/TAU/DENSE_STEPS/DENSE_LAYERS/QCHUNK`). Candidate:
`candidates/hunyuan_video_sol_v2_only.toml`.

## Correctness findings (scripts/_hunyuan_sol_v2_correctness.py, GB200)
1. **The suspected LSE-merge-convention bug does NOT exist.** Injecting a pure
   dense fake kernel (same `(out, lse)` contract) through the full v1/v2 paths
   reproduces dense SDPA to bf16 error (rel_l2≈0.003) on 4 sequence structures
   incl. ragged grid and no-padding.
2. The old `_hunyuan_sol_correctness.py` premise ("density→1.0 must equal
   dense") is unachievable: the colmask route saturates at ~0.53–0.61 density
   even at tau=0, and random-tensor softmax is near-uniform, so its large
   errors were a false alarm, not a merge bug.
3. Kernel LSE units verified: exp(lse_kernel − lse_dense_video) ∈ (0.51, 0.80]
   — natural log of q·k·scale over routed keys, merge-compatible.
4. Real bug found+fixed in v2 (v1 has it too, left untouched): with ZERO valid
   text keys, softmax(all −inf)=NaN poisoned the merge via 0×NaN. v2 zeroes the
   NaN rows (their merge weight is exactly 0); degeneracy test now exact.

## End-to-end run (job 5464947, runs/20260718-165950-hunyuan_video_sol_v2_only)
Canonical config, density 0.15, no compile/cache. vs baseline
`runs/20260716-025321` (generate 863.10s):
- generate_s **391.89** → **2.20x** (v1 sol_only 20260717: 383.18s / 2.25x)
- peak mem 57.88 GiB alloc / 72.02 reserved; no dense fallback; 129 frames.

## Quality vs baseline frames (SSIM/PSNR/LPIPS(alex), per-frame, n=129)
- v2: SSIM 0.7793±0.0241, PSNR 15.73 dB, LPIPS **0.4050**±0.0428
- v1: identical to 4 decimals (paths are numerically equivalent).

**Conclusion:** the deviation is intrinsic to the sparse video×video
approximation at density 0.15 on HunyuanVideo — not a plumbing/merge bug.
LPIPS 0.405 ≫ the TeaCache delivery gate range (0.03–0.37), so SOL-only at
d=0.15 is NOT deliverable as-is. Levers already wired in v2: raise density,
`HUNYUAN_SOLV2_DENSE_STEPS` (keep early denoising exact),
`HUNYUAN_SOLV2_DENSE_LAYERS`. A density/dense_steps sweep is the next step.
