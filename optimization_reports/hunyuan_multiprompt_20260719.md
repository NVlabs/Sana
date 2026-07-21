# HunyuanVideo SOL — multi-prompt evaluation on official prompts (2026-07-19)

Prompts: the official Tencent demo prompt ("A cat walks on the grass, realistic
style.") + 3 from the official **Penguin Video Benchmark**
(`assets/PenguinVideoBenchmark.csv`, 602 prompts): #98 clown (human/mood),
#298 greenhouse (scene + camera pan), #398 sparks (high-frequency detail).
Canonical config (1280x720, 129f, 50 steps, seed 42), density 0.15, per-prompt
baseline reference. v3 here is the FULLY upstream-aligned version (text queries
exact dense, matching Sparse-VideoGen's integration) + morton3d order.

## Speed (generate_s; baseline ≈ 863s on every prompt)
- v2 colmask+Morton: 379–390s → **2.22–2.28x**
- v3-aligned+Morton: 392–394s → **2.19–2.20x**

## Trajectory fidelity vs baseline (SSIM / PSNR / LPIPS(alex), 129 frames)

| prompt | v2 SSIM | v3m SSIM | v2 PSNR | v3m PSNR | v2 LPIPS | v3m LPIPS |
|---|---|---|---|---|---|---|
| cat | 0.216 | **0.246** | 15.19 | **15.47** | 0.356 | **0.322** |
| clown | 0.399 | **0.432** | 11.12 | **11.57** | 0.667 | **0.636** |
| greenhouse | 0.170 | **0.174** | **9.39** | 8.58 | **0.642** | 0.653 |
| sparks | 0.470 | **0.494** | 11.75 | **12.23** | 0.577 | **0.569** |
| **mean (official)** | 0.314 | **0.337** | 11.86 | **11.96** | 0.561 | **0.545** |
| (lion, custom, pre-fix v3) | **0.779** | 0.726 | **15.73** | 13.46 | **0.405** | 0.497 |

Runs: `runs/20260719-0822*` (jobs 5476303–5476316).

## Conclusions
1. **Multi-prompt flips the single-prompt ranking**: the fully-aligned
   v3+Morton beats v2 on 3/4 official prompts and on the mean (LPIPS 0.545 vs
   0.561, SSIM 0.337 vs 0.314). The lion prompt (v2 strongly ahead) was an
   outlier; per-prompt variance dwarfs the variant gap.
2. **Both variants diverge heavily from the baseline trajectory on official
   prompts** (LPIPS 0.32–0.67, SSIM 0.17–0.49): at density 0.15 with zero
   dense warmup steps, 50-step trajectories separate into visibly different
   videos regardless of routing scheme. Trajectory-fidelity gates cannot be
   passed by density tuning of the routing alone.
3. Next lever, in priority order: `*_DENSE_STEPS` (exact early denoising — the
   trajectory forks in the first steps), then density. Both variants expose
   these knobs; upstream ships the same (`pisa_dense_first_steps`).
