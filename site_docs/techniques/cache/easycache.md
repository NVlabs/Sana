# EasyCache

EasyCache is the cache policy selected for SANA-Video. It is a runtime-adaptive feature reuse method that balances speed and quality online instead of relying on a fixed offline schedule.

## Sol-Engine placement

Sol-Engine uses EasyCache in the SANA-Video denoising transformer path with
`--easycache 0.1` in the full optimization stack.

SANA-Video already uses an efficient linear-attention architecture, so EasyCache becomes the main algorithm-level acceleration component before kernel-level optimization is applied.

## Tunable knobs

- cache threshold: controls reuse aggressiveness.
- warmup: keeps early denoising steps dense when needed.
- subsampling: estimates feature change with reduced overhead.

## Validation

Use the same prompt, seed, scheduler, resolution, and frame count when comparing baseline and full optimization outputs.

## Wan / LingBot usage

- **Wan (5B / 14B).** EasyCache is the strongest single lever on both: 5B at threshold 0.036 reuses ~47% of steps (1.90×); 14B at threshold 0.30 reuses ~14/40 steps (1.42×).
- **Per-stage cache (LingBot).** The two-stage LingBot pipeline caches **each stage independently** with its own threshold: the 40-step base (threshold 0.08, ~14/40 reused, 1.18×) and the 8-step refiner (threshold 0.25, ~2/8 reused, 1.10×). The refiner needs a looser threshold because its few coarse steps change more per step; its incremental quality is high (SSIM ≈ 0.98 vs base-only-cache). The reuse decision is a pure function of the replicated latents, so it is identical on every context-parallel rank (no collective divergence).

## References

- [Sol-Engine paper](http://arxiv.org/abs/2606.23743)
- [EasyCache](https://github.com/H-EmbodVis/EasyCache)
