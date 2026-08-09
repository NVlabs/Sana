# HunyuanVideo diffusers — acceleration release matrix

Run: `fanout_hunyuan_20260620T183315Z` · target `hunyuanvideo-community/HunyuanVideo`
(local `Hunyuan-Diffusers` submodule). Speedup convention = **generation time**
(`generate_s`, the diffusers `pipe()` denoise+VAE-decode call), excluding the
~79s one-time model load/placement, per the Cosmos `[baseline]` precedent.

## Canonical baseline (default official config)
`1280x720 · 129 frames · 24 fps · 50 steps · guidance 6.0 · true_cfg 1.0 ·
max_seq 256 · seed 42` — Slurm job 3467620,
`runs/20260620-163957-hunyuan_diffusers_baseline-gpu-default-video`.
- generation `881.85s`, wall `1007.47s`, peak memory `51.0` alloc / `62.21` reserved GiB.

## Delivery matrix (all DEFAULT config; all Gemini-pass / no new artifacts)

| Tier | Speedup (gen) | gen_s | wall_s | Peak mem (alloc/resv GiB) | LPIPS_max | Gemini | Profile / flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **LOW** ≥1.5x | **1.88x** | 469.90 | 590.29 | 51.0 / 62.21 | 0.366 | pass/none | `teacache_low_delivery` — TeaCache on `time_text_embed`, online auto-min threshold, 1 hit |
| **MEDIUM** ≥2.0x | **2.10x** | 419.17 | 539.16 | 51.0 / 62.21 | 0.031 | pass/none | `teacache_medium_delivery` — reuse start@10, maxhits 2 (quality-first pick) |
| **HIGH** ≥3.0x (default) | **3.28x** | 268.95 | 389.53 | 51.0 / 62.21 | 0.109 | pass/none | `teacache_high_delivery` — reuse start@3, maxhits 3 |
| **HIGH** — fastest validated alt | **3.81x** | 231.29 | 318.80 | 51.0 / 62.21 | 0.109 | pass/none | `teacache_temb_start3_maxhits4` — reuse start@3, maxhits 4 |

Peak memory is unchanged vs baseline (TeaCache reuses cached activations; it does
not raise peak memory). MEDIUM is the quality-first pick among >=2.0x profiles
(LPIPS 0.031); a faster 2.63x medium (LPIPS 0.092) is also Gemini-pass if speed
is preferred over LPIPS. For HIGH, `maxhits3` (3.28x) is the conservative default
and `maxhits4` (3.81x) is the fastest validated alternate at essentially equal
quality (LPIPS 0.109 vs 0.109).

## Mechanism (single winning dimension: step_cache / TeaCache)
All tiers are TeaCache on the Hunyuan `time_text_embed` output (timestep+text
embedding) with a relative-L1 controller, implemented in
`Hunyuan-Diffusers/hunyuan_diffusers/step_cache_runtime.py`, driven by
`gpu_infer.py`. Knobs: `start` (warm-up calls run full before any reuse) and
`maxhits` (max consecutive cached steps); higher = faster, slightly higher LPIPS.
Token-prune, sparse-attention, NVFP4-FFN and KWL-fusion did not yield a
quality-preserving speedup beyond ~1.0-1.3x on HunyuanVideo (aggressive settings
were true quality cliffs at LPIPS 0.9+), so they are not in the delivery.

## Quality gate
Authoritative aligned gate: OFF identity where applicable + aligned LPIPS over the
129 canonical baseline frames + aligned pairwise NVIDIA-Gemini, `--model
hunyuan_diffusers`. The coordinator gate is hardened against the MI-6 pairwise
hallucinated false-fail (see `docs/mechanism-issues.md`). HIGH winners were
independently re-gated.

## Rollback
Every profile is OFF-guarded: disabling the TeaCache flag recovers the baseline
denoise path exactly (OFF identity). Rollback = run with TeaCache disabled.

## Evidence (assess_verdict.json)
- LOW: `output/fanout_runs/fanout_hunyuan_20260620T183315Z/integration/runs/20260621-020325-hunyuan_teacache_low_delivery/`
- MEDIUM: `.../integration/runs/20260621-014909-hunyuan_teacache_medium_delivery/`
- HIGH (maxhits3): `.../integration/runs/20260621-013714-hunyuan_teacache_high_delivery/`
- HIGH (maxhits4 alt): `.../step_cache/runs/20260621-011511-hunyuan_teacache_temb_start3_maxhits4/`
  (fan-out gate + `assess_verdict_orchestrator.json` hardened-gate re-confirm)
- Fan-out: 76 config gated across 5 dimensions; step_cache swept all tiers.
