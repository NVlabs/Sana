# Acceptance

## Required Gates

- Independent test:
  `~/lustre/miniconda3/envs/sana/bin/python loops/sparse_attention/test_sparse_attention.py`
  exits 0.
- Search enumeration:
  `~/lustre/miniconda3/envs/sana/bin/python search/search.py --model <target-model>`
  lists this dimension as composable when the target model declares
  `SWAPPABLE_ATTENTION`, or skipped when it does not.
- Official config comes from the target model profile and matches
  `evals/profiles/official_video_t2v.toml` when that profile is selected.
- OFF identity: with sparse env disabled or all attention routed to dense
  fallback, same seed, same prompt, and same official config recover the
  baseline path.
- Performance: record total and denoise time; promote at >= 1.10x denoise
  speedup and treat >= 1.03x as exploratory.
- Quantitative quality: `quality.json` passes frame count, duration, sharpness,
  and temporal jitter thresholds from the official profile.
- Visual artifact gate: no new medium/high artifacts relative to baseline.

## Rejection Conditions

- `outputs/out.mp4` is missing or empty.
- Candidate config changes official config without a new baseline.
- Target model claims `SWAPPABLE_ATTENTION` before the runtime attention seam is
  wired.
- Sparse mode cannot be disabled cleanly.
- New visual artifacts exceed the official gate.
