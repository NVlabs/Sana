# Acceptance

## Required Gates

- Independent test:
  `~/lustre/miniconda3/envs/sana/bin/python loops/sparse_attention/test_sparse_attention.py`
  exits 0.
- Launcher dry-run: `scripts/launch_candidate.py candidates/sparse_attention.toml
  --mode dry-run` prepares a run bundle without submitting GPU work.
- Official config matches `evals/profiles/official_video_t2v.toml`.
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
- Candidate changes official config without a new baseline.
- Cosmos3 claims `SWAPPABLE_ATTENTION` before the runtime attention seam is
  wired.
- Sparse mode cannot be disabled cleanly.
- New visual artifacts exceed the official gate.
