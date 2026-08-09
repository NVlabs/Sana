# Evals

This directory defines how acceleration transfeat are accepted or rejected.

The design is intentionally small: every transfeat points at one eval profile,
and every collector/report should speak the same vocabulary.

## Core Gates

| Gate | Purpose | Required for promotion |
| --- | --- | --- |
| `artifact` | Did the run produce the expected files? | yes |
| `official_config` | Did it use the comparable target-model settings? | yes |
| `performance` | Is it faster than the selected baseline? | yes, threshold is transfeat-specific |
| `off_identity` | Does disabling the technique recover baseline behavior? | yes for code/patch transfeat |
| `quantitative_quality` | Are simple video metrics within tolerance? | yes when available |
| `visual_artifact` | Does a vision judge see new artifacts? | yes |

## Minimal Promotion Rule

A transfeat may be promoted only when:

```text
artifact == pass
official_config == pass
performance == pass
off_identity == pass or not_applicable
quantitative_quality == pass or explicitly_deferred
visual_artifact == pass
```

For early M2/M3 bring-up, `quantitative_quality` may be deferred only if the
report says why and includes side-by-side review artifacts.

## Profiles

- `profiles/official_video_t2v.toml`: official text-to-video baseline/comparison profile for the current target model.
- `rubrics/gemini_visual_artifact_gate.md`: prompt for a Gemini or equivalent
  multimodal judge.

## Current Gap

The current repo can launch and collect runs, and includes a dry-runnable
NVIDIA-hosted Gemini judge wrapper under `tools/vision/`. It does not yet compute
quantitative frame metrics automatically.
