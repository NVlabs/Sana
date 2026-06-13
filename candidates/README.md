# Candidates

Candidate manifests describe what to run. They do not contain large logs,
videos, model weights, or generated frames.

Use one TOML file per candidate. The launcher currently supports baseline-style
manifests and is intentionally conservative: it prepares a run bundle first, and
only executes when explicitly asked.

## Current M1 Candidate

- `baseline.toml`: official Cosmos3-Super baseline using
  `Sol-LTX-Infer/scripts/run_cosmos3_sglang.sh`.

## Future M2-M5 Candidates

These should be added as independent manifests/goals:

- sparse attention PISA
- step cache
- token pruning
- TeaCache
- KWL operator fusion
- NVFP4 FFN
- full-stack composed preset

Keep acceleration implementation details in `Sol-LTX-Infer`. This directory
should only describe the candidate and how to launch or validate it.
