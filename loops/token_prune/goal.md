# Goal: token_prune

Wire feature-norm token pruning into the Cosmos3 DiT denoise block loop through
the shared `efficiency/` engine.

## Objective

Implement the Cosmos3 runtime path for `TokenPrune(keep_ratio=0.5,
method="feat_norm", compensation="prev")`, guarded by env/config, and active
only on the chosen denoise steps after a full seed step.

## Required Work

- Refine the Cosmos3 model spec so `prunable_segment` names only generated video
  tokens.
- Add model-specific gather/scatter helpers if the DiT forward needs side
  tensors to shrink and restore with hidden states.
- Compose `TokenPrune` around the Cosmos3 block loop in
  `runtime/models/dits/cosmos3video.py`.
- Preserve an OFF mode where unset/disabled prune config follows the baseline
  path.
- Produce launcher artifacts using the official video T2V eval profile.

## Done When

- `python loops/token_prune/test_token_prune.py` passes in the sana env.
- `candidates/token_prune.toml` launches in dry-run mode.
- OFF mode matches the baseline for the same prompt, seed, and official config.
- ON mode produces `outputs/out.mp4`, `outputs/benchmark.json`,
  `outputs/quality.json`, `outputs/risk_notes.md`, and
  `outputs/patch_summary.md`.
- Candidate timing shows at least experimental speedup without quality-gate
  regression.
