# Acceptance

- Independent gate passes:
  `~/lustre/miniconda3/envs/sana/bin/python loops/token_prune/test_token_prune.py`.
- Search candidates compose through `efficiency/` on any target model that
  declares the `prunable_tokens` capability, or are skipped cleanly.
- OFF mode is baseline: same official config, prompt, seed, and disabled token
  pruning recover the baseline path.
- ON mode uses `keep_ratio=0.5`, `method=feat_norm`, `compensation=prev`, and a
  documented denoise-step schedule.
- Official config matches `evals/profiles/official_video_t2v.toml`.
- Artifacts follow `docs/artifact-contract.md`: `outputs/out.mp4`,
  `outputs/benchmark.json`, `outputs/quality.json`, `outputs/risk_notes.md`,
  `outputs/patch_summary.md`, and `outputs/collection.json`.
- Experimental gate: denoise speedup is at least `1.03x` versus the accepted
  baseline run.
- Promotion gate: warmed denoise speedup is at least `1.10x` and the visual
  artifact gate reports no new artifacts.
