# Goal: <loop-id>

## Objective

<Define the bounded per-dimension search objective. This is a loop, not a
one-candidate target.>

## Context

- Execution repo: `Sol-LTX-Infer`
- Orchestration repo: `autovideo`
- Eval profile: `evals/profiles/official_video_t2v.toml`

## Constraints

- Preserve the eval profile's official target-model config for comparable numbers.
- Do not modify unrelated implementation areas.
- Keep candidate OFF behavior identical or explain why not applicable.
- Follow `docs/fanout-loop-contract.md`: propose, implement, gate, learn, and
  loop until max_iters, real blocker, or explicit orchestrator release.
- A failed candidate gate must be recorded and followed by a meaningfully
  different next hypothesis; it does not complete the goal.
- Retain a candidate when quality improves or speed/memory improves; discard it
  only when neither quality nor speed/memory improves.
- Default fan-out budget is fixed `max_iters = 40`; `early_stop_patience = 0`
  disables patience early stop. `no_improve_count` is telemetry.
- A structured-negative decision is recorded as a proposal/failure signature; it
  does not stop the default fixed-budget loop by itself.
- Final low/medium/high selection uses 1.5x/2.0x/3.0x speed targets and ranks
  quality with both aligned pairwise Gemini and LPIPS.

## Loop

1. Read current `SEARCH_JOURNAL.md`, prior failures, and frontier candidates.
2. Write the next hypothesis and expected improvement.
3. Implement exactly one candidate.
4. Run preflight and OFF identity when applicable.
5. Launch, collect, and assess with the canonical `sana` gate:
   `/lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/envs/hunyuanvideo15/bin/python search/plan_eval.py --assess <run_dir> --baseline-frames /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/agent_deploy/Sol-LTX-Infer/runs/20260613-175619-baseline/outputs/frames --out <run_dir>/assess_verdict.json`
6. Retain frontier, discard/log/loop, reject/log/loop, block, or log a structured-negative proposal and continue.

## Done When

- loop stopped for max_iters, real blocker, or explicit orchestrator release
- `SEARCH_JOURNAL.md` records each candidate, gate result, and failure signature
- `AGENT-STATUS.json` records status, iters_used, frontier_candidates,
  discarded/rejected candidates, no_improve_count, remaining_hypotheses,
  agent_recommendation, and next_commands
- `SUMMARY.md` explains retained candidates, discarded/rejected candidates, artifacts, and blockers
