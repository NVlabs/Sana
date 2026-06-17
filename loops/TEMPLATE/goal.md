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
  loop until max_iters, early_stop, real blocker, or structured-negative evidence.
- A failed candidate gate must be recorded and followed by a meaningfully
  different next hypothesis; it does not complete the goal.
- A successful candidate updates best_per_tier and the loop continues unless the
  orchestrator releases the session.

## Loop

1. Read current `SEARCH_JOURNAL.md`, prior failures, and best_per_tier.
2. Write the next hypothesis and expected improvement.
3. Implement exactly one candidate.
4. Run preflight and OFF identity when applicable.
5. Launch, collect, and assess with the canonical `sana` gate:
   `/home/haozhel/lustre/miniconda3/envs/sana/bin/python search/plan_eval.py --assess <run_dir> --baseline-frames /home/haozhel/lustre/auto-video/runs/20260613-175619-baseline/outputs/frames`
6. Promote/keep, reject/log/loop, block, or structured-negative stop.

## Done When

- loop stopped for max_iters, early_stop, real blocker, structured negative, or
  explicit orchestrator release
- `SEARCH_JOURNAL.md` records each candidate, gate result, and failure signature
- `AGENT-STATUS.json` records status, iters_used, best_per_tier, rejects, and next_commands
- `SUMMARY.md` explains promoted candidates, rejected candidates, artifacts, and blockers
