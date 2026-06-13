# Goal: <loop-id>

## Objective

<Define the bounded implementation or experiment objective.>

## Context

- Execution repo: `Sol-LTX-Infer`
- Orchestration repo: `autovideo`
- Eval profile: `evals/profiles/official_video_t2v.toml`

## Constraints

- Preserve the eval profile's official target-model config for comparable numbers.
- Do not modify unrelated implementation areas.
- Keep candidate OFF behavior identical or explain why not applicable.

## Done When

- candidate can be launched through `scripts/launch_candidate.py`
- run can be collected through `scripts/collect_run.py`
- eval gates are reported
- blockers are explicit
