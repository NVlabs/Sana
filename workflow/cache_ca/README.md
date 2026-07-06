# Cache CA Cache Workflow

Workflow uid: `cache_ca`.

This workflow has one decision-making Codex executor plus an independent blind
Codex visual-evidence node for cache methods on the target model. It is intentionally
not a generic graph runner. The visual reviewer receives only attached A/B
images and cannot make optimization or completion decisions.

## Required Search Direction

TeaCache, EasyCache, and TaylorSeer are the complete candidate set for this
workflow, not examples or anchor families. Other cache methods and cross-family
hybrids are out of scope. Fixed step schedules may be used for calibration but
cannot be retained or ranked. Workflow completion requires faithful
model-adapted candidates from every applicable family and evidence-driven
parameter refinement. The executor maintains
`CACHE-SEARCH-STATE.json`, linking each child parameter point to the full-run
speed, cache statistics, LPIPS, and Codex visual result that motivated it. The
executor owns the final evidence-backed decisions; no reviewer agent is
launched.

The search is adaptive rather than a static grid: quality-passing points may be
made cautiously more aggressive, quality-failing points must be tightened or
narrowed, zero-hit points require signal repair, and hit-positive/no-speed
points require payload or bookkeeping changes.

The optimization target is quality at matched measured E2E inference time.
Candidates are normalized by `candidate_total_s / baseline_total_s` and may be
compared across families only when those ratios are within 2% relative under
the same full-run conditions. The executor tunes each family toward shared time
targets, then uses LPIPS and blind Codex review over all five prompts to select the
quality-preserving recipe and Pareto frontier. Unmatched-speed points cannot
establish a family winner.

## Evaluation Rule

Cache methods must be judged by full generated videos, not single-DiT,
module-only, or microbench evidence.

The ordinary loop is:

```text
executor -> codex_visual_reviewer(full diffusion + LPIPS + blind images)
         -> eval_gate
         -> done when AGENT-STATUS.status=complete
         -> resume_prompt -> executor otherwise
```

`eval_gate` only accepts durable `assess_verdict.json` evidence merged by the
workflow-owned Codex visual reviewer when all of these hold:

- full target-model run completed with `outputs/benchmark.json`;
- frames or `outputs/out.mp4` exist;
- fixed prompt/config contract is preserved: the first five prompts of the
  target model's validation set at the model's official eval profile
  (resolution, duration, frame count, fps, steps, guidance, flow shift, motion
  score);
- numeric `baseline_total_s`, `candidate_total_s`, `speedup`, and `lpips_max`;
- `visual_provider=codex` and `codex_visual_overall=pass`;
- a valid independent `codex_visual_verdict.json`;
- no quality or infrastructure blockers.

## Decision Rule

The executor may retain or discard a concrete cache candidate only from
complete full-evaluation evidence. Discard requires all of these:

- full-workload LPIPS/Codex-visual evidence exists with durable artifacts;
- there is no viable speed/quality tradeoff, or quality failure cannot be fixed
  without removing the method's benefit;
- the negative result is not caused by Slurm, filesystem, quota, collection,
  missing LPIPS/Codex-image evidence, prompt/config mismatch,
  missing logs, or another out-of-method condition;
- the executor documents that no credible cache-level refinement remains;
- for a family-level decision, a faithful seed and adaptive child have been
  evaluated, or concrete source evidence proves structural inapplicability.

Infrastructure and incomplete-assessment failures are always retried or
repaired. The workflow reaches `done` only when the executor writes
`AGENT-STATUS.json.status=complete` and `eval_gate` finds a valid full
assessment.

## Run

Run one node:

```bash
python3 workflow/cache_ca/workflow.py run \
  --experiment-json output/experiments/<id>/experiment.json \
  --experiment-uid <model_id>-cache_ca-0001 \
  --once
```

Run until terminal or max cycles:

```bash
python3 workflow/cache_ca/workflow.py run \
  --experiment-json output/experiments/<id>/experiment.json \
  --experiment-uid <model_id>-cache_ca-0001
```

State is written inside the experiment worktree:

```text
state/workflow-cache_ca-state.json
state/workflow-cache_ca-events.jsonl
```

All executable nodes for this workflow live under `workflow/cache_ca/nodes/`.
Do not import nodes from another workflow at runtime.
