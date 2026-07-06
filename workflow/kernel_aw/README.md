# Kernel AW Kernel Workflow

Workflow uid: `kernel_aw`.

This workflow is a direct executor/eval/reviewer loop for kernel-agent
experiments where methods must not be discarded casually. It is intentionally
not a generic graph runner. The experiment container remains separate; this
workflow consumes an existing experiment worktree and goal.

The executor first profiles the registry-resolved full DiT and reproduces the
known same-scope global-attention kernel frontier from the read-only target
model reference. It then performs profile-ranked novel search. Module microbenchmarks
screen individual mechanisms; cumulative full-DiT OFF/ON benchmarks determine
composition priority.

## Discard Rule

Executor may implement, retry, repair, refine, and request review, but executor
may not make a final discard decision.

A method can be discarded only by the reviewer and only when all of these hold:

- smooth single-DiT/module-level evidence exists with durable artifacts;
- there is no meaningful speed, memory, or correctness/quality proxy improvement
  at that level;
- the negative result is not caused by Slurm, filesystem, quota, collection,
  missing API key, missing logs, or another out-of-method condition;
- reviewer judges there is no remaining credible operator/module-level
  optimization space for that method.

Microbench numerical drift alone is not a discard reason. If the algorithm is
mathematically correct, keep the method for reviewer tolerance judgment. If
there is a semantic implementation error, executor rewrites and reruns rather
than discarding.

Retention does not imply immediate refinement. The reviewer may retain and park
a candidate while the executor switches to a higher-impact profiled family.

## Evaluation Shape

The ordinary loop is:

```text
executor -> eval_gate(current candidate) -> reviewer
```

The first iteration and composition checkpoints require a registry-resolved
full-DiT gate. A synthetic or module-only gate is screening evidence and must be
labeled accordingly. The evaluator reads `active_candidate_id` and
`active_gate`; it does not reuse an older smooth result.

Full denoising/full diffusion is not part of this loop. When the reviewer writes
an exit decision, the workflow runs:

```text
reviewer -> final_full_eval(full diffusion + Gemini) -> done
```

If terminal full evaluation is missing, blocked, or Gemini visual quality does
not pass, the workflow writes a reviewer resume prompt. The reviewer should then
send concrete follow-ups back to the executor.

## Run

Run one node:

```bash
python3 workflow/kernel_aw/workflow.py run \
  --experiment-json output/experiments/<id>/experiment.json \
  --experiment-uid <model_id>-kernel_aw-0001 \
  --once
```

Run until terminal or max cycles:

```bash
python3 workflow/kernel_aw/workflow.py run \
  --experiment-json output/experiments/<id>/experiment.json \
  --experiment-uid <model_id>-kernel_aw-0001 \
  --max-cycles 400
```

`max_cycles` counts workflow node transitions, not optimization candidate
iterations. Candidate iterations are tracked separately in
`AGENT-STATUS.json.candidate_iteration`.

For legacy experiment ids that do not follow `<task>-kernel_aw-0000`, pass
`--allow-legacy-experiment-id`.

State is written inside the experiment worktree:

```text
state/workflow-kernel_aw-state.json
state/workflow-kernel_aw-events.jsonl
```

All executable nodes for this workflow live under `workflow/kernel_aw/nodes/`.
Do not import nodes from another workflow at runtime.
