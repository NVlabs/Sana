# KR Kernel Retention Workflow

Workflow uid: `kr`.

This workflow is a direct executor/eval/reviewer loop for kernel-agent
experiments where methods must not be discarded casually. It is intentionally
not a generic graph runner. The experiment container remains separate; this
workflow consumes an existing experiment worktree and goal.

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

## Evaluation Shape

The ordinary loop is:

```text
executor -> eval_gate(single-DiT/module) -> reviewer
```

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
python3 workflow/kr/workflow.py run \
  --experiment-json output/experiments/<id>/experiment.json \
  --experiment-uid hunyuan-kr-0001 \
  --once
```

Run until terminal or max cycles:

```bash
python3 workflow/kr/workflow.py run \
  --experiment-json output/experiments/<id>/experiment.json \
  --experiment-uid hunyuan-kr-0001
```

For legacy experiment ids that do not follow `<task>-kr-0000`, pass
`--allow-legacy-experiment-id`.

State is written inside the experiment worktree:

```text
state/workflow-kr-state.json
state/workflow-kr-events.jsonl
```

All executable nodes for this workflow live under `workflow/kr/nodes/`.
Do not import nodes from another workflow at runtime.
