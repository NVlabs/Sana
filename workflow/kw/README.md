# KWL Kernel Workflow

Workflow uid: `kw`.

This first-stage workflow is a direct, centralized executor/eval/reviewer loop.
It is intentionally not a generic graph runner. The experiment container remains
separate; this workflow consumes an existing experiment worktree and goal.

Run one node:

```bash
python3 workflow/kw/workflow.py run \
  --experiment-json output/experiments/<id>/experiment.json \
  --experiment-uid hunyuan-kw-0001 \
  --once
```

Run until terminal or max cycles:

```bash
python3 workflow/kw/workflow.py run \
  --experiment-json output/experiments/<id>/experiment.json \
  --experiment-uid hunyuan-kw-0001
```

For legacy experiment ids that do not follow `<task>-kw-0000`, pass
`--allow-legacy-experiment-id`.

State is written inside the experiment worktree:

```text
state/workflow-kw-state.json
state/workflow-kw-events.jsonl
```

All executable nodes for this workflow live under `workflow/kw/nodes/`.
Do not import nodes from another workflow at runtime.
