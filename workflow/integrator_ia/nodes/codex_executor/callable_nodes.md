# Workflow-Local Callable Nodes

Use only callable contracts under `workflow/integrator_ia/nodes/callable/`.
They describe capabilities available to this executor; they are not donor
workflow imports.

- `delivery_materializer`: port pinned implementation files and write the
  source lock.
- `composition_probe`: measure all eight integrated-source toggle combinations
  and dispatch interactions under the warm-sample timing contract.
- `full_diffusion_eval`: run the fixed workload for conservative, balanced, and
  aggressive recipes, timing only warm sample inference while producing visual
  artifacts outside the timer.

The explicit workflow graph invokes `codex_visual_reviewer` after the executor
exits. Do not launch another visual reviewer yourself.
