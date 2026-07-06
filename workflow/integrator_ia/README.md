# Integrator IA Workflow

Workflow uid: `integrator_ia`.

This workflow materializes one pinned kernel canonical stack, one measured PISA
implementation, and one measured cache implementation inside a new isolated
experiment for the target model. Donor experiments are read-only. The workflow measures
their interactions and emits conservative, balanced, and aggressive recipes in
one hash-pinned downstream interface, `INTEGRATION-DELIVERY.json`.

## Explicit Graph

```text
check_sources
  -> executor
  -> check_integration
  -> codex_visual_reviewer
  -> final_gate
  -> done

Any repairable integration or quality failure:
  -> resume_prompt -> executor
```

`check_sources` adapts existing workflow-specific outputs into a pinned source
inventory. Runtime node implementations are not imported from donor workflows.
All executable nodes are owned by `workflow/integrator_ia/nodes/`.

## Callable Interfaces

The graph is fixed by `workflow.py`. These workflow-local callable contracts
are exposed to the executor but are not unconditional scheduler phases:

- `delivery_materializer`: port every pinned snapshot into the integration
  worktree and write file-level provenance;
- `composition_probe`: measure all eight kernel/PISA/cache toggle combinations
  from one integrated source tree;
- `full_diffusion_eval`: run the fixed five-prompt workload at the target
  model's official eval profile for all three delivery recipes and emit
  warm-sample timings, configuration, component counters, and videos.

Their contracts live under `nodes/callable/`. The executor decides when to use
them, while `integration_gate`, `codex_visual_reviewer`, and `final_gate`
programmatically decide whether their evidence is sufficient.

## Source Interfaces

- Kernel: `AGENT-STATUS.json.canonical_on_manifest` plus its selected manifest
  and latest integrated full-DiT gate.
- PISA: one measured entry in `PISA-RECIPES.json`, defaulting to
  `visually_indistinguishable`.
- Cache: one explicit candidate id from `AGENT-STATUS.json.candidates` with a
  passing Codex visual assessment.

Selection is explicit and hash-pinned at workflow start. Selected manifests,
assessments, and declared implementation files are copied into
`state/integration-source-snapshots/`. The executor reads only those local
snapshots, so it never tracks a donor's moving `latest` state after the
inventory is written.

## Required Integration Evidence

The executor must port all three components into the integration worktree and
produce:

- file-level source/destination provenance;
- all-off identity from the integrated source tree;
- all eight measured toggle conditions, including all pairwise combinations;
- enabled-component activity with zero fallbacks and disabled-component zero
  activity;
- distinct conservative, balanced, and aggressive recipes with strictly
  increasing measured speedups;
- the fixed five-prompt run at the target model's official eval profile for every recipe;
- LPIPS plus independent blind Codex review for every recipe;
- final `INTEGRATION-DELIVERY.json` with hashes and per-recipe activation.

Performance means only warm per-sample inference from text-encoder computation
through synchronized VAE-decode completion. Process startup, model/text
encoder/VAE loading, one-time compile and warmup, frame extraction, video
encoding/writing, upload, and teardown are excluded. Visual artifacts are
produced outside the timer.

Quality is tiered: conservative accepts through low severity; balanced accepts
an isolated medium regression; aggressive accepts broader medium differences.
High, critical, and inconclusive evidence are rejected. LPIPS is diagnostic and
has no universal hard threshold. Slurm, filesystem, frame extraction, and
visual-review failures are retried as infrastructure.

The only stable downstream interface is `INTEGRATION-DELIVERY.json`, described
by `contracts/integration_delivery.schema.json`. Donor status files and the
executor's intermediate files are not downstream runtime APIs.

## Create Experiment

```bash
python3 scripts/create_model_experiment.py \
  --model <model_id> \
  --workflow-uid integrator_ia \
  --experiment-uid <model_id>-integrator_ia-0001
```

## Run With Current Deliveries

The following example selects the current quality-preserving PISA recipe and a
measured TaylorSeer cache candidate. Paths may point to an experiment directory,
`experiment.json`, or its worktree.

```bash
python3 workflow/integrator_ia/workflow.py run \
  --experiment-json output/experiments/<model_id>-integrator_ia-0001/experiment.json \
  --experiment-uid <model_id>-integrator_ia-0001 \
  --kernel-delivery output/experiments/<model_id>-kernel_aw-0005 \
  --pisa-delivery output/experiments/<model_id>-attention_pa-0002 \
  --pisa-recipe visually_indistinguishable \
  --cache-delivery output/experiments/<model_id>-cache_ca-0005 \
  --cache-candidate <cache-candidate-id> \
  --max-cycles 100
```

State is written to:

```text
state/integration-source-inventory.json
state/integration-source-snapshots/
state/workflow-integrator_ia-state.json
state/workflow-integrator_ia-events.jsonl
state/integration-gate.json
state/final-gate.json
```

The current donor workflows may continue running after the source inventory is
pinned; their later changes do not alter this integration experiment.
