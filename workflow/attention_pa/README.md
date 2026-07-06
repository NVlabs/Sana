# Attention PA PISA Workflow

Workflow uid: `attention_pa`.

This workflow uses one persistent decision-making Codex executor to port and
tune PISA on the experiment-local target model transformer. A separate blind
Codex visual reviewer receives only attached A/B frame images and contributes
quality evidence; it cannot tune, retain, discard, or complete recipes.

Authoritative implementation:

- `/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/layers/attention/backends/piecewise_attn.py`
- validated commit: `7546a4bd1d382923ef4876945172655a84d23686`
- validated file SHA-256:
  `bfad198d834d21254492676ad210e6d5393c88b236bd3b4b793c99a6ac960fb3`
- GB200 microbenchmark establishing PISA-backend viability at a representative
  video-attention shape

The executor copies and adapts this local implementation into the isolated
experiment. It does not reimplement PISA from a paper or public repository, and
it never modifies the shared authoritative checkout.

## Search Product

The executor discovers a measured policy over:

```text
attention mode = dense | pisa
policy axes    = layer x denoising step x attention type
PISA density   = fraction of blocks computed exactly
PISA sparsity  = fraction handled by the approximate remainder
density        = 1 - sparsity
```

It must verify a faithful PISA exact-or-approximate backend. Simply dropping
unselected blocks, setting unused environment variables, or falling back to
dense attention does not count.

The search first maps the target model's attention shapes and costs, then profiles layer and
step sensitivity, brackets density in tolerant regions, and finally measures
composed policies. Cross/text attention stays dense unless separately proven.

## Required Recipes

`PISA-RECIPES.json` must contain three full-evaluation-backed recipes:

- `visually_indistinguishable`: no new visible blind-Codex artifact;
- `acceptable_loss`: only documented low-severity non-temporal differences;
- `aggressive`: maximum measured speed with all quality loss disclosed.

Every recipe includes the concrete backend and source hash, block size,
layer/step schedule, density/sparsity, dense fallback, dispatch/fallback counts,
full end-to-end speed, LPIPS, Codex visual severity, run config, and evidence paths.

## Evaluation

Recipe-relevant candidates run the target model's fixed official workload: the
first five prompts of its validation prompt set at the model's official eval
profile (resolution, duration, frame count, fps, steps, guidance, flow shift,
and motion score). LPIPS and aligned blind Codex assessment are mandatory.

A valid visual failure is useful evidence for quality-boundary search. Missing
frames, Codex image attachments/verdict, LPIPS output, Slurm completion, or benchmark
data is infrastructure and must be repaired rather than classified as PISA
quality.

The loop is:

```text
executor -> codex_visual_reviewer(full diffusion + LPIPS + blind images)
         -> eval_gate(recipe schema)
         -> done when AGENT-STATUS.status=complete and all recipes validate
         -> resume_prompt -> executor otherwise
```

## Run

Create an isolated experiment for the target model with an id such as
`<model_id>-attention_pa-0001`:

```bash
python3 scripts/create_model_experiment.py \
  --model <model_id> \
  --workflow-uid attention_pa \
  --experiment-uid <model_id>-attention_pa-0001
```

Then run:

```bash
python3 workflow/attention_pa/workflow.py run \
  --experiment-json output/experiments/<model_id>-attention_pa-0001/experiment.json \
  --experiment-uid <model_id>-attention_pa-0001 \
  --workflow-uid attention_pa \
  --max-cycles 400
```

State is written to:

```text
state/workflow-attention_pa-state.json
state/workflow-attention_pa-events.jsonl
```

All executable nodes are owned by `workflow/attention_pa/nodes/`; runtime code
does not import implementations from another workflow.
