## Workflow-Local Callable Nodes

You may use workflow-local callable node contracts when deciding how to test a
target-model transformer-kernel candidate. Callable nodes are not global shared code;
use only the copies under `workflow/kernel_aw/nodes/callable/`.

- `kernel_microbench`: use as the ordinary loop evaluation for a concrete
  target-model transformer-kernel candidate. Module-level evidence is a screening gate; a
  retained component must later be measured through the cumulative full-DiT
  path.
- `dit_profile`: use before the first novel candidate and at composition
  checkpoints. It profiles and benchmarks the registry-resolved full target-model DiT
  for one diffusion step without launching full diffusion or video generation.
- `full_diffusion_eval`: do not use during the ordinary executor/eval/reviewer
  loop. Use it only when the reviewer/final gate has explicitly requested
  terminal full diffusion validation.
- `plan_assess`: use after terminal full run outputs exist to produce
  `assess_verdict.json` with canonical baseline frames and Gemini visual
  judgment.

Do not treat a self-reported completion message as workflow completion. Durable
JSON artifacts and `AGENT-STATUS.json` are the source of truth.

`AGENT-STATUS.json` must identify the current invocation's
`active_candidate_id` and `active_gate`. The workflow evaluator must not reuse a
smooth gate from an older candidate.

Callable node outcomes are not final discard decisions. A failed DiT-level gate,
cancelled terminal full run, no-output Slurm allocation, missing assess file,
missing API key, or numerical drift is evidence to repair, retry, or request
reviewer judgment. Only the reviewer can decide that a method is discarded, and
terminal full diffusion/Gemini must pass before the workflow actually exits.
