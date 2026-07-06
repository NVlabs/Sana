## Workflow-Local Callable Nodes

You may use workflow-local callable node contracts when deciding how to test a
KWL candidate. Callable nodes are not global shared code; use only the copies
under `workflow/kr/nodes/callable/`.

- `kwl_microbench`: use as the ordinary loop evaluation for a concrete KWL
  candidate. It must produce a warm paired OFF/ON single-DiT or module-level
  artifact and gate JSON.
- `full_diffusion_eval`: do not use during the ordinary executor/eval/reviewer
  loop. Use it only when the reviewer/final gate has explicitly requested
  terminal full diffusion validation.
- `plan_assess`: use after terminal full run outputs exist to produce
  `assess_verdict.json` with canonical baseline frames and Gemini visual
  judgment.

Do not treat a self-reported completion message as workflow completion. Durable
JSON artifacts and `AGENT-STATUS.json` are the source of truth.

Callable node outcomes are not final discard decisions. A failed DiT-level gate,
cancelled terminal full run, no-output Slurm allocation, missing assess file,
missing API key, or numerical drift is evidence to repair, retry, or request
reviewer judgment. Only the reviewer can decide that a method is discarded, and
terminal full diffusion/Gemini must pass before the workflow actually exits.
