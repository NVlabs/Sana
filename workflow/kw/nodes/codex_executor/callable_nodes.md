## Workflow-Local Callable Nodes

You may use workflow-local callable node contracts when deciding how to test a
KWL candidate. Callable nodes are not global shared code; use only the copies
under `workflow/kw/nodes/callable/`.

- `kwl_microbench`: use before full diffusion for a concrete KWL candidate. It
  must produce a warm paired OFF/ON microbench artifact and gate JSON.
- `full_diffusion_eval`: use after microbench promotion to launch or collect
  the official full Hunyuan diffusion run.
- `plan_assess`: use after full run outputs exist to produce
  `assess_verdict.json` with canonical baseline frames.

Do not treat a self-reported completion message as workflow completion. Durable
JSON artifacts and `AGENT-STATUS.json` are the source of truth.
