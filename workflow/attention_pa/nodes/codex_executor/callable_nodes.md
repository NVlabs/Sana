## Workflow-Local Callable Nodes

Use only callable-node contracts owned by `workflow/attention_pa`.

- `pisa_preflight`: map the real target model attention paths and prove that the
  backend executes both PISA exact and approximate phases before GPU recipe
  search.
- `full_diffusion_eval`: run every PISA transfeat that may influence a recipe.
  It produces the complete fixed five-prompt workload at the target model's
  official eval profile.

After the executor exits, the explicit graph invokes its independent
`codex_visual_reviewer` node. Do not call Gemini, launch another visual judge,
or self-author the visual verdict. The graph attaches blinded comparison images
to a separate Codex session and merges its result with LPIPS/runtime evidence.
Both visual passes and failures are useful measured boundary evidence.

Isolated attention, module, or single-DiT benchmarks may be created as
screening artifacts, but they do not replace these callable contracts. Durable
JSON artifacts, dispatch/fallback counters, and source provenance are the
source of truth.

Callable failures caused by infrastructure are retried. As the sole executor,
you may discard only a fully evaluated concrete PISA operating point, never an
incomplete run or the whole PISA family because one point failed.
