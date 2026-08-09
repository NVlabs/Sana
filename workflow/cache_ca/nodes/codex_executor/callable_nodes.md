## Workflow-Local Callable Nodes

You may use workflow-local callable node contracts when deciding how to test a
target-model cache transfeat. Callable nodes are not global shared code; use only the
copies under `workflow/cache_ca/nodes/callable/`.

- `full_diffusion_eval`: use for every cache transfeat as the ordinary loop
  evaluation. It must produce a completed full target-model run with
  `outputs/benchmark.json`, frames or `out.mp4`, and the fixed prompt/config at
  the model's official eval profile.

After the executor exits, the explicit workflow graph invokes its independent
`codex_visual_reviewer` node. Do not call Gemini, launch a second visual judge,
or self-author the visual verdict. The graph attaches blinded comparison images
to a separate Codex session and merges its verdict with LPIPS/runtime evidence.

Do not treat a self-reported completion message as workflow completion. Durable
JSON artifacts and `AGENT-STATUS.json` are the source of truth.

Callable node outcomes alone are not final discard decisions. A failed full run,
cancelled Slurm allocation, missing assess file, missing LPIPS/Codex-visual
evidence, quality failure, or numerical drift is evidence to repair or
retry. The sole executor may discard a method only after complete full-run
evidence and a documented finding that no credible refinement remains.
