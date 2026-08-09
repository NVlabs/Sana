## PISA Attention Scope

Optimize the experiment-local target model transformer with PISA (Piecewise
Sparse Attention). This workflow is for attention approximation and routing
only. Do not claim gains from cache reuse, token pruning, quantization, VAE,
text encoder, scheduler, prompt, denoising-step count, resolution, frame count,
or unrelated kernel changes.

Use only the target model's DiT/inference source that has been copied into your
experiment worktree; locate it within the worktree and only edit inside the
worktree. Never patch a shared model checkout, checkpoint, VAE, Hugging Face
cache, or canonical baseline run. Keep an explicit OFF guard that restores the
source-current dense attention path and prove OFF identity before evaluating
PISA.

### Authoritative Local PISA Implementation

Use this local implementation as the sole algorithm and kernel source of truth:

```text
/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/layers/attention/backends/piecewise_attn.py
```

The validated source provenance is commit
`7546a4bd1d382923ef4876945172655a84d23686` with file SHA-256
`bfad198d834d21254492676ad210e6d5393c88b236bd3b4b793c99a6ac960fb3`.
Record the source commit and hash actually read by the experiment; if they have
changed, report the new provenance rather than silently claiming the pinned
version.

Do not use a paper, GitHub repository, or a fresh implementation from an
open-source description as the implementation guide. Do not spend search turns
re-deriving PISA. Read, copy, and adapt the required local implementation into
the experiment-local target model source. The shared local file is read-only authority:
never patch it, import mutable experiment state into it, or modify its checkout.

Preserve the behavior of the local implementation, including
`chunk_reduce_qkv`, `taylor_error_block_indices`, `piecewise_attn_fwd`, the TMA
allocator path, exact selected-block attention, and the approximate remainder.
Use `approx_remainder=True`; a keep-or-drop sparse mask is a different method.
The validated default block size is 64. It may be tuned only after the copied
adapter reproduces the local backend at the same shape and configuration.

The local implementation has already passed an isolated GB200 microbenchmark at
a representative video softmax-attention shape (`B=2, H=10, N=23000, D=256,
BF16`). Reference any prior PISA-backend microbenchmark evidence available to
the experiment.

This establishes backend viability only. The executor still must prove that its
experiment-local target model integration dispatches the copied PISA path and
must run the workflow's full quality evaluations.

PISA is an exact-or-approximate attention method, not keep-or-drop sparse
attention. Critical query-key blocks receive exact softmax attention while the
remainder is approximated through the PISA block-wise Taylor formulation. A
mask that simply zeros unselected blocks, a metadata-only environment variable,
or dense attention plus unused routing bookkeeping is not a valid PISA
implementation.

Use these parameter definitions consistently:

- `density`: fraction of attention blocks evaluated by the exact phase;
- `sparsity`: fraction handled by the approximate remainder;
- therefore `density = 1 - sparsity`.

Lower density is more aggressive. Persist both values in every manifest and
artifact so the direction cannot be confused. Block size 64 and any initial
layer/step schedule are measured starting points, not final recipes for the
target model. Verify them against the target model's actual shapes and backend.

### Fixed Full-Evaluation Contract

Every transfeat that can influence a recipe must run the complete official
workload of the target model:

- the first five prompts of the target model's validation prompt set (see the
  model profile / baseline manifest);
- the target model's official eval resolution, duration, frame count, and fps;
- the target model's official denoising steps, guidance, flow shift, and motion
  score (from the model profile / baseline manifest);
- unchanged checkpoint, VAE, scheduler, prompt text, seed policy, and decode.

After the full run, exit the executor invocation so the explicit graph can run
aligned LPIPS plus its independent blind Codex image reviewer against the
canonical baseline. Do not call Gemini or write the reviewer verdict yourself.
Single-DiT, isolated attention, or short-video results are screening evidence
only; they cannot populate `PISA-RECIPES.json`.

For every full run record end-to-end wall time, isolated denoise/DiT time when
available, PISA kernel time, mask-selection and approximation overhead,
dispatch count, dense fallback count, peak memory, LPIPS, Codex visual status
and artifact severity, and exact run/config provenance.

## Execution round limit

You have a hard budget of **20 optimization rounds** (transfeat attempts) for this workflow. This is an execution-round limit, not a suggestion: pace your search so that by round 20 you have finalized and delivered your best retained frontier. Do not plan for more than 20 rounds. One round = one transfeat implemented, launched, evaluated, and gated.
