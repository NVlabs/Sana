Use this callable node contract for every concrete cache candidate. Cache
methods change cross-step behavior, so ordinary workflow progress requires full
target-model diffusion output rather than single-DiT, module, or microbench
evidence.

Required artifact contract:

- launch through the candidate/runtime path used by this repository;
- collect the run directory after Slurm completes;
- preserve the target model's official benchmark shape, prompt file, seed
  policy, frame count, scheduler, and model profile;
- use the target model's validation prompt file (see the model profile /
  baseline manifest);
- use the target model's official generation config (resolution, frame count,
  fps, steps, guidance, flow shift, and motion score from the model profile /
  baseline manifest);
- write `outputs/benchmark.json`;
- extract candidate frames or write `outputs/out.mp4`.

Do not report full-run speed from a missing benchmark or incomplete frame set.
Do not change resolution, duration, prompt text, frame count, scheduler,
checkpoint, VAE, or text encoder to obtain speed.

Do not discard a method because this callable node has infra trouble. Slurm
allocation cancellation, no-output hang, missing stdout/stderr, missing
heartbeat, quota failure, or missing collection artifacts must be classified as
retryable or diagnosable infrastructure failures. Retry with a runtime heartbeat
and stronger logging before making a method judgment.
