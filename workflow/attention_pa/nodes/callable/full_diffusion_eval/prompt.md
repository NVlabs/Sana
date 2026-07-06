Use this callable contract for every PISA candidate that may influence a final
recipe. PISA changes attention outputs across the denoising trajectory, so
module or single-DiT evidence is not sufficient for visual classification.

Required execution contract:

- run the experiment-local target model source and candidate guard;
- use the first five prompts of the target model's validation prompt set;
- use the target model's official eval profile (resolution, duration, frame
  count, fps, steps, guidance, flow shift, motion score) and the unchanged
  checkpoint/VAE/scheduler/seed policy;
- write `outputs/benchmark.json`, `outputs/run_config.json`, and frames or
  `outputs/out.mp4`;
- persist the exact PISA backend, source hash, block size, density/sparsity,
  layer map, step map, attention-type policy, dispatches, and fallbacks;
- record dense/PISA call counts and mask-selection/approximation overhead so a
  metadata-only or fallback-only candidate cannot claim acceleration.

Do not report speed from an incomplete frame set or changed workload. Slurm,
filesystem, quota, logging, or collection failure is retryable infrastructure,
not PISA quality or speed evidence.
