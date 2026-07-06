Use this callable node contract only after the reviewer or terminal final gate
has explicitly requested full Hunyuan diffusion validation. Do not use it as an
ordinary loop evaluation after every microbench promotion.

Required artifact contract:

- launch through the candidate/runtime path used by this repository;
- collect the run directory after Slurm completes;
- preserve official Hunyuan benchmark shape, prompt, seed, frame count,
  scheduler, and model profile;
- write `outputs/benchmark.json`;
- extract candidate frames or write `outputs/out.mp4`.

Do not report full-run speed from a missing benchmark or incomplete frame set.
Do not use this node to decide ordinary loop progress; it is a terminal
validation input for `final_full_eval`.

Do not discard a method because this callable node has infra trouble. Slurm
allocation cancellation, no-output hang, missing stdout/stderr, missing
heartbeat, quota failure, or missing collection artifacts must be classified as
retryable or diagnosable infrastructure failures. Retry with a runtime heartbeat
and stronger logging before asking reviewer for method judgment.
