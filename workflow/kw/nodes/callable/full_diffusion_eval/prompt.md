Use this callable node contract only after a KWL candidate has a passing
workflow-local microbench gate.

Required artifact contract:

- launch through the candidate/runtime path used by this repository;
- collect the run directory after Slurm completes;
- preserve official Hunyuan benchmark shape, prompt, seed, frame count,
  scheduler, and model profile;
- write `outputs/benchmark.json`;
- extract candidate frames or write `outputs/out.mp4`.

Do not report full-run speed from a missing benchmark or incomplete frame set.
