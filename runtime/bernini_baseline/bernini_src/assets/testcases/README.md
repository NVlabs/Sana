# Test cases

Each task has a directory under `assets/testcases/` holding one or more
**case files** plus their source media and generated outputs. A case file is a
JSON that bundles one example's routing and inputs, so a run is a one-line
`--case` instead of a command line carrying a long prompt:

```bash
python infer_single_gpu.py \
    --high_noise_ckpt <path> --low_noise_ckpt <path> \
    --case assets/testcases/v2v/v2v_case1.json
```

## Layout

```
assets/testcases/
  t2v/    text-to-video
  t2i/    text-to-image
  v2v/    video editing             (v2v_case1.json, v2v_case2.json, ...)
  i2i/    image editing
  r2v/    reference-to-video
  rv2v/   reference + video editing  (rv2v_case1.json, rv2v_case2.json, ...)
```

Each directory contains the case file(s), the source media they reference
(`source*.mp4`, `ref*.jpg`, `source.png`, ...) and the generated `*_out.*`
results. Paths inside a case file are resolved relative to the current
directory, so run from the repository root.

## Format

A case file is a single JSON object. Supported keys (all optional except
`prompt`):

| Key             | Meaning                                              |
|-----------------|------------------------------------------------------|
| `task_type`     | task type — drives the prompt-enhancement template   |
| `guidance_mode` | sampling-time guidance mode                          |
| `prompt`        | text prompt / editing instruction (**required**)     |
| `video`         | source video path, or a list of paths                |
| `image`         | single source image path (image editing)             |
| `images`        | reference image path(s)                              |
| `output`        | output file path                                     |

The case file fully defines those fields; generation parameters (`--seed`,
`--num_frames`, `--omega_*`, ...) are still passed on the command line.
`--case` cannot be combined with `--inputs` (batch mode).

## Image tasks

The image tasks (`t2i`, `i2i`) generate a single frame, so they must be run
with `--num_frames 1`:

```bash
python infer_single_gpu.py \
    --high_noise_ckpt <path> --low_noise_ckpt <path> \
    --case assets/testcases/i2i/i2i.json --num_frames 1
```
