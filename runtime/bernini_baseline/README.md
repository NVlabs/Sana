# runtime/bernini_baseline

Self-contained runtime for the Bernini t2v baseline. Parallel to
`runtime/sana_video_baseline`.

## Files

- `bernini_src/` — **vendored Bernini inference code**, a clean copy of the
  upstream pristine tree (no `.git` history). This is the code the baseline runs
  and that experiment worktrees copy for editing.
- `scripts/run_bernini_gpu.sh` — the `run_script`. `launch.sh` exports `OUT_DIR`
  + the model `[env]` and calls this shim, which hands off to `gpu_infer.py`
  (all output teed to `run.log`).
- `gpu_infer.py` — orchestrator (stdlib only). Runs a warmup pass + a measured
  pass over the 5-prompt validation set via a 4-way `torchrun` of
  `bernini_hot_infer.py` inside `bernini_src`, then normalizes the measured
  (hot) calls into `out.mp4` + `videos/<prompt>.mp4` + `frames/` +
  `benchmark.json` (schema 2) + `run_config.json`.
- `bernini_hot_infer.py` — timing-only driver: times planner / text / diffusion
  / VAE per call, prints `[HOT_TIMING]`, writes the timing JSON.

## Metric

`benchmark.json.total_s` = **median `text_to_vae_decode`** over the validation
set (start of text/condition encode → end of VAE decode). Baseline reference
≈ **129s**. `denoise_s` = DiT diffusion loop, `decode_s` = VAE decode,
`text_encoder_s` = t5, `vit_mllm_s` = semantic planner.

## External references (absolute-path only — never in the code tree)

- `BERNINI_WEIGHTS` → `--config` (192 GB `ByteDance/Bernini-Diffusers`; sub-weights
  load from subfolders of this dir).
- `BERNINI_DEPS` → `PYTHONPATH` (vendored third-party libs: torch 2.5.1+cu124,
  diffusers 0.35.2, …).

Runs on **GB200 / aarch64** (Slurm partition `batch_long`, `nvl72*` nodes) with
`/usr/bin/python3.12`.
