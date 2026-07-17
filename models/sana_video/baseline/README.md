# sana_video_baseline

Baseline wrapper for the private `yitongl/sana_video` minimal Sana 5B video
inference bundle.

The Hugging Face dataset provides the runnable source zip:

```text
yitongl/sana_video
standalone/sana5b_720p193_minimal_infer.zip
```

and the checkpoint:

```text
Sana_5B_480px_QwenNext_ltxvae23_selfflow_pertoken_subattnres_v2_multires_multifps_sft/checkpoints/epoch_7_step_2107015/model_ema.pth
```

This runtime does not vendor the 201-file bundle or the 17 GB checkpoint into
the repo. On first run it downloads/extracts the small zip into
`$SANA_VIDEO_ASSET_ROOT` and then executes the bundled inference entrypoint. The
checkpoint and VAE are reference-only by default; set `SANA_VIDEO_PREPARE_ASSETS=1`
to let the bundled downloader materialize them under the shared asset root, or
set `SANA_VIDEO_CKPT` and `SANA_VIDEO_VAE_ROOT` to pre-existing paths.

If a complete standalone bundle already exists, set `SANA_VIDEO_BUNDLE_ROOT` to
that directory. The active profile uses the known-good standalone bundle under
`/lustre/fs1/.../code/sana_video_2/Sana/output/standalone/` because it contains
the full source closure needed by the Sana inference script.

The wrapper normalizes the bundle output into the repo's standard artifact
contract:

```text
outputs/run.log
outputs/out.mp4
outputs/frames/
outputs/benchmark.json
outputs/run_config.json
```

`benchmark.json` uses the hot inference contract
`warm_single_sample_text_encoder_through_vae_decode`. The model, text encoder,
and VAE are loaded before timing. One unmeasured sample with the official shape
warms the same path, then each fixed prompt is CUDA-synchronized and timed from
text-encoder inference through the end of VAE decode. Process startup, model
loading, CPU postprocessing, MP4 encoding, and filesystem writes are excluded.
The bundled process wall clock remains available only under `diagnostics`.

The model contract installs the experiment-local timing entrypoint through
`baseline.overlay_copy`. Optimization agents continue editing the installed
copy under `external/sana_standalone`; the shared standalone reference is never
modified.

Workflow baseline launchers write `job-started.json` before entering the
runtime. A running job that never writes this sentinel is canceled and retried
with a bounded attempt count. Retryable Slurm terminal states such as timeout,
preemption, or node failure are recorded as infrastructure attempts rather than
method failures. A completed canonical run may be copied into another workflow
with `CANONICAL_BASELINE_RUN`, avoiding duplicate baseline GPU submissions.

Baseline acceleration knobs are disabled by default:

```text
FORWARD_CACHE_METHOD=none
```

Runtime environment note: the checked local ltx23 environment is enough for
some existing video baselines, but not for this Sana bundle. A real Sana run
needs a Python environment with the bundle's dependencies, including
`accelerate`, `diffusers>=0.38`, `transformers>=4.57`, `pyrallis`,
`mmcv==1.7.2`, `flash-linear-attention==0.5.0`, and a compatible Torch/CUDA
stack. Set `PYTHON_BIN` or `SANA_VIDEO_INFER_PYTHON` to that environment before
submitting GPU inference.
