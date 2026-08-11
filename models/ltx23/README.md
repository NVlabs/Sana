# LTX-2.3 — 22B two-stage HQ (1088x1920, 241 frames)

Two arms, one official config:

| Arm | Runtime root | What it turns on |
| --- | --- | --- |
| `baseline` | `models/ltx23/baseline` | nothing — official two-stage, every seam explicitly zeroed |
| `optimized` | `models/ltx23/optimized` | KWL fusion + stage-1 SCSP cache + stage-2 PISA + NVFP4 video FFN + stage-2 token prune |

Published speedup for the optimized arm is **2.40x** (`site_docs/pipelines/ltx.md`),
measured on GB200 at 1088x1920 / 241 frames with warmup excluded. The upstream
launcher comment claims 2.47x for the same stack; the two figures have never
been reconciled in-repo, so re-measure before quoting either.

## Layout

```text
models/ltx23/
  model.toml                       directory contract (minimal-copy allowlist)
  run_ltx23_common.sh              shared launch body — official config + CLI
  prompts/{default,negative}.txt   versioned prompt pair
  baseline/env.sh                  every acceleration knob = 0
  baseline/scripts/run_ltx23_gpu.sh
  optimized/env.sh                 the validated full stack
  optimized/scripts/run_ltx23_gpu.sh
models/ltx23.toml                  flat profile (official config, env, seams)
config/ltx23_{baseline,fullopt}.toml
evals/profiles/official_video_t2v_ltx23.toml
```

The two arms share `run_ltx23_common.sh` on purpose: they differ **only** by the
technique env, never by sampling config or CLI. A baseline that drifts from the
optimized arm's config stops being a control.

Both `env.sh` files write out every knob explicitly, including the ones that stay
off. That is not verbosity — a stale `SGLANG_LTX2_*` export in the caller's shell
would otherwise silently switch part of the optimized stack on inside a
"baseline" run, and the resulting speedup would be measured against the wrong
control.

## Running

`scripts/launch_config.py` generates `launch.sh`, which cd's to the runtime
root, exports the model `[env]` plus `OUT_DIR`, and calls the arm's shim:

```bash
python3 scripts/create_model_experiment.py \
  --model ltx23 --workflow-uid kernel_aw --experiment-uid ltx23-kernel_aw-0001
```

To run an arm directly:

```bash
OUT_DIR=/path/to/out \
SGLANG_RUNTIME_ROOT=/path/to/sglang-runtime \
bash models/ltx23/baseline/scripts/run_ltx23_gpu.sh
```

Override `MODEL_PATH`, `DISTILLED_LORA` or `SPATIAL_UPSAMPLER` when the weights
live outside the default cache location.

**Keep `WARMUP=true` for any comparison run.** The optimized arm pays a one-time
`torch.compile`/autotune cost on its first forward; without a warmup pass that
lands inside the timed window and the measured speedup reads far below steady
state.

## What lives where

This directory holds the *profile and the two runnable arms*. The 22B model code
itself — `LTX2TwoStageHQPipeline`, the ltx2 DiT/VAE/vocoder modules and the
`ltx2_*` Triton kernels — stays in `the SGLang runtime`, pinned
in `model.toml` to `b0b7eb4d0` (`elm/v1_formal`) and reached through
`SGLANG_RUNTIME_ROOT`. It is declared `reference_only` and is never copied into an
experiment worktree.

## Baseline timing is not filled in

`models/ltx23.toml [baseline]` is zeroed with `measured = false`, and the eval
profile deliberately omits `baseline_total_s`. The one published LTX-2.3 timing
(119.811s Diffusers no-compile / 59.332s SGLang, from
`docs/diffusion/ltx2_1080p_speedup.md` in the SGLang runtime) is for a **30-step**
stage-1 config; these arms run the **15-step** HQ config. The numbers are not
comparable, so nothing was copied in. Fill them from a measured run of
`config/ltx23/baseline.toml`.

## History

The LTX-2.3 spec entered this repo on 2026-06-13 as
`efficiency/models/ltx2_spec.py` (`6a0d38f`, "Port LTX-2.3 efficiency framework
into repo (efficiency/) + Cosmos3 spec") and was deleted a week later on
2026-06-20 by `5b725f0` ("Add model-agnostic efficiency config and audits"),
which moved every model onto manifest-declared capabilities. Cosmos3 came back
as `models/cosmos3.toml` in that reorg; LTX-2.3 did not, and `models/` carried no
LTX entry at all until this directory. This is a re-add in the current
directory-contract form, not a revert of the deletion.
