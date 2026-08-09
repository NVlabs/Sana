# `models/` — runtime model profiles and baseline contracts

This directory stores two related model artifacts:

- flat model profiles such as `models/hunyuan_diffusers.toml`, used by existing
  launch and evaluation scripts;
- directory-style model contracts such as `models/hunyuan_diffusers/model.toml`,
  used to materialize experiment-local baseline copies.

The actual runtime adapter code lives under `models/<model_uid>/baseline/` and
`models/<model_uid>/optimized/`; reusable algorithm helpers live in
`techniques/`. (Both moved in the 2026-07-17 `efficiency/` -> `techniques/`
reorg; there is no longer a top-level `runtime/` or `efficiency/` directory.)

## A model profile contains

- official config and runtime env
- baseline timing/quality metadata
- run script and submodule commit
- human seam/status notes for what the current runtime consumes

Transfeat manifests declare their required capabilities. During dry-run,
`techniques.transfeat_manifest` builds a minimal manifest-derived `ModelSpec`
and `compose()` checks the selected technique/transform against that contract.
Because `scripts/launch_transfeat.py` imports `techniques.transfeat_manifest` at
module scope, every contract's copy allowlist must include
`techniques/transfeat_manifest.py` — otherwise the copied worktree is created
successfully and only fails later, on import.

Current local profiles include:

- `hunyuan_diffusers`: HunyuanVideo diffusers baseline and kernel workflow
  experiments.
- `sana_video`: Sana 5B 720p193 baseline wrapper for the private
  `yitongl/sana_video` minimal inference bundle.
- `lingbot_video`: LingBot-Video MoE 30B-A3B two-stage T2V on 4x GB200, with
  physically isolated CP4/FA2 baseline and cuDNN-optimized runtimes.
- `ltx23`: LTX-2.3 22B two-stage HQ at 1088x1920/241f on a single GB200, with a
  dense baseline and a KWL-fusion + stage-1 cache + PISA + NVFP4 + token-prune
  optimized arm.
- `cosmos3`: Cosmos3-Super 64B T2V at 1280x720/189f on 4x GB200, with a dense
  baseline and a TeaCache + step-selective NVFP4 optimized arm.

`ltx23` and `cosmos3` both keep their model code in the separate
`Efficient-Large-Model/Sol-LTX-Infer` repo, declared `reference_only` and pinned
by commit in `model.toml`; what lives here is the profile plus the two runnable
arms. Each arm is an `env.sh` (the technique declaration, with every knob written
out explicitly including the zeros) plus a shim that sources it and then the
model's shared launch body — so the two arms can differ by technique and by
nothing else.

## Directory-Style Model Contracts

A directory-style contract defines the minimal baseline runnable closure for one
model. Creating an experiment from this contract copies only allowlisted baseline
files into the experiment worktree:

```text
models/<model_uid>/model.toml
  -> output/experiments/<experiment_uid>/worktree/
```

The copy scope should include baseline runtime code, the baseline transfeat
manifest, launch/collect/eval helper scripts, and evaluation profiles. It should
not include generated transfeat, historical runs, search spaces, cache methods,
compiled artifacts, model weights, Conda environments, or Torch/Triton caches.

Use `scripts/create_model_experiment.py` to create this kind of experiment:

```bash
python3 scripts/create_model_experiment.py \
  --model hunyuan_diffusers \
  --workflow-uid kernel_aw \
  --experiment-uid hunyuan-kernel_aw-0001
```

For Sana Video:

```bash
python3 scripts/create_model_experiment.py \
  --model sana_video \
  --workflow-uid kernel_aw \
  --experiment-uid sana-kernel_aw-0001
```

The workflow is named with its aspect prefix, but aspect-specific behavior is
implemented inside the workflow directory rather than a separate runtime
`aspect/` directory.
