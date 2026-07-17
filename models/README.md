# `models/` — runtime model profiles and baseline contracts

This directory stores two related model artifacts:

- flat model profiles such as `models/hunyuan_diffusers.toml`, used by existing
  launch and evaluation scripts;
- directory-style model contracts such as `models/hunyuan_diffusers/model.toml`,
  used to materialize experiment-local baseline copies.

The actual runtime adapter code lives under `runtime/`; reusable algorithm
helpers live in `efficiency/`.

## A model profile contains

- official config and runtime env
- baseline timing/quality metadata
- run script and submodule commit
- human seam/status notes for what the current runtime consumes

Candidate manifests declare their required capabilities. During dry-run,
`efficiency.candidate_manifest` builds a minimal manifest-derived `ModelSpec`
and `compose()` checks the selected technique/transform against that contract.

Current local profiles include:

- `hunyuan_diffusers`: HunyuanVideo diffusers baseline and kernel workflow
  experiments.
- `sana_video`: Sana 5B 720p193 baseline wrapper for the private
  `yitongl/sana_video` minimal inference bundle.
- `lingbot_video`: LingBot-Video MoE 30B-A3B two-stage T2V on 4x GB200, with
  physically isolated CP4/FA2 baseline and cuDNN-optimized runtimes.

## Directory-Style Model Contracts

A directory-style contract defines the minimal baseline runnable closure for one
model. Creating an experiment from this contract copies only allowlisted baseline
files into the experiment worktree:

```text
models/<model_uid>/model.toml
  -> output/experiments/<experiment_uid>/worktree/
```

The copy scope should include baseline runtime code, the baseline candidate
manifest, launch/collect/eval helper scripts, and evaluation profiles. It should
not include generated candidates, historical runs, search spaces, cache methods,
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
