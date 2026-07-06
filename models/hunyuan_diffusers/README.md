# Hunyuan Diffusers Model Contract

This directory is the directory-style model contract for `hunyuan_diffusers`.
The legacy flat profile remains at `models/hunyuan_diffusers.toml` because
existing launch and evaluation scripts still read it.

The contract in `model.toml` defines the minimum baseline runnable closure that
is copied into a new experiment worktree. It intentionally copies only the
baseline inference runtime, baseline manifest, model/eval profiles, and small
local launch/evaluation helpers. It does not copy historical runs, generated
candidates, optional cache seams, cache methods, search-space libraries, or
experiment outputs.

Large external state such as model weights, Hugging Face cache, Conda
environments, Torch/Triton caches, and Slurm allocations is reference-only and
must not be copied into experiments.
