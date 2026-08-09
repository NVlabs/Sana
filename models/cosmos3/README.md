# Cosmos3-Super — 64B T2V (1280x720, 189 frames, 4-GPU sequence parallel)

Two arms, one official config:

| Arm | Runtime root | What it turns on |
| --- | --- | --- |
| `baseline` | `models/cosmos3/baseline` | nothing — official 35-step pipeline, seams explicitly off |
| `optimized` | `models/cosmos3/optimized` | TeaCache 1.15 / start 10 / max 3 + step-selective NVFP4 |

Published speedup for the optimized arm is **2.26x**
(`site_docs/pipelines/cosmos3.md`), measured on 4x GB200 at 1280x720 / 189
frames / 35 steps with warmup excluded. The baseline reference is **130.41s**
(denoise 121.4198s + decode 5.8017s, run `20260612-175151-baseline-official-baseline-autodl`).

## Layout

```text
models/cosmos3/
  model.toml                       directory contract (minimal-copy allowlist)
  run_cosmos3_common.sh            shared launch body — official config + env
  prompts/{default,negative}.txt   versioned prompt pair
  baseline/env.sh                  TeaCache off, NVFP4 off
  baseline/scripts/run_cosmos3_gpu.sh
  optimized/env.sh                 TeaCache + NVFP4, with the dense-step policy
  optimized/scripts/run_cosmos3_gpu.sh
models/cosmos3.toml                flat profile (official config, env, seams)
candidates/cosmos3_{baseline,fullopt}.toml
evals/profiles/official_video_t2v_cosmos3.toml
```

The two arms share `run_cosmos3_common.sh` on purpose: they differ **only** by
the technique env, never by sampling config. Both `env.sh` files write out every
knob explicitly, including the ones that stay off, so a stale
`SGLANG_COSMOS3_FP4_*` export in the caller's shell cannot switch FP4 on inside
a "baseline" run.

## Running

```bash
python3 scripts/create_model_experiment.py \
  --model cosmos3 --workflow-uid cache_aw --experiment-uid cosmos3-cache_aw-0001
```

To run an arm directly (needs 4 GPUs; on Slurm submit it, don't run it on a
login node):

```bash
OUT_DIR=/path/to/out \
SOL_LTX_INFER_ROOT=/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer \
bash models/cosmos3/baseline/scripts/run_cosmos3_gpu.sh
```

## What lives where

This directory holds the *profile and the two runnable arms*. The 64B model code
— `cosmos3_pipeline.py`, the `Cosmos3OmniTransformer` DiT, `cosmos3_teacache.py`
and the `run_cosmos3_cache_matrix.sh` driver that parses the cache-variant
strings — stays in `Efficient-Large-Model/Sol-LTX-Infer`, pinned in `model.toml`
to `b0b7eb4d0` (`elm/v1_formal`) and reached through `SOL_LTX_INFER_ROOT`.

## Why the pin moved

`models/cosmos3.toml` previously pinned `base_commit = 29d0d9e4...`. That commit
sits on **no branch** in Sol-LTX-Infer — `git branch -r --contains` returns
nothing for it — so an ordinary `git fetch` never brings it down and every
checkout on the fleet reported it as a bad object. It is not lost: GitHub still
serves it to an explicit `git fetch <remote> 29d0d9e4...`. But pinning a
model profile to a commit that no branch reaches makes the profile look broken,
so both models now pin to `elm/v1_formal`, which is the branch carrying
`scripts/ltx/` and `scripts/cosmos/` in the layout the site docs reference.

## Seam status is inherited, not re-verified

`models/cosmos3.toml [seam_status]` still says `prunable_tokens = "declared"`,
`swappable_attention = "declared"` and `residual_tuple = "todo"`. Those were
written against the old spec file and are **not** re-validated by this re-add.
The two arms here only exercise `teacache_signal` and `ffn_precision`, which is
why `candidates/cosmos3_fullopt.toml` requires exactly those two capabilities.
