# Configs

Config manifests describe what to run. They do not contain large logs,
videos, model weights, or generated frames.

Use one TOML file per config. The launcher currently supports baseline-style
manifests and is intentionally conservative: it prepares a run bundle first, and
only executes when explicitly asked.

## Current Baseline

- `baseline.toml`: official Cosmos3-Super baseline using
  `sglang-runtime/scripts/run_cosmos3_sglang.sh`.

## Model-Agnostic Efficiency Configs

The model-agnostic config set is now materialized as one TOML manifest per
config under:

- `step_cache/`: 5 config
- `token_prune/`: 5 config
- `sparse_attention/`: 9 config
- `nvfp4_ffn/`: 5 config
- `kwl_fusion/`: 6 config

Each manifest records the public/canonical reference, local generic
implementation, model adapter example, runtime hook example, required model
capabilities, and verification policy. The generic implementation boundary must
remain outside `sglang-runtime`; the runtime paths are integration hooks for the
current Cosmos3 validation target.

Resolve a config before spending an allocation:

```bash
python3 scripts/run.py config/<model>/<arm>.toml --print
```

That renders the run bundle and checks the repo-side paths -- runtime root, run
script, prompt file -- without running anything. It is the check worth doing;
the audit and alignment scripts that used to be named here were one-off probes
and are not part of what this repository ships.

`scripts/launch_config.py --mode dry-run` is expected to work for every
config. `--mode local` and `--mode sbatch` are intentionally blocked for
Cosmos3 config whose advertised optimization is only a pure policy without
a runtime consumer, wiring probe, unconsumed env/config adapter, or a wired path
with a missing runtime dependency; pass `--allow-unsupported-gpu` only for
explicit diagnostic export checks.
