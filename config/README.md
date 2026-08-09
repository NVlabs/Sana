# Configs

Config manifests describe what to run. They do not contain large logs,
videos, model weights, or generated frames.

Use one TOML file per config. The launcher currently supports baseline-style
manifests and is intentionally conservative: it prepares a run bundle first, and
only executes when explicitly asked.

## Current Baseline

- `baseline.toml`: official Cosmos3-Super baseline using
  `Sol-LTX-Infer/scripts/run_cosmos3_sglang.sh`.

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
remain outside `Sol-LTX-Infer`; the runtime paths are integration hooks for the
current Cosmos3 validation target.

Method-family starting points live in `loops/<dimension>/dimension.toml` as
`[[method_baseline]]` entries. Use `scripts/test_config_manifests.py`,
`scripts/audit_config_soundness.py`, and
`scripts/audit_public_reference_alignment.py` before launching GPU work.

`scripts/launch_config.py --mode dry-run` is expected to work for every
config. `--mode local` and `--mode sbatch` are intentionally blocked for
Cosmos3 config whose advertised optimization is only a pure policy without
a runtime consumer, wiring probe, unconsumed env/config adapter, or a wired path
with a missing runtime dependency; pass `--allow-unsupported-gpu` only for
explicit diagnostic export checks.
