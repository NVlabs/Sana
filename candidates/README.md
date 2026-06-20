# Candidates

Candidate manifests describe what to run. They do not contain large logs,
videos, model weights, or generated frames.

Use one TOML file per candidate. The launcher currently supports baseline-style
manifests and is intentionally conservative: it prepares a run bundle first, and
only executes when explicitly asked.

## Current Baseline

- `baseline.toml`: official Cosmos3-Super baseline using
  `Sol-LTX-Infer/scripts/run_cosmos3_sglang.sh`.

## Model-Agnostic Efficiency Candidates

The model-agnostic candidate set is now materialized as one TOML manifest per
candidate under:

- `step_cache/`: 5 candidates
- `token_prune/`: 5 candidates
- `sparse_attention/`: 9 candidates
- `nvfp4_ffn/`: 5 candidates
- `kwl_fusion/`: 6 candidates

Each manifest records the public/canonical reference, local generic
implementation, model adapter example, runtime hook example, required model
capabilities, and verification policy. The generic implementation boundary must
remain outside `Sol-LTX-Infer`; the runtime paths are integration hooks for the
current Cosmos3 validation target.

Method-family starting points live in `loops/<dimension>/dimension.toml` as
`[[method_baseline]]` entries. Use `scripts/test_candidate_manifests.py`,
`scripts/audit_candidate_soundness.py`, and
`scripts/audit_public_reference_alignment.py` before launching GPU work.

`scripts/launch_candidate.py --mode dry-run` is expected to work for every
candidate. `--mode local` and `--mode sbatch` are intentionally blocked for
Cosmos3 candidates whose advertised optimization is only a pure policy without
a runtime consumer, wiring probe, unconsumed env/config adapter, or a wired path
with a missing runtime dependency; pass `--allow-unsupported-gpu` only for
explicit diagnostic export checks.
