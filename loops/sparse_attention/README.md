# Loop: sparse_attention

## Purpose

Bring the LTX-2.3 PISA sparse-attention recipe into the autovideo loop system and
verify that the in-repo `efficiency/` transform emits the expected
`SGLANG_HQ_*` backend configuration.

## LTX-2.3 Provenance

This loop migrates the sparse-attention pieces from the read-only
`Sol-LTX-Infer` checkout at `29d0d9e`:

- `scripts/run_ltx23_sglang_hq_1080p10s.sh`: piecewise/PISA env setup for
  `piecewise_attn`.
- `docs/ltx23_sglang_hq_variants.md`: sparse-attention notes and the 1080p10s
  matrix where `kwl_sparse` reached 1.126x total and 1.136x denoise speedup
  versus the KWL baseline.

The migrated reference snippets are in `reference/`. They are local excerpts
with provenance headers, not runtime dependencies on the LTX checkout.

## Mapping To `efficiency/`

The generic implementation is already present at
`efficiency/transforms/sparse_attention.py` as `SparseAttention`. It is a
build-time transform requiring `Capability.SWAPPABLE_ATTENTION`, writing
`Seam.ATTENTION_BACKEND`, and delegating to the existing SGLang HQ pipeline by
setting:

- `SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS`
- `SGLANG_HQ_ATTENTION_BACKEND_CONFIG`

The local test composes `SparseAttention(dense_steps=3, stage2_dense_layers="0")`
against a loop-local `ModelSpec` fixture with `BLOCKS` and
`SWAPPABLE_ATTENTION`, then asserts the generated env selects
`transformer_2=piecewise_attn` with the LTX-2.3 sparse config.

## Cosmos3 Wiring Step

`efficiency/models/cosmos3_spec.py` intentionally does not declare
`Capability.SWAPPABLE_ATTENTION` yet. To run this on Cosmos3, a future Codex
task must wire an attention-backend seam in the Cosmos3 runtime, then add that
capability to the Cosmos3 spec. That task must also confirm the Cosmos3
component names and layer guards because the LTX names `transformer` and
`transformer_2` may not map directly to Cosmos3 generation layers.

## Candidate

The launcher-runnable manifest is `candidates/sparse_attention.toml`. The loop
copy is `candidate.toml`. The candidate is `kind = "env_only"` because it
delegates to `SGLANG_HQ_*` environment configuration rather than carrying a
runtime patch in this worktree.

## Test

Run the independent gate with the torch-capable sana environment:

```bash
~/lustre/miniconda3/envs/sana/bin/python loops/sparse_attention/test_sparse_attention.py
```

Optional launcher dry-run:

```bash
python3 scripts/launch_candidate.py candidates/sparse_attention.toml --mode dry-run
```

## Status

`ready-for-codex`
