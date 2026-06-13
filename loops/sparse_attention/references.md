# References

## Migrated Source Files

- `Sol-LTX-Infer/scripts/run_ltx23_sglang_hq_1080p10s.sh` @ `29d0d9e`:
  `enable_stage2_sparse_env` and the `SGLANG_HQ_*` attention-backend snippet,
  migrated to `reference/recipe.sh` and summarized in `reference/recipe.md`.
- `Sol-LTX-Infer/docs/ltx23_sglang_hq_variants.md` @ `29d0d9e`:
  sparse-attention variant notes and 1080p10s benchmark excerpt, migrated to
  `reference/report.md`.

## In-Repo Framework References

- `efficiency/transforms/sparse_attention.py`: `SparseAttention` transform,
  referenced but not copied.
- `efficiency/selftest.py`: section `[7]` checks
  `transformer_2=piecewise_attn` in
  `SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS`.
- `efficiency/models/cosmos3_spec.py`: target spec that currently declares
  `BLOCKS` and `PRUNABLE_TOKENS`, but not `SWAPPABLE_ATTENTION`.

## Upstream Names

- Source checkout: `Sol-LTX-Infer` detached at `29d0d9e`.
- Report names: `ltx-stage1-sparse-schedule` and
  `ltx-sparse-attn-bringup`.
- Report file: `docs/ltx23_sglang_hq_variants.md`.
