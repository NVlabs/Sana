# References

## Source Checkout

- `/home/haozhel/lustre/auto-video/Sol-LTX-Infer` at `29d0d9e` detached HEAD.

## In-Repo Framework

- `efficiency/transforms/kwl_fusions.py`: `KWLFusions` env-writing transform.
- `efficiency/selftest.py`: section `[7]` asserts `SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE`.
- `efficiency/skills/operator_fusion.md`: KWL methodology and pitfalls.
- `efficiency/models/cosmos3_spec.py`: target Cosmos3 `ModelSpec`.

## Migrated LTX Files

- `Sol-LTX-Infer/scripts/run_ltx23_sglang_hq_kwl_1080p10s.sh`:
  wrapper selecting `SGLANG_HQ_VARIANT=kwl`.
- `Sol-LTX-Infer/scripts/run_ltx23_sglang_hq_1080p10s.sh`:
  `SGLANG_HQ_KWL_*` to `SGLANG_LTX2_*` env mapping.
- `Sol-LTX-Infer/scripts/ltx23_official_kwl_ops.py`:
  official HQ KWL module-op installer, migrated to `reference/kwl_ops.py`.
- `Sol-LTX-Infer/docs/ltx23_official_hq_kwl_report.md`:
  official HQ KWL benchmark and quality report.
- `Sol-LTX-Infer/docs/diffusion/ltx2_dit_fusion_report.md`:
  fusion catalog, lossless interpretation, and retained-switch notes.

## Upstream Report Names

- `ltx23_official_hq_kwl_report`
- `ltx2_dit_fusion_report`
