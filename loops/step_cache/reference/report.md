# Cache Report Shape

Provenance: ported from Sol-LTX-Infer
`scripts/make_ltx23_cache_report.py` @
`29d0d9e464000a2472345dcad51054b15aacca8d`, with the report requirements
summarized in `snippets/sol-ltx-infer-reference.md`.

A cache report must make the acceleration mechanism auditable, not just report a
single speedup number.

## Required Sections

- Baseline run ID and candidate run ID.
- Official config checksum or parameter table.
- Baseline total, denoise, decode, and stage seconds.
- Candidate total, denoise, decode, and stage seconds.
- Total, denoise, and per-stage speedup ratios.
- Cache mechanism summary: StepCache schedule, TeaCache threshold/start, PAB
  windows, or Cache-DiT policy.
- Parsed cache stats from `run.log`: calls, computes, hits, and skipped steps.
- Quality result from `quality.json`, including visual artifact status when
  available.
- Links to `out.mp4`, sampled frames, `run.log`, `benchmark.json`,
  `quality.json`, and `risk_notes.md`.
- Explicit notes on eligible stages/layers and stages where cache is disabled.

## Canonical Autovideo Artifacts

Use the artifact names from `docs/artifact-contract.md`:

```text
outputs/run.log
outputs/out.mp4
outputs/benchmark.json
outputs/quality.json
outputs/risk_notes.md
outputs/patch_summary.md
outputs/collection.json
```

The migrated helper in this directory reads `benchmark.json`, `quality.json`,
and `run.log`, then writes the human report to `patch_summary.md` by default.
