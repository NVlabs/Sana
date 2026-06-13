# References

## Migrated Source Files

- `Sol-LTX-Infer/scripts/run_ltx23_sglang_nonhq_cache_10s.sh` at
  `29d0d9e464000a2472345dcad51054b15aacca8d`: source for non-HQ cache variants,
  env knobs, semantics JSON fields, and cache algorithm labels.
- `Sol-LTX-Infer/scripts/run_ltx23_teacache_hq_nonhq_matrix_10s.sh` at
  `29d0d9e464000a2472345dcad51054b15aacca8d`: source for the HQ/non-HQ
  TeaCache matrix shape and compare-video/report orchestration.
- `Sol-LTX-Infer/scripts/make_ltx23_cache_report.py` at
  `29d0d9e464000a2472345dcad51054b15aacca8d`: source for timing extraction,
  speedup tables, TeaCache stats parsing, and report sections.
- `snippets/sol-ltx-infer-reference.md`: local summary of the cache-report
  shape expected by autovideo.

## In-Repo Generic Technique References

- `efficiency/techniques/step_cache.py`: `StepCache`
- `efficiency/techniques/teacache.py`: `TeaCache`
- `efficiency/presets.py`: LTX full-opt stage-1 skip cluster
- `efficiency/selftest.py`: section `[7]` per-stage StepCache active/inactive
  assertion shape

## Existing Branches And Report Names

- Upstream branch/family name: `sglang-ltx-cache`
- LTX report outputs from the helper script:
  `benchmark_summary.json`, `benchmark_summary.md`, and
  `benchmark_report.html`
- Autovideo canonical report artifact:
  `outputs/patch_summary.md`
