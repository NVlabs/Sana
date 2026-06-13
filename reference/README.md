# `reference/` — the LTX-2.3 acceleration corpus (self-contained, in-repo)

The proven **LTX-2.3** acceleration experience, **copied into this project** — not
referenced from anywhere external. It is the prior/seed knowledge the generic
search dimensions in `loops/*/dimension.toml` build on (their `[[seeds]]` cite it).

This is the single home for that corpus. Organized one folder per technique:

```
reference/<technique>/   recipe (env/run knobs) + report (LTX results + acceptance
                         language) + any helper script, each with a provenance header.
```

## What stays external — and what doesn't
- **Knowledge (this folder): in-repo.** Nothing here points into another repo at
  use time; these are local copies. The search reads only `reference/` + `efficiency/`.
- **`Sol-LTX-Infer` submodule: execution runtime only.** It remains solely as the
  model/runtime code needed to actually RUN a model on GPU. The reference corpus no
  longer depends on it.

## Provenance (copied from `Sol-LTX-Infer` @ `29d0d9e`)
| `reference/<technique>/` | source files |
| --- | --- |
| `step_cache` | `scripts/run_ltx23_sglang_nonhq_cache_10s.sh`, `scripts/run_ltx23_teacache_hq_nonhq_matrix_10s.sh`, `scripts/make_ltx23_cache_report.py` |
| `token_prune` | `efficiency/techniques/token_prune.py` + `efficiency/presets.py` (`ltx_full_opt` stage-2 midpoint prune) |
| `sparse_attention` | `scripts/run_ltx23_sglang_hq_1080p10s.sh`, `docs/ltx23_sglang_hq_variants.md` |
| `nvfp4_ffn` | `scripts/slurm_ltx23_best_nvfp4_*.sh`, `scripts/bench_te_nvfp4_gelu_epilogue.py`, `docs/diffusion/quantization.md` |
| `kwl_fusion` | `scripts/run_ltx23_sglang_hq_kwl_1080p10s.sh`, `scripts/ltx23_official_kwl_ops.py`, `docs/ltx23_official_hq_kwl_report.md` |

## Relationship to the rest of the repo
- **Generic engine**: `efficiency/` (the technique/transform implementations).
- **Reference corpus**: this folder (the proven LTX-2.3 recipes/reports = priors).
- **Search dimensions**: `loops/<technique>/` (cite `reference/<technique>/` in seeds).
- **Model adapter**: `models/<id>.toml` + `efficiency/models/<id>_spec.py`.
