# `efficiency/`

Model-agnostic efficiency config code for video-generation inference.

The package owns the reusable algorithm/policy layer. A model adapter supplies
the runtime seams: denoise-step loop, transformer-block loop, token layout,
attention backend, FFN linear modules, and any model-specific CFG/RoPE/text
prefix details.

## Entry Flow

```text
config/<dimension>/<config>.toml
  -> scripts/launch_config.py --mode dry-run
  -> efficiency.config_manifest.dry_run_manifest()
  -> registry.build_technique/build_transform()
  -> compose(items, manifest-derived ModelSpec)
  -> transform env preview or runtime Technique plan
  -> model runtime hook consumes that plan
```

Public-reference and soundness gates live in `scripts/audit_*.py` and
`scripts/probe_*.py`. Their Markdown reports are generated on demand and are
not required repo state.

## Core Framework

- `technique.py`: `Technique`, `Capability`, `Seam`, `Phase`,
  `TechniqueContext`.
- `transform.py`: build/load-time `ModelTransform`.
- `compose.py`: capability checks, seam-conflict checks, and ordered `Plan`.
- `schedule.py`: step/stage schedule DSL.
- `registry.py`: technique/transform registration plus optional external
  model-spec registration.
- `config_manifest.py`: TOML schema checks, capability resolution, and
  manifest-derived dry-run specs.

## Runtime Techniques

- `techniques/step_cache.py`: whole-step skip/reuse and delta forecast.
- `techniques/teacache.py`: TeaCache rel-L1/poly controller, whole-step
  TeaCache, and block-residual TeaCache replay.
- `techniques/payload_cache.py`: PAB-style attention/MLP payload cache
  controller.
- `techniques/token_prune.py`: feature-norm pruning, region-density pruning,
  CAT-style cluster/stale selection, ToMe merge/unmerge, ToMeSD random-2D
  merge/unmerge, and gather/scatter compensation.

## Build/Load Transforms

- `transforms/sparse_attention.py`: sparse-attention route/backend env policy.
- `sparse_attention_policies.py`: PISA/SVG/SpargeAttn/MInference route masks,
  SAP dynamic map/permutation helpers, and block-index conversion.
- `transforms/nvfp4_ffn.py`: NVFP4 FFN load-time policy.
- `nvfp4_profile.py`: profiled NVFP4 layer selection and dense guards.
- `transforms/kwl_fusions.py`: generic KWL backend and compile/capture policy.

## Model Capability Contract

This repo no longer ships built-in per-model spec files. Config dry-runs
synthesize a minimal `ModelSpec` from `[requirements].capabilities` in the
manifest, then `compose()` checks the selected technique or transform against
that contract. The concrete model adapter remains in the runtime code under
`Sol-LTX-Infer/`.

## Verify

```bash
# Every config resolves, and its repo-side paths exist:
for c in config/*/*.toml; do python3 scripts/run.py "$c" --print >/dev/null || echo "FAILED $c"; done

# The technique registry agrees with itself:
PYTHONNOUSERSITE=1 python3 techniques/selftest.py
```
