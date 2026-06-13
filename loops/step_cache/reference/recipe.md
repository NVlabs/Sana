# LTX-2.3 Cache Recipe

Provenance: ported from Sol-LTX-Infer
`scripts/run_ltx23_sglang_nonhq_cache_10s.sh` and
`scripts/run_ltx23_teacache_hq_nonhq_matrix_10s.sh` @
`29d0d9e464000a2472345dcad51054b15aacca8d`.

These snippets are reference material only. Do not execute them from this loop.

## Non-HQ Cache Variants

The non-HQ runner accepted:

```bash
SGLANG_NONHQ_VARIANT=dense
SGLANG_NONHQ_VARIANT=kwl
SGLANG_NONHQ_VARIANT=cache_pab_late12_w3
SGLANG_NONHQ_VARIANT=cache_teacache_c04_s6
SGLANG_NONHQ_VARIANT=cache_teacache_c06_s5
SGLANG_NONHQ_VARIANT=cache_teacache_c08_s5
SGLANG_NONHQ_VARIANT=cache_dbcache_aggressive
SGLANG_NONHQ_VARIANT=kwl_cache_teacache_c04_s6
SGLANG_NONHQ_VARIANT=kwl_cache_teacache_c06_s5
SGLANG_NONHQ_VARIANT=kwl_cache_teacache_c08_s5
```

Cache env was cleared before each variant:

```bash
export SGLANG_LTX2_PAB_ENABLED=0
export SGLANG_CACHE_DIT_ENABLED=0
export SGLANG_LTX2_TEACACHE_ENABLED=0
unset SGLANG_LTX2_PAB_SPATIAL_WINDOW SGLANG_LTX2_PAB_TEMPORAL_WINDOW
unset SGLANG_LTX2_PAB_CROSS_WINDOW SGLANG_LTX2_PAB_START_STEP
unset SGLANG_LTX2_PAB_END_STEP SGLANG_LTX2_PAB_STAGE2_ENABLED
unset SGLANG_CACHE_DIT_WARMUP SGLANG_CACHE_DIT_RDT
unset SGLANG_CACHE_DIT_MC SGLANG_CACHE_DIT_FN SGLANG_CACHE_DIT_BN
unset SGLANG_LTX2_TEACACHE_THRESH SGLANG_LTX2_TEACACHE_START
unset SGLANG_LTX2_TEACACHE_END SGLANG_LTX2_TEACACHE_STAGE2_DISABLE
unset SGLANG_LTX2_TEACACHE_MAX_CONTINUOUS_HITS
unset SGLANG_LTX2_TEACACHE_STAGE1_ENABLED
unset SGLANG_LTX2_TEACACHE_PERIODIC_RECOMPUTE_STEPS
```

## PAB Late-12 Window-3 Variant

```bash
export SGLANG_LTX2_PAB_ENABLED=1
export SGLANG_LTX2_PAB_SPATIAL_WINDOW=3
export SGLANG_LTX2_PAB_TEMPORAL_WINDOW=3
export SGLANG_LTX2_PAB_CROSS_WINDOW=3
export SGLANG_LTX2_PAB_START_STEP=12
export SGLANG_LTX2_PAB_END_STEP=-1
export SGLANG_LTX2_PAB_DISABLE_AUDIO_VIDEO_CROSS=1
export SGLANG_LTX2_PAB_A2V_WINDOW=1
export SGLANG_LTX2_PAB_V2A_WINDOW=1
export SGLANG_LTX2_PAB_STAGE2_ENABLED=0
```

## TeaCache Variants

Common TeaCache knobs:

```bash
export SGLANG_LTX2_TEACACHE_ENABLED=1
export SGLANG_LTX2_TEACACHE_STAGE1_ENABLED=1
export SGLANG_LTX2_TEACACHE_END=-1
export SGLANG_LTX2_TEACACHE_STAGE2_DISABLE=1
export SGLANG_LTX2_TEACACHE_MAX_CONTINUOUS_HITS=1
export SGLANG_LTX2_TEACACHE_PERIODIC_RECOMPUTE_STEPS=0
```

Tuned threshold/start variants:

```bash
# c04_s6
export SGLANG_LTX2_TEACACHE_THRESH=0.04
export SGLANG_LTX2_TEACACHE_START=6

# c06_s5
export SGLANG_LTX2_TEACACHE_THRESH=0.06
export SGLANG_LTX2_TEACACHE_START=5

# c08_s5
export SGLANG_LTX2_TEACACHE_THRESH=0.08
export SGLANG_LTX2_TEACACHE_START=5
```

## Cache-DiT Aggressive Variant

```bash
export SGLANG_CACHE_DIT_ENABLED=1
export SGLANG_CACHE_DIT_WARMUP=4
export SGLANG_CACHE_DIT_RDT=0.24
export SGLANG_CACHE_DIT_MC=3
export SGLANG_CACHE_DIT_FN=1
export SGLANG_CACHE_DIT_BN=0
```

## Matrix Shape

The TeaCache matrix ran two prompts through HQ and non-HQ variants:

```bash
HQ_VARIANTS="kwl kwl_teacache_c04_s6 kwl_teacache_c06_s5 kwl_teacache_c08_s5"
NONHQ_VARIANTS="kwl kwl_cache_teacache_c04_s6 kwl_cache_teacache_c06_s5 kwl_cache_teacache_c08_s5"
```

After generation, it built compare videos when outputs existed, then ran:

```bash
python scripts/make_ltx23_cache_report.py \
  --root "$ROOT" \
  --prompt-count "$PROMPT_COUNT" \
  --hq-variants "$HQ_VARIANTS" \
  --nonhq-variants "$NONHQ_VARIANTS"
```

## Cosmos3 Translation

The closest generic `efficiency` translation for the first Cosmos3 patch is a
disabled-by-default `StepCache` schedule:

```python
from efficiency import TechniqueContext, at_steps, by_stage, compose
from efficiency.techniques.step_cache import StepCache

cache = StepCache(
    skip=by_stage({"stage1": at_steps("16-28", True, False)}, default=False),
    delta_scale=0.0,
)
plan = compose([cache], cosmos3_spec)
out = plan.on_step(
    TechniqueContext(step=step, stage=stage, spec=cosmos3_spec, cache_key=cache_key, scratch=scratch),
    run_step,
)
```

Inactive plan steps do not seed `StepCache`; if the first active step has no
cached output, it computes once and stores that output. Tune the effective skip
cluster against Cosmos3 timing and visual gates before promotion.
