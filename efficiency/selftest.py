#!/usr/bin/env python3
"""Self-test for runtime/efficiency: schedule, capability check, conflict
detection, and the token-prune off==identity invariant + real prune/scatter."""
import sys
import types

import torch
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from efficiency import (  # noqa: E402
    Capability,
    CompositionError,
    ModelSpec,
    Phase,
    Seam,
    Technique,
    TechniqueContext,
    at_steps,
    before,
    build_technique,
    compose,
    const,
)
from efficiency.techniques.token_prune import (  # noqa: E402
    TokenPrune,
    tome_bipartite_soft_matching,
    tomesd_random2d_matching,
)

ok = 0
fail = 0


def check(name, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}")


# ---- 1. Schedule DSL ----
print("[1] schedule")
s = before(2, "bf16", "nvfp4")  # first 2 steps high precision
check("before(2): step0=bf16", s.at(0) == "bf16")
check("before(2): step3=nvfp4", s.at(3) == "nvfp4")
sset_sched = at_steps("1-2", True, False)
check("at_steps('1-2'): step1 active", sset_sched.at(1) is True and sset_sched.at(3) is False)
check("at_steps('1-2'): truthy_steps", sorted(at_steps("1-2,5", True, False).truthy_steps(8)) == [1, 2, 5])

# ---- model spec (1-GPU: whole sequence prunable) ----
spec = ModelSpec(
    name="DummyDiT",
    capabilities=frozenset({Capability.PRUNABLE_TOKENS, Capability.BLOCKS}),
    seq_dim=1,
)

# ---- 2. capability (type) check ----
print("[2] capability check")
class NeedsAttn(Technique):
    name = "needs_attn"
    phase = Phase.WRAP_ATTENTION
    required_capabilities = frozenset({Capability.SWAPPABLE_ATTENTION})

try:
    compose([NeedsAttn()], spec)
    check("missing-capability rejected", False)
except CompositionError as e:
    check("missing-capability rejected", "swappable_attention" in str(e))

check("token_prune accepted (has PRUNABLE_TOKENS)", isinstance(compose([TokenPrune(keep_ratio=0.5)], spec), object))

# ---- 3. conflict (effect) detection ----
print("[3] conflict detection")
class FreezeTokens(Technique):
    name = "freeze_tokens"
    phase = Phase.PRE_BLOCKS
    writes = frozenset({Seam.TOKEN_SET})
    required_capabilities = frozenset({Capability.PRUNABLE_TOKENS})

try:  # two writers of exclusive TOKEN_SET, both always-on -> conflict
    compose([TokenPrune(keep_ratio=0.5), FreezeTokens()], spec)
    check("exclusive-seam conflict detected", False)
except CompositionError as e:
    check("exclusive-seam conflict detected",
          "token_set" in str(e) and "multiple active writers" in str(e))

# same two techniques but DISJOINT schedules -> provably safe
ft = FreezeTokens(); ft.enabled = at_steps("5-6", True, False)
tp = TokenPrune(keep_ratio=0.5, enabled=at_steps("1-2", True, False))
try:
    compose([tp, ft], spec)
    check("disjoint schedules -> no conflict", True)
except CompositionError:
    check("disjoint schedules -> no conflict", False)

# ---- 4. token-prune off == identity ----
print("[4] off == byte-identical")
torch.manual_seed(0)
hidden = torch.randn(2, 16, 8)
plan = compose([TokenPrune(keep_ratio=1.0)], spec)  # OFF (ratio>=1)
ctx = TechniqueContext(step=3, spec=spec, cache_key="k")
h2, carries = plan.before_blocks(ctx, hidden)
h2 = plan.after_blocks(ctx, h2, carries)
check("ratio=1.0 is identity (before/after no-op)", torch.equal(h2, hidden))

# ---- 5. token-prune real gather/scatter shape round-trip ----
print("[5] prune gather->scatter round-trip")
tp = TokenPrune(keep_ratio=0.5, method="feat_norm", compensation="prev",
                enabled=const(True))
plan = compose([tp], spec)
# step 0: seed (runs full)
c0 = TechniqueContext(step=0, spec=spec, cache_key="k", scratch={})
h, car = plan.before_blocks(c0, hidden)
check("step0 seed: no gather (full S)", h.shape[1] == 16 and car == [(tp, None)])
h = plan.after_blocks(c0, h, car)
# step 1: prune to K=8 inside the loop, scatter back to 16
c1 = TechniqueContext(step=1, spec=spec, cache_key="k", scratch=c0.scratch)
hg, car = plan.before_blocks(c1, hidden)
check("step1 gather: K=8 tokens", hg.shape[1] == 8)
# pretend the block loop ran (identity here), then scatter
hs = plan.after_blocks(c1, hg, car)
check("step1 scatter: back to S=16", hs.shape[1] == 16)

# ToMe merge/unmerge is a real merge path, so it does not need a previous
# hidden buffer before reducing the first active step.
tp_tome = TokenPrune(
    keep_ratio=0.75,
    method="tome_merge_restore",
    compensation="prev",
    enabled=const(True),
)
plan_tome = compose([tp_tome], spec)
c_tome = TechniqueContext(step=0, spec=spec, cache_key="tome", scratch={})
hm, car = plan_tome.before_blocks(c_tome, hidden)
check("ToMe merge: first active step reduces tokens", hm.shape[1] == 12)
hr = plan_tome.after_blocks(c_tome, hm, car)
check("ToMe unmerge: restores original token count", hr.shape[1] == 16)
tome_plan = tome_bipartite_soft_matching(hidden, remove=4)
check(
    "ToMe helper: merge/unmerge plan exists",
    tome_plan is not None and tome_plan.merge(hidden).shape[1] == 12,
)
shape_plan = tomesd_random2d_matching(hidden, remove=4, no_rand=True)
check(
    "ToMeSD random2D helper: merge/unmerge plan exists",
    shape_plan is not None and shape_plan.merge(hidden).shape[1] == 12,
)

tp_shape = TokenPrune(
    keep_ratio=0.75,
    method="shape_stable_compute_mask",
    compensation="prev",
    enabled=const(True),
)
plan_shape = compose([tp_shape], spec)
c_shape = TechniqueContext(step=0, spec=spec, cache_key="shape", scratch={})
hs_m, hs_car = plan_shape.before_blocks(c_shape, hidden)
check("shape-stable ToMeSD merge: first active step reduces tokens", hs_m.shape[1] == 12)
hs_r = plan_shape.after_blocks(c_shape, hs_m, hs_car)
check("shape-stable ToMeSD unmerge: restores original token count", hs_r.shape[1] == 16)

# ---- 6. registry ----
print("[6] registry")
t = build_technique("token_prune", keep_ratio=0.7)
check("build_technique('token_prune')", isinstance(t, TokenPrune))

# ---- 7. full-opt assembly (the 5-component config) ----
print("[7] generic full-opt preset")
from efficiency.presets import ltx_full_opt  # noqa: E402
from efficiency.transform import TransformContext  # noqa: E402
from efficiency.transforms.kwl_fusions import KWLFusions  # noqa: E402
from efficiency.transforms.nvfp4_ffn import NVFP4FFN  # noqa: E402

video_spec = ModelSpec(
    name="VideoDiTManifestSpec",
    capabilities=frozenset(
        {
            Capability.BLOCKS,
            Capability.PRUNABLE_TOKENS,
            Capability.SWAPPABLE_ATTENTION,
            Capability.SUPPORTS_STEP_CACHE,
            Capability.SUPPORTS_NVFP4_LINEAR,
        }
    ),
    seq_dim=1,
)
check("generic video spec constructed", video_spec.name == "VideoDiTManifestSpec")

items = ltx_full_opt()  # 2 techniques + 3 transforms
plan = compose(items, video_spec)  # must NOT raise (all 5 compose cleanly)
check("full-opt composes (5 items, no conflict)",
      len(plan.transforms) == 3 and len(plan.techniques) == 2)

# transforms set the exact existing env (delegate, not reimplement)
env = {}
plan.apply_transforms(None, stage="stage2", env=env)
check("KWL env set", env.get("SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE") == "1")
check("NVFP4 env set", env.get("SGLANG_HQ_ENABLE_TE_NVFP4_FFN") == "1")
check("PISA backend = transformer_2 piecewise",
      "transformer_2=piecewise_attn" in env.get("SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS", ""))

# Generic transforms must not smuggle model-specific replay glue. The explicit
# LTX2 preset may request that glue; the default model-agnostic transforms may
# only expose neutral strategy axes.
generic_kwl_env = {}
compose([KWLFusions()], video_spec).apply_transforms(None, stage="stage2", env=generic_kwl_env)
check("generic KWL: no LTX2 adapter marker",
      "SGLANG_HQ_KWL_ADAPTER" not in generic_kwl_env)
check("generic KWL: full bundle flags default off",
      all(v == "0" for k, v in generic_kwl_env.items() if k.startswith("SGLANG_HQ_KWL_")))

explicit_kwl_env = {}
compose([KWLFusions(kwl_adapter="ltx2")], video_spec).apply_transforms(
    None, stage="stage2", env=explicit_kwl_env
)
check("explicit KWL LTX2 adapter: marker set",
      explicit_kwl_env.get("SGLANG_HQ_KWL_ADAPTER") == "ltx2")
check("explicit KWL LTX2 adapter: bundle enabled",
      explicit_kwl_env.get("SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE") == "1")

generic_nvfp4_env = {}
compose([NVFP4FFN(fused_proj_in_gelu=True, fused_proj_out_bias_gate=True)], video_spec).apply_transforms(
    None, stage="stage2", env=generic_nvfp4_env
)
check("generic NVFP4: no LTX2 TE adapter env",
      not any(k.startswith("SGLANG_LTX2_TE_NVFP4_") for k in generic_nvfp4_env))

explicit_nvfp4_env = {}
compose([NVFP4FFN(fused_proj_in_gelu=True, fused_proj_out_bias_gate=True, te_adapter="ltx2")], video_spec).apply_transforms(
    None, stage="stage2", env=explicit_nvfp4_env
)
check("explicit NVFP4 LTX2 adapter: env gated",
      explicit_nvfp4_env.get("SGLANG_LTX2_TE_NVFP4_VIDEO_FFN") == "1"
      and explicit_nvfp4_env.get("SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_IN_GELU") == "1"
      and explicit_nvfp4_env.get("SGLANG_LTX2_TE_NVFP4_FUSED_PROJ_OUT_BIAS_GATE") == "1")

# per-stage gating: TokenPrune active in stage2 step1, inactive in stage1
tp = [t for t in plan.techniques if t.name == "token_prune"][0]
check("prune active stage2 step1",
      tp.is_active(TechniqueContext(step=1, stage="stage2", spec=video_spec)))
check("prune inactive stage1 step1",
      not tp.is_active(TechniqueContext(step=1, stage="stage1", spec=video_spec)))
sc = [t for t in plan.techniques if t.name == "step_cache"][0]
check("step_cache active stage1 step20 (skip cluster)",
      sc.is_active(TechniqueContext(step=20, stage="stage1", spec=video_spec)))
check("step_cache inactive stage2",
      not sc.is_active(TechniqueContext(step=1, stage="stage2", spec=video_spec)))

# no-FP4 variant drops the NVFP4 transform
env2 = {}
compose(ltx_full_opt(nvfp4=False), video_spec).apply_transforms(None, "stage2", env2)
check("no-fp4 variant: NVFP4 env NOT set", "SGLANG_HQ_ENABLE_TE_NVFP4_FFN" not in env2)

print(f"\n=== {ok} passed, {fail} failed ===")
sys.exit(1 if fail else 0)
