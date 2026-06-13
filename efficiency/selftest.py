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
from efficiency.techniques.token_prune import TokenPrune  # noqa: E402

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

# ---- 6. registry ----
print("[6] registry")
t = build_technique("token_prune", keep_ratio=0.7)
check("build_technique('token_prune')", isinstance(t, TokenPrune))

# ---- 7. LTX-2 full-opt assembly (the 5-component config) ----
print("[7] ltx full-opt preset")
from efficiency import get_model_spec  # noqa: E402
from efficiency.presets import ltx_full_opt  # noqa: E402
from efficiency.transform import TransformContext  # noqa: E402

ltx_spec = get_model_spec("LTX2")
check("LTX2 spec registered", ltx_spec is not None and ltx_spec.name == "LTX2")

items = ltx_full_opt()  # 2 techniques + 3 transforms
plan = compose(items, ltx_spec)  # must NOT raise (all 5 compose cleanly)
check("full-opt composes (5 items, no conflict)",
      len(plan.transforms) == 3 and len(plan.techniques) == 2)

# transforms set the exact existing env (delegate, not reimplement)
env = {}
plan.apply_transforms(None, stage="stage2", env=env)
check("KWL env set", env.get("SGLANG_HQ_KWL_FUSED_CA_DUAL_MODULATE") == "1")
check("NVFP4 env set", env.get("SGLANG_HQ_ENABLE_TE_NVFP4_FFN") == "1")
check("PISA backend = transformer_2 piecewise",
      "transformer_2=piecewise_attn" in env.get("SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS", ""))

# per-stage gating: TokenPrune active in stage2 step1, inactive in stage1
tp = [t for t in plan.techniques if t.name == "token_prune"][0]
check("prune active stage2 step1",
      tp.is_active(TechniqueContext(step=1, stage="stage2", spec=ltx_spec)))
check("prune inactive stage1 step1",
      not tp.is_active(TechniqueContext(step=1, stage="stage1", spec=ltx_spec)))
sc = [t for t in plan.techniques if t.name == "step_cache"][0]
check("step_cache active stage1 step20 (skip cluster)",
      sc.is_active(TechniqueContext(step=20, stage="stage1", spec=ltx_spec)))
check("step_cache inactive stage2",
      not sc.is_active(TechniqueContext(step=1, stage="stage2", spec=ltx_spec)))

# no-FP4 variant drops the NVFP4 transform
env2 = {}
compose(ltx_full_opt(nvfp4=False), ltx_spec).apply_transforms(None, "stage2", env2)
check("no-fp4 variant: NVFP4 env NOT set", "SGLANG_HQ_ENABLE_TE_NVFP4_FFN" not in env2)

print(f"\n=== {ok} passed, {fail} failed ===")
sys.exit(1 if fail else 0)
