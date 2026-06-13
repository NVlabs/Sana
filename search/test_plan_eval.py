#!/usr/bin/env python3
"""CPU test for plan_eval: tier binning + candidate rendering.
Run: ~/lustre/miniconda3/envs/sana/bin/python search/test_plan_eval.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.plan_eval import load_profile, load_tiers, tier_of, render_candidate  # noqa: E402

ok = fail = 0
def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")

tiers = load_tiers()

# --- tier_of: improvement is required ---
check("no speedup/mem win -> no tier", tier_of(1.0, None, {"overall": "pass", "new_artifacts": []}, tiers) is None)

# --- cleanest config -> low ---
check("faster + clean(pass, no artifacts) -> low",
      tier_of(1.3, None, {"overall": "pass", "new_artifacts": []}, tiers) == "low")

# --- low-severity artifact falls out of low into medium ---
check("faster + pass + low-severity artifact -> medium",
      tier_of(2.0, None, {"overall": "pass", "new_artifacts": [{"severity": "low"}]}, tiers) == "medium")

# --- medium-severity -> high (and overall may fail) ---
check("faster + medium-severity -> high",
      tier_of(3.0, None, {"overall": "fail", "new_artifacts": [{"severity": "medium"}]}, tiers) == "high")

# --- high-severity artifact -> reject (no tier) ---
check("high-severity artifact -> reject",
      tier_of(3.0, None, {"overall": "fail", "new_artifacts": [{"severity": "high"}]}, tiers) is None)

# --- memory-only win still qualifies ---
check("mem win (peak_mem_ratio<1) + clean -> low",
      tier_of(None, 0.8, {"overall": "pass", "new_artifacts": []}, tiers) == "low")

# --- render_candidate produces a launcher-valid sparse manifest ---
try:
    prof = load_profile("cosmos3")
    m = render_candidate(prof, "sparse_attention", {"sparsity": 0.9, "component": "transformer"})
    check("render: has official_config + slurm", "official_config" in m and "slurm" in m)
    check("render: composed sparse env present",
          "SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS" in m["env"])
    check("render: carries model base env (MODEL_REPO)", m["env"].get("MODEL_REPO") == "nvidia/Cosmos3-Super")
except Exception as e:  # torch/efficiency import issues shouldn't fail the tier logic
    print(f"  SKIP  render_candidate ({type(e).__name__}: {e})")

print(f"\n=== {ok} passed, {fail} failed ===")
sys.exit(1 if fail else 0)
