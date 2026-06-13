#!/usr/bin/env python3
"""Independent test for the model-agnostic search harness (CPU; needs torch).

Run: ~/lustre/miniconda3/envs/sana/bin/python search/test_search.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.search import load_dimensions, load_model_profile, search  # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}")


# model profile loads + names a registered spec
prof = load_model_profile("cosmos3")
check("cosmos3 profile loads", prof["spec"] == "Cosmos3")
check("profile carries official_config + baseline", "official_config" in prof and "baseline" in prof)

# the search runs and is model-agnostic: dimensions never name a model
results = search("cosmos3", verbose=False)
check("search returns results once a dimension exists", isinstance(results, list))

# whatever dimensions exist, each result is composable-or-rejected with a reason,
# and eligibility is decided by the model spec's declared capabilities (not by the
# dimension naming the model)
for r in results:
    check(f"{r['dimension']}/{r['technique']} has a verdict",
          r["composable"] + r["rejected"] == r["candidates"])

# the dimensions on disk must be model-agnostic: no model id / no SGLANG_<MODEL>_ env
import pathlib  # noqa: E402
import re  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
leak = re.compile(r"cosmos3|SGLANG_COSMOS3|nvidia/Cosmos3", re.I)
for dim_id, _ in load_dimensions():
    txt = (REPO / "loops" / dim_id / "dimension.toml").read_text()
    check(f"loops/{dim_id}/dimension.toml is model-agnostic (no model identity)",
          not leak.search(txt))

print(f"\n=== {ok} passed, {fail} failed ===")
sys.exit(1 if fail else 0)
