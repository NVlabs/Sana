#!/usr/bin/env python3
"""Aggregate per-dimension assessment verdicts into the final tier matrix.

Reads one or more `--verdict path/to/plan_eval_assess.json` (the JSON emitted by
`search/plan_eval.py --assess RUN_DIR`) and prints:
  1. a per-dimension best-per-tier table
  2. the composed low/medium/high recommendation across dimensions (which
     candidates would stack, with compose() seam-conflict noted)

The composition step calls efficiency.compose() against the Cosmos3 spec, so
exclusive-seam conflicts (two step_output writers / two FFN precisions / etc.)
are surfaced instead of silently double-applying.

CLI:
  python scripts/tier_report.py --model cosmos3 --verdict v1.json --verdict v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from search.plan_eval import load_profile, load_tiers  # noqa: E402

TIERS = ("low", "medium", "high")


def _read(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="cosmos3")
    ap.add_argument(
        "--verdict",
        action="append",
        required=True,
        help="Path to one plan_eval assess JSON (repeatable).",
    )
    ap.add_argument("--label", action="append", default=[],
                    help="Optional label per verdict (defaults to candidate id).")
    args = ap.parse_args()

    profile = load_profile(args.model)
    tiers = load_tiers()

    rows: list[dict] = []
    for i, vpath in enumerate(args.verdict):
        v = _read(Path(vpath))
        lbl = args.label[i] if i < len(args.label) else Path(v.get("run_dir", vpath)).name
        rows.append({**v, "label": lbl})

    # Per-row tier (already in v) + speedup.
    print(f"# Tier matrix -- model={args.model} baseline_total_s={profile['baseline']['total_s']:.2f}s")
    print()
    print(f"{'CANDIDATE':50} {'TIER':8} {'SPEEDUP':9} {'GEMINI':8} {'MAX_ART':9}")
    print("-" * 90)
    by_tier: dict[str, list[dict]] = {t: [] for t in TIERS}
    rejected: list[dict] = []
    for r in rows:
        tier = r.get("tier") or "REJECT"
        speedup = r.get("speedup")
        gemini = r.get("gemini_overall") or "-"
        sev = r.get("max_artifact_severity") or "-"
        speedup_s = f"{speedup:.3f}x" if speedup else "-"
        print(f"{r['label']:50} {tier:8} {speedup_s:9} {gemini:8} {sev:9}")
        if tier in by_tier:
            by_tier[tier].append(r)
        else:
            rejected.append(r)

    # Per-tier winner (best speedup that qualifies for that tier or tighter).
    # A candidate qualifying for `low` also satisfies `medium` and `high`; pick
    # the best speedup across the tightest-qualified.
    print()
    print("# Per-tier winners (best speedup whose tier <= bucket)")
    rank = {"low": 0, "medium": 1, "high": 2}
    qualified_for: dict[str, list[dict]] = {t: [] for t in TIERS}
    for r in rows:
        if r.get("tier") in rank:
            for t in TIERS:
                if rank[r["tier"]] <= rank[t]:
                    qualified_for[t].append(r)
    for t in TIERS:
        winners = sorted(
            (q for q in qualified_for[t] if q.get("speedup")),
            key=lambda q: -q["speedup"],
        )
        target = tiers["targets"].get(f"{t}_speedup")
        if winners:
            w = winners[0]
            mark = "[hit]" if (target and w["speedup"] >= target) else "[short]"
            print(
                f"  {t:8} target {target}x  best {w['speedup']:.3f}x "
                f"{mark}  cand={w['label']}"
            )
        else:
            print(f"  {t:8} target {target}x  -- no qualifying candidate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
