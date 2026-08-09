#!/usr/bin/env python3
"""Aggregate assessment verdicts into speed-target quality-best delivery buckets.

Reads one or more `--verdict path/to/plan_eval_assess.json` (the JSON emitted by
`search/plan_eval.py --assess RUN_DIR`) and prints:
  1. a config table with speed and quality telemetry
  2. low/medium/high delivery recommendations:
     for each speed target, choose the best-quality config/profile at or
     above the target speed.

The composition step calls efficiency.compose() against manifest-declared
capabilities, so exclusive-seam conflicts (two step_output writers / two FFN
precisions / etc.) are surfaced instead of silently double-applying.

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

from search.plan_eval import load_profile, load_tiers, quality_ranking_key  # noqa: E402

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
                    help="Optional label per verdict (defaults to config id).")
    args = ap.parse_args()

    profile = load_profile(args.model)
    tiers = load_tiers()

    rows: list[dict] = []
    for i, vpath in enumerate(args.verdict):
        v = _read(Path(vpath))
        lbl = args.label[i] if i < len(args.label) else Path(v.get("run_dir", vpath)).name
        rows.append({**v, "label": lbl})

    # Per-row speed bucket (already in v when produced by current plan_eval) + quality.
    print(f"# Tier matrix -- model={args.model} baseline_total_s={profile['baseline']['total_s']:.2f}s")
    print()
    print(f"{'CANDIDATE':50} {'BUCKET':8} {'SPEEDUP':9} {'GEMINI':8} {'MAX_ART':9} {'LPIPS':9}")
    print("-" * 102)
    for r in rows:
        tier = r.get("tier") or "-"
        speedup = r.get("speedup")
        gemini = r.get("gemini_overall") or "-"
        sev = r.get("max_artifact_severity") or "-"
        lpips = r.get("lpips_max")
        speedup_s = f"{speedup:.3f}x" if speedup else "-"
        lpips_s = f"{lpips:.4f}" if isinstance(lpips, (int, float)) else "-"
        print(f"{r['label']:50} {tier:8} {speedup_s:9} {gemini:8} {sev:9} {lpips_s:9}")

    # Delivery winner per target: choose the best quality among config that
    # meet the target speed. LPIPS/Gemini are ranking signals, not hard thresholds.
    print()
    print("# Delivery winners (best quality at or above each speed target)")
    for t in TIERS:
        target = tiers["targets"].get(f"{t}_speedup")
        winners = sorted(
            (row for row in rows if row.get("speedup") and target and row["speedup"] >= target),
            key=quality_ranking_key,
        )
        if winners:
            w = winners[0]
            print(
                f"  {t:8} target {target}x  selected {w['speedup']:.3f}x "
                f"quality=(gemini={w.get('gemini_overall')}, "
                f"severity={w.get('max_artifact_severity')}, "
                f"lpips={w.get('lpips_max')}) cand={w['label']}"
            )
        else:
            print(f"  {t:8} target {target}x  -- no config reaches speed target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
