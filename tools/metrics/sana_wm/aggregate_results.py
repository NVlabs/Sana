"""Aggregate SANA-WM benchmark results into a comparison table.

Walks `results/<method>/{eval/<split>/summary.json,
<split>/eval_poses.json, eval/<split>/temporal_degradation.json}` and emits
markdown and JSON summaries.

Columns (per method × split):
  VBench Total / Quality / Semantic
  Revisit PSNR / SSIM / LPIPS
  Camera RotErr° / TransErr_rel / CamMC_rel
  Temporal degradation (first → last imaging_quality)

Usage:
  python tools/metrics/sana_wm/aggregate_results.py \
      --results_root results \
      --md_out results/sana_wm_benchmark.md \
      --json_out results/sana_wm_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

RESULTS_ROOT = "results"
OUT_MD = os.path.join("results", "sana_wm_benchmark.md")
OUT_JSON = os.path.join("results", "sana_wm_benchmark.json")


def _safe_load(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _rot_errors_as_degrees(rows: list[dict[str, Any]]) -> list[float]:
    """Return RotErr values in degrees, supporting old rad-valued JSON files."""
    out: list[float] = []
    for row in rows:
        value = float(row["RotErr"])
        unit = str(row.get("RotErr_unit", "")).lower()
        if unit == "deg":
            out.append(value)
        elif unit == "rad" or abs(value) <= math.pi + 1e-6:
            out.append(value * 180.0 / math.pi)
        else:
            out.append(value)
    return out


# Mirrors VBENCH_DIM_WEIGHTS / NORMALIZE_DIC / compute_vbench_total in
# eval_unified.py. Kept local to avoid heavy torch imports during aggregation.
_VBENCH_QUALITY_DIMS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "dynamic_degree",
]
_VBENCH_SEMANTIC_DIMS = [
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
]
_VBENCH_DIM_WEIGHTS = {d: 1.0 for d in _VBENCH_QUALITY_DIMS + _VBENCH_SEMANTIC_DIMS}
_VBENCH_DIM_WEIGHTS["dynamic_degree"] = 0.5
_NORMALIZE_DIC = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "temporal_flickering": (0.6293, 1.0),
    "motion_smoothness": (0.7060, 0.9975),
    "dynamic_degree": (0.0, 1.0),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
    "object_class": (0.0, 1.0),
    "multiple_objects": (0.0, 1.0),
    "human_action": (0.0, 1.0),
    "color": (0.0, 1.0),
    "spatial_relationship": (0.0, 1.0),
    "scene": (0.0, 1.0),
    "appearance_style": (0.0, 1.0),
    "temporal_style": (0.0, 1.0),
    "overall_consistency": (0.0, 1.0),
}


def _normalize_score(dim: str, raw: float) -> float:
    if dim not in _NORMALIZE_DIC:
        return raw
    mn, mx = _NORMALIZE_DIC[dim]
    if mx - mn < 1e-8:
        return raw
    return max(0.0, min(1.0, (raw - mn) / (mx - mn)))


def _compute_vbench_total_local(scores: dict[str, float]) -> dict[str, float]:
    norm = {d: _normalize_score(d, s) for d, s in scores.items()}
    q_dims = [d for d in _VBENCH_QUALITY_DIMS if d in norm]
    if q_dims:
        q_w = sum(_VBENCH_DIM_WEIGHTS.get(d, 1.0) for d in q_dims)
        quality = sum(norm[d] * _VBENCH_DIM_WEIGHTS.get(d, 1.0) for d in q_dims) / q_w
    else:
        quality = 0.0
    s_dims = [d for d in _VBENCH_SEMANTIC_DIMS if d in norm]
    if s_dims:
        s_w = sum(_VBENCH_DIM_WEIGHTS.get(d, 1.0) for d in s_dims)
        semantic = sum(norm[d] * _VBENCH_DIM_WEIGHTS.get(d, 1.0) for d in s_dims) / s_w
    else:
        semantic = 0.0
    total = (4.0 * quality + 1.0 * semantic) / 5.0
    return {
        "quality_score": round(quality, 4),
        "semantic_score": round(semantic, 4),
        "total_score": round(total, 4),
    }


def _recompute_vbench_from_raw(eval_dir: str) -> dict[str, float]:
    """Recover per-dimension VBench scores from raw eval_<dim>_eval_results.json files.

    Used as a fallback when summary.json was overwritten without a `vbench` block
    (e.g. by a later metric run such as LPIPS revisit).
    """
    scores: dict[str, float] = {}
    if not os.path.isdir(eval_dir):
        return scores
    import math

    for fname in os.listdir(eval_dir):
        if not fname.startswith("eval_") or not fname.endswith("_eval_results.json"):
            continue
        path = os.path.join(eval_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for dim, payload in data.items():
            if dim in scores:
                continue
            if not isinstance(payload, list) or len(payload) == 0:
                continue
            try:
                val = float(payload[0])
            except (TypeError, ValueError):
                continue
            if math.isnan(val):
                continue
            scores[dim] = val
    return scores


def _gather_one(results_root: str, method: str, split: str) -> dict[str, Any]:
    """Collect every available metric for one (method, split)."""
    base = os.path.join(results_root, method)
    rec: dict[str, Any] = {"method": method, "split": split}

    # method_info
    mi = _safe_load(os.path.join(base, "method_info.json"))
    if mi:
        rec["method_info"] = {
            "model": mi.get("model"),
            "variant": mi.get("variant"),
            "fps": mi.get("fps"),
            "resolution": mi.get("resolution"),
        }

    # summary.json (vbench + revisit)
    summary = _safe_load(os.path.join(base, "eval", split, "summary.json"))
    eval_dir = os.path.join(base, "eval", split)
    if summary and "vbench" in summary:
        v = summary["vbench"]
        rec["vbench"] = {
            "total": v.get("total_score"),
            "quality": v.get("quality_score"),
            "semantic": v.get("semantic_score"),
        }
    else:
        # Fallback: load vbench_scores.json if present
        vs = _safe_load(os.path.join(eval_dir, "vbench_scores.json"))
        if vs:
            rec["vbench"] = {
                "total": vs.get("total_score"),
                "quality": vs.get("quality_score"),
                "semantic": vs.get("semantic_score"),
            }
        else:
            # Deepest fallback: recompute from raw eval_*_eval_results.json
            scores = _recompute_vbench_from_raw(eval_dir)
            if scores:
                total = _compute_vbench_total_local(scores)
                rec["vbench"] = {
                    "total": total["total_score"],
                    "quality": total["quality_score"],
                    "semantic": total["semantic_score"],
                }

    # revisit (might be in summary, or in revisit_consistency.json)
    revisit = _safe_load(os.path.join(base, "eval", split, "revisit_consistency.json"))
    if revisit and "summary" in revisit:
        s = revisit["summary"]
        rec["revisit"] = {
            "psnr": s.get("overall_mean_psnr"),
            "ssim": s.get("overall_mean_ssim"),
            "lpips": s.get("overall_mean_lpips"),
            "n_pairs": s.get("n_total_pairs"),
            "per_category": s.get("per_category"),
        }
    elif summary and "revisit" in summary:
        s = summary["revisit"]
        rec["revisit"] = {
            "psnr": s.get("overall_mean_psnr"),
            "ssim": s.get("overall_mean_ssim"),
            "lpips": s.get("overall_mean_lpips"),
        }

    # camera (eval_poses.json sits in result_folder = base/split). The schema is
    # a flat dict {scene_id: {"RotErr": ..., "TransErr_rel": ..., "CamMC_rel": ...,
    # "scene_type": ..., "n_frames": ...}, ...}. Aggregate here.
    poses = _safe_load(os.path.join(base, split, "eval_poses.json"))
    if poses:
        import numpy as _np

        rows: list[dict[str, Any]] = []
        trn, cmc = [], []
        per_cat: dict[str, dict[str, list[float]]] = {}
        for scene_id, v in poses.items():
            if not isinstance(v, dict) or "RotErr" not in v:
                continue
            rows.append(v)
            trn.append(float(v.get("TransErr_rel", 0.0)))
            cmc.append(float(v.get("CamMC_rel", 0.0)))
            cat = v.get("scene_type", "unknown")
            per_cat.setdefault(cat, {"rot": [], "trn": [], "cmc": []})
            per_cat[cat]["trn"].append(trn[-1])
            per_cat[cat]["cmc"].append(cmc[-1])
        rot_deg = _rot_errors_as_degrees(rows)
        for row, rot in zip(rows, rot_deg):
            cat = row.get("scene_type", "unknown")
            per_cat[cat]["rot"].append(rot)
        if rot_deg:
            rec["camera"] = {
                "rot_err_deg": float(_np.mean(rot_deg)),
                "trans_err_rel": float(_np.mean(trn)),
                "cam_mc_rel": float(_np.mean(cmc)),
                "n_scenes": len(rot_deg),
                "per_category": {
                    cat: {
                        "rot_err_deg": float(_np.mean(d["rot"])),
                        "trans_err_rel": float(_np.mean(d["trn"])),
                        "cam_mc_rel": float(_np.mean(d["cmc"])),
                        "n_scenes": len(d["rot"]),
                    }
                    for cat, d in per_cat.items()
                },
            }

    # temporal degradation. Schema: {"windows": {wname: {dim: score}}, "trend": {dim: {...}}}
    temp = _safe_load(os.path.join(base, "eval", split, "temporal_degradation.json"))
    if temp:
        trend = temp.get("trend", {})
        windows = temp.get("windows", {})
        iq = trend.get("imaging_quality", {})
        rec["temporal"] = {
            "n_windows": len(windows),
            "imaging_quality_first": iq.get("first_window"),
            "imaging_quality_last": iq.get("last_window"),
            "imaging_quality_drop": iq.get("degradation"),
            "trend": trend,
        }

    return rec


def _format_cell(value: Any, fmt: str = "{:.3f}") -> str:
    if value is None:
        return "—"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def _build_md(records: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("# SANA-WM Benchmark Evaluation Results\n")
    lines.append("Auto-generated by `tools/metrics/sana_wm/aggregate_results.py`.\n")
    lines.append("")

    # Split into per-split tables
    for split in ("simple_60s", "hard_60s"):
        sub = [r for r in records if r["split"] == split]
        if not sub:
            continue
        lines.append(f"## Split: `{split}`\n")
        lines.append(
            "| Method | VBench Total | Quality | Semantic | "
            "Revisit PSNR↑ | SSIM↑ | LPIPS↓ | "
            "RotErr°↓ | TransErr_rel↓ | CamMC_rel↓ | "
            "Temp IQ first | last | drop |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(sub, key=lambda x: x["method"]):
            v = r.get("vbench", {})
            rv = r.get("revisit", {})
            cm = r.get("camera", {})
            tm = r.get("temporal", {})
            lines.append(
                f"| `{r['method']}` "
                f"| {_format_cell(v.get('total'), '{:.4f}')} "
                f"| {_format_cell(v.get('quality'), '{:.4f}')} "
                f"| {_format_cell(v.get('semantic'), '{:.4f}')} "
                f"| {_format_cell(rv.get('psnr'), '{:.2f}')} "
                f"| {_format_cell(rv.get('ssim'), '{:.4f}')} "
                f"| {_format_cell(rv.get('lpips'), '{:.4f}')} "
                f"| {_format_cell(cm.get('rot_err_deg'), '{:.2f}')} "
                f"| {_format_cell(cm.get('trans_err_rel'), '{:.4f}')} "
                f"| {_format_cell(cm.get('cam_mc_rel'), '{:.4f}')} "
                f"| {_format_cell(tm.get('imaging_quality_first'), '{:.4f}')} "
                f"| {_format_cell(tm.get('imaging_quality_last'), '{:.4f}')} "
                f"| {_format_cell(tm.get('imaging_quality_drop'), '{:+.4f}')} |"
            )
        lines.append("")

    # Per-category revisit breakdown (if available)
    cat_rows: list[str] = []
    for r in records:
        rv = r.get("revisit", {}) or {}
        per_cat = rv.get("per_category") or {}
        for cat, cs in per_cat.items():
            cat_rows.append(
                f"| `{r['method']}` | {r['split']} | {cat} "
                f"| {_format_cell(cs.get('mean_psnr'), '{:.2f}')} "
                f"| {_format_cell(cs.get('mean_ssim'), '{:.4f}')} "
                f"| {_format_cell(cs.get('mean_lpips'), '{:.4f}')} "
                f"| {cs.get('n_scenes', '—')} |"
            )
    if cat_rows:
        lines.append("## Revisit per-category breakdown\n")
        lines.append("| Method | Split | Category | PSNR↑ | SSIM↑ | LPIPS↓ | N scenes |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        lines.extend(cat_rows)
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default=RESULTS_ROOT, help="Root containing method/ subdirs")
    parser.add_argument("--md_out", default=OUT_MD)
    parser.add_argument("--json_out", default=OUT_JSON)
    args = parser.parse_args()

    methods = sorted(d for d in os.listdir(args.results_root) if os.path.isdir(os.path.join(args.results_root, d)))

    records: list[dict[str, Any]] = []
    for method in methods:
        for split in ("simple_60s", "hard_60s", "simple", "hard"):
            video_dir = os.path.join(args.results_root, method, split)
            if not os.path.isdir(video_dir):
                continue
            # Skip `eval` subdir as a split
            if split == "eval":
                continue
            rec = _gather_one(args.results_root, method, split)
            # Only keep if any metric present
            if any(k in rec for k in ("vbench", "revisit", "camera", "temporal")):
                records.append(rec)

    json_dir = os.path.dirname(args.json_out)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump({"records": records}, f, indent=2)

    md = _build_md(records)
    md_dir = os.path.dirname(args.md_out)
    if md_dir:
        os.makedirs(md_dir, exist_ok=True)
    with open(args.md_out, "w") as f:
        f.write(md)

    print(f"Wrote {len(records)} records:")
    print(f"  Markdown: {args.md_out}")
    print(f"  JSON:     {args.json_out}")


if __name__ == "__main__":
    main()
