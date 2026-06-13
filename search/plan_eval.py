#!/usr/bin/env python3
"""GPU-stage eval for the acceleration search.

Drives one candidate (or a dimension's bounded loop) through the real pipeline:
render a candidate manifest from the model profile + composed technique env ->
launch (scripts/launch_candidate.py, sbatch) -> collect (scripts/collect_run.py)
-> quality (tools/vision/nvidia_gemini_judge.py) -> compare vs the model baseline
-> bin into a risk tier (evals/tiers.toml).

The orchestration logic (render_candidate / assess / tier_of / search_loop) lives
here; the GPU work is the existing scripts. `assess()` also runs standalone on an
already-completed run dir (no GPU) — used to validate the harness on an existing run.

CLI:
  python search/plan_eval.py --assess RUN_DIR [--baseline-frames DIR] [--model cosmos3]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # py<3.11 (sana env) ships tomli
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[1]
_SEV = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _load(p: Path) -> dict:
    with open(p, "rb") as f:
        return tomllib.load(f)


def load_profile(model_id: str) -> dict:
    return _load(REPO / "models" / f"{model_id}.toml")


def load_tiers() -> dict:
    return _load(REPO / "evals" / "tiers.toml")


def max_gemini_severity(gemini: dict | None) -> str:
    arts = (gemini or {}).get("new_artifacts") or []
    if not arts:
        return "none"
    return max((a.get("severity", "low") for a in arts), key=lambda s: _SEV.get(s, 1))


def tier_of(speedup, peak_mem_ratio, gemini, tiers, lpips_delta=None):
    """Cleanest (low-first) tier the candidate qualifies for, or None.

    Requires beating the baseline on latency OR peak memory, AND meeting a tier's
    quality budget. Tiers are quality budgets that loosen low->high, so a clean
    candidate is classified into the *tightest* (low) tier it satisfies — that is
    the safest config to ship; a more-degraded one can only be offered higher-risk.
    """
    improved = (speedup and speedup > 1.0 + 1e-6) or (
        peak_mem_ratio and peak_mem_ratio < 1.0 - 1e-6
    )
    if not improved:
        return None  # no speed/mem win -> not a tier candidate
    overall = (gemini or {}).get("overall")
    if overall in (None, "inconclusive"):
        return None  # no usable quality verdict -> cannot tier (never auto-promote to high)
    sev = _SEV[max_gemini_severity(gemini)]
    for name in ("low", "medium", "high"):  # cleanest budget first
        t = tiers[name]
        if sev > _SEV.get(t.get("gemini_max_artifact_severity", "high"), 3):
            continue
        if t.get("gemini_overall", "pass_or_fail") == "pass" and overall != "pass":
            continue
        if lpips_delta is not None and lpips_delta > t.get("lpips_delta_max", 1.0):
            continue
        return name
    return None  # fails even the high budget -> reject


def render_candidate(profile: dict, technique: str, cfg: dict, kind: str = "build_transform",
                     candidate_id: str | None = None, out_path: Path | None = None) -> dict:
    """Model profile + the composed technique env -> a launcher-valid manifest."""
    sys.path.insert(0, str(REPO))
    from efficiency import compose, get_model_spec
    from efficiency.registry import build_technique, build_transform

    spec = get_model_spec(profile["spec"])
    item = build_transform(technique, **cfg) if kind == "build_transform" else build_technique(technique, **cfg)
    plan = compose([item], spec)
    tech_env: dict = {}
    if kind == "build_transform":
        plan.apply_transforms(None, "stage2", tech_env)  # transforms set SGLANG_HQ_* env
    cid = candidate_id or f"{profile['id']}__{technique}"
    manifest = {
        "id": cid,
        "kind": "methodology",
        "description": f"{technique} candidate on {profile['display_name']} (search-rendered).",
        "submodule": profile["submodule"],
        "base_commit": profile["base_commit"],
        "run_script": profile["run_script"],
        "eval_profile": profile["eval_profile"],
        "official_config": profile["official_config"],
        "env": {**profile.get("env", {}), **tech_env},
        "artifacts": {"output_dir": "outputs", "video": "out.mp4", "log": "run.log",
                      "benchmark": "benchmark.json", "quality": "quality.json",
                      "frames_dir": "frames", "patch_summary": "patch_summary.md"},
        "slurm": {"account": "nvr_elm_llm", "partition": "batch", "nodes": 1,
                  "gpus_per_node": profile["official_config"].get("num_gpus", 4),
                  "cpus_per_task": 64, "mem": "0", "time": "04:00:00",
                  "job_name": f"autovideo-{cid}", "exclusive": True},
    }
    if out_path:
        _write_toml(manifest, out_path)
    return manifest


def assess(run_dir, profile: dict, tiers: dict, baseline_frames: str | None = None,
           gemini: bool = True) -> dict:
    """Collect a finished run, judge quality vs baseline, bin into a tier."""
    run_dir = Path(run_dir)
    subprocess.run([sys.executable, str(REPO / "scripts/collect_run.py"), str(run_dir)],
                   cwd=str(REPO), capture_output=True, text=True)
    bench_p = run_dir / "outputs/benchmark.json"
    bench = json.load(open(bench_p)) if bench_p.exists() else {}
    base = profile.get("baseline", {})
    cand_total = bench.get("total_s") or bench.get("denoise_s")
    base_total = base.get("total_s")
    speedup = (base_total / cand_total) if (cand_total and base_total) else None

    # Gemini verdict: prefer a rigorous pairwise judge (candidate frames vs the real
    # baseline frames); fall back to the collector's quality.json verdict.
    gem = None
    qp = run_dir / "outputs/quality.json"
    cand_frames = sorted((run_dir / "outputs/frames").glob("*.png"))
    base_fr = sorted(Path(baseline_frames).glob("*.png")) if baseline_frames else []
    if gemini and base_fr and cand_frames:
        n = min(len(base_fr), len(cand_frames), 4)
        pj = run_dir / "outputs/quality_pairwise.json"
        cmd = [sys.executable, str(REPO / "tools/vision/nvidia_gemini_judge.py"),
               "--out", str(pj), "--max-tokens", "1024"]
        for i in range(n):
            cmd += ["--baseline-frame", str(base_fr[i]), "--candidate-frame", str(cand_frames[i])]
        subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        if pj.exists():
            gem = json.load(open(pj))
    if gem is None and qp.exists():  # fall back to the collector's gemini verdict (nested)
        q = json.load(open(qp))
        gem = ((q.get("judges") or {}).get("nvidia_gemini") or {}).get("result")
    tier = tier_of(speedup, None, gem, tiers)
    return {
        "run_dir": str(run_dir),
        "baseline_total_s": base_total,
        "candidate_total_s": cand_total,
        "speedup": round(speedup, 4) if speedup else None,
        "gemini_overall": (gem or {}).get("overall"),
        "max_artifact_severity": max_gemini_severity(gem) if gem else None,
        "tier": tier,
        "note": (None if tier else "no latency/mem improvement vs baseline -> not a tier winner "
                 "(expected until a technique is wired into the model runtime)"),
    }


def _write_toml(d: dict, path: Path) -> None:
    """Minimal TOML writer for the flat candidate manifest (no extra deps)."""
    def val(v):
        if isinstance(v, bool): return "true" if v else "false"
        if isinstance(v, (int, float)): return repr(v)
        return '"' + str(v).replace('"', '\\"') + '"'
    lines = []
    tables = {k: v for k, v in d.items() if isinstance(v, dict)}
    for k, v in d.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {val(v)}")
    for tname, t in tables.items():
        lines.append(f"\n[{tname}]")
        for k, v in t.items():
            lines.append(f"{k} = {val(v)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="cosmos3")
    ap.add_argument("--assess", metavar="RUN_DIR", help="assess an already-completed run dir")
    ap.add_argument("--baseline-frames", default=None, help="baseline frames dir for the Gemini judge")
    ap.add_argument("--no-gemini", action="store_true")
    args = ap.parse_args()
    prof, tiers = load_profile(args.model), load_tiers()
    if args.assess:
        verdict = assess(args.assess, prof, tiers, baseline_frames=args.baseline_frames,
                         gemini=not args.no_gemini)
        print(json.dumps(verdict, indent=2))
    else:
        print("provide --assess RUN_DIR (GPU search_loop orchestration: see docstring)")
