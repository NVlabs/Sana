#!/usr/bin/env python3
"""GPU-stage eval for the acceleration search.

Drives one candidate (or a dimension's bounded loop) through the real pipeline:
render a candidate manifest from the model profile + composed technique env ->
launch (scripts/launch_candidate.py, sbatch) -> collect (scripts/collect_run.py)
-> quality (tools/vision/nvidia_gemini_judge.py) -> compare vs the model baseline
-> bin into a speed-target delivery bucket (evals/tiers.toml [targets]).

The orchestration logic (render_candidate / assess / tier_of / search_loop) lives
here; the GPU work is the existing scripts. `assess()` also runs standalone on an
already-completed run dir (no GPU) — used to validate the harness on an existing run.

CLI:
  python search/plan_eval.py --assess RUN_DIR [--baseline-frames DIR] [--model cosmos3] [--out RUN_DIR/assess_verdict.json]
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
_OVERALL = {"pass": 0, "fail": 1, "inconclusive": 2, None: 3}

# Runtime-technique env mapping. Build-transforms set their own SGLANG_HQ_* env
# via Plan.apply_transforms(); runtime techniques (Phase.ON_STEP etc.) do not
# (their hooks live in-process), so the search has to publish their config to
# the runtime out-of-band. Each entry maps a registered technique name to:
#   {param_name -> (env_var_name, stringifier)}
# Adding a runtime technique here makes plan_eval able to drive it through the
# render_candidate -> launcher -> Sol-LTX-Infer pipeline without further glue.
_RUNTIME_TECHNIQUE_ENV: dict[str, dict[str, tuple[str, callable]]] = {
    "step_cache": {
        "skip": ("SGLANG_HQ_STEP_CACHE_SKIP", str),
        "delta_scale": ("SGLANG_HQ_STEP_CACHE_DELTA", lambda v: f"{float(v)}"),
    },
    "teacache": {
        "threshold": ("SGLANG_HQ_TEACACHE_THRESHOLD", lambda v: f"{float(v)}"),
        "start_step": ("SGLANG_HQ_TEACACHE_START_STEP", lambda v: f"{int(v)}"),
        "max_continuous_hits": ("SGLANG_HQ_TEACACHE_MAX_HITS", lambda v: f"{int(v)}"),
    },
    "token_prune": {
        "keep_ratio": ("SGLANG_HQ_TOKEN_PRUNE_KEEP_RATIO", lambda v: f"{float(v)}"),
        "steps": ("SGLANG_HQ_TOKEN_PRUNE_STEPS", str),
        "method": ("SGLANG_HQ_TOKEN_PRUNE_METHOD", str),
        "compensation": ("SGLANG_HQ_TOKEN_PRUNE_COMP", str),
    },
}


def _runtime_technique_env(technique: str, cfg: dict) -> dict[str, str]:
    """Map a runtime-technique cfg to SGLANG_HQ_* env vars per the table above."""
    mapping = _RUNTIME_TECHNIQUE_ENV.get(technique)
    if not mapping:
        return {}
    env: dict[str, str] = {}
    for k, v in cfg.items():
        spec = mapping.get(k)
        if spec is None:
            continue
        env_key, fmt = spec
        env[env_key] = fmt(v)
    return env


def _load(p: Path) -> dict:
    with open(p, "rb") as f:
        return tomllib.load(f)


def _load_json_if_present(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.load(open(path))
    except json.JSONDecodeError:
        return None


def usable_gemini_verdict(gemini: dict | None) -> bool:
    return (gemini or {}).get("overall") not in (None, "inconclusive")


def reconcile_quality_blockers(blockers: list[str], gemini: dict | None) -> list[str]:
    """Pairwise Gemini from plan_eval overrides collector-only Gemini failures."""

    if not usable_gemini_verdict(gemini):
        return blockers
    return [blocker for blocker in blockers if not blocker.startswith("nvidia_gemini:")]


def load_profile(model_id: str) -> dict:
    return _load(REPO / "models" / f"{model_id}.toml")


def load_tiers() -> dict:
    return _load(REPO / "evals" / "tiers.toml")


def max_gemini_severity(gemini: dict | None) -> str:
    arts = (gemini or {}).get("new_artifacts") or []
    if not arts:
        return "none"
    return max((a.get("severity", "low") for a in arts), key=lambda s: _SEV.get(s, 1))


def quality_ranking_key(row: dict) -> tuple:
    """Sort key for final speed-target winners: Gemini first, LPIPS second."""
    severity = _SEV.get(row.get("max_artifact_severity") or "high", 3)
    overall = _OVERALL.get(row.get("gemini_overall"), 3)
    lpips = row.get("lpips_max")
    lpips_value = float(lpips) if isinstance(lpips, (int, float)) else float("inf")
    speedup = row.get("speedup") or 0.0
    return (severity, overall, lpips_value, -speedup)


def tier_of(speedup, peak_mem_ratio, gemini, tiers, lpips_delta=None):
    """Speed-target delivery bucket the candidate reaches, or None.

    Fan-out retention is frontier-based, not LPIPS-threshold based. Final
    low/medium/high delivery profiles are speed target buckets (1.5x / 2x / 3x
    by default); within each bucket, `quality_ranking_key()` picks by aligned
    Gemini severity/status and LPIPS together rather than requiring an absolute
    LPIPS cutoff here.
    """
    improved = (speedup and speedup > 1.0 + 1e-6) or (
        peak_mem_ratio and peak_mem_ratio < 1.0 - 1e-6
    )
    if not improved:
        return None
    if not speedup:
        return None
    targets = tiers.get("targets", {})
    if speedup >= targets.get("high_speedup", 3.0):
        return "high"
    if speedup >= targets.get("medium_speedup", 2.0):
        return "medium"
    if speedup >= targets.get("low_speedup", 1.5):
        return "low"
    return None


def promotion_note(tier, quality_blockers, speedup, peak_mem_ratio, gemini, tiers, lpips_delta=None):
    """Human-readable reason a candidate did not reach a delivery speed bucket."""
    if tier:
        return None
    if quality_blockers:
        return (
            "missing or blocked quality evidence: "
            + ", ".join(quality_blockers)
            + " -> retain/backfill frontier evidence, but do not select final profile yet"
        )

    improved = (speedup and speedup > 1.0 + 1e-6) or (
        peak_mem_ratio and peak_mem_ratio < 1.0 - 1e-6
    )
    if not improved:
        return (
            "no latency/mem improvement vs baseline -> not a tier winner "
            "(expected until a technique is wired into the model runtime)"
        )

    target = (tiers.get("targets") or {}).get("low_speedup", 1.5)
    if speedup:
        return (
            f"speed/mem improved, but speedup {speedup:.4g}x is below the "
            f"lowest delivery target {target}x -> retained frontier evidence, "
            "not a final speed-bucket winner"
        )
    return "memory improved but no speed target bucket reached -> retained frontier evidence"


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
    else:
        # runtime techniques: publish their cfg through SGLANG_HQ_* env so the
        # Sol-LTX-Infer side can rebuild the same technique inside the denoise loop.
        tech_env.update(_runtime_technique_env(technique, cfg))
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
    collect_cmd = [sys.executable, str(REPO / "scripts/collect_run.py"), str(run_dir)]
    base_fr = sorted(Path(baseline_frames).glob("*.png")) if baseline_frames else []
    for frame in base_fr:
        collect_cmd.extend(["--baseline-frame", str(frame)])
    subprocess.run(collect_cmd, cwd=str(REPO), capture_output=True, text=True)
    bench_p = run_dir / "outputs/benchmark.json"
    bench = json.load(open(bench_p)) if bench_p.exists() else {}
    qp = run_dir / "outputs/quality.json"
    quality = json.load(open(qp)) if qp.exists() else {}
    collector_quality_status = quality.get("status")
    collector_quality_blockers = list(quality.get("promotion_blockers") or [])
    base = profile.get("baseline", {})
    cand_total = bench.get("total_s") or bench.get("denoise_s")
    base_total = base.get("total_s")
    speedup = (base_total / cand_total) if (cand_total and base_total) else None

    # Gemini verdict: prefer a rigorous pairwise judge (candidate frames vs the real
    # baseline frames); fall back to the collector's quality.json verdict.
    gem = None
    cand_frames = sorted((run_dir / "outputs/frames").glob("*.png"))
    pj = run_dir / "outputs/quality_pairwise.json"
    if gemini:
        gem = _load_json_if_present(pj)
    if gemini and gem is None and base_fr and cand_frames:
        total = min(len(base_fr), len(cand_frames))
        n = min(total, 32)
        cmd = [sys.executable, str(REPO / "tools/vision/nvidia_gemini_judge.py"),
               "--out", str(pj), "--max-tokens", "4096",
               "--context",
               "Pairwise tier check with stratified chronological frames. "
               "Prioritize temporal flicker, patch-boundary discontinuity, "
               "broken motion coherence, blur/detail loss, ghosting, snow/static, "
               "and severe degradation."]
        indices = [0] if n == 1 else [round(i * (total - 1) / (n - 1)) for i in range(n)]
        for i in indices:
            cmd += ["--baseline-frame", str(base_fr[i]), "--candidate-frame", str(cand_frames[i])]
        subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        gem = _load_json_if_present(pj)
    if gem is None and qp.exists():  # fall back to the collector's gemini verdict (nested)
        gem = ((quality.get("judges") or {}).get("nvidia_gemini") or {}).get("result")
    lpips_result = ((quality.get("judges") or {}).get("lpips") or {}).get("result") or {}
    lpips_delta = lpips_result.get("max") if lpips_result.get("status") == "ok" else None
    quality_blockers = reconcile_quality_blockers(collector_quality_blockers, gem)
    quality_status = collector_quality_status
    if collector_quality_blockers and not quality_blockers and usable_gemini_verdict(gem):
        quality_status = "available"
    tier = None if quality_blockers else tier_of(speedup, None, gem, tiers, lpips_delta=lpips_delta)
    note = promotion_note(tier, quality_blockers, speedup, None, gem, tiers, lpips_delta=lpips_delta)
    return {
        "run_dir": str(run_dir),
        "baseline_total_s": base_total,
        "candidate_total_s": cand_total,
        "speedup": round(speedup, 4) if speedup else None,
        "gemini_overall": (gem or {}).get("overall"),
        "max_artifact_severity": max_gemini_severity(gem) if gem else None,
        "quality_status": quality_status,
        "quality_blockers": quality_blockers,
        "collector_quality_status": collector_quality_status,
        "collector_quality_blockers": collector_quality_blockers,
        "lpips_max": lpips_delta,
        "tier": tier,
        "note": note,
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
    ap.add_argument("--out", default=None, help="write the assess verdict JSON to this path")
    args = ap.parse_args()
    prof, tiers = load_profile(args.model), load_tiers()
    if args.assess:
        verdict = assess(args.assess, prof, tiers, baseline_frames=args.baseline_frames,
                         gemini=not args.no_gemini)
        text = json.dumps(verdict, indent=2)
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n")
        print(text)
    else:
        print("provide --assess RUN_DIR (GPU search_loop orchestration: see docstring)")
