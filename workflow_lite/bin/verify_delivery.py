#!/usr/bin/env python3
"""Independently verify an executor's DELIVERY.json (anti-fabrication).

This is the trusted check the master runs — it does NOT trust the executor's
numbers. For each frontier point it: (1) confirms the run_dir + out.mp4 +
benchmark.json exist and were produced by a real run (provenance), (2) RE-RUNS
`plan_eval --assess` on that run against the FROZEN baseline frames, and (3)
compares the independently-recomputed speedup / LPIPS to what the delivery
claimed. Mismatches, missing artifacts, or fabricated runs are reported.

For a LOSSLESS technique (e.g. kernel), correctness is MATHEMATICAL / ALGORITHMIC
— a property of the METHOD, judged by reasoning, NOT by comparing outputs. So for
those techniques this deterministic check does NOT compare outputs at all (no
bit-identity, no latent/tensor diff, no floating-point tolerance, no LPIPS): two
correct implementations of the same algorithm may diverge numerically and both
are equally correct. It only confirms the STRUCTURAL invariants any
semantics-preserving implementation must keep (denoising-step count and
DiT/model-call count unchanged) and surfaces the recorded method/semantics
argument. The MASTER then independently REASONS about that argument + the actual
code changes to accept algorithmic-semantic correctness (see master.md); it must
never reject a lossless candidate merely because its output moved.

plan_eval is invoked with $PLAN_EVAL_PYTHON if set (the eval env python), else
this interpreter. Prints JSON: {objective_ok, issues, points}.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEEDUP_TOL = 0.05   # 5% relative tolerance on the reported speedup

# Lossless techniques are gated on MATHEMATICAL / ALGORITHMIC correctness (a
# method property), never on any output metric. For these the executor records a
# method/semantics argument + structural counts, re-checked here (structure only)
# and independently REASONED about by the master (see master.md).
LOSSLESS_TECHS = {"kernel", "kernel_aw"}


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _num(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
    return None


def find_equivalence(fp: dict, run_dir: Path) -> dict:
    """Locate + MERGE the executor's recorded correctness evidence.

    A delivery may carry an inline summary (status + an `artifact` pointer) while
    the structural counts + method argument live in a JSON file. Merge both — the
    artifact file's fields fill in, the inline dict overlays — so the gate sees the
    actual evidence, not just a pointer.
    """
    inline: dict = {}
    for key in ("equivalence", "correctness", "lossless_evidence"):
        v = fp.get(key)
        if isinstance(v, dict) and v:
            inline = v
            break
    cands = [run_dir / "equivalence.json", run_dir / "outputs" / "equivalence.json",
             run_dir / "correctness.json", run_dir / "outputs" / "correctness.json",
             run_dir / "equivalence_report.json", run_dir / "outputs" / "equivalence_report.json"]
    ptr = inline.get("artifact") if isinstance(inline, dict) else None
    if ptr:
        p = Path(str(ptr))
        if p.is_absolute():
            cands.append(p)
        else:  # pointer may be worktree-relative, not run_dir-relative
            for base in (run_dir, run_dir.parent, run_dir.parent.parent, run_dir.parent.parent.parent):
                cands.append(base / p)
    for art in (fp.get("artifacts") or []):
        ap = Path(str(art))
        if "equival" in ap.name.lower() or "correct" in ap.name.lower() or "lossless" in ap.name.lower():
            cands.append(ap if ap.is_absolute() else run_dir / ap)
    artifact: dict = {}
    for c in cands:
        d = load(c)
        if d:
            artifact = d
            break
    if not artifact and not inline:
        return {}
    merged = dict(artifact)
    for k, v in inline.items():
        merged.setdefault(k, v)
    return merged


def check_correctness(fp: dict, run_dir: Path) -> tuple[list[str], dict]:
    """STRUCTURAL correctness gate for a LOSSLESS point — NO output comparison.

    Correctness is mathematical / algorithmic-semantic (a method property) and is
    REASONED about by the master. This deterministic pass does NOT look at any
    output artifact (no bit / latent / fp-tolerance / LPIPS). It only:
      - confirms the structural invariants a semantics-preserving implementation
        must keep: denoising-step count and DiT/model-call count unchanged (a
        change here means the *work* changed → an algorithmic change, not just a
        different implementation);
      - surfaces whether a method/semantics argument was recorded for the master.
    It NEVER flags a candidate for numeric output divergence.
    """
    ev = find_equivalence(fp, run_dir)
    if not ev:
        return ["correctness_evidence_missing"], {}
    issues: list[str] = []
    cs = _num(ev, "candidate_steps", "on_denoising_steps", "steps")
    bs = _num(ev, "baseline_steps", "off_denoising_steps", "expected_denoising_steps")
    cc = _num(ev, "candidate_dit_calls", "on_dit_calls", "dit_calls")
    bc = _num(ev, "baseline_dit_calls", "off_dit_calls", "expected_dit_calls")
    if ev.get("steps_match") is False or (cs is not None and bs is not None and cs != bs):
        issues.append("step_count_changed")          # work changed -> algorithmic change
    if ev.get("dit_calls_match", ev.get("calls_match")) is False or (cc is not None and bc is not None and cc != bc):
        issues.append("dit_call_count_changed")
    argument = next((ev.get(k) for k in ("method_argument", "semantics_argument", "justification",
                                          "rationale", "reference_path", "candidate_path")
                     if isinstance(ev.get(k), str) and ev.get(k).strip()), None)
    return issues, {"steps": cs, "dit_calls": cc, "method_argument_present": bool(argument),
                    "note": "correctness = algorithmic semantics (master reasons); output NOT compared"}


def reassess(run_dir: Path, model_id: str, baseline_frames: str) -> dict:
    py = os.environ.get("PLAN_EVAL_PYTHON", sys.executable)
    out = run_dir / "reverify_verdict.json"
    # objective only: LPIPS + speedup. Visual is judged by the master's own
    # multimodal vision (no external vision API), so skip Gemini here.
    cmd = [py, "search/plan_eval.py", "--model", model_id, "--no-gemini",
           "--assess", str(run_dir), "--out", str(out)]
    if baseline_frames:
        cmd += ["--baseline-frames", baseline_frames]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=3600)
    verdict = load(out)
    verdict["_plan_eval_rc"] = proc.returncode
    if not verdict:
        verdict["_plan_eval_tail"] = (proc.stdout or "")[-500:]
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tech", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--lossless", action="store_true",
                    help="Gate on mathematical/algorithmic correctness (structural + method "
                         "argument), NOT any output metric. Auto-enabled for lossless techniques.")
    args = ap.parse_args()

    require_correctness = args.lossless or (args.tech in LOSSLESS_TECHS)

    wt = Path(args.worktree)
    baseline = load(Path(args.baseline))
    base_total = baseline.get("total_s")
    base_frames = str(baseline.get("baseline_frames") or "")

    delivery = load(wt / "DELIVERY.json")
    issues: list[str] = []
    points_out: list[dict] = []

    if not delivery:
        issues.append("delivery_missing_or_unparseable")
    elif delivery.get("schema_version") != 2 or delivery.get("status") != "complete":
        issues.append("delivery_schema_or_status_invalid")
    if delivery.get("component") not in {None, args.tech}:
        issues.append(f"delivery_component_mismatch:{delivery.get('component')}")
    if delivery.get("model_id") not in {None, args.model}:
        issues.append(f"delivery_model_id_mismatch:{delivery.get('model_id')}")

    fps = delivery.get("frontier_points") or []
    if not fps:
        issues.append("no_frontier_points")

    for i, fp in enumerate(fps):
        pid = fp.get("candidate_id", f"point_{i}")
        run_dir = wt / str(fp.get("run_dir", ""))
        p_issues: list[str] = []
        if not run_dir.is_dir():
            p_issues.append("run_dir_missing")
        else:
            if not (run_dir / "outputs" / "out.mp4").exists():
                p_issues.append("out_mp4_missing")
            if not (run_dir / "outputs" / "benchmark.json").exists():
                p_issues.append("benchmark_missing")
            # provenance: a real launched run leaves metadata + a start sentinel
            meta = load(run_dir / "metadata.json")
            if not (meta.get("slurm_job_id") or (run_dir / "job-started.json").exists()):
                p_issues.append("no_run_provenance")
        reverify = {}
        if not p_issues:
            reverify = reassess(run_dir, args.model, base_frames)
            # plan_eval may exit non-zero because quality is "blocked" without
            # Gemini -- EXPECTED here (visual is the master's multimodal job).
            # Treat the re-eval as successful iff it produced a numeric speedup.
            if not isinstance(reverify.get("speedup"), (int, float)):
                p_issues.append("plan_eval_reverify_no_speedup")
            # speedup claimed vs independently recomputed
            claimed = ((fp.get("performance") or {}).get("speedup"))
            recomputed = reverify.get("speedup")
            if isinstance(claimed, (int, float)) and isinstance(recomputed, (int, float)) and recomputed:
                if abs(claimed - recomputed) / recomputed > SPEEDUP_TOL:
                    p_issues.append(f"speedup_misreport claimed={claimed} recomputed={recomputed:.4f}")
            # baseline reference must be the frozen one
            claimed_base = ((fp.get("performance") or {}).get("baseline_total_s"))
            if isinstance(base_total, (int, float)) and isinstance(claimed_base, (int, float)):
                if abs(claimed_base - base_total) / base_total > 0.01:
                    p_issues.append(f"wrong_baseline claimed={claimed_base} frozen={base_total}")

        # LOSSLESS correctness gate: STRUCTURAL + method argument only, NO output compare.
        correctness: dict = {}
        if require_correctness and run_dir.is_dir():
            c_issues, correctness = check_correctness(fp, run_dir)
            p_issues.extend(c_issues)

        points_out.append({
            "candidate_id": pid, "run_dir": str(run_dir), "objective_ok": not p_issues,
            "issues": p_issues,
            "reverify": {k: reverify.get(k) for k in ("speedup", "lpips_max", "tier")},
            "lossless_required": require_correctness,
            "correctness": correctness,
            "candidate_frames": str(run_dir / "outputs" / "frames"),
            "baseline_frames": base_frames,
            "visual_check": "pending_master_multimodal_review",
        })
        issues.extend(f"{pid}:{x}" for x in p_issues)

    ok = not issues and any(p["objective_ok"] for p in points_out)
    correctness_note = (
        " This is a LOSSLESS technique: correctness is MATHEMATICAL / ALGORITHMIC, "
        "judged by REASONING about the method — NOT by any output comparison. This gate "
        "only checks structure (denoising-step + DiT-call counts unchanged) and that a "
        "method/semantics argument was recorded. The master MUST independently REASON about "
        "that argument + the actual code changes (same algorithm? no approximation, sparsity, "
        "step-skip, sub-16-bit quant, or reduced work?) and MUST NOT reject a candidate merely "
        "because its numeric output moved." if require_correctness else "")
    print(json.dumps({
        "objective_ok": ok, "issues": issues, "points": points_out,
        "lossless_required": require_correctness,
        "note": ("Objective checks only (speedup + provenance"
                 + (" + STRUCTURAL correctness" if require_correctness else " + LPIPS") + "). The master MUST "
                 "independently VIEW each point's candidate_frames vs baseline_frames with its "
                 "own multimodal vision, per evals/rubrics/gemini_visual_artifact_gate.md, "
                 "before accepting." + correctness_note),
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
