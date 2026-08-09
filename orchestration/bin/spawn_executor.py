#!/usr/bin/env python3
"""Spawn ONE executor sub-agent for one registered optimization technique.

Thin primitive called by the master orchestrator. It (1) materializes the
model's experiment worktree via create_model_experiment, (2) assembles the
executor prompt = seed goal.md + the (de-sana'd) technique scope + the shared
loop_and_gate_contract + the frozen-baseline block, (3) launches one detached
codex executor session via codex_goal_session. It does NOT poll or verify.

Prints a JSON line: {worktree, goal_dir, name, delivery_path}.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LITE = ROOT / "orchestration"


def load_techniques() -> dict[str, dict[str, str]]:
    path = LITE / "techniques.toml"
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    techniques = raw.get("techniques")
    if not isinstance(techniques, dict) or not techniques:
        raise RuntimeError(f"invalid technique registry: {path}")
    normalized: dict[str, dict[str, str]] = {}
    for name, spec in techniques.items():
        if not isinstance(spec, dict):
            raise RuntimeError(f"invalid technique entry {name!r}: {path}")
        required = ("workflow_uid", "scope", "correctness")
        if any(not isinstance(spec.get(key), str) or not spec[key] for key in required):
            raise RuntimeError(f"incomplete technique entry {name!r}: {path}")
        normalized[str(name)] = {key: str(spec[key]) for key in required}
    workflow_uids = [spec["workflow_uid"] for spec in normalized.values()]
    if len(workflow_uids) != len(set(workflow_uids)):
        raise RuntimeError(f"technique workflow_uid values must be unique: {path}")
    for name, spec in normalized.items():
        expected_prefix = f"workflow/{spec['workflow_uid']}/"
        if not spec["scope"].startswith(expected_prefix):
            raise RuntimeError(
                f"technique {name!r} scope must be owned by {expected_prefix}: {path}"
            )
    return normalized


TECH = load_techniques()


def read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def owned_worktree_path(worktree: Path, raw: object, *, label: str) -> Path:
    """Resolve one concrete experiment-local path without absolute/`..` escape."""
    if not isinstance(raw, str) or not raw.strip():
        raise SystemExit(f"[spawn_executor] {label} is missing")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"[spawn_executor] {label} must be worktree-relative: {raw!r}")
    root = worktree.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"[spawn_executor] {label} escapes the worktree: {raw!r}") from exc
    return resolved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tech", required=True, choices=sorted(TECH))
    ap.add_argument("--experiment-uid", required=True)
    ap.add_argument("--baseline", required=True, help="path to the frozen BASELINE.json")
    ap.add_argument("--experiments-root", default="output/experiments")
    ap.add_argument("--no-launch", action="store_true",
                    help="Create the experiment + assemble the prompt but do NOT start the codex session (shakedown).")
    args = ap.parse_args()

    technique = TECH[args.tech]
    workflow_uid = technique["workflow_uid"]
    scope_rel = technique["scope"]
    baseline = json.loads(Path(args.baseline).read_text())

    # 1) materialize the experiment worktree (model-aware goal.md seed included)
    proc = subprocess.run(
        [sys.executable, "scripts/create_model_experiment.py",
         "--model", args.model, "--workflow-uid", workflow_uid,
         "--experiment-uid", args.experiment_uid,
         "--experiments-root", args.experiments_root],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        # experiment may already exist; try to reuse it
        exp_json = (ROOT / args.experiments_root / args.experiment_uid / "experiment.json")
        if not exp_json.exists():
            raise SystemExit(f"[spawn_executor] create_model_experiment failed: {proc.stderr.strip() or proc.stdout.strip()}")
        meta = json.loads(exp_json.read_text())
    else:
        meta = json.loads(proc.stdout)

    expected_meta = {
        "experiment_uid": args.experiment_uid,
        "workflow_uid": workflow_uid,
        "model_id": args.model,
    }
    mismatches = [
        f"{key}={meta.get(key)!r} (expected {value!r})"
        for key, value in expected_meta.items()
        if meta.get(key) != value
    ]
    if mismatches:
        raise SystemExit(
            "[spawn_executor] refusing mismatched experiment metadata: "
            + "; ".join(mismatches)
        )

    experiments_root = Path(args.experiments_root)
    if not experiments_root.is_absolute():
        experiments_root = ROOT / experiments_root
    expected_experiment_dir = (experiments_root / args.experiment_uid).resolve()
    expected_worktree = expected_experiment_dir / "worktree"
    expected_goal_dir = expected_worktree / "goals" / workflow_uid
    expected_paths = {
        "experiment_dir": expected_experiment_dir,
        "worktree": expected_worktree,
        "goal_dir": expected_goal_dir,
    }
    path_mismatches = []
    for key, expected in expected_paths.items():
        raw = meta.get(key)
        if not isinstance(raw, str) or Path(raw).resolve() != expected:
            path_mismatches.append(f"{key}={raw!r} (expected {str(expected)!r})")
    if path_mismatches:
        raise SystemExit(
            "[spawn_executor] refusing non-canonical experiment paths: "
            + "; ".join(path_mismatches)
        )

    worktree = Path(meta["worktree"]).resolve()
    goal_dir = Path(meta["goal_dir"]).resolve()
    model_id = str(meta.get("model_id") or args.model)
    baseline_meta = meta.get("baseline") if isinstance(meta.get("baseline"), dict) else {}
    contract_path = (ROOT / "models" / model_id / "model.toml").resolve()
    if not contract_path.is_file():
        raise SystemExit(f"[spawn_executor] model contract is missing: {contract_path}")
    with contract_path.open("rb") as handle:
        model_contract = tomllib.load(handle)
    declared_baseline = (
        model_contract.get("baseline")
        if isinstance(model_contract.get("baseline"), dict)
        else {}
    )
    expected_contract = contract_path.relative_to(ROOT).as_posix()
    if meta.get("model_contract") != expected_contract:
        raise SystemExit(
            "[spawn_executor] refusing mismatched model contract: "
            f"{meta.get('model_contract')!r} (expected {expected_contract!r})"
        )
    for key in ("manifest", "runtime_root"):
        if baseline_meta.get(key) != declared_baseline.get(key):
            raise SystemExit(
                "[spawn_executor] refusing mismatched baseline metadata: "
                f"{key}={baseline_meta.get(key)!r} "
                f"(expected {declared_baseline.get(key)!r})"
            )

    manifest_path = owned_worktree_path(
        worktree, baseline_meta.get("manifest"), label="baseline manifest"
    )
    runtime_path = owned_worktree_path(
        worktree, baseline_meta.get("runtime_root"), label="baseline runtime_root"
    )
    launcher_path = owned_worktree_path(
        worktree, "scripts/launch_transfeat.py", label="transfeat launcher"
    )
    context_path = owned_worktree_path(
        worktree,
        (goal_dir / "context.json").relative_to(worktree).as_posix(),
        label="executor context",
    )
    missing_closure = []
    if not manifest_path.is_file():
        missing_closure.append(str(manifest_path))
    if not runtime_path.is_dir():
        missing_closure.append(str(runtime_path))
    for required_file in (launcher_path, context_path):
        if not required_file.is_file():
            missing_closure.append(str(required_file))
    if missing_closure:
        raise SystemExit(
            "[spawn_executor] refusing incomplete experiment runnable closure: "
            + ", ".join(missing_closure)
        )

    # 2) assemble the executor prompt
    frozen_frames = str(baseline.get("baseline_frames") or "")
    seed_goal = read(goal_dir / "goal.md")
    scope_path = ROOT / scope_rel
    scope = read(scope_path)
    if not scope.strip():
        raise SystemExit(f"[spawn_executor] technique scope is missing or empty: {scope_path}")
    contract = read(LITE / "prompts" / "loop_and_gate_contract.md")
    contract = contract.replace("<model_id>", model_id).replace("<baseline_frames>", frozen_frames)
    frozen_block = (
        "## Frozen baseline (do not re-run)\n\n```json\n"
        + json.dumps(baseline, indent=2) + "\n```\n"
    )
    prompt = "\n\n".join(p for p in (seed_goal, scope, contract, frozen_block) if p.strip())
    goal_dir.mkdir(parents=True, exist_ok=True)
    (goal_dir / "goal.md").write_text(prompt)
    context_path = goal_dir / "context.json"
    context = json.loads(context_path.read_text()) if context_path.exists() else {}
    context.update(
        {
            "technique": args.tech,
            "technique_scope": scope_rel,
            "correctness_mode": technique["correctness"],
        }
    )
    context_path.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n")

    # 3) launch one detached executor session
    if args.no_launch:
        print(json.dumps({
            "tech": args.tech, "workflow_uid": workflow_uid, "name": args.experiment_uid,
            "worktree": str(worktree), "goal_dir": str(goal_dir),
            "delivery_path": str(worktree / "DELIVERY.json"), "launched": False,
            "correctness": technique["correctness"],
        }))
        return 0
    # Executors run the DEFAULT workspace-write + on-request sandbox (org policy
    # forbids bypass; `[sandbox_workspace_write] network_access = true` in
    # ~/.codex/config.toml unblocks Slurm/sockets). Pass the env through so
    # PLAN_EVAL_PYTHON etc. reach the executor; strip any stray bypass flag.
    launch_env = {**os.environ}
    launch_env.pop("SYMPOSIUM_AUTORUN_BYPASS", None)
    launch = subprocess.run(
        [sys.executable, "tools/symposium/codex_goal_session.py", "start",
         str(goal_dir), "--name", args.experiment_uid, "--worktree", str(worktree)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=launch_env,
    )
    sys.stderr.write(launch.stdout or "")
    if launch.returncode != 0:
        raise SystemExit(f"[spawn_executor] codex_goal_session start failed (rc={launch.returncode})")

    print(json.dumps({
        "tech": args.tech,
        "workflow_uid": workflow_uid,
        "name": args.experiment_uid,
        "worktree": str(worktree),
        "goal_dir": str(goal_dir),
        "delivery_path": str(worktree / "DELIVERY.json"),
        "correctness": technique["correctness"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
