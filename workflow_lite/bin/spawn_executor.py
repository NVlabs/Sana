#!/usr/bin/env python3
"""Spawn ONE executor sub-agent for a single technique (kernel|cache|pisa).

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
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LITE = ROOT / "workflow_lite"

TECH = {
    "kernel": ("kernel_aw", "workflow/kernel_aw/nodes/codex_executor/kernel_scope.md"),
    "cache": ("cache_ca", "workflow/cache_ca/nodes/codex_executor/cache_scope.md"),
    "pisa": ("attention_pa", "workflow/attention_pa/nodes/codex_executor/attention_scope.md"),
}


def read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


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

    workflow_uid, scope_rel = TECH[args.tech]
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

    worktree = Path(meta["worktree"])
    goal_dir = Path(meta["goal_dir"])
    model_id = str(meta.get("model_id") or args.model)

    # 2) assemble the executor prompt
    frozen_frames = str(baseline.get("baseline_frames") or "")
    seed_goal = read(goal_dir / "goal.md")
    scope = read(ROOT / scope_rel)
    contract = read(LITE / "prompts" / "loop_and_gate_contract.md")
    contract = contract.replace("<model_id>", model_id).replace("<baseline_frames>", frozen_frames)
    frozen_block = (
        "## Frozen baseline (do not re-run)\n\n```json\n"
        + json.dumps(baseline, indent=2) + "\n```\n"
    )
    prompt = "\n\n".join(p for p in (seed_goal, scope, contract, frozen_block) if p.strip())
    goal_dir.mkdir(parents=True, exist_ok=True)
    (goal_dir / "goal.md").write_text(prompt)

    # 3) launch one detached executor session
    if args.no_launch:
        print(json.dumps({
            "tech": args.tech, "workflow_uid": workflow_uid, "name": args.experiment_uid,
            "worktree": str(worktree), "goal_dir": str(goal_dir),
            "delivery_path": str(worktree / "DELIVERY.json"), "launched": False,
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
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
