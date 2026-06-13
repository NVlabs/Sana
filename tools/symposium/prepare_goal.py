#!/usr/bin/env python3
"""Create a Symposium/Codex interactive goal bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sanitize(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    out = "".join(ch if ch in allowed else "-" for ch in value.strip())
    return out.strip("-") or "goal"


def render_goal_md(args: argparse.Namespace, candidate_rel: str) -> str:
    return f"""# Goal: {args.goal_id}

You are working in an isolated autovideo goal context.

## Objective

{args.objective}

## Symposium Step

Use Symposium `interview-harness` first if the objective is still ambiguous.
Produce a final Seed with:

- goal
- constraints
- acceptance criteria
- ontology boundary

## Candidate Contract

- Candidate manifest: `{candidate_rel}`
- Launch dry-run: `python3 scripts/launch_candidate.py {candidate_rel} --mode dry-run`
- Collect: `python3 scripts/collect_run.py runs/<run-id>`

## Branching Contract

- Root branch: `{args.root_branch}`
- Submodule branch: `{args.submodule_branch}`

## Done When

- the goal has a clear Seed or implementation plan
- candidate launch/collection commands are identified
- blockers are explicit
- no unrelated files are changed
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--root-branch", default="")
    parser.add_argument("--submodule-branch", default="")
    parser.add_argument("--goals-root", default="goals")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = project_root()
    goal_id = sanitize(args.goal_id)
    args.goal_id = goal_id
    if not args.root_branch:
        args.root_branch = f"codex/{goal_id}"
    if not args.submodule_branch:
        args.submodule_branch = f"codex/{goal_id}-sol"

    candidate = (root / args.candidate).resolve()
    if not candidate.exists():
        raise SystemExit(f"Candidate manifest does not exist: {candidate}")
    candidate_rel = str(candidate.relative_to(root))

    goal_dir = (root / args.goals_root / goal_id).resolve()
    if goal_dir.exists() and not args.overwrite:
        raise SystemExit(f"Goal already exists: {goal_dir} (use --overwrite)")
    goal_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(candidate, goal_dir / "candidate.toml")
    (goal_dir / "goal.md").write_text(render_goal_md(args, candidate_rel))
    context = {
        "goal_id": goal_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "tools/symposium/prepare_goal.py",
        "target_agent": "codex",
        "mode": "interactive-goal",
        "candidate_manifest": candidate_rel,
        "root_branch": args.root_branch,
        "submodule_branch": args.submodule_branch,
        "objective": args.objective,
        "symposium": {
            "skill": "interview-harness",
            "vendor": "tools/symposium/vendor/Symposium",
        },
    }
    (goal_dir / "context.json").write_text(json.dumps(context, indent=2, sort_keys=True) + "\n")
    print(goal_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
