#!/usr/bin/env python3
"""Resume one executor sub-agent with an orchestrator correction.

Thin primitive for the master. It appends the correction to the executor's
goal.md (so a re-woken session re-reads its full prompt PLUS the correction),
also drops a STOP_HOOK_RESUME.md marker, and restarts the detached session
(codex_goal_session start --force).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--goal-dir", required=True)
    ap.add_argument("--feedback", required=True, help="specific problems the sub-agent must fix")
    ap.add_argument("--mode", choices=["reject", "continue"], default="reject",
                    help="reject: prior delivery was bad, fix + re-deliver (default). "
                         "continue: prior delivery is not final, keep optimizing, do NOT re-deliver yet.")
    args = ap.parse_args()

    goal_dir = Path(args.goal_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.mode == "continue":
        block = (
            f"\n\n## Orchestrator correction ({stamp})\n\n"
            "Do NOT treat your last delivery as final, and do NOT re-deliver yet — the "
            "orchestrator is re-opening this workflow to push further. Continue "
            "optimizing per the direction below, using your remaining round budget; "
            "write a new DELIVERY.json ONLY after you have genuinely acted on it:\n\n"
            + args.feedback.strip() + "\n"
        )
    else:
        block = (
            f"\n\n## Orchestrator correction ({stamp})\n\n"
            "Your previous delivery was independently re-verified and REJECTED. Fix "
            "exactly these problems, re-run the affected transfeat(s) end-to-end "
            "(launch + collect + plan_eval), and rewrite DELIVERY.json with honest, "
            "re-measured numbers:\n\n" + args.feedback.strip() + "\n"
        )
    goal_md = goal_dir / "goal.md"
    if goal_md.exists():
        goal_md.write_text(goal_md.read_text() + block)
    else:
        goal_md.write_text(block)
    (goal_dir / "STOP_HOOK_RESUME.md").write_text(block)

    launch = subprocess.run(
        [sys.executable, "tools/symposium/codex_goal_session.py", "start",
         str(goal_dir), "--name", args.name, "--worktree", args.worktree, "--force"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    sys.stderr.write(launch.stdout or "")
    if launch.returncode != 0:
        raise SystemExit(f"[resume_executor] restart failed (rc={launch.returncode})")
    print(f"[resume_executor] resumed {args.name} with correction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
