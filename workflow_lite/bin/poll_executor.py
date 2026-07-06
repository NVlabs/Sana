#!/usr/bin/env python3
"""Poll one executor sub-agent. Thin primitive for the master orchestrator.

Prints JSON: {delivered, alive, delivery_path, delivery_status}. `delivered` is
the authoritative signal (a valid DELIVERY.json exists). `alive` is best-effort
liveness from codex_goal_session status.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--goal-dir", default="")
    args = ap.parse_args()

    delivery_path = Path(args.worktree) / "DELIVERY.json"
    delivered = False
    delivery_status = None
    if delivery_path.exists():
        try:
            d = json.loads(delivery_path.read_text())
            delivery_status = d.get("status")
            delivered = d.get("schema_version") == 2 and d.get("status") == "complete"
        except (OSError, json.JSONDecodeError):
            delivered = False

    alive = None
    goal_dir = args.goal_dir or str(Path(args.worktree) / "goals")
    try:
        st = subprocess.run(
            [sys.executable, "tools/symposium/codex_goal_session.py", "status",
             goal_dir, "--name", args.name, "--worktree", args.worktree],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
        )
        out = (st.stdout or "").lower()
        alive = st.returncode == 0 and not any(k in out for k in ("not found", "dead", "no session", "inactive"))
    except Exception:
        alive = None

    print(json.dumps({
        "name": args.name,
        "delivered": delivered,
        "delivery_status": delivery_status,
        "alive": alive,
        "delivery_path": str(delivery_path),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
