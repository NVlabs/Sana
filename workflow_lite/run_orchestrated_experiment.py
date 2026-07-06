#!/usr/bin/env python3
"""Bootstrap + heartbeat watchdog for the master-agent-orchestrated experiment.

The ONLY deterministic Python scheduling that remains. It:
  1. freezes the baseline ONCE (reuse the model profile's recorded [baseline]
     run, or launch it once) into a read-only BASELINE.json for the whole run;
  2. assembles the master orchestrator prompt and launches ONE master agent;
  3. runs a thin heartbeat watchdog that restarts the master if it dies, until
     the master writes the integrated delivery.

The master agent does everything else: spawn 3 executor sub-agents, poll,
independently verify (anti-fabrication), resume on bad delivery, and integrate.
Heavy workflow/ is untouched.

    python workflow_lite/run_orchestrated_experiment.py --model bernini
    python workflow_lite/run_orchestrated_experiment.py --model bernini --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LITE = ROOT / "workflow_lite"
MODEL_ALIASES = {"sana": "sana_video", "sana_video": "sana_video", "bernini": "bernini",
                 "hunyuan": "hunyuan_diffusers", "hunyuan_diffusers": "hunyuan_diffusers"}
MODEL_PREFIX = {"sana_video": "sana", "bernini": "bernini", "hunyuan_diffusers": "hunyuan"}


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def freeze_baseline(model_id: str, out_path: Path, override_run_dir: str | None) -> dict:
    """Reuse the recorded canonical baseline (or an override); freeze to a file."""
    profile = tomllib.loads((ROOT / "models" / f"{model_id}.toml").read_text())
    b = profile.get("baseline", {})
    if override_run_dir:
        run_dir = Path(override_run_dir)
    elif b.get("run_id"):
        run_dir = ROOT / "runs" / str(b["run_id"])
    else:
        raise SystemExit(
            f"no recorded [baseline] for {model_id}. Run it once first:\n"
            f"  python scripts/launch_candidate.py candidates/{model_id}_baseline.toml --mode sbatch --confirm-submit\n"
            f"then re-run with --baseline-run-dir runs/<id>, or record it in models/{model_id}.toml [baseline]."
        )
    run_dir = run_dir.resolve()
    frames = run_dir / "outputs" / "frames"
    if not run_dir.is_dir():
        raise SystemExit(f"baseline run dir not found: {run_dir}")
    baseline = {
        "model_id": model_id,
        "total_s": b.get("total_s"),
        "denoise_s": b.get("denoise_s"),
        "timing_scope": b.get("timing_scope"),
        "run_dir": str(run_dir),
        "baseline_frames": str(frames),
        "baseline_video": str(run_dir / "outputs" / "out.mp4"),
        "frozen_at": utc(),
        "source": "override_run_dir" if override_run_dir else "recorded_profile_baseline",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(baseline, indent=2) + "\n")
    return baseline


def master_alive(goal_dir: Path, name: str) -> bool:
    """True only if codex_goal_session status reports the session alive.

    status prints a JSON object with a top-level "alive" boolean. Parse it —
    do NOT keyword-scan the output: an inactive session prints
    `{"alive": false, ...}`, which contains none of the "dead/not found"
    keywords and would be read as a false positive.
    """
    try:
        st = subprocess.run(
            [sys.executable, "tools/symposium/codex_goal_session.py", "status",
             str(goal_dir), "--name", name, "--worktree", str(ROOT)],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
        )
        out = st.stdout or ""
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            lo, hi = out.find("{"), out.rfind("}")
            data = json.loads(out[lo:hi + 1]) if 0 <= lo < hi else {}
        return bool(data.get("alive"))
    except Exception:
        return False


def start_master(goal_dir: Path, name: str, force: bool = False) -> None:
    cmd = [sys.executable, "tools/symposium/codex_goal_session.py", "start",
           str(goal_dir), "--name", name, "--worktree", str(ROOT)]
    if force:
        cmd.append("--force")
    r = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tag = "OK" if r.returncode == 0 else f"FAILED rc={r.returncode}"
    print(f"[orchestrate] start_master({name}, force={force}) -> {tag}", flush=True)
    if r.stdout:
        print("\n".join(f"[start_master] {ln}" for ln in r.stdout.strip().splitlines()[-8:]), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="bernini")
    ap.add_argument("--seq", default="0001")
    ap.add_argument("--baseline-run-dir", default=None, help="reuse a specific baseline run dir instead of the recorded one")
    ap.add_argument("--poll-sec", type=float, default=120.0)
    ap.add_argument("--max-hours", type=float, default=24.0)
    ap.add_argument("--techs", default="kernel,cache,pisa",
                    help="comma-separated executor techniques to run (e.g. 'kernel,cache' to skip pisa)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # SANDBOX: codex agents run the DEFAULT workspace-write + on-request sandbox
    # with the approver daemon. Do NOT set SYMPOSIUM_AUTORUN_BYPASS — the org
    # policy (/etc/codex/requirements.toml) FORBIDS bypass/danger-full-access and
    # silently degrades it to a locked-down read-only sandbox (no tmux/Slurm/socket
    # /workspace writes). Instead, tmux + Slurm + AF_UNIX access is unblocked by
    #   [sandbox_workspace_write]  network_access = true
    # in ~/.codex/config.toml (user-settable; NOT requirements-enforceable).
    # Verified end-to-end via a capability probe (tmux/sinfo/socket/write all OK).

    # The master runs in the coordinator checkout, where its OWN live executor
    # experiment dirs (output/experiments/<uid>) live. The startup hygiene step
    # globs output/experiments/* as "stale records" and rmtree's them — which on
    # a watchdog RESTART would try to delete the running executors' worktrees
    # (EBUSY on their live .codex). Preserve history + skip the stale-record
    # refusal for the master: the repo is pre-cleaned before launch and each
    # executor worktree is a fresh clean closure, so runtime hygiene is a no-op
    # here anyway — but it must not nuke live sub-agents on restart.
    os.environ["SYMPOSIUM_PRESERVE_HISTORY_RECORDS"] = "1"
    os.environ["SYMPOSIUM_ALLOW_HISTORY_RECORDS"] = "1"

    model_id = MODEL_ALIASES.get(args.model, args.model)
    if not (ROOT / "models" / model_id / "model.toml").exists():
        raise SystemExit(f"unknown model {args.model!r} (known: {', '.join(sorted(set(MODEL_ALIASES)))})")
    prefix = MODEL_PREFIX.get(model_id, model_id.split("_")[0])
    techs = [t.strip() for t in args.techs.split(",") if t.strip()]
    if not techs:
        raise SystemExit("--techs must list at least one technique")
    techs_fmt = ", ".join(f"`{t}`" for t in techs)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    exp_root = ROOT / "output" / "orchestrated" / f"{prefix}-{stamp}"
    baseline_json = exp_root / "BASELINE.json"
    integrated = exp_root / "INTEGRATED-DELIVERY.json"
    master_goal = exp_root / "master"
    master_name = f"{prefix}-master-{args.seq}"
    state_path = exp_root / "MASTER-STATE.json"

    print(f"[orchestrate] model={model_id} prefix={prefix} techs={techs} root={exp_root}", flush=True)
    if args.dry_run:
        print(json.dumps({"model_id": model_id, "prefix": prefix, "exp_root": str(exp_root),
                          "baseline_json": str(baseline_json), "integrated_delivery": str(integrated),
                          "master_name": master_name, "seq": args.seq}, indent=2))
        return 0

    # 1) freeze baseline once
    baseline = freeze_baseline(model_id, baseline_json, args.baseline_run_dir)
    print(f"[orchestrate] frozen baseline: total_s={baseline['total_s']} run_dir={baseline['run_dir']}", flush=True)

    # 2) assemble master prompt + launch master
    master_goal.mkdir(parents=True, exist_ok=True)
    tpl = (LITE / "prompts" / "master.md").read_text()
    for k, v in {"{MODEL_ID}": model_id, "{ROOT}": str(ROOT), "{BASELINE_JSON}": str(baseline_json),
                 "{SEQ}": args.seq, "{PREFIX}": prefix, "{INTEGRATED_DELIVERY}": str(integrated),
                 "{TECHS}": techs_fmt}.items():
        tpl = tpl.replace(k, v)
    (master_goal / "goal.md").write_text(tpl)
    # codex_goal_session requires BOTH goal.md and context.json in the goal dir.
    (master_goal / "context.json").write_text(json.dumps({
        "schema_version": 1, "goal_id": master_name, "experiment_uid": master_name,
        "created_by": "run_orchestrated_experiment", "target_agent": "codex",
        "mode": "master-orchestrator", "model_uid": model_id, "role": "master",
    }, indent=2))
    print(f"[orchestrate] launching master agent {master_name}", flush=True)
    start_master(master_goal, master_name, force=False)

    # 3) heartbeat watchdog: restart master if it dies, until integrated delivery
    deadline = time.time() + args.max_hours * 3600
    restarts = 0
    while time.time() < deadline:
        done = integrated.exists()
        alive = master_alive(master_goal, master_name)
        state_path.write_text(json.dumps({
            "updated_at_utc": utc(), "model_id": model_id, "master_name": master_name,
            "master_alive": alive, "restarts": restarts, "integrated_delivery_present": done,
            "baseline_json": str(baseline_json),
        }, indent=2) + "\n")
        if done:
            print(f"[orchestrate] DONE: integrated delivery at {integrated}", flush=True)
            return 0
        if not alive:
            restarts += 1
            print(f"[orchestrate] master not alive; restart #{restarts}", flush=True)
            start_master(master_goal, master_name, force=True)
        time.sleep(max(args.poll_sec, 30.0))
    print("[orchestrate] deadline reached without integrated delivery", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
