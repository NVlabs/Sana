#!/usr/bin/env python3
"""Create and manage isolated indexed Codex experiments.

Each experiment id owns a separate git worktree, goal directory, state files,
run artifacts, and compile/cache directories. This is the hard isolation layer
above prompts: a fresh experiment is created from a shared baseline commit
instead of reusing a dirty fanout worktree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_ID = re.compile(r"[^A-Za-z0-9_.-]+")
DEFAULT_EXPERIMENTS_ROOT = "output/experiments"
DEFAULT_DIMENSION = "kwl_fusion"
DEFAULT_GOAL_ID = "kwl-fusion"
DEFAULT_MODEL_ID = "hunyuan_diffusers"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize(value: str) -> str:
    cleaned = VALID_ID.sub("-", value.strip())
    return cleaned.strip("-") or "experiment"


def run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "command failed"
        raise SystemExit(f"{' '.join(args)}\n{detail}")
    return proc


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["git", *args], cwd=root, check=check)


def resolve_commit(root: Path, ref: str) -> str:
    proc = git(root, "rev-parse", "--verify", f"{ref}^{{commit}}")
    return proc.stdout.strip()


def branch_exists(root: Path, branch: str) -> bool:
    return (
        git(root, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}", check=False)
        .returncode
        == 0
    )


def rel_to(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def experiments_root(root: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def experiment_dir(root: Path, raw_root: str, experiment_id: str) -> Path:
    return experiments_root(root, raw_root) / sanitize(experiment_id)


def metadata_path(exp_dir: Path) -> Path:
    return exp_dir / "experiment.json"


def load_metadata(path_or_dir: Path) -> dict[str, Any]:
    path = path_or_dir if path_or_dir.name == "experiment.json" else metadata_path(path_or_dir)
    if not path.exists():
        raise SystemExit(f"Experiment metadata does not exist: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid experiment metadata: {path}") from exc


def write_metadata(exp_dir: Path, data: dict[str, Any]) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    metadata_path(exp_dir).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def default_transfeat(model_id: str) -> str:
    return f"transfeat/{model_id}_baseline.toml"


def default_objective(model_id: str, dimension: str) -> str:
    if model_id == DEFAULT_MODEL_ID and dimension == DEFAULT_DIMENSION:
        return (
            "Implement kernel-wise (KWL) fusion for HunyuanVideo in "
            "runtime/hunyuan_diffusers_baseline from a fresh isolated "
            "experiment worktree; follow search_space/05_kernel_fusion.md; "
            "start with module/DiT microbenchmarks; target a quality-preserving "
            "speedup and prove OFF identity."
        )
    return (
        f"Run an isolated {dimension} optimization experiment for {model_id} "
        "from the shared baseline commit, preserving the model algorithm and "
        "recording durable speed/quality evidence."
    )


def experiment_env(meta: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    caches = meta["caches"]
    exp_id = meta["experiment_id"]
    env.update(
        {
            "SYMPOSIUM_EXPERIMENT_ID": exp_id,
            "SYMPOSIUM_CURRENT_RUN_ID": exp_id,
            "AUTO_VIDEO_RUN_ID": exp_id,
            "RUN_ID": exp_id,
            "AUTO_VIDEO_EXPERIMENT_ROOT": meta["experiment_dir"],
            "AUTO_VIDEO_RUNS_ROOT": meta["runs_dir"],
            "TMPDIR": caches["tmp"],
            "TRITON_CACHE_DIR": caches["triton"],
            "TORCH_EXTENSIONS_DIR": caches["torch_extensions"],
            # The experiment worktree is already isolated. Preserve current
            # experiment state on resume instead of deleting it as "stale".
            "SYMPOSIUM_PRESERVE_HISTORY_RECORDS": "1",
            "SYMPOSIUM_ALLOW_HISTORY_RECORDS": "1",
        }
    )
    return env


def local_state_paths(worktree: Path) -> list[Path]:
    paths: list[Path] = []
    for name in ("AGENT-STATUS.json", "SEARCH_JOURNAL.md", "SUMMARY.md"):
        path = worktree / name
        if path.exists():
            paths.append(path)
    runs = worktree / "runs"
    baseline_run_entries = {"README.md", ".gitkeep", ".gitignore"}
    if runs.exists() and any(path.name not in baseline_run_entries for path in runs.iterdir()):
        paths.append(runs)
    return paths


def create(args: argparse.Namespace, *, root: Path | None = None) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    exp_id = sanitize(args.experiment_id)
    goal_id = sanitize(args.goal_id or DEFAULT_GOAL_ID)
    dimension = args.dimension or DEFAULT_DIMENSION
    model_id = args.model_id or DEFAULT_MODEL_ID
    transfeat = args.transfeat or default_transfeat(model_id)
    objective = args.objective or default_objective(model_id, dimension)
    exp_dir = experiment_dir(root, args.experiments_root, exp_id)
    worktree = exp_dir / "worktree"

    if exp_dir.exists():
        raise SystemExit(f"Experiment already exists: {exp_dir}")

    base_ref = args.base_ref or "HEAD"
    base_sha = resolve_commit(root, base_ref)
    branch = args.branch or f"exp/{exp_id}/{goal_id}"
    if not args.detached and branch_exists(root, branch):
        raise SystemExit(f"Branch already exists: {branch}")

    exp_dir.mkdir(parents=True)
    worktree_args = ["worktree", "add"]
    if args.detached:
        worktree_args.append("--detach")
    else:
        worktree_args.extend(["-b", branch])
    worktree_args.extend([str(worktree), base_sha])
    git(root, *worktree_args)

    for rel in (
        "goals",
        "runs",
        "state",
        "caches/tmp",
        "caches/triton",
        "caches/torch_extensions",
    ):
        (worktree / rel).mkdir(parents=True, exist_ok=True)

    dirty = git(root, "status", "--short", check=False).stdout.splitlines()
    meta = {
        "schema_version": 1,
        "experiment_id": exp_id,
        "created_at_utc": now_utc(),
        "status": "created",
        "coordinator_root": str(root),
        "experiment_dir": str(exp_dir),
        "worktree": str(worktree),
        "goal_dir": str(worktree / "goals" / goal_id),
        "state_dir": str(worktree / "state"),
        "runs_dir": str(worktree / "runs"),
        "caches": {
            "tmp": str(worktree / "caches" / "tmp"),
            "triton": str(worktree / "caches" / "triton"),
            "torch_extensions": str(worktree / "caches" / "torch_extensions"),
        },
        "base_ref": base_ref,
        "base_sha": base_sha,
        "branch": "" if args.detached else branch,
        "detached": bool(args.detached),
        "goal_id": goal_id,
        "dimension": dimension,
        "role": args.role,
        "model_id": model_id,
        "transfeat": transfeat,
        "objective": objective,
        "session_name": args.session_name or f"exp-{exp_id}-{goal_id}",
        "isolation": {
            "fresh_worktree_from_base_sha": True,
            "refuse_existing_experiment_id": True,
            "start_requires_resume_for_prior_local_state": True,
            "per_experiment_runs_dir": "runs",
            "per_experiment_caches": ["TMPDIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"],
        },
        "coordinator_dirty_status_at_create": dirty,
    }
    write_metadata(exp_dir, meta)

    if not args.skip_goal:
        prepare_goal(meta, overwrite=args.overwrite_goal)
        meta["status"] = "goal_prepared"
        meta["goal_prepared_at_utc"] = now_utc()
        write_metadata(exp_dir, meta)

    return meta


def prepare_goal(meta: dict[str, Any], *, overwrite: bool) -> None:
    worktree = Path(meta["worktree"])
    cmd = [
        sys.executable,
        "tools/symposium/prepare_goal.py",
        "--goal-id",
        meta["goal_id"],
        "--transfeat",
        meta["transfeat"],
        "--objective",
        meta["objective"],
        "--dimension",
        meta["dimension"],
        "--role",
        meta["role"],
        "--model-id",
        meta["model_id"],
        "--run-id",
        meta["experiment_id"],
        "--root-branch",
        meta["branch"] or f"detached/{meta['experiment_id']}",
        "--submodule-branch",
        f"{meta['branch'] or 'detached/' + meta['experiment_id']}-sol",
        "--goals-root",
        "goals",
    ]
    if overwrite:
        cmd.append("--overwrite")
    run(cmd, cwd=worktree, env=experiment_env(meta))


def resolve_experiment(args: argparse.Namespace, *, root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    root = (root or project_root()).resolve()
    exp_id = sanitize(args.experiment_id)
    exp_dir = experiment_dir(root, args.experiments_root, exp_id)
    return exp_dir, load_metadata(exp_dir)


def start(args: argparse.Namespace, *, root: Path | None = None) -> dict[str, Any]:
    exp_dir, meta = resolve_experiment(args, root=root)
    worktree = Path(meta["worktree"])
    goal_dir = Path(meta["goal_dir"])
    if not goal_dir.exists():
        raise SystemExit(f"Goal has not been prepared: {goal_dir}")
    prior_state = local_state_paths(worktree)
    if prior_state and not args.resume:
        rels = ", ".join(rel_to(worktree, path) for path in prior_state)
        raise SystemExit(
            "Experiment contains prior local state. Use --resume to continue "
            f"this experiment, or create a new id. State: {rels}"
        )

    cmd = [
        sys.executable,
        "tools/symposium/codex_goal_session.py",
        "start",
        "--worktree",
        str(worktree),
        "--name",
        meta["session_name"],
        rel_to(worktree, goal_dir),
    ]
    if args.force:
        cmd.append("--force")
    proc = run(cmd, cwd=Path(meta["coordinator_root"]), env=experiment_env(meta))
    meta["status"] = "started"
    meta["started_at_utc"] = now_utc()
    write_metadata(exp_dir, meta)
    return {"experiment": meta, "session_start": json.loads(proc.stdout)}


def status(args: argparse.Namespace, *, root: Path | None = None) -> dict[str, Any]:
    _, meta = resolve_experiment(args, root=root)
    worktree = Path(meta["worktree"])
    goal_dir = Path(meta["goal_dir"])
    cmd = [
        sys.executable,
        "tools/symposium/codex_goal_session.py",
        "status",
        "--worktree",
        str(worktree),
        "--name",
        meta["session_name"],
        rel_to(worktree, goal_dir),
    ]
    proc = run(cmd, cwd=Path(meta["coordinator_root"]), env=experiment_env(meta), check=False)
    session_status: Any
    if proc.returncode == 0 and proc.stdout.strip():
        session_status = json.loads(proc.stdout)
    else:
        session_status = {"error": proc.stderr.strip() or proc.stdout.strip()}
    git_status = git(worktree, "status", "--short", check=False).stdout.splitlines()
    return {
        "experiment": meta,
        "session": session_status,
        "dirty_files": git_status,
        "local_state": [rel_to(worktree, path) for path in local_state_paths(worktree)],
    }


def stop(args: argparse.Namespace, *, root: Path | None = None) -> dict[str, Any]:
    exp_dir, meta = resolve_experiment(args, root=root)
    worktree = Path(meta["worktree"])
    goal_dir = Path(meta["goal_dir"])
    cmd = [
        sys.executable,
        "tools/symposium/codex_goal_session.py",
        "stop",
        "--worktree",
        str(worktree),
        "--name",
        meta["session_name"],
        rel_to(worktree, goal_dir),
    ]
    proc = run(cmd, cwd=Path(meta["coordinator_root"]), env=experiment_env(meta), check=False)
    meta["status"] = "stopped"
    meta["stopped_at_utc"] = now_utc()
    write_metadata(exp_dir, meta)
    result: Any
    if proc.returncode == 0 and proc.stdout.strip():
        result = json.loads(proc.stdout)
    else:
        result = {"error": proc.stderr.strip() or proc.stdout.strip(), "returncode": proc.returncode}
    return {"experiment": meta, "session_stop": result}


def list_experiments(args: argparse.Namespace, *, root: Path | None = None) -> list[dict[str, Any]]:
    root = (root or project_root()).resolve()
    base = experiments_root(root, args.experiments_root)
    if not base.exists():
        return []
    items = []
    for path in sorted(base.glob("*/experiment.json")):
        try:
            items.append(load_metadata(path))
        except SystemExit:
            continue
    return items


def print_result(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiments-root", default=DEFAULT_EXPERIMENTS_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    create_p = sub.add_parser("create", help="Create a fresh indexed experiment worktree")
    add_common(create_p)
    create_p.add_argument("--experiment-id", required=True)
    create_p.add_argument("--base-ref", default="HEAD")
    create_p.add_argument("--branch")
    create_p.add_argument("--detached", action="store_true")
    create_p.add_argument("--goal-id", default=DEFAULT_GOAL_ID)
    create_p.add_argument("--dimension", default=DEFAULT_DIMENSION)
    create_p.add_argument("--role", choices=("implementation", "gate", "integration"), default="implementation")
    create_p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    create_p.add_argument("--transfeat")
    create_p.add_argument("--objective")
    create_p.add_argument("--session-name")
    create_p.add_argument("--skip-goal", action="store_true")
    create_p.add_argument("--overwrite-goal", action="store_true")
    create_p.set_defaults(func=create)

    for name, help_text, func in (
        ("start", "Start the experiment's Codex session", start),
        ("status", "Show experiment and session status", status),
        ("stop", "Stop the experiment's Codex session", stop),
    ):
        p = sub.add_parser(name, help=help_text)
        add_common(p)
        p.add_argument("--experiment-id", required=True)
        if name == "start":
            p.add_argument("--resume", action="store_true")
            p.add_argument("--force", action="store_true")
        p.set_defaults(func=func)

    list_p = sub.add_parser("list", help="List indexed experiments")
    add_common(list_p)
    list_p.set_defaults(func=list_experiments)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print_result(args.func(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
