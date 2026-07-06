#!/usr/bin/env python3
"""Run kernel, PISA, and cache in parallel for a chosen model, then launch their integrator.

The model is selected at launch with ``--model`` (``bernini`` | ``sana`` |
``hunyuan``, or a full model id under ``models/<id>/``). No per-model source
edits or prompt-file edits are needed: the model flows into
``create_model_experiment.py`` (which seeds a model-aware ``goal.md``) and into
the derived experiment / chain ids.

    python scripts/run_workflow_chain.py --model bernini
    python scripts/run_workflow_chain.py --model sana --seq 0008
    python scripts/run_workflow_chain.py --model hunyuan --chain-id hunyuan-opt-try1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

# component -> workflow uid (model-agnostic; the model is chosen at launch)
WORKFLOWS = {"kernel": "kernel_aw", "pisa": "attention_pa", "cache": "cache_ca"}
INTEGRATOR_WORKFLOW = "integrator_ia"

# friendly --model name -> model id (must resolve to models/<id>/model.toml)
MODEL_ALIASES = {
    "sana": "sana_video",
    "sana_video": "sana_video",
    "bernini": "bernini",
    "hunyuan": "hunyuan_diffusers",
    "hunyuan_diffusers": "hunyuan_diffusers",
}
# model id -> short prefix used in experiment / chain ids
MODEL_PREFIX = {"sana_video": "sana", "bernini": "bernini", "hunyuan_diffusers": "hunyuan"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def run_checked(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "command failed")
    try:
        value = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"command returned invalid JSON: {' '.join(command)}") from exc
    return value if isinstance(value, dict) else {}


def create_experiment(
    experiments_root: Path, model_id: str, workflow_uid: str, experiment_uid: str
) -> dict[str, Any]:
    exp_dir = experiments_root / experiment_uid
    existing = read_json(exp_dir / "experiment.json")
    if existing:
        return existing
    return run_checked(
        [
            sys.executable,
            "scripts/create_model_experiment.py",
            "--model",
            model_id,
            "--workflow-uid",
            workflow_uid,
            "--experiment-uid",
            experiment_uid,
            "--experiments-root",
            str(experiments_root),
        ]
    )


def experiment_env(meta: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    caches = meta.get("caches") if isinstance(meta.get("caches"), dict) else {}
    env.update(
        {
            "SYMPOSIUM_EXPERIMENT_ID": str(meta.get("experiment_uid") or meta.get("experiment_id")),
            "SYMPOSIUM_CURRENT_RUN_ID": str(meta.get("experiment_uid") or meta.get("experiment_id")),
            "AUTO_VIDEO_EXPERIMENT_ROOT": str(meta.get("experiment_dir") or ""),
            "AUTO_VIDEO_RUNS_ROOT": str(meta.get("runs_dir") or ""),
            "TMPDIR": str(caches.get("tmp") or Path(meta["worktree"]) / "caches/tmp"),
            "TRITON_CACHE_DIR": str(caches.get("triton") or Path(meta["worktree"]) / "caches/triton"),
            "TORCH_EXTENSIONS_DIR": str(caches.get("torch_extensions") or Path(meta["worktree"]) / "caches/torch_extensions"),
            "CODEX_AUTORUN_MODEL": "gpt-5.6-sol",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def workflow_state(meta: dict[str, Any], workflow_uid: str) -> dict[str, Any]:
    path = Path(meta["worktree"]) / "state" / f"workflow-{workflow_uid}-state.json"
    return read_json(path)


def valid_delivery(meta: dict[str, Any], component: str) -> bool:
    delivery = read_json(Path(meta["worktree"]) / "DELIVERY.json")
    return (
        delivery.get("schema_version") == 2
        and delivery.get("status") == "complete"
        and delivery.get("component") == component
    )


def launch_workflow(
    meta: dict[str, Any],
    workflow_uid: str,
    log_path: Path,
    extra: list[str] | None = None,
    env_overrides: dict[str, str] | None = None,
    max_cycles: int = 400,
) -> subprocess.Popen[str]:
    command = [
        sys.executable,
        str(ROOT / "workflow" / workflow_uid / "workflow.py"),
        "run",
        "--experiment-json",
        str(Path(meta["experiment_dir"]) / "experiment.json"),
        "--experiment-uid",
        str(meta.get("experiment_uid") or meta.get("experiment_id")),
        "--max-cycles",
        str(max_cycles),
        "--autorun-model",
        "gpt-5.6-sol",
        "--model-id",
        str(meta.get("model_id") or meta.get("model_uid") or "sana_video"),
        "--baseline-manifest",
        str((meta.get("baseline") or {}).get("manifest") or "candidates/sana_video_baseline.toml"),
    ]
    command.extend(extra or [])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a")
    env = experiment_env(meta)
    env.update(env_overrides or {})
    return subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def run_workflow_once(meta: dict[str, Any], workflow_uid: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "workflow" / workflow_uid / "workflow.py"),
        "run",
        "--experiment-json",
        str(Path(meta["experiment_dir"]) / "experiment.json"),
        "--experiment-uid",
        str(meta.get("experiment_uid") or meta.get("experiment_id")),
        "--max-cycles",
        "400",
        "--autorun-model",
        "gpt-5.6-sol",
        "--model-id",
        str(meta.get("model_id") or meta.get("model_uid") or "sana_video"),
        "--baseline-manifest",
        str((meta.get("baseline") or {}).get("manifest") or "candidates/sana_video_baseline.toml"),
        "--once",
    ]
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=experiment_env(meta),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "baseline workflow step failed")
    return workflow_state(meta, workflow_uid)


def ensure_canonical_baseline(
    meta: dict[str, Any],
    chain_id: str,
    state_path: Path,
    poll_sec: float,
) -> Path:
    workflow_uid = "kernel_aw"
    while True:
        state = workflow_state(meta, workflow_uid)
        phase = str(state.get("phase") or "baseline_run")
        if phase not in {"baseline_run", "baseline_gate"}:
            if phase in {"blocked", "failed"}:
                raise RuntimeError(f"canonical baseline failed in phase {phase}: {state.get('terminal_reason')}")
            lock = read_json(Path(meta["worktree"]) / "BASELINE-LOCK.json")
            run_raw = str(lock.get("run_dir") or "")
            run_dir = Path(run_raw)
            if not run_dir.is_absolute():
                run_dir = Path(meta["worktree"]) / run_dir
            if not run_dir.is_dir():
                raise RuntimeError(f"canonical baseline lock has no valid run_dir: {run_raw}")
            return run_dir.resolve()

        state = run_workflow_once(meta, workflow_uid)
        write_json(
            state_path,
            {
                "schema_version": 1,
                "chain_id": chain_id,
                "updated_at_utc": utc_now(),
                "status": "canonical_baseline_running",
                "canonical_baseline": {
                    "experiment_uid": meta.get("experiment_uid") or meta.get("experiment_id"),
                    "phase": state.get("phase"),
                    "status": state.get("status"),
                    "job_id": state.get("baseline_job_id"),
                    "run_dir": state.get("baseline_run"),
                },
            },
        )
        if str(state.get("phase") or "") in {"baseline_run", "baseline_gate"}:
            time.sleep(max(min(poll_sec, 10.0), 2.0))


def process_snapshot(process: subprocess.Popen[str] | None) -> dict[str, Any]:
    if process is None:
        return {"pid": None, "returncode": None, "alive": False}
    return {"pid": process.pid, "returncode": process.poll(), "alive": process.poll() is None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default="sana",
        help="Model to optimize: bernini | sana | hunyuan (or a full model id under models/<id>/).",
    )
    parser.add_argument("--experiments-root", default="output/experiments")
    parser.add_argument("--seq", default="0001", help="4-digit experiment sequence for the default ids.")
    parser.add_argument("--chain-id", default=None, help="Default: <prefix>-opt")
    parser.add_argument("--kernel-id", default=None, help="Default: <prefix>-kernel_aw-<seq>")
    parser.add_argument("--pisa-id", default=None, help="Default: <prefix>-attention_pa-<seq>")
    parser.add_argument("--cache-id", default=None, help="Default: <prefix>-cache_ca-<seq>")
    parser.add_argument("--integrator-id", default=None, help="Default: <prefix>-integrator_ia-<seq>")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Per-component optimization round budget the agents are told to respect "
                             "(default 20; stated in the executor scope prompts). Also sets a workflow "
                             "max-cycles backstop. The integrator is not round-limited.")
    parser.add_argument("--poll-sec", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved plan (model, ids, chain-id) and exit without creating anything.")
    args = parser.parse_args()

    model_id = MODEL_ALIASES.get(args.model, args.model)
    if not (ROOT / "models" / model_id / "model.toml").exists():
        raise SystemExit(
            f"unknown model {args.model!r}: no models/{model_id}/model.toml "
            f"(known: {', '.join(sorted(set(MODEL_ALIASES)))})"
        )
    prefix = MODEL_PREFIX.get(model_id, model_id.split("_")[0])
    seq = args.seq
    chain_id = args.chain_id or f"{prefix}-opt"
    integrator_id = args.integrator_id or f"{prefix}-{INTEGRATOR_WORKFLOW}-{seq}"
    # The 20-round limit is the agent-facing instruction (executor scope prompts);
    # this is a workflow max-cycles (node-transition) backstop that comfortably
    # allows ~<rounds> rounds without letting a runaway agent go unbounded.
    component_max_cycles = max(args.rounds * 5, 60)

    experiments_root = Path(args.experiments_root)
    if not experiments_root.is_absolute():
        experiments_root = (ROOT / experiments_root).resolve()
    chain_dir = ROOT / "output" / "workflow_chains" / chain_id
    state_path = chain_dir / "CHAIN-STATE.json"
    ids = {
        "kernel": args.kernel_id or f"{prefix}-kernel_aw-{seq}",
        "pisa": args.pisa_id or f"{prefix}-attention_pa-{seq}",
        "cache": args.cache_id or f"{prefix}-cache_ca-{seq}",
    }
    print(f"[chain] model={model_id} prefix={prefix} chain_id={chain_id}", flush=True)
    print(f"[chain] experiments: {ids} integrator={integrator_id}", flush=True)
    print(f"[chain] rounds={args.rounds} (component max-cycles backstop={component_max_cycles})", flush=True)
    if args.dry_run:
        print(json.dumps(
            {"model_id": model_id, "prefix": prefix, "chain_id": chain_id,
             "experiments": ids, "integrator": integrator_id,
             "rounds": args.rounds, "component_max_cycles": component_max_cycles,
             "experiments_root": str(experiments_root)},
            indent=2,
        ))
        return 0
    metas: dict[str, dict[str, Any]] = {}
    processes: dict[str, subprocess.Popen[str] | None] = {}

    try:
        for component, experiment_uid in ids.items():
            workflow_uid = WORKFLOWS[component]
            metas[component] = create_experiment(experiments_root, model_id, workflow_uid, experiment_uid)
        canonical_baseline = ensure_canonical_baseline(
            metas["kernel"],
            chain_id,
            state_path,
            args.poll_sec,
        )
        for component, meta in metas.items():
            workflow_uid = WORKFLOWS[component]
            if valid_delivery(meta, component):
                processes[component] = None
                continue
            baseline_env = {} if component == "kernel" else {"CANONICAL_BASELINE_RUN": str(canonical_baseline)}
            processes[component] = launch_workflow(
                meta,
                workflow_uid,
                chain_dir / "logs" / f"{component}.log",
                env_overrides=baseline_env,
                max_cycles=component_max_cycles,
            )

        while True:
            component_state: dict[str, Any] = {}
            all_done = True
            failed: list[str] = []
            for component, meta in metas.items():
                workflow_uid = WORKFLOWS[component]
                state = workflow_state(meta, workflow_uid)
                delivered = valid_delivery(meta, component)
                process = processes.get(component)
                snapshot = process_snapshot(process)
                component_state[component] = {
                    "experiment_uid": ids[component],
                    "phase": state.get("phase"),
                    "status": state.get("status"),
                    "delivery_complete": delivered,
                    "process": snapshot,
                }
                all_done = all_done and delivered
                if not delivered and state.get("phase") in {"failed", "blocked"}:
                    failed.append(component)
                if not delivered and process is not None and process.poll() is not None:
                    failed.append(component)
            chain_state = {
                "schema_version": 1,
                "chain_id": chain_id,
                "model_id": model_id,
                "updated_at_utc": utc_now(),
                "status": "component_failed" if failed else ("components_complete" if all_done else "components_running"),
                "components": component_state,
                "canonical_baseline_run": str(canonical_baseline),
                "integrator_experiment_uid": integrator_id,
                "integrator_started": False,
                "failures": sorted(set(failed)),
            }
            write_json(state_path, chain_state)
            if failed:
                return 2
            if all_done:
                break
            time.sleep(max(args.poll_sec, 5.0))

        integrator = create_experiment(experiments_root, model_id, INTEGRATOR_WORKFLOW, integrator_id)
        donor_args = [
            "--kernel-delivery",
            str(Path(metas["kernel"]["worktree"]) / "DELIVERY.json"),
            "--pisa-delivery",
            str(Path(metas["pisa"]["worktree"]) / "DELIVERY.json"),
            "--cache-delivery",
            str(Path(metas["cache"]["worktree"]) / "DELIVERY.json"),
        ]
        integrator_process = launch_workflow(
            integrator,
            INTEGRATOR_WORKFLOW,
            chain_dir / "logs" / "integrator.log",
            donor_args,
            env_overrides={"CANONICAL_BASELINE_RUN": str(canonical_baseline)},
        )
        while True:
            state = workflow_state(integrator, INTEGRATOR_WORKFLOW)
            delivered = valid_delivery(integrator, "integrator")
            failed = state.get("phase") in {"failed", "blocked"}
            process = process_snapshot(integrator_process)
            chain_state.update(
                {
                    "updated_at_utc": utc_now(),
                    "status": "complete" if delivered else ("integrator_failed" if failed or (not process["alive"] and not delivered) else "integrator_running"),
                    "integrator_started": True,
                    "integrator": {
                        "experiment_uid": integrator_id,
                        "phase": state.get("phase"),
                        "status": state.get("status"),
                        "delivery_complete": delivered,
                        "process": process,
                    },
                }
            )
            write_json(state_path, chain_state)
            if delivered:
                return 0
            if chain_state["status"] == "integrator_failed":
                return 3
            time.sleep(max(args.poll_sec, 5.0))
    except Exception as exc:
        state = read_json(state_path)
        state.update(
            {
                "schema_version": 1,
                "chain_id": chain_id,
                "model_id": model_id,
                "updated_at_utc": utc_now(),
                "status": "orchestrator_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
