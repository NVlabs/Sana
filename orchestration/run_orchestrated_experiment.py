#!/usr/bin/env python3
"""Bootstrap + heartbeat watchdog for the master-agent-orchestrated experiment.

The ONLY deterministic Python scheduling that remains. It:
  1. freezes the baseline ONCE (reuse the model profile's recorded [baseline]
     run, or launch it once) into a read-only BASELINE.json for the whole run;
  2. assembles the master orchestrator prompt and launches ONE master agent;
  3. runs a thin heartbeat watchdog that restarts the master if it dies, until
     the master writes the integrated delivery.

The master agent does everything else: spawn the configured executor sub-agents,
poll, independently verify (anti-fabrication), resume on bad delivery, and
integrate. Heavy workflow state machines remain separate; this runner consumes
their workflow-owned technique scopes.

    python orchestration/run_orchestrated_experiment.py --model bernini
    python orchestration/run_orchestrated_experiment.py --model bernini --dry-run
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
LITE = ROOT / "orchestration"


def load_technique_registry() -> tuple[dict[str, dict[str, str]], tuple[str, ...]]:
    path = LITE / "techniques.toml"
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    techniques = raw.get("techniques")
    default_order = raw.get("default_order")
    if not isinstance(techniques, dict) or not techniques:
        raise RuntimeError(f"invalid technique registry: {path}")
    if not isinstance(default_order, list) or not default_order:
        raise RuntimeError(f"technique registry has no default_order: {path}")
    normalized: dict[str, dict[str, str]] = {}
    for name, spec in techniques.items():
        if not isinstance(spec, dict):
            raise RuntimeError(f"invalid technique entry {name!r}: {path}")
        required = ("workflow_uid", "scope", "correctness")
        if any(not isinstance(spec.get(key), str) or not spec[key] for key in required):
            raise RuntimeError(f"incomplete technique entry {name!r}: {path}")
        normalized[str(name)] = {key: str(spec[key]) for key in required}
    defaults = tuple(str(item) for item in default_order)
    workflow_uids = [spec["workflow_uid"] for spec in normalized.values()]
    if len(workflow_uids) != len(set(workflow_uids)):
        raise RuntimeError(f"technique workflow_uid values must be unique: {path}")
    for name, spec in normalized.items():
        expected_prefix = f"workflow/{spec['workflow_uid']}/"
        if not spec["scope"].startswith(expected_prefix):
            raise RuntimeError(
                f"technique {name!r} scope must be owned by {expected_prefix}: {path}"
            )
    unknown_defaults = [name for name in defaults if name not in normalized]
    if unknown_defaults or len(defaults) != len(set(defaults)):
        raise RuntimeError(f"invalid default_order in {path}: {default_order!r}")
    return normalized, defaults


TECHNIQUES, DEFAULT_TECHNIQUES = load_technique_registry()
MODEL_ALIASES = {"sana": "sana_video", "sana_video": "sana_video", "bernini": "bernini",
                 "hunyuan": "hunyuan_diffusers", "hunyuan_diffusers": "hunyuan_diffusers",
                 "wan22_ti2v_5b": "wan22_ti2v_5b", "wan5b": "wan22_ti2v_5b",
                 "wan22_t2v_a14b": "wan22_t2v_a14b", "wan14b": "wan22_t2v_a14b"}
# Distinct prefixes per model — used for exp_root + master session name. wan22
# models MUST NOT both fall back to "wan22" (split("_")[0]) or two parallel
# masters would share an exp_root + session name and clobber each other.
MODEL_PREFIX = {"sana_video": "sana", "bernini": "bernini", "hunyuan_diffusers": "hunyuan",
                "wan22_ti2v_5b": "wan5b", "wan22_t2v_a14b": "wan14b"}


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def freeze_baseline(model_id: str, out_path: Path, override_run_dir: str | None) -> dict:
    """Reuse the recorded canonical baseline (or an override); freeze to a file."""
    profile = tomllib.loads((ROOT / "models" / f"{model_id}.toml").read_text())
    b = profile.get("baseline", {})
    official = profile.get("official_config", {})
    slurm = profile.get("slurm", {})
    orchestration = profile.get("orchestration", {})
    world_size = orchestration.get("inference_world_size")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
        world_size = official.get("num_gpus")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
        nodes = slurm.get("nodes")
        gpus_per_node = slurm.get("gpus_per_node")
        if all(isinstance(value, int) and not isinstance(value, bool) and value > 0
               for value in (nodes, gpus_per_node)):
            world_size = nodes * gpus_per_node
        else:
            world_size = None
    world_size_source = "model_profile"
    observed_world_size = False
    if override_run_dir:
        run_dir = Path(override_run_dir)
    elif b.get("run_id"):
        run_dir = ROOT / "runs" / str(b["run_id"])
    else:
        raise SystemExit(
            f"no recorded [baseline] for {model_id}. Run it once first:\n"
            f"  python scripts/launch_config.py config/{model_id}_baseline.toml --mode sbatch --confirm-submit\n"
            f"then re-run with --baseline-run-dir runs/<id>, or record it in models/{model_id}.toml [baseline]."
        )
    run_dir = run_dir.resolve()
    frames = run_dir / "outputs" / "frames"
    if not run_dir.is_dir():
        raise SystemExit(f"baseline run dir not found: {run_dir}")
    benchmark_path = run_dir / "outputs" / "benchmark.json"
    try:
        run_benchmark = json.loads(benchmark_path.read_text())
    except (OSError, json.JSONDecodeError):
        run_benchmark = {}
    if not isinstance(run_benchmark, dict):
        run_benchmark = {}
    run_total = run_benchmark.get("total_s")
    if (
        not isinstance(run_total, (int, float))
        or isinstance(run_total, bool)
        or run_total <= 0
    ):
        run_total = b.get("total_s") if not override_run_dir else None
    if (
        not isinstance(run_total, (int, float))
        or isinstance(run_total, bool)
        or run_total <= 0
    ):
        source = "override baseline benchmark" if override_run_dir else "recorded baseline/profile"
        raise SystemExit(f"{source} has no positive total_s: {benchmark_path}")
    run_timing_scope = run_benchmark.get("timing_scope")
    if not isinstance(run_timing_scope, str) or not run_timing_scope.strip():
        run_timing_scope = b.get("timing_scope") if not override_run_dir else None
    if not isinstance(run_timing_scope, str) or not run_timing_scope.strip():
        source = "override baseline benchmark" if override_run_dir else "recorded baseline/profile"
        raise SystemExit(f"{source} has no timing_scope: {benchmark_path}")

    run_denoise = run_benchmark.get("denoise_s")
    if not isinstance(run_denoise, (int, float)) or isinstance(run_denoise, bool):
        run_denoise = b.get("denoise_s") if not override_run_dir else None
    peak_memory = run_benchmark.get("max_device_memory_used_mib")
    if not isinstance(peak_memory, (int, float)) or isinstance(peak_memory, bool):
        memory = run_benchmark.get("memory")
        peak_memory = (
            memory.get("max_device_memory_used_mib")
            if isinstance(memory, dict)
            else None
        )
    if not isinstance(peak_memory, (int, float)) or isinstance(peak_memory, bool):
        peak_memory = None
    for artifact_path in (
        run_dir / "outputs" / "run_config.json",
        run_dir / "outputs" / "benchmark.json",
        run_dir / "metadata.json",
    ):
        try:
            artifact = json.loads(artifact_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        config = artifact.get("config") if isinstance(artifact.get("config"), dict) else artifact
        for key in ("world_size", "num_gpus", "nproc"):
            value = config.get(key) if isinstance(config, dict) else None
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                world_size = value
                world_size_source = f"run_artifact:{artifact_path.name}:{key}"
                observed_world_size = True
                break
        else:
            continue
        break
    if override_run_dir and not observed_world_size:
        raise SystemExit(
            "override baseline run does not record an observed world_size/num_gpus/nproc "
            f"in run_config.json, benchmark.json, or metadata.json: {run_dir}"
        )
    allocated_gpus = None
    nodes = slurm.get("nodes")
    gpus_per_node = slurm.get("gpus_per_node")
    if all(isinstance(value, int) and not isinstance(value, bool) and value > 0
           for value in (nodes, gpus_per_node)):
        allocated_gpus = nodes * gpus_per_node
    baseline = {
        "model_id": model_id,
        # An explicit --baseline-run-dir is authoritative and must never be
        # combined with stale profile timings.  Legacy recorded runs may fall
        # back to their profile only when an old benchmark omitted a field.
        "total_s": float(run_total),
        "denoise_s": float(run_denoise) if run_denoise is not None else None,
        "timing_scope": run_timing_scope,
        "peak_memory_mib": float(peak_memory) if peak_memory is not None else None,
        "run_dir": str(run_dir),
        "baseline_frames": str(frames),
        "baseline_video": str(run_dir / "outputs" / "out.mp4"),
        "world_size": world_size,
        "world_size_source": world_size_source,
        "resource_envelope": {
            "nodes": slurm.get("nodes"),
            "gpus_per_node": slurm.get("gpus_per_node"),
            "world_size": world_size,
            "allocated_gpus": allocated_gpus,
            "hardware": b.get("hardware"),
        },
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


def clean_stale_arg0() -> int:
    """Remove empty stale codex arg0 self-extraction dirs before a codex launch.

    codex re-execs its ~248MB musl binary via a temp dir under ~/.codex/tmp/arg0
    and is supposed to clean them up, but its own cleanup chokes on a non-empty
    dir and bails, so EMPTY leftovers accumulate (2000+ over weeks). On startup
    codex stat()s every one on NFS; once that scan exceeds its 5s health probe,
    the session launch flakes and the pane dies ~30s in. Empty dirs are safe to
    drop (a live extraction dir is non-empty, so rmdir refuses it). Best-effort,
    never fatal.
    """
    arg0 = Path.home() / ".codex" / "tmp" / "arg0"
    if not arg0.is_dir():
        return 0
    removed = 0
    try:
        for d in arg0.iterdir():
            if d.is_dir() and d.name.startswith("codex-arg0"):
                try:
                    d.rmdir()  # only succeeds when empty -> never touches a live extraction
                    removed += 1
                except OSError:
                    pass
    except OSError:
        return 0
    if removed:
        print(f"[orchestrate] cleaned {removed} stale codex arg0 dirs (~/.codex/tmp/arg0)", flush=True)
    return removed


def start_master(goal_dir: Path, name: str, force: bool = False) -> None:
    clean_stale_arg0()  # keep the arg0 pile from blowing codex's 5s health probe
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
    ap.add_argument(
        "--techs",
        default=None,
        help=(
            "comma-separated executor techniques to run "
            f"(available: {', '.join(TECHNIQUES)}; default comes from the model "
            "profile or the registry)"
        ),
    )
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
    profile = tomllib.loads((ROOT / "models" / f"{model_id}.toml").read_text())
    workflow_defaults = profile.get("orchestration", {})
    configured_defaults = (
        workflow_defaults.get("default_techniques")
        if isinstance(workflow_defaults, dict)
        else None
    )
    if args.techs is None:
        selected = configured_defaults if isinstance(configured_defaults, list) else DEFAULT_TECHNIQUES
        techs = [str(tech).strip() for tech in selected if str(tech).strip()]
        tech_selection = "model_profile" if isinstance(configured_defaults, list) else "registry_default"
    else:
        techs = [t.strip() for t in args.techs.split(",") if t.strip()]
        tech_selection = "cli"
    if not techs:
        raise SystemExit("--techs must list at least one technique")
    unknown_techs = [tech for tech in techs if tech not in TECHNIQUES]
    if unknown_techs:
        raise SystemExit(
            f"unknown --techs value(s): {', '.join(unknown_techs)}; "
            f"available: {', '.join(TECHNIQUES)}"
        )
    if len(techs) != len(set(techs)):
        raise SystemExit("--techs must not contain duplicates")
    techs_fmt = ", ".join(f"`{t}`" for t in techs)
    tech_specs = "\n".join(
        "- "
        f"`{tech}` -> workflow_uid `{TECHNIQUES[tech]['workflow_uid']}`; "
        f"correctness `{TECHNIQUES[tech]['correctness']}`"
        for tech in techs
    )
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
                          "master_name": master_name, "seq": args.seq, "techs": techs,
                          "tech_selection": tech_selection,
                          "technique_specs": {tech: TECHNIQUES[tech] for tech in techs}}, indent=2))
        return 0

    # 1) freeze baseline once
    baseline = freeze_baseline(model_id, baseline_json, args.baseline_run_dir)
    if "topology" in techs and (
        not isinstance(baseline.get("world_size"), int) or baseline["world_size"] < 2
    ):
        raise SystemExit(
            "topology executor requires a frozen multi-rank baseline; "
            f"observed world_size={baseline.get('world_size')!r}. "
            "Remove topology from --techs or provide a real multi-rank baseline run."
        )
    print(f"[orchestrate] frozen baseline: total_s={baseline['total_s']} run_dir={baseline['run_dir']}", flush=True)

    # 2) assemble master prompt + launch master
    master_goal.mkdir(parents=True, exist_ok=True)
    tpl = (LITE / "prompts" / "master.md").read_text()
    replacements = {
        "{MODEL_ID}": model_id,
        "{ROOT}": str(ROOT),
        "{BASELINE_JSON}": str(baseline_json),
        "{SEQ}": args.seq,
        "{PREFIX}": prefix,
        "{INTEGRATED_DELIVERY}": str(integrated),
        "{TECHS}": techs_fmt,
        "{TECH_SPECS}": tech_specs,
    }
    for k, v in replacements.items():
        tpl = tpl.replace(k, v)
    unresolved = [placeholder for placeholder in replacements if placeholder in tpl]
    if unresolved:
        raise RuntimeError(f"unresolved master prompt placeholders: {', '.join(unresolved)}")
    (master_goal / "goal.md").write_text(tpl)
    # codex_goal_session requires BOTH goal.md and context.json in the goal dir.
    (master_goal / "context.json").write_text(json.dumps({
        "schema_version": 1, "goal_id": master_name, "experiment_uid": master_name,
        "created_by": "run_orchestrated_experiment", "target_agent": "codex",
        "mode": "master-orchestrator", "model_uid": model_id, "role": "master",
        "techniques": techs, "technique_selection": tech_selection,
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
            "baseline_json": str(baseline_json), "techniques": techs,
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
