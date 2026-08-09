#!/usr/bin/env python3
"""Create a model-contract experiment from a minimal baseline copy.

This is the model x aspect/workflow x experiment materializer. It creates an
experiment directory, copies only the model contract's baseline runnable
closure into the experiment worktree, writes metadata, and leaves workflow
execution to a separate workflow runner.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # python < 3.11
    # draco and cs ship python 3.9. Refusing outright made this script unusable
    # on the two clusters where the A100 work happens; the tomli backport reads
    # the same dialect, so try it before giving up.
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "Reading TOML needs python 3.11+ (tomllib) or the tomli backport; "
            "this interpreter has neither"
        ) from exc


WORKFLOW_UID_RE = re.compile(r"^([a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*)_([A-Za-z]{2})$")
LEGACY_WORKFLOW_UID_RE = re.compile(r"^[A-Za-z]{2}$")
EXPERIMENT_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
GLOB_MAGIC = set("*?[")
DEFAULT_EXPERIMENTS_ROOT = "output/experiments"
DEFAULT_EXCLUDES = {
    ".git/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.pytest_cache/**",
}
EXPERIMENT_RUNTIME_SUPPORT_INCLUDES = [
    "search_space/**",
    "tools/symposium/**",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def rel_posix(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def validate_repo_relative_pattern(raw: str) -> str:
    if not raw or raw.strip() != raw:
        raise SystemExit(f"Invalid copy pattern: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise SystemExit(f"Copy pattern must be repo-relative and stay inside repo: {raw}")
    return raw


def validate_worktree_relative_path(raw: str, *, label: str) -> str:
    if not raw or raw.strip() != raw:
        raise SystemExit(f"Invalid {label}: {raw!r}")
    if has_glob_magic(raw):
        raise SystemExit(f"{label} must be a concrete worktree-relative path: {raw}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise SystemExit(f"{label} must stay inside the experiment worktree: {raw}")
    return raw.rstrip("/")


def has_glob_magic(pattern: str) -> bool:
    return any(ch in pattern for ch in GLOB_MAGIC)


def excluded(rel: str, patterns: list[str]) -> bool:
    if "/__pycache__/" in f"/{rel}/" or rel.endswith(".pyc"):
        return True
    return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)


def sorted_unique(paths: list[Path]) -> list[Path]:
    return sorted(set(paths), key=lambda path: path.as_posix())


def expand_include(root: Path, pattern: str) -> list[Path]:
    pattern = validate_repo_relative_pattern(pattern)
    if has_glob_magic(pattern):
        matches = [path for path in root.glob(pattern) if path.is_file()]
    else:
        path = root / pattern
        if path.is_file():
            matches = [path]
        elif path.is_dir():
            matches = [config for config in path.rglob("*") if config.is_file()]
        else:
            matches = []
    if not matches:
        raise SystemExit(f"Copy include matched no files: {pattern}")
    return matches


def contract_path(root: Path, model: str, explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = root / path
        return path.resolve()
    return (root / "models" / model / "model.toml").resolve()


def load_model_contract(root: Path, model: str, explicit: str | None = None) -> dict[str, Any]:
    path = contract_path(root, model, explicit)
    if not path.exists():
        raise SystemExit(f"Model contract does not exist: {path}")
    data = load_toml(path)
    if data.get("id") != model:
        raise SystemExit(f"Model contract id mismatch: expected {model}, got {data.get('id')!r}")
    data["_contract_path"] = rel_posix(root, path)
    return data


def copy_patterns(contract: dict[str, Any]) -> tuple[list[str], list[str]]:
    baseline = contract.get("baseline") if isinstance(contract.get("baseline"), dict) else {}
    copy = baseline.get("copy") if isinstance(baseline.get("copy"), dict) else {}
    includes = [str(item) for item in copy.get("include", [])]
    excludes = [str(item) for item in copy.get("exclude", [])]
    if not includes:
        raise SystemExit("Model contract baseline.copy.include is empty")
    return includes, sorted(set(excludes).union(DEFAULT_EXCLUDES))


def external_copy_specs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = contract.get("baseline") if isinstance(contract.get("baseline"), dict) else {}
    raw_specs = baseline.get("external_copy", [])
    if isinstance(raw_specs, dict):
        raw_specs = [raw_specs]
    if not isinstance(raw_specs, list):
        raise SystemExit("Model contract baseline.external_copy must be a table array")
    specs: list[dict[str, Any]] = []
    for index, raw_spec in enumerate(raw_specs):
        if not isinstance(raw_spec, dict):
            raise SystemExit(f"baseline.external_copy[{index}] must be a table")
        source = str(raw_spec.get("source") or "").strip()
        dest = str(raw_spec.get("dest") or "").strip()
        if not source:
            raise SystemExit(f"baseline.external_copy[{index}] is missing source")
        if not dest:
            raise SystemExit(f"baseline.external_copy[{index}] is missing dest")
        excludes = [str(item) for item in raw_spec.get("exclude", [])]
        specs.append(
            {
                "name": str(raw_spec.get("name") or f"external_copy_{index}"),
                "source": source,
                "dest": validate_worktree_relative_path(
                    dest,
                    label=f"baseline.external_copy[{index}].dest",
                ),
                "reason": str(raw_spec.get("reason") or ""),
                "exclude": sorted(set(excludes).union(DEFAULT_EXCLUDES)),
            }
        )
    return specs


def overlay_copy_specs(root: Path, contract: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = contract.get("baseline") if isinstance(contract.get("baseline"), dict) else {}
    raw_specs = baseline.get("overlay_copy", [])
    if isinstance(raw_specs, dict):
        raw_specs = [raw_specs]
    if not isinstance(raw_specs, list):
        raise SystemExit("baseline.overlay_copy must be a table array")
    specs: list[dict[str, Any]] = []
    for index, raw_spec in enumerate(raw_specs):
        if not isinstance(raw_spec, dict):
            raise SystemExit(f"baseline.overlay_copy[{index}] must be a table")
        source = validate_repo_relative_pattern(str(raw_spec.get("source") or "").strip())
        if has_glob_magic(source):
            raise SystemExit(f"baseline.overlay_copy[{index}].source must be a concrete file: {source}")
        source_path = (root / source).resolve()
        if not source_path.is_file():
            raise SystemExit(f"Baseline overlay source does not exist: {source_path}")
        dest = validate_worktree_relative_path(
            str(raw_spec.get("dest") or "").strip(),
            label=f"baseline.overlay_copy[{index}].dest",
        )
        specs.append(
            {
                "name": str(raw_spec.get("name") or f"overlay_copy_{index}"),
                "source": source,
                "source_path": source_path,
                "dest": dest,
                "reason": str(raw_spec.get("reason") or ""),
            }
        )
    return specs


def baseline_copy_plan(root: Path, contract: dict[str, Any]) -> list[str]:
    includes, excludes = copy_patterns(contract)
    files: list[Path] = []
    for pattern in includes:
        files.extend(expand_include(root, pattern))
    rels = []
    for path in sorted_unique(files):
        rel = rel_posix(root, path)
        if not excluded(rel, excludes):
            rels.append(rel)
    if not rels:
        raise SystemExit("Copy plan is empty after exclusions")
    return rels


def experiment_runtime_support_copy_plan(root: Path) -> list[str]:
    files: list[Path] = []
    for pattern in EXPERIMENT_RUNTIME_SUPPORT_INCLUDES:
        files.extend(expand_include(root, pattern))
    rels = []
    for path in sorted_unique(files):
        rel = rel_posix(root, path)
        if not excluded(rel, sorted(DEFAULT_EXCLUDES)):
            rels.append(rel)
    if not rels:
        raise SystemExit("Experiment runtime support copy plan is empty")
    return rels


def resolve_external_source(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def expand_external_source(source: Path, excludes: list[str]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    if source.is_file():
        rel = source.name
        if not excluded(rel, excludes):
            files.append((rel, source))
        return files
    if not source.is_dir():
        raise SystemExit(f"External copy source does not exist: {source}")
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source).as_posix()
        if not excluded(rel, excludes):
            files.append((rel, path))
    return files


def external_copy_plan(root: Path, contract: dict[str, Any]) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for spec in external_copy_specs(contract):
        source = resolve_external_source(root, spec["source"])
        files = expand_external_source(source, spec["exclude"])
        if not files:
            raise SystemExit(f"External copy source produced no files: {source}")
        copied_paths = [f"{spec['dest']}/{rel}" for rel, _ in files]
        plans.append(
            {
                **spec,
                "source_path": source,
                "files": files,
                "copied_paths": copied_paths,
            }
        )
    return plans


def validate_workflow_uid(workflow_uid: str, allow_legacy: bool = False) -> tuple[str, str]:
    match = WORKFLOW_UID_RE.fullmatch(workflow_uid)
    if match:
        return match.group(1), match.group(2)
    if allow_legacy and LEGACY_WORKFLOW_UID_RE.fullmatch(workflow_uid):
        return "legacy", workflow_uid
    raise SystemExit(
        "Workflow uid must be <aspect>_<two_letter_code>, for example kernel_aw; "
        f"got {workflow_uid!r}"
    )


def validate_experiment_uid(experiment_uid: str, workflow_uid: str) -> None:
    marker = f"-{workflow_uid}-"
    if marker not in experiment_uid:
        raise SystemExit(
            "Experiment uid must be <model_or_task_code>-<workflow_uid>-<0000>; "
            f"got {experiment_uid!r} for workflow {workflow_uid!r}"
        )
    prefix, sequence = experiment_uid.rsplit(marker, 1)
    if not EXPERIMENT_PREFIX_RE.fullmatch(prefix):
        raise SystemExit(f"Invalid experiment uid prefix: {prefix!r}")
    if not re.fullmatch(r"[0-9]{4}", sequence):
        raise SystemExit(f"Experiment uid must end with a four-digit sequence: {experiment_uid!r}")


def resolve_output_root(root: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def ensure_empty_experiment(exp_dir: Path) -> None:
    if exp_dir.exists():
        raise SystemExit(f"Experiment already exists; refusing to overwrite: {exp_dir}")


def copy_files(root: Path, worktree: Path, rels: list[str]) -> None:
    for rel in rels:
        src = root / rel
        dst = worktree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_external_files(worktree: Path, plans: list[dict[str, Any]]) -> None:
    for plan in plans:
        dest_root = worktree / plan["dest"]
        for rel, src in plan["files"]:
            dst = dest_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def copy_overlay_files(worktree: Path, specs: list[dict[str, Any]]) -> None:
    for spec in specs:
        dst = worktree / spec["dest"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec["source_path"], dst)


def summarize_external_copy_plans(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for plan in plans:
        summaries.append(
            {
                "name": plan["name"],
                "source": str(plan["source_path"]),
                "dest": plan["dest"],
                "reason": plan["reason"],
                "copied_path_count": len(plan["files"]),
                "copied_paths": plan["copied_paths"],
            }
        )
    return summaries


def create_standard_dirs(worktree: Path, contract: dict[str, Any]) -> dict[str, str]:
    defaults = contract.get("experiment_defaults") if isinstance(contract.get("experiment_defaults"), dict) else {}
    state_dir = str(defaults.get("state_dir") or "state")
    runs_dir = str(defaults.get("runs_dir") or "runs")
    logs_dir = str(defaults.get("logs_dir") or "logs")
    caches = [str(item) for item in defaults.get("caches", ["tmp", "triton", "torch_extensions"])]
    for rel in [state_dir, runs_dir, logs_dir, *[f"caches/{name}" for name in caches]]:
        (worktree / rel).mkdir(parents=True, exist_ok=True)
    return {
        "state_dir": state_dir,
        "runs_dir": runs_dir,
        "logs_dir": logs_dir,
        "caches_dir": "caches",
    }


def write_goal_seed(worktree: Path, metadata: dict[str, Any]) -> None:
    goal_dir = Path(metadata["goal_dir"])
    goal_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "schema_version": 1,
        "experiment_uid": metadata["experiment_uid"],
        "model_uid": metadata["model_uid"],
        "workflow_uid": metadata["workflow_uid"],
        "aspect": metadata["aspect"],
        "baseline": metadata["baseline"],
        "model_contract": metadata["model_contract"],
    }
    write_json(goal_dir / "context.json", context)
    goal = f"""# {metadata["experiment_uid"]}

Run workflow `{metadata["workflow_uid"]}` for model `{metadata["model_uid"]}`
under aspect `{metadata["aspect"]}` using the experiment-local baseline copy.

The model baseline has already been materialized into this worktree from
`{metadata["model_contract"]}`. Workflow-specific executor/reviewer prompts may
extend this goal, but runtime mutations must stay inside this experiment.
"""
    (goal_dir / "goal.md").write_text(goal)


def write_worktree_readme(worktree: Path, metadata: dict[str, Any]) -> None:
    text = f"""# Experiment {metadata["experiment_uid"]}

This worktree was created from model contract
`{metadata["model_contract"]}` using minimal baseline copy mode.

It contains the baseline runnable closure for model `{metadata["model_uid"]}`
and is owned by workflow `{metadata["workflow_uid"]}`. It intentionally does
not contain historical runs, generated config, or other workflow search
spaces unless a workflow creates them inside this experiment.
"""
    (worktree / "README.md").write_text(text)


def build_metadata(
    *,
    root: Path,
    contract: dict[str, Any],
    experiment_uid: str,
    workflow_uid: str,
    aspect: str,
    workflow_code: str,
    exp_dir: Path,
    baseline_copied_paths: list[str],
    runtime_support_copied_paths: list[str],
    copied_paths: list[str],
    external_copies: list[dict[str, Any]],
    overlay_copies: list[dict[str, Any]],
    dirs: dict[str, str],
) -> dict[str, Any]:
    baseline = contract.get("baseline") if isinstance(contract.get("baseline"), dict) else {}
    references = baseline.get("reference_only", []) if isinstance(baseline.get("reference_only"), list) else []
    return {
        "schema_version": 2,
        "created_at_utc": utc_now(),
        "status": "created",
        "experiment_uid": experiment_uid,
        "experiment_id": experiment_uid,
        "model_uid": contract["id"],
        "model_id": contract["id"],
        "model_contract": contract["_contract_path"],
        "legacy_model_profile": contract.get("legacy_profile"),
        "workflow_uid": workflow_uid,
        "aspect": aspect,
        "workflow_code": workflow_code,
        "coordinator_root": str(root),
        "experiment_dir": str(exp_dir),
        "worktree": str(exp_dir / "worktree"),
        "goal_dir": str(exp_dir / "worktree" / "goals" / workflow_uid),
        "state_dir": str(exp_dir / "worktree" / dirs["state_dir"]),
        "runs_dir": str(exp_dir / "worktree" / dirs["runs_dir"]),
        "logs_dir": str(exp_dir / "worktree" / dirs["logs_dir"]),
        "caches": {
            "tmp": str(exp_dir / "worktree" / "caches" / "tmp"),
            "triton": str(exp_dir / "worktree" / "caches" / "triton"),
            "torch_extensions": str(exp_dir / "worktree" / "caches" / "torch_extensions"),
        },
        "baseline": {
            "manifest": baseline.get("manifest"),
            "runtime_root": baseline.get("runtime_root"),
            "eval_profile": baseline.get("eval_profile"),
            "copy_mode": "allowlist_minimal_runnable_closure",
            "copied_path_count": len(baseline_copied_paths),
            "external_copy_count": sum(item["copied_path_count"] for item in external_copies),
            "overlay_copy_count": len(overlay_copies),
        },
        "experiment_runtime_support": {
            "copy_mode": "allowlist_runtime_support_closure",
            "copied_path_count": len(runtime_support_copied_paths),
            "copied_paths": runtime_support_copied_paths,
        },
        "reference_only": references,
        "external_sources": external_copies,
        "baseline_overlays": [
            {key: item[key] for key in ("name", "source", "dest", "reason")}
            for item in overlay_copies
        ],
        "copied_paths": copied_paths,
        "external_copied_paths": [
            path for item in external_copies for path in item["copied_paths"]
        ],
        "isolation": {
            "model_baseline_minimal_copy": True,
            "editable_external_source_copied": bool(external_copies),
            "refuse_existing_experiment_id": True,
            "workflow_owns_experiment": True,
            "workflow_runtime_imports_across_workflows": False,
            "large_external_state_reference_only": True,
        },
    }


def create_experiment(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    aspect, workflow_code = validate_workflow_uid(args.workflow_uid, args.allow_legacy_workflow_uid)
    validate_experiment_uid(args.experiment_uid, args.workflow_uid)
    contract = load_model_contract(root, args.model, args.model_contract)
    baseline_copied_paths = baseline_copy_plan(root, contract)
    runtime_support_copied_paths = experiment_runtime_support_copy_plan(root)
    copied_paths = sorted(set(baseline_copied_paths + runtime_support_copied_paths))
    external_plans = external_copy_plan(root, contract)
    external_copies = summarize_external_copy_plans(external_plans)
    overlay_copies = overlay_copy_specs(root, contract)
    output_root = resolve_output_root(root, args.experiments_root)
    exp_dir = output_root / args.experiment_uid
    worktree = exp_dir / "worktree"

    if args.dry_run:
        return {
            "dry_run": True,
            "experiment_uid": args.experiment_uid,
            "model_uid": args.model,
            "workflow_uid": args.workflow_uid,
            "aspect": aspect,
            "model_contract": contract["_contract_path"],
            "experiment_dir": str(exp_dir),
            "baseline_copied_path_count": len(baseline_copied_paths),
            "runtime_support_copied_path_count": len(runtime_support_copied_paths),
            "copied_path_count": len(copied_paths),
            "external_copy_count": sum(item["copied_path_count"] for item in external_copies),
            "overlay_copy_count": len(overlay_copies),
            "copied_paths": copied_paths,
            "runtime_support_copied_paths": runtime_support_copied_paths,
            "external_sources": external_copies,
            "baseline_overlays": [
                {key: item[key] for key in ("name", "source", "dest", "reason")}
                for item in overlay_copies
            ],
        }

    ensure_empty_experiment(exp_dir)
    worktree.mkdir(parents=True)
    copy_files(root, worktree, copied_paths)
    copy_external_files(worktree, external_plans)
    copy_overlay_files(worktree, overlay_copies)
    dirs = create_standard_dirs(worktree, contract)
    metadata = build_metadata(
        root=root,
        contract=contract,
        experiment_uid=args.experiment_uid,
        workflow_uid=args.workflow_uid,
        aspect=aspect,
        workflow_code=workflow_code,
        exp_dir=exp_dir,
        baseline_copied_paths=baseline_copied_paths,
        runtime_support_copied_paths=runtime_support_copied_paths,
        copied_paths=copied_paths,
        external_copies=external_copies,
        overlay_copies=overlay_copies,
        dirs=dirs,
    )
    write_json(exp_dir / "experiment.json", metadata)
    write_json(worktree / "state" / "baseline_copy_manifest.json", {
        "created_at_utc": metadata["created_at_utc"],
        "model_contract": metadata["model_contract"],
        "copy_mode": metadata["baseline"]["copy_mode"],
        "baseline_copied_paths": baseline_copied_paths,
        "runtime_support_copied_paths": runtime_support_copied_paths,
        "copied_paths": copied_paths,
        "external_sources": external_copies,
        "external_copied_paths": metadata["external_copied_paths"],
        "baseline_overlays": metadata["baseline_overlays"],
    })
    write_goal_seed(worktree, metadata)
    write_worktree_readme(worktree, metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model uid, e.g. sana_video")
    parser.add_argument("--workflow-uid", required=True, help="Workflow uid, e.g. kernel_aw")
    parser.add_argument("--experiment-uid", required=True, help="Experiment uid, e.g. sana-kernel_aw-0001")
    parser.add_argument("--model-contract", help="Optional explicit model contract TOML")
    parser.add_argument("--experiments-root", default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--allow-legacy-workflow-uid", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = create_experiment(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
